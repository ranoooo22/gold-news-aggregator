"""News pipeline for the gold terminal.

Every run (triggered every 15 min):
  1. Fetch RSS feeds (publisher feeds + Google News searches).
  2. Keep articles that mention gold-relevant topics (keyword filter).
  3. Group articles about the same event into clusters (sentence embeddings).
  4. Have Gemini write a strictly factual summary for new/updated clusters.
  5. Write data.json for the website.
It also saves a snapshot of OANDA's public order book (oanda_book.json),
which OANDA itself only refreshes every 20 minutes.

Market prices and exchange order books are NOT fetched here: the website
loads them live in the browser.
"""
import calendar
import hashlib
import html
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import quote_plus

import feedparser
import numpy as np
import requests

DATA_FILE = "data.json"
OANDA_FILE = "oanda_book.json"

RETENTION_HOURS = 48        # clusters older than this (since last article) are dropped
JOIN_WINDOW_HOURS = 12      # an article may only join a cluster updated within this window
SIM_THRESHOLD = 0.62        # cosine similarity to a cluster's centroid needed to join it
MAX_CLUSTER_SIZE = 40       # beyond this a story continues in a new cluster
BATCH_SIZE = 6              # clusters per Gemini call
MAX_BATCHES_PER_RUN = 4
ARTICLES_PER_PROMPT = 10    # first report + newest 9

GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0"}
GOOGLE_NEWS_GAP_S = 1.5        # pause between Google News searches (it rate-limits bursts)
RATE_LIMIT_RETRY_S = 8         # wait before the single retry after an HTTP 429


def google_news(query):
    q = quote_plus(f"{query} when:1d")
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


# Publisher feeds, plus Google News searches. Google News brings in Reuters, AP,
# Bloomberg, FT, Kitco etc. and outlets whose own feeds block bots (Times of Israel, Arab News).
FEEDS = {
    "CNBC Economy": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "Investing.com": "https://www.investing.com/rss/news_11.rss",
    "CNBC Technology": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19854910",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "Middle East Eye": "https://www.middleeasteye.net/rss",
    "Tehran Times": "https://www.tehrantimes.com/rss",
    "Mehr News": "https://en.mehrnews.com/rss",
    "GN Gold": google_news('gold price OR bullion OR "precious metals"'),
    "GN Fed & Rates": google_news('"Federal Reserve" OR FOMC OR Powell OR "Treasury yields"'),
    "GN US Data": google_news('"consumer prices" OR "nonfarm payrolls" OR "jobs report" OR "PCE inflation" OR "US GDP"'),
    "GN Central Banks": google_news('"central bank" gold OR ECB OR "Bank of Japan" OR PBOC'),
    "GN Oil": google_news('"oil prices" OR OPEC OR Brent crude'),
    "GN Middle East": google_news('Iran OR Israel OR Hormuz OR Houthi OR "Red Sea"'),
    "GN Times of Israel": google_news("site:timesofisrael.com"),
    "GN Arab News": google_news("site:arabnews.com"),
    "GN AI & Tech": google_news('Nvidia OR "AI chips" OR semiconductors OR OpenAI OR "AI spending" OR "data centers"'),
    "GN Stock Markets": google_news('"stock market" OR Nasdaq OR "S&P 500" OR "Wall Street"'),
}

# Google News returns hundreds of publishers, many of them SEO sites, broker promos and
# price-forecast blogs. Only these publishers are accepted from Google News results
# (matched case-insensitively, ignoring a trailing ".com"). Direct feeds above are always accepted.
GOOGLE_NEWS_PUBLISHERS = {
    # wires & business press
    "reuters", "bloomberg", "associated press", "ap news", "financial times", "wsj",
    "the wall street journal", "cnbc", "marketwatch", "barron's", "the economist", "nikkei asia",
    "kitco", "mining", "mining weekly", "world gold council", "oilprice", "s&p global",
    "argus media", "gcaptain", "lloyd's list", "tradewinds news", "investing", "yahoo finance",
    "morningstar", "axios", "politico", "fortune", "business insider", "investor's business daily",
    # tech press
    "techcrunch", "the verge", "the information", "wired", "ars technica", "semafor",
    # general news
    "bbc", "the guardian", "the new york times", "the washington post", "npr", "cnn", "nbc news",
    "cbs news", "abc news", "fox news", "france 24", "dw", "euronews", "south china morning post",
    "the independent", "the telegraph", "time", "newsweek",
    # Middle East
    "the times of israel", "the jerusalem post", "haaretz", "i24news", "arab news", "al jazeera",
    "al-monitor", "middle east eye", "the national", "thenationalnews", "gulf news", "iran international",
    "the arab weekly", "the media line", "anadolu ajansı", "anadolu agency", "daily sabah",
}

# SEO filler and speculation, not news: dropped from every feed.
FILLER_RE = re.compile(
    r"\b(?:price forecast|price prediction|predictions?:|technical analysis|price today|prices today"
    r"|price of (?:gold|oil|silver) today|current price of|should you buy|to buy now|best (?:places|way)s? to buy"
    r"|stocks? to buy|elliott wave|better buy|buy the dip|is it time to (?:buy|sell)|should you sell"
    r"|top \d+ stocks|millionaire|could make you rich|price target|how to trade|analysts? (?:believe|see|expect)\b.*\bcould\b|could (?:rally|soar|surge|double|triple))\b", re.IGNORECASE)

# Outlets editorially controlled by a government. Shown with a label; their claims get attributed.
STATE_AFFILIATED = {
    "tehran times", "mehr news", "mehr news agency", "irna", "tasnim", "tasnim news agency",
    "press tv", "presstv", "fars news", "fars news agency", "irib", "saba", "saba news agency",
    "al masirah", "rt", "tass", "sputnik", "xinhua", "cgtn", "global times", "china daily",
    "anadolu ajansı", "anadolu agency", "daily sabah",
}

KEYWORDS = [
    # gold & metals
    r"gold", r"bullion", r"xau", r"precious metals?", r"silver",
    # rates, dollar, data
    r"fed", r"federal reserve", r"fomc", r"powell", r"interest rates?", r"rate (?:cut|hike)s?",
    r"treasur(?:y|ies)", r"yields?", r"dollar", r"dxy", r"inflation", r"cpi", r"pce", r"ppi",
    r"payrolls?", r"jobs report", r"jobless", r"unemployment", r"gdp", r"recession",
    r"central banks?", r"ecb", r"bank of japan", r"boj", r"pboc", r"bank of england",
    r"tariffs?", r"trade war", r"sanctions?", r"debt ceiling", r"shutdown",
    # energy
    r"oil", r"crude", r"brent", r"wti", r"opec\+?", r"lng", r"natural gas",
    # geopolitics
    r"iran(?:ian)?", r"tehran", r"hormuz", r"irgc", r"israel(?:i)?", r"gaza", r"hezbollah", r"lebanon",
    r"houthis?", r"yemen", r"red sea", r"bab al-mandab", r"saudi", r"missiles?", r"air ?strikes?",
    r"drones?", r"ceasefire", r"truce", r"nuclear", r"ukraine", r"russia", r"taiwan", r"war",
    r"invasion", r"military",
    # AI, tech & equity markets
    r"ai", r"artificial intelligence", r"openai", r"nvidia", r"nvda", r"tsmc", r"amd", r"broadcom",
    r"semiconductors?", r"chips?", r"chipmakers?", r"data cent(?:er|re)s?", r"hyperscalers?",
    r"big tech", r"magnificent seven", r"nasdaq", r"s&p 500", r"stock market", r"wall street",
]
KEYWORD_RE = re.compile(r"\b(?:" + "|".join(KEYWORDS) + r")\b", re.IGNORECASE)

TOPICS = [
    "Gold & Precious Metals", "Fed & US Rates", "Economic Data", "Central Banks",
    "Oil & Energy", "Middle East", "Trade & Sanctions", "Other Geopolitics",
    "AI & Tech", "Stock Markets", "Other",
]


def passes_filter(text):
    return bool(KEYWORD_RE.search(text)) and not FILLER_RE.search(text)


def publisher_key(name):
    return re.sub(r"\.com$", "", name.strip().lower().removeprefix("www."))


def utc_str(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def entry_timestamp(entry):
    """Publish time of a feed entry (UTC), or None. Also handles Investing.com's
    non-standard "Nov 20, 2026 17:06 GMT", which feedparser does not recognise."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        return calendar.timegm(parsed)
    raw = (entry.get("published") or entry.get("updated") or "").strip()
    for fmt in ("%b %d, %Y %H:%M GMT", "%b %d, %Y %H:%M:%S GMT"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    return None


def clean_text(raw):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw or ""))).strip()


def is_state_affiliated(source):
    return source.strip().lower() in STATE_AFFILIATED


# ---------------------------------------------------------------- fetching

def fetch_feed(item):
    name, url = item
    is_gn = name.startswith("GN ")
    stats = {"feed": name, "status": "ok", "items": 0, "kept": 0}
    articles = []
    try:
        res = requests.get(url, headers=HTTP_HEADERS, timeout=12)
        if res.status_code == 429:  # rate limited: wait once and retry
            time.sleep(RATE_LIMIT_RETRY_S)
            res = requests.get(url, headers=HTTP_HEADERS, timeout=12)
        if res.status_code != 200:
            stats["status"] = f"HTTP {res.status_code}"
            return articles, stats
        feed = feedparser.parse(res.content)
        stats["items"] = len(feed.entries)
        cutoff = time.time() - RETENTION_HOURS * 3600
        for e in feed.entries[:100]:
            title = clean_text(e.get("title", ""))
            link = e.get("link", "")
            if not title or not link:
                continue
            if is_gn:
                # Google News: real publisher is in <source>; title ends with " - Publisher"
                source = (e.get("source") or {}).get("title") or "Google News"
                if title.endswith(f" - {source}"):
                    title = title[: -len(source) - 3].strip()
                if publisher_key(source) not in GOOGLE_NEWS_PUBLISHERS:
                    continue
                desc = ""
            else:
                source = name
                desc = clean_text(e.get("summary", e.get("description", "")))[:400]
            if not passes_filter(f"{title} {desc}"):
                continue
            ts = entry_timestamp(e) or time.time()
            ts = min(ts, time.time())
            if ts < cutoff:
                continue
            articles.append({
                "title": title,
                "link": link,
                "source": source,
                "state_affiliated": is_state_affiliated(source),
                "ts": ts,
                "time_utc": utc_str(ts),
                "description": desc,
            })
        stats["kept"] = len(articles)
    except Exception as ex:
        stats["status"] = f"error: {type(ex).__name__}"
    return articles, stats


def fetch_sequentially(items):
    results = []
    for i, item in enumerate(items):
        if i:
            time.sleep(GOOGLE_NEWS_GAP_S)
        results.append(fetch_feed(item))
    return results


def fetch_all_feeds():
    """Publisher feeds in parallel; Google News searches one after another (it rate-limits bursts)."""
    google = [i for i in FEEDS.items() if i[0].startswith("GN ")]
    direct = [i for i in FEEDS.items() if not i[0].startswith("GN ")]
    with ThreadPoolExecutor(max_workers=len(direct) + 1) as pool:
        google_future = pool.submit(fetch_sequentially, google)
        results = list(pool.map(fetch_feed, direct)) + google_future.result()
    articles, health = [], []
    for arts, stats in results:
        articles.extend(arts)
        health.append(stats)
    for s in health:
        print(f"  feed {s['feed']:<20} {s['status']:<14} items={s['items']:<4} kept={s['kept']}")
    return articles, health


# ---------------------------------------------------------------- clustering

def norm_title(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def make_cluster(article):
    cid = hashlib.sha1(article["link"].encode()).hexdigest()[:12]
    return {"id": cid, "first_ts": article["ts"], "last_ts": article["ts"],
            "articles": [article], "summary": None, "needs_summary": True}


def cluster_articles(clusters, new_articles):
    """Assign each new article to the most similar recent cluster, or start a new one."""
    from fastembed import TextEmbedding
    model = TextEmbedding(EMBED_MODEL, cache_dir=os.environ.get("FASTEMBED_CACHE", ".model_cache"))

    def embed(titles):
        return np.array(list(model.embed(titles))) if titles else np.zeros((0, 384))

    centroid_sum = [embed([a["title"] for a in c["articles"]]).sum(0) for c in clusters]
    new_vecs = embed([a["title"] for a in new_articles])
    seen_titles = {norm_title(a["title"]) for c in clusters for a in c["articles"]}

    for art, vec in sorted(zip(new_articles, new_vecs), key=lambda p: p[0]["ts"]):
        nt = norm_title(art["title"])
        best, best_sim = -1, -1.0
        for i, c in enumerate(clusters):
            if art["ts"] - c["last_ts"] > JOIN_WINDOW_HOURS * 3600 or len(c["articles"]) >= MAX_CLUSTER_SIZE:
                continue
            cen = centroid_sum[i]
            sim = float(cen @ vec / (np.linalg.norm(cen) + 1e-9))
            if sim > best_sim:
                best, best_sim = i, sim
        if best >= 0 and best_sim >= SIM_THRESHOLD:
            c = clusters[best]
            if nt in seen_titles and any(norm_title(a["title"]) == nt for a in c["articles"]):
                continue  # same headline syndicated twice
            c["articles"].append(art)
            c["first_ts"] = min(c["first_ts"], art["ts"])
            c["last_ts"] = max(c["last_ts"], art["ts"])
            c["needs_summary"] = True
            centroid_sum[best] = centroid_sum[best] + vec
        else:
            clusters.append(make_cluster(art))
            centroid_sum.append(vec.copy())
        seen_titles.add(nt)
    return clusters


# ---------------------------------------------------------------- summaries

PROMPT = """You write summaries for a news panel on a market data terminal. The reader wants facts only.

Rules:
- Report only what the articles state. No analysis, no market commentary, no predictions, no statements
  about what something means for gold, prices or markets (never write things like "safe-haven demand",
  "bullish", "could weigh on", "investors fear").
- Attribute claims that are not independently confirmed by several outlets ("according to Reuters",
  "Mehr News reported"). Articles marked STATE-AFFILIATED come from government-controlled media:
  ALWAYS attribute their claims to them by name.
- If sources contradict each other, give both versions, each attributed.
- headline: neutral, max 14 words, describes the most recent development.
- summary: 1-3 sentences describing the core event.
- key_points: up to 6 distinct facts, newest first, each starting with "[HH:MM UTC, Source] ".
- topic: the single best category.
- on_topic: true if the story is about gold or precious metals markets, US monetary policy or rates,
  major economic data, central banks, the US dollar, oil or energy supply, armed conflict or sanctions
  involving major economies or energy producers, trade policy between major economies, broad stock market
  moves, or major AI/technology developments: earnings, guidance and leadership changes at the largest
  tech/AI companies (e.g. Nvidia, Microsoft, Alphabet, Amazon, Meta, Apple, TSMC, OpenAI), AI investment
  and data-center spending, chip export controls.
  false for: other individual company news (including mining exploration/drill results and small-cap stocks),
  product launches and reviews, stock picks and investing advice, local crime, domestic politics without
  economic or military relevance, sports, culture, opinion pieces.

Clusters:
{clusters}
"""


def summary_schema():
    from google.genai import types
    S, T = types.Schema, types.Type
    return S(type=T.ARRAY, items=S(type=T.OBJECT, properties={
        "cluster_id": S(type=T.INTEGER),
        "on_topic": S(type=T.BOOLEAN),
        "topic": S(type=T.STRING, enum=TOPICS),
        "headline": S(type=T.STRING),
        "summary": S(type=T.STRING),
        "key_points": S(type=T.ARRAY, items=S(type=T.STRING)),
    }, required=["cluster_id", "on_topic", "topic", "headline", "summary", "key_points"]))


def format_cluster_for_prompt(idx, cluster):
    arts = sorted(cluster["articles"], key=lambda a: a["ts"])
    if len(arts) > ARTICLES_PER_PROMPT:
        arts = arts[:1] + arts[-(ARTICLES_PER_PROMPT - 1):]
    lines = []
    for a in arts:
        tag = " | STATE-AFFILIATED" if a.get("state_affiliated") else ""
        desc = f" - {a['description'][:300]}" if a.get("description") else ""
        lines.append(f"  [{a['time_utc']} UTC | {a['source']}{tag}] {a['title']}{desc}")
    return f"=== cluster_id {idx} ({len(cluster['articles'])} articles) ===\n" + "\n".join(lines)


def summarize_batch(client, batch):
    from google.genai import types
    prompt = PROMPT.format(clusters="\n\n".join(format_cluster_for_prompt(i, c) for i, c in enumerate(batch)))
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=summary_schema())
    for model in GEMINI_MODELS:
        try:
            res = client.models.generate_content(model=model, contents=prompt, config=config)
            items = json.loads(res.text)
            out = {}
            for it in items:
                cid = it.get("cluster_id")
                if isinstance(cid, int) and 0 <= cid < len(batch) and cid not in out:
                    out[cid] = {k: it.get(k) for k in ("on_topic", "topic", "headline", "summary", "key_points")}
            return out
        except Exception as ex:
            print(f"  Gemini {model} failed: {ex}")
    return {}


def summarize_pending(clusters):
    api_key = os.environ.get("GEMINI_API_KEY")
    pending = sorted((c for c in clusters if c["needs_summary"]), key=lambda c: -c["last_ts"])
    if not api_key:
        print("GEMINI_API_KEY missing: summaries skipped")
        return
    from google import genai
    client = genai.Client(api_key=api_key)
    for b in range(min(MAX_BATCHES_PER_RUN, -(-len(pending) // BATCH_SIZE))):
        batch = pending[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        results = summarize_batch(client, batch)
        for i, c in enumerate(batch):
            if i in results:
                c["summary"] = results[i]
                c["summary"]["n_articles"] = len(c["articles"])
                c["needs_summary"] = False
    left = sum(c["needs_summary"] for c in clusters)
    print(f"Summaries: {len(pending) - left} written, {left} queued for next run")


# ---------------------------------------------------------------- OANDA order book

def fetch_oanda_book():
    key = os.environ.get("OANDA_API_KEY")
    if not key:
        return None
    try:
        res = requests.get("https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/orderBook",
                           headers={"Authorization": f"Bearer {key}"}, timeout=10)
        res.raise_for_status()
        ob = res.json()["orderBook"]
        price = float(ob["price"])
        buckets = [
            {"price": float(b["price"]),
             "long_pct": float(b["longCountPercent"]),
             "short_pct": float(b["shortCountPercent"])}
            for b in ob["buckets"] if abs(float(b["price"]) - price) / price <= 0.06
        ]
        return {"time": ob["time"], "price": price, "bucket_width": float(ob["bucketWidth"]), "buckets": buckets}
    except Exception as ex:
        print(f"OANDA order book failed: {ex}")
        return None


# ---------------------------------------------------------------- main

def load_clusters():
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("version") == 2:
            return data["clusters"]
    except (OSError, ValueError):
        pass
    return []  # missing file or old format: start fresh


def main():
    now = time.time()
    clusters = [c for c in load_clusters() if now - c["last_ts"] <= RETENTION_HOURS * 3600]
    known_links = {a["link"] for c in clusters for a in c["articles"]}

    print("Fetching feeds...")
    fetched, health = fetch_all_feeds()
    new_articles, seen = [], set()
    for a in fetched:
        if a["link"] not in known_links and a["link"] not in seen:
            seen.add(a["link"])
            new_articles.append(a)
    print(f"{len(new_articles)} new articles")

    if new_articles:
        clusters = cluster_articles(clusters, new_articles)
    summarize_pending(clusters)

    clusters.sort(key=lambda c: -c["last_ts"])
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"version": 2, "generated_utc": utc_str(now), "feeds": health, "clusters": clusters},
                  f, ensure_ascii=False, separators=(",", ":"))

    book = fetch_oanda_book()
    if book:
        with open(OANDA_FILE, "w", encoding="utf-8") as f:
            json.dump(book, f, separators=(",", ":"))

    import extras  # Treasury yields, OKX taker volume for the volume profile
    extras.run_all()
    print(f"Done: {len(clusters)} clusters")


if __name__ == "__main__":
    main()
