"""Data jobs run after the news update. A failure in one never stops the other
or the news update.

- US Treasury: 10Y real yield (TIPS) and 10Y breakeven inflation -> macro.json
- OKX: hourly aggressive (taker) buy and sell volume of the XAU/USDT perp -> taker_volume.json
  (OKX keeps 60 days; the file keeps building up history beyond that)
- Yahoo Finance: hourly COMEX gold futures (GC=F) candles with volume, aligned to the spot
  index -> comex_volume.json, for the volume profile on the chart
- Gold market analysis and commentary from analyst publishers (RSS feeds, Kitco pages, and
  Google News for the rest) -> analysis.json (listed as published: no clustering, no AI summary)
"""
import calendar
import csv
import html
import io
import json
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import feedparser

from aggregator import entry_timestamp

import requests

MACRO_FILE = "macro.json"
TAKER_FILE = "taker_volume.json"

TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                "daily-treasury-rates.csv/{year}/all?type={kind}&field_tdr_date_value={year}&page&_format=csv")
MACRO_REFRESH_HOURS = 6       # daily data; no need to fetch every 15 min
MACRO_HISTORY_DAYS = 400

TAKER_URL = "https://www.okx.com/api/v5/rubik/stat/taker-volume-contract"
TAKER_INST = "XAU-USDT-SWAP"
TAKER_KEEP_DAYS = 100         # enough for the 3-month volume profile
TAKER_MAX_PAGES = 16          # 100 hours per page; OKX serves at most ~60 days

COMEX_FILE = "comex_volume.json"
COMEX_URL = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
COMEX_KEEP_DAYS = 100         # enough for the 3-month volume profile
OKX_INDEX_URL = "https://www.okx.com/api/v5/market/history-index-candles"

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0"}


def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def data_age_hours(path):
    """Age from the file's own generated_utc field (git checkouts reset file timestamps)."""
    try:
        stamp = datetime.strptime(read_json(path, {})["generated_utc"], "%Y-%m-%d %H:%M")
        return (datetime.now(timezone.utc) - stamp.replace(tzinfo=timezone.utc)).total_seconds() / 3600
    except (KeyError, TypeError, ValueError):
        return float("inf")


# ---------------------------------------------------------------- US Treasury yields

def fetch_treasury_column(kind, column, years):
    """{iso_date: value} for one column of a Treasury daily rates CSV."""
    out = {}
    for year in years:
        res = requests.get(TREASURY_URL.format(year=year, kind=kind), headers=HTTP_HEADERS, timeout=40)
        res.raise_for_status()
        for row in csv.DictReader(io.StringIO(res.text)):
            value = (row.get(column) or "").strip()
            if not value:
                continue
            date = datetime.strptime(row["Date"], "%m/%d/%Y").date().isoformat()
            out[date] = float(value)
    return out


def update_macro():
    if read_json(MACRO_FILE, {}).get("series") and data_age_hours(MACRO_FILE) < MACRO_REFRESH_HOURS:
        return
    now = datetime.now(timezone.utc)
    years = sorted({now.year, (now - timedelta(days=MACRO_HISTORY_DAYS)).year})
    real = fetch_treasury_column("daily_treasury_real_yield_curve", "10 YR", years)
    nominal = fetch_treasury_column("daily_treasury_yield_curve", "10 Yr", years)
    start = (now - timedelta(days=MACRO_HISTORY_DAYS)).date().isoformat()
    dates = sorted(d for d in real if d >= start)
    series = {
        "REAL10": {"label": "US 10Y real yield (TIPS)", "unit": "%",
                   "points": [[d, real[d]] for d in dates]},
        # breakeven = nominal 10Y yield minus real 10Y yield, same day
        "BE10": {"label": "US 10Y breakeven inflation", "unit": "%",
                 "points": [[d, round(nominal[d] - real[d], 2)] for d in dates if d in nominal]},
    }
    if not series["REAL10"]["points"]:
        raise ValueError("no Treasury real yield data")
    write_json(MACRO_FILE, {"generated_utc": utc_now_str(), "source": "U.S. Department of the Treasury",
                            "series": series})
    last = series["REAL10"]["points"][-1]
    print(f"Treasury: 10Y real yield {last[1]}% on {last[0]}")


# ---------------------------------------------------------------- OKX taker buy/sell volume

def write_taker_file(rows):
    """One row per line so each hourly update is a small git diff."""
    lines = ",\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
    with open(TAKER_FILE, "w", encoding="utf-8") as f:
        f.write('{"inst":"%s","unit":"contracts","columns":["ts_ms","taker_sell","taker_buy"],"rows":[\n%s\n]}\n'
                % (TAKER_INST, lines))


def update_taker_volume():
    stored = {r[0]: r for r in read_json(TAKER_FILE, {}).get("rows", [])}
    newest_stored = max(stored) if stored else 0
    # only completed hours: the current hour is still changing
    current_hour = int(time.time() // 3600 * 3600 * 1000)
    fetched, end = {}, ""
    for _ in range(TAKER_MAX_PAGES):
        params = {"instId": TAKER_INST, "period": "1H", "limit": "100"}
        if end:
            params["end"] = end
        res = requests.get(TAKER_URL, params=params, headers=HTTP_HEADERS, timeout=20)
        res.raise_for_status()
        body = res.json()
        if body.get("code") != "0":
            raise ValueError(body.get("msg") or "OKX error")
        page = body.get("data") or []
        if not page:
            break
        for ts, sell, buy in page:
            ts = int(ts)
            if ts < current_hour:
                fetched[ts] = [ts, round(float(sell)), round(float(buy))]
        end = page[-1][0]
        if int(end) <= newest_stored:
            break  # reached hours we already have
        time.sleep(0.45)  # OKX allows 5 requests per 2 s on this endpoint
    merged = {**stored, **fetched}
    cutoff = (time.time() - TAKER_KEEP_DAYS * 86400) * 1000
    rows = sorted(r for ts, r in merged.items() if ts >= cutoff)
    if rows != sorted(stored.values()):
        write_taker_file(rows)
        print(f"Taker volume: {len(rows)} hours stored ({len(rows) - len(stored)} new)")


# ---------------------------------------------------------------- COMEX gold futures volume

def fetch_index_closes(since_ms):
    """{hour_ms: close} of the OKX XAU/USD spot index for completed hours since since_ms."""
    closes, after = {}, ""
    for _ in range(30):  # 100 hours per page, ~3 months at most
        params = {"instId": "XAU-USD", "bar": "1H", "limit": "100"}
        if after:
            params["after"] = after
        res = requests.get(OKX_INDEX_URL, params=params, headers=HTTP_HEADERS, timeout=20)
        res.raise_for_status()
        page = res.json().get("data") or []
        if not page:
            break
        for r in page:
            closes[int(r[0])] = float(r[4])
        after = page[-1][0]
        if int(after) <= since_ms:
            break
        time.sleep(0.12)
    return closes


def write_comex_file(rows):
    """One row per line so each update is a small git diff."""
    lines = ",\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
    with open(COMEX_FILE, "w", encoding="utf-8") as f:
        f.write('{"symbol":"GC=F","source":"Yahoo Finance","unit":"contracts (100 oz)",'
                '"note":"low/high shifted by each hour\'s futures-minus-spot gap",'
                '"columns":["ts_ms","low_spot","high_spot","volume"],"rows":[\n%s\n]}\n' % lines)


def update_comex():
    stored = {r[0]: r for r in read_json(COMEX_FILE, {}).get("rows", [])}
    newest = max(stored) if stored else 0
    # small request once history exists; 3 months on the first run
    rng = "5d" if newest and time.time() * 1000 - newest < 4 * 86400000 else "3mo"
    res = requests.get(COMEX_URL, params={"range": rng, "interval": "1h"}, headers=HTTP_HEADERS, timeout=30)
    res.raise_for_status()
    result = res.json()["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    now_ms = time.time() * 1000
    candles = []
    for ts, high, low, close, vol in zip(result["timestamp"], quote["high"], quote["low"],
                                         quote["close"], quote["volume"]):
        ts_ms = ts * 1000
        # completed hourly candles only, with volume, and not already stored
        if ts % 3600 or ts_ms + 3600000 > now_ms or not vol or None in (high, low, close) or ts_ms in stored:
            continue
        candles.append((ts_ms, high, low, close, vol))
    if not candles:
        return
    index = fetch_index_closes(min(c[0] for c in candles))
    for ts_ms, high, low, close, vol in candles:
        spot = index.get(ts_ms)
        if spot is None:
            continue
        basis = close - spot  # futures premium over spot at that hour
        stored[ts_ms] = [ts_ms, round(low - basis, 2), round(high - basis, 2), int(vol)]
    cutoff = now_ms - COMEX_KEEP_DAYS * 86400000
    rows = sorted(r for ts, r in stored.items() if ts >= cutoff)
    write_comex_file(rows)
    print(f"COMEX volume: {len(rows)} hours stored, latest {datetime.fromtimestamp(rows[-1][0] / 1000, timezone.utc):%Y-%m-%d %H:%M} UTC")


# ---------------------------------------------------------------- analysis & commentary
#
# Gold market analysis written by analysts, listed as published (no clustering, no AI summary).
# Publishers with an RSS feed or crawlable pages are read directly (with a teaser text);
# the rest come via Google News (title only).

ANALYSIS_FILE = "analysis.json"
ANALYSIS_KEEP_DAYS = 7
ANALYSIS_TEASER_CHARS = 320
BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}

# (publisher, RSS feed, only items by this author or None); Investing.com feeds have no teaser
ANALYSIS_FEEDS = [
    ("ING Think", "https://think.ing.com/rss/", None),
    ("FXEmpire", "https://www.fxempire.com/api/v1/en/articles/rss/forecasts", None),
    ("Investing.com", "https://www.investing.com/rss/commodities.rss", None),      # analysis & opinion
    ("Investing.com", "https://www.investing.com/rss/news_11.rss", "Investing.com"),  # own daily reports, not wires
]
# Kitco has no RSS feed; its listing pages are crawlable (robots.txt allows all).
KITCO_PAGES = [
    ("https://www.kitco.com/opinion/", r"/opinion/\d{4}-\d{2}-\d{2}/[a-z0-9-]+"),
    ("https://www.kitco.com/news/", r"/news/(?:article|off-the-wire)/\d{4}-\d{2}-\d{2}/[a-z0-9-]+"),
]
KITCO_MAX_NEW_PER_RUN = 20
# Google News for publishers without a usable feed (FXStreet and StoneX block direct access)
ANALYSIS_QUERIES = [
    "gold (site:fxstreet.com OR site:stonex.com)",
    "(gold OR commodities OR metals) (site:home.saxo OR site:gold.org)",
    'gold (outlook OR forecast OR analysis OR "week ahead" OR weekly)',
]
# publisher names as Google News reports them -> name shown on the site
ANALYSIS_PUBLISHERS = {
    "kitco": "Kitco", "investing.com": "Investing.com", "fxstreet": "FXStreet", "fxempire": "FXEmpire",
    "ing think": "ING Think", "saxo": "Saxo", "world gold council": "World Gold Council",
    "stonex": "StoneX", "investinglive": "investingLive", "reuters": "Reuters", "bloomberg": "Bloomberg",
    "cnbc": "CNBC", "marketwatch": "MarketWatch", "barron's": "Barron's", "financial times": "Financial Times",
    "wsj": "WSJ", "the wall street journal": "WSJ", "mining.com": "Mining.com",
}
ANALYSIS_TOPIC_RE = re.compile(r"\b(gold|bullion|xau|precious metals?|metals?|silver|commodit\w*)\b", re.IGNORECASE)
# landing pages, ads, SEO price pages, filings and equity market wraps are not analysis
ANALYSIS_JUNK_RE = re.compile(r"trade gold on|news, analysis & commentary|methodology|price today|prices today"
                              r"|current price of|live price|price chart|how to buy|best way to buy"
                              r"|^form \d|\|.*\|"
                              r"|\b(ftse|tsx|asx|nikkei|dax|stoxx|sensex|nifty)\b",
                              re.IGNORECASE)
AI_DISCLAIMER_RE = re.compile(r"AI-generated content[^.]*\.?|For informational and educational purposes only\.?"
                              r"|Not investment advice\.?", re.IGNORECASE)


def title_key(title):
    return re.sub(r"[^a-z0-9]", "", title.lower())[:120]


def clean_teaser(text):
    s = html.unescape(html.unescape(text or ""))
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\.[a-z][\w-]*[^{}]*\{[^}]*\}", " ", s)          # inline CSS that leaks into descriptions
    s = re.sub(r"^\s*\(Kitco [A-Za-z]+\)\s*-\s*", "", s)           # "(Kitco News) - " prefix
    s = re.sub(r"([a-z0-9][.!?”\"])([A-Z])", r"\1 \2", s)          # "published.The" -> "published. The"
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > ANALYSIS_TEASER_CHARS:
        s = s[:ANALYSIS_TEASER_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return s


def analysis_publisher(name):
    key = re.sub(r"\s+(uk|india|south africa|nigeria|australia|canada)$", "", name.strip().lower())
    return ANALYSIS_PUBLISHERS.get(key)


def page_meta(page, key):
    m = (re.search(r'<meta[^>]+(?:property|name)="%s"[^>]+content="([^"]*)"' % re.escape(key), page)
         or re.search(r'<meta[^>]+content="([^"]*)"[^>]+(?:property|name)="%s"' % re.escape(key), page))
    return html.unescape(m.group(1)) if m else ""


def collect_feeds(add):
    for source, url, only_author in ANALYSIS_FEEDS:
        try:
            res = requests.get(url, headers=BROWSER_HEADERS, timeout=20)
            res.raise_for_status()
            for e in feedparser.parse(res.content).entries:
                if only_author and (e.get("author") or "").strip() != only_author:
                    continue
                title = html.unescape(e.get("title", "")).strip()
                raw = e.get("summary", "") or e.get("description", "")
                ai = bool(re.search(r"AI-generated", raw, re.IGNORECASE))
                teaser = clean_teaser(AI_DISCLAIMER_RE.sub(" ", raw))
                ts = entry_timestamp(e)
                if ts is None:
                    continue  # without a date the article can't be placed in the list
                add(title, source, e.get("link", ""), ts,
                    teaser=teaser, author=(e.get("author") or "").strip(), ai=ai, topic_text=f"{title} {teaser}")
        except Exception as ex:
            print(f"Analysis feed {source} failed: {type(ex).__name__}")


def collect_kitco(add, known_links):
    new_links = []
    for page_url, pattern in KITCO_PAGES:
        try:
            res = requests.get(page_url, headers=BROWSER_HEADERS, timeout=20)
            res.raise_for_status()
            for path in dict.fromkeys(re.findall(pattern, res.text)):
                link = "https://www.kitco.com" + path
                if link not in known_links and link not in new_links:
                    new_links.append(link)
        except Exception as ex:
            print(f"Kitco listing {page_url} failed: {type(ex).__name__}")
    for link in new_links[:KITCO_MAX_NEW_PER_RUN]:
        try:
            page = requests.get(link, headers=BROWSER_HEADERS, timeout=20).text
            title = page_meta(page, "og:title")
            published = page_meta(page, "article:published_time")
            try:
                ts = datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp()
            except ValueError:
                ts = time.time()
            authors = re.findall(r'"author[s]?"\s*:\s*\[?\s*\{[^}]*?"name"\s*:\s*"([^"]+)"', page)
            add(title, "Kitco", link, ts, teaser=clean_teaser(page_meta(page, "description")),
                author=authors[0] if authors else "", kind="commentary" if "/opinion/" in link else "")
            time.sleep(0.3)
        except Exception as ex:
            print(f"Kitco article failed: {type(ex).__name__}")


def collect_google_news(add):
    for q in ANALYSIS_QUERIES:
        url = ("https://news.google.com/rss/search?q=" + quote_plus(f"{q} when:2d")
               + "&hl=en-US&gl=US&ceid=US:en")
        try:
            res = requests.get(url, headers=HTTP_HEADERS, timeout=20)
            if res.status_code == 429:
                time.sleep(8)
                res = requests.get(url, headers=HTTP_HEADERS, timeout=20)
            res.raise_for_status()
            for e in feedparser.parse(res.content).entries:
                raw_source = (e.get("source") or {}).get("title") or ""
                source = analysis_publisher(raw_source)
                title = html.unescape(e.get("title", "")).strip()
                if title.endswith(f" - {raw_source}"):
                    title = title[: -len(raw_source) - 3].strip()
                parsed = e.get("published_parsed")
                if source:
                    add(title, source, e.get("link", ""), calendar.timegm(parsed) if parsed else time.time())
        except Exception as ex:
            print(f"Analysis Google News query failed: {type(ex).__name__}")
        time.sleep(1.5)  # Google News rate-limits bursts


def update_analysis():
    # re-apply the junk filter to stored items too, so filter changes take effect immediately
    stored = {i["key"]: i for i in read_json(ANALYSIS_FILE, {}).get("items", [])
              if not ANALYSIS_JUNK_RE.search(i["title"])}
    added = 0

    def add(title, source, link, ts, teaser="", author="", ai=False, kind="", topic_text=None):
        nonlocal added
        if not title or ANALYSIS_JUNK_RE.search(title) or not ANALYSIS_TOPIC_RE.search(topic_text or title):
            return
        key = title_key(title)
        old = stored.get(key)
        if old and (old.get("teaser") or not teaser):
            return  # already have it (same story via several sources): keep the version with a teaser
        stored[key] = {"key": key, "title": title, "source": source, "link": link,
                       "ts": int(min(ts, time.time())), "teaser": teaser, "author": author,
                       "ai": ai, "kind": kind}
        added += 0 if old else 1

    collect_feeds(add)
    collect_kitco(add, {i["link"] for i in stored.values()})
    collect_google_news(add)
    cutoff = time.time() - ANALYSIS_KEEP_DAYS * 86400
    items = sorted((i for i in stored.values() if i["ts"] >= cutoff), key=lambda i: -i["ts"])
    write_json(ANALYSIS_FILE, {"generated_utc": utc_now_str(), "items": items})
    with_teaser = sum(1 for i in items if i.get("teaser"))
    print(f"Analysis: {len(items)} articles in the last {ANALYSIS_KEEP_DAYS} days ({added} new, {with_teaser} with teaser)")


def run_all():
    for job in (update_macro, update_taker_volume, update_comex, update_analysis):
        try:
            job()
        except Exception as ex:
            print(f"{job.__name__} failed: {ex}")
