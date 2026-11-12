import os
import json
import time
import requests
import feedparser
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util
from google import genai
from google.genai import types

# --- CONFIGURATION ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OANDA_API_KEY = os.environ.get("OANDA_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None
    print("WARNING: GEMINI_API_KEY not found.")

FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-1.5-flash",
    "gemini-3.5-flash-lite"
]

BATCH_SIZE = 6
MAX_BATCH_CALLS_PER_RUN = 2

MIN_EVAL_PROBABILITY = 51       # Evaluate all directional signals > 50% (51%+)
EVAL_HORIZON_HOURS = 20.0       # Evaluation window in hours
MAX_DRAWDOWN_ATR_MULT = 1.8     # Catastrophe stop limit (1.8x Daily ATR adverse move)

FEEDS = {
    "CNBC Commodities": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000118",
    "Investing.com": "https://www.investing.com/rss/news_11.rss",
    "Yahoo Finance Gold": "https://finance.yahoo.com/rss/headline?s=GC=F",
    "Yahoo Finance Oil": "https://finance.yahoo.com/rss/headline?s=CL=F",
    "CNBC Tech & AI": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19854910",
    "CNBC Economy": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "Yahoo Finance Nasdaq": "https://finance.yahoo.com/rss/headline?s=%5EIXIC,NVDA,MSFT,AAPL,TSM",
    "MarketWatch Macro": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Zawya": "https://www.zawya.com/en/rss",
    "Arab News": "https://www.arabnews.com/cat/4/rss.xml",
    "Al Jazeera English": "https://www.aljazeera.com/xml/rss/all.xml",
    "Middle East Eye": "https://www.middleeasteye.net/rss",
    "Tehran Times": "https://www.tehrantimes.com/rss",
    "Mehr News Agency": "https://en.mehrnews.com/rss",
    "The Times of Israel": "https://www.timesofisrael.com/feed/",
    "Saba News Agency": "https://www.saba.ye/en/rss"
}

IMPACT_KEYWORDS = [
    "gold", "xau", "bullion", "oil", "brent", "crude", "opec", "energy", 
    "hormuz", "red sea", "yemen", "houthi", "iran", "tehran", "israel", 
    "idf", "missile", "strike", "attack", "drone", "sanction", "nuclear", 
    "irgc", "escalat", "retaliat", "war", "truce", "ceasefire", "saudi",
    "fed", "inflation", "rate", "central bank", "brics", "dollar", "treasury", 
    "tariff", "recession", "gdp", "unemployment", "payroll", "cpi", "pmi", 
    "stimulus", "china", "default", "liquidity", "debt", "yield",
    "nvidia", "nvda", "nasdaq", "s&p", "semiconductor", "chip", "taiwan", 
    "tsmc", "artificial intelligence", "ai ", "big tech", "microsoft", "apple",
    "earnings", "selloff", "sell-off", "crash", "bubble", "rout", "margin"
]

DATA_FILE = "data.json"
STATS_FILE = "performance_stats.json"
SIMILARITY_THRESHOLD = 0.54

def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)

def classify_news_category(headline, source=""):
    text = f"{headline} {source}".lower()
    if any(k in text for k in ["iran", "israel", "idf", "houthi", "yemen", "red sea", "hormuz", "missile", "strike", "attack", "drone", "war", "truce", "ceasefire", "nuclear", "irgc", "sanction", "tehran", "middle east", "al jazeera", "times of israel", "saba", "mehr"]):
        return "Geopolitics & War"
    if any(k in text for k in ["nvidia", "nvda", "nasdaq", "semiconductor", "chip", "tsmc", "taiwan", "ai ", "artificial intelligence", "microsoft", "apple", "big tech", "tech"]):
        return "Tech, AI & Equities"
    if any(k in text for k in ["fed", "inflation", "rate", "cpi", "pmi", "payroll", "unemployment", "gdp", "treasury", "yield", "dollar", "central bank", "recession", "tariff", "debt", "stimulus"]):
        return "Fed, Rates & Macro"
    return "Commodities & Energy"

def detect_specific_narrative(text):
    t = text.lower()
    if any(k in t for k in ["iran", "tehran", "hormuz", "irgc", "nuclear"]):
        return ("Iran & Strait of Hormuz", "Geopolitics & War")
    if any(k in t for k in ["houthi", "yemen", "red sea", "saba"]):
        return ("Red Sea & Yemen Conflict", "Geopolitics & War")
    if any(k in t for k in ["israel", "idf", "gaza", "lebanon", "hezbollah"]):
        return ("Israel & Middle East Escalation", "Geopolitics & War")
    if any(k in t for k in ["tariff", "trade war", "sanction", "brics"]):
        return ("Tariffs, Sanctions & Geoeconomics", "Fed, Rates & Macro")
    if any(k in t for k in ["fed", "powell", "fomc", "rate", "cpi", "inflation", "payroll", "yield", "treasury"]):
        return ("US Inflation (CPI) & Fed Rate Path", "Fed, Rates & Macro")
    if any(k in t for k in ["nvidia", "nvda", "semiconductor", "chip", "tsmc", "ai ", "artificial intelligence", "nasdaq"]):
        return ("AI, Semiconductors & Tech Liquidity", "Tech, AI & Equities")
    if any(k in t for k in ["oil", "brent", "crude", "opec", "energy"]):
        return ("Crude Oil & Energy Supply", "Commodities & Energy")
    if any(k in t for k in ["recession", "gdp", "debt", "default", "stimulus", "china"]):
        return ("Global Growth, Debt & China Macro", "Fed, Rates & Macro")
    return ("Gold Bullion & Central Bank Demand", "Commodities & Energy")

def fetch_single_feed(source_url):
    source, url = source_url
    articles = []
    now = datetime.now()
    cutoff_time = now - timedelta(hours=24)
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if response.status_code != 200:
            return articles
        feed = feedparser.parse(response.content)
        for entry in feed.entries[:20]:
            raw_desc = entry.get("summary", entry.get("description", ""))
            clean_desc = clean_html(raw_desc).strip()
            title = entry.title

            full_text = f"{title} {clean_desc}".lower()
            if not any(kw in full_text for kw in IMPACT_KEYWORDS):
                continue

            pub_dt = now
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    pub_dt = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                except Exception:
                    pass

            if pub_dt < cutoff_time:
                continue

            articles.append({
                "title": title,
                "link": entry.link,
                "source": source,
                "date": pub_dt.strftime("%Y-%m-%d %H:%M"),
                "description": clean_desc[:450],
                "_timestamp": pub_dt.timestamp()
            })
    except Exception as e:
        print(f"Timeout/Error on {source}: {e}")
    return articles

def fetch_new_articles(existing_urls):
    articles = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(fetch_single_feed, FEEDS.items())
        for feed_articles in results:
            for a in feed_articles:
                if a["link"] not in existing_urls:
                    articles.append(a)
    return articles

def fetch_cftc_cot_institutional():
    """
    Fetches official weekly CFTC Commitment of Traders (COT) data for COMEX Gold (Code 088691).
    Shows real institutional positioning (Hedge Funds / Managed Money vs. Bullion Banks / Swap Dealers)
    and Open Interest changes over the multi-week horizon.
    """
    url = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json?$where=cftc_contract_market_code='088691'&$order=report_date_as_yyyy_mm_dd\%20DESC&$limit=2"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if data and len(data) >= 1:
                latest = data[0]
                prev = data[1] if len(data) >= 2 else latest

                mm_long = float(latest.get("m_money_positions_long_all", 0))
                mm_short = float(latest.get("m_money_positions_short_all", 0))
                mm_tot = mm_long + mm_short
                mm_long_pct = round((mm_long / mm_tot) * 100.0, 1) if mm_tot > 0 else 50.0
                mm_short_pct = round(100.0 - mm_long_pct, 1)
                mm_net = int(mm_long - mm_short)

                prev_net = int(float(prev.get("m_money_positions_long_all", 0)) - float(prev.get("m_money_positions_short_all", 0)))
                net_change_wow = mm_net - prev_net

                oi_all = int(float(latest.get("open_interest_all", 0)))
                prev_oi = int(float(prev.get("open_interest_all", 0)))
                oi_change = oi_all - prev_oi

                rep_date = str(latest.get("report_date_as_yyyy_mm_dd", ""))[:10]

                cot_obj = {
                    "report_date": rep_date,
                    "mm_long_pct": mm_long_pct,
                    "mm_short_pct": mm_short_pct,
                    "mm_net_contracts": mm_net,
                    "mm_net_change_wow": net_change_wow,
                    "comex_open_interest": oi_all,
                    "comex_oi_change_wow": oi_change
                }
                prompt_str = (
                    f"CFTC COMEX Institutional Positioning (Report {rep_date}):\n"
                    f"- Hedge Funds (Managed Money): {mm_long_pct}% Long / {mm_short_pct}% Short (Net: {mm_net:+,} contracts, WoW Change: {net_change_wow:+,})\n"
                    f"- Total COMEX Open Interest: {oi_all:,} contracts (WoW Change: {oi_change:+,})"
                )
                print(f"CFTC COT loaded ({rep_date}): Managed Money {mm_long_pct}% Long (Net {mm_net:+,}, WoW {net_change_wow:+,})")
                return prompt_str, cot_obj
    except Exception as e:
        print(f"Note: CFTC COT API unreachable ({e})")

    return "CFTC Institutional Positioning: Temporarily unavailable.", {}

def fetch_okx_l2_orderbook_zones(reference_price, zone_width=10.0):
    """
    Fetches live Level-2 Resting Limit Orderbook (up to 400 depth levels per contract)
    from OKX (XAU-USDT-SWAP & XAUT-USDT-SWAP) + Bybit (PAXGUSDT linear perp) to provide
    real-time algorithmic market-maker liquidity walls alongside OANDA and COMEX.
    """
    l2_zones = {}
    headers = {"User-Agent": "Mozilla/5.0"}
    raw_orders = []
    total_vol = 0.0

    # 1. OKX L2 Orderbook Depth (400 bids + 400 asks)
    for inst in ["XAU-USDT-SWAP", "XAUT-USDT-SWAP"]:
        try:
            url = f"https://www.okx.com/api/v5/market/books?instId={inst}&sz=400"
            res = requests.get(url, headers=headers, timeout=6)
            if res.status_code == 200:
                data = res.json().get("data", [])
                if data:
                    asks = data[0].get("asks", [])
                    bids = data[0].get("bids", [])
                    for a in asks:
                        p = float(a[0])
                        sz = float(a[1])
                        if reference_price <= 0 or abs(p - reference_price) / reference_price <= 0.20:
                            raw_orders.append((p, 0.0, sz))
                            total_vol += sz
                    for b in bids:
                        p = float(b[0])
                        sz = float(b[1])
                        if reference_price <= 0 or abs(p - reference_price) / reference_price <= 0.20:
                            raw_orders.append((p, sz, 0.0))
                            total_vol += sz
        except Exception:
            pass

    # 2. Bybit L2 Orderbook Depth (200 levels
