import os
import json
import time
import math
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
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash"
]
DEAD_MODELS = set()

CLUSTER_SUMMARY_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "Cluster_ID": types.Schema(type=types.Type.INTEGER),
            "Is_Gold_Relevant": types.Schema(type=types.Type.BOOLEAN),
            "Dominant_Driver": types.Schema(type=types.Type.STRING),
            "Event_Timestamp": types.Schema(type=types.Type.STRING),
            "Facts": types.Schema(type=types.Type.STRING),
            "Macro_Context": types.Schema(type=types.Type.STRING),
            "Impact_Timing": types.Schema(type=types.Type.STRING),
            "Timeframe": types.Schema(type=types.Type.STRING),
            "XAUUSD_Impact": types.Schema(type=types.Type.STRING),
            "Probability_Percent": types.Schema(type=types.Type.INTEGER),
            "Orderflow_Price_Target": types.Schema(type=types.Type.STRING),
        },
        required=[
            "Cluster_ID", "Is_Gold_Relevant", "Dominant_Driver", "Event_Timestamp",
            "Facts", "Macro_Context", "Impact_Timing", "Timeframe",
            "XAUUSD_Impact", "Probability_Percent", "Orderflow_Price_Target"
        ]
    )
)

# Schema for Tab [4] Master AI Trade Decision & Active Trade Review
MASTER_TRADE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "Decision": types.Schema(type=types.Type.STRING),  # "LONG", "SHORT", or "NO_TRADE"
        "Confidence_Percent": types.Schema(type=types.Type.INTEGER),
        "Take_Profit_Price": types.Schema(type=types.Type.NUMBER),
        "Setup_Title": types.Schema(type=types.Type.STRING),
        "Detailed_Rationale": types.Schema(type=types.Type.STRING),
        "Chart_Orderbook_Evidence": types.Schema(type=types.Type.STRING),
        "News_Catalyst_Evidence": types.Schema(type=types.Type.STRING),
        "Surface_3D_Evidence": types.Schema(type=types.Type.STRING),
        "Experience_Adjustment": types.Schema(type=types.Type.STRING),
        "Invalidation_Note": types.Schema(type=types.Type.STRING),
        "Cited_Sources": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "type": types.Schema(type=types.Type.STRING),   # "NEWS_LINK", "GRAPH_DATA", "ORDERBOOK", "WINRATE_STATS"
                    "label": types.Schema(type=types.Type.STRING),
                    "detail": types.Schema(type=types.Type.STRING),
                    "url": types.Schema(type=types.Type.STRING)
                },
                required=["type", "label", "detail", "url"]
            )
        )
    },
    required=[
        "Decision", "Confidence_Percent", "Take_Profit_Price", "Setup_Title",
        "Detailed_Rationale", "Chart_Orderbook_Evidence", "News_Catalyst_Evidence",
        "Surface_3D_Evidence", "Experience_Adjustment", "Invalidation_Note", "Cited_Sources"
    ]
)

ACTIVE_TRADE_REVIEW_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "Still_Valid": types.Schema(type=types.Type.BOOLEAN),
        "Status_Note": types.Schema(type=types.Type.STRING)
    },
    required=["Still_Valid", "Status_Note"]
)

BATCH_SIZE = 6
MAX_BATCH_CALLS_PER_RUN = 4

MIN_EVAL_PROBABILITY = 51
EVAL_HORIZON_HOURS = 20.0
MAX_DRAWDOWN_ATR_MULT = 1.8

# Tab [4] Master Trade Rules
TRADE_EXPIRY_HOURS = 20.0
TRADE_SL_ATR_MULT = 1.2
TRADE_MIN_TP_DISTANCE_USD = 10.0

# Maximum distance for orderbook zones (2.5% ~ $95 around spot to prevent chart stretching)
MAX_ZONE_RADIUS_PCT = 0.025

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

CORE_MACRO_GEO_KEYWORDS = [
    "gold", "xau", "bullion", "oil", "brent", "crude", "opec", "energy", 
    "hormuz", "red sea", "yemen", "houthi", "iran", "tehran", "israel", 
    "idf", "missile", "strike", "attack", "drone", "sanction", "nuclear", 
    "irgc", "escalat", "retaliat", "war", "truce", "ceasefire", "saudi",
    "fed", "inflation", "rate", "central bank", "brics", "dollar", "treasury", 
    "tariff", "recession", "gdp", "unemployment", "payroll", "cpi", "pmi", 
    "stimulus", "china", "default", "liquidity", "debt", "yield"
]

TECH_ENTITIES = [
    "nvidia", "nvda", "nasdaq", "s&p", "semiconductor", "chip", "taiwan", 
    "tsmc", "artificial intelligence", "ai ", "big tech", "microsoft", "apple", "supermicro"
]

TECH_SYSTEMIC_TRIGGERS = [
    "crash", "selloff", "sell-off", "rout", "bubble", "margin", "plunge", "turmoil",
    "scandal", "resign", "doj", "probe", "investigation", "ban", "curb", "export",
    "tariff", "sanction", "china", "taiwan", "beijing", "trump", "fed", "inflation",
    "liquidity", "yield", "guidance", "shock", "record", "trillion", "war", "spy"
]

DATA_FILE = "data.json"
STATS_FILE = "performance_stats.json"
TRADE_JOURNAL_FILE = "trade_journal.json"
SIMILARITY_THRESHOLD = 0.54

def fetch_capitalcom_tv_price():
    url = "https://scanner.tradingview.com/global/scan"
    payload = {
        "symbols": {
            "tickers": ["CAPITALCOM:GOLD", "OANDA:XAUUSD", "FOREXCOM:XAUUSD"]
        },
        "columns": ["close", "high", "low", "open", "change", "change_abs"]
    }
    headers = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=6)
        if res.status_code == 200:
            rows = res.json().get("data", [])
            for row in rows:
                sym = row.get("s", "")
                d = row.get("d", [])
                if d and d[0] is not None and float(d[0]) > 0:
                    price = round(float(d[0]), 2)
                    print(f"TradingView Live Price loaded ({sym}): ${price:.2f}")
                    return price, sym
    except Exception as e:
        print(f"TradingView Scanner API note: {e}")
    return 0.0, "N/A"

def passes_stage1_filter(title, description):
    full_text = f"{title} {description}".lower()
    if any(kw in full_text for kw in CORE_MACRO_GEO_KEYWORDS):
        return True
    if any(te in full_text for te in TECH_ENTITIES):
        if any(trig in full_text for trig in TECH_SYSTEMIC_TRIGGERS):
            return True
    return False

def clean_html(raw_html):
    return re.sub(re.compile('<.*?>'), '', raw_html)

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
            if not passes_stage1_filter(title, clean_desc):
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
        for feed_articles in executor.map(fetch_single_feed, FEEDS.items()):
            for a in feed_articles:
                if a["link"] not in existing_urls:
                    articles.append(a)
    return articles

def fetch_live_perp_price():
    headers = {"User-Agent": "Mozilla/5.0"}
    for inst, lbl in [("XAU-USDT-SWAP", "OKX XAUUSDT.P"), ("XAUT-USDT-SWAP", "OKX XAUTUSDT.P")]:
        try:
            res = requests.get(f"https://www.okx.com/api/v5/market/ticker?instId={inst}", headers=headers, timeout=6)
            if res.status_code == 200:
                data = res.json().get("data", [])
                if data and data[0].get("last"):
                    return float(data[0]["last"]), lbl
        except Exception:
            pass
    try:
        res = requests.get("https://api.bybit.com/v5/market/tickers?category=linear&symbol=PAXGUSDT", headers=headers, timeout=6)
        if res.status_code == 200:
            lst = res.json().get("result", {}).get("list", [])
            if lst and lst[0].get("lastPrice"):
                return float(lst[0]["lastPrice"]), "Bybit Gold Perp"
    except Exception:
        pass
    return 0.0, "N/A"

def fetch_crypto_gold_basis(spot_price):
    crypto_price, source_used = fetch_live_perp_price()
    if crypto_price > 0 and spot_price <= 0:
        spot_price = crypto_price

    now_utc = datetime.utcnow()
    is_weekend = (now_utc.weekday() == 4 and now_utc.hour >= 22) or (now_utc.weekday() == 5) or (now_utc.weekday() == 6 and now_utc.hour < 22)

    if crypto_price > 0 and spot_price > 0:
        spread_usd = round(crypto_price - spot_price, 2)
        spread_pct = round((spread_usd / spot_price) * 100.0, 2)
        if is_weekend:
            regime = "WEEKEND GAP INDICATOR (Spot Closed)"
            signal = f"Bullish Sunday Gap (+${spread_usd:.2f})" if spread_usd >= 5.0 else (f"Bearish Sunday Gap (${spread_usd:+.2f})" if spread_usd <= -5.0 else f"Flat Weekend (${spread_usd:+.2f})")
        else:
            regime = "LIVE INTRADAY BASIS (Spot Open)"
            signal = f"High Perp Premium (+${spread_usd:.2f})" if spread_usd >= 8.0 else (f"Perp Discount (${spread_usd:+.2f})" if spread_usd <= -8.0 else f"Tight Parity (${spread_usd:+.2f})")

        prompt_str = (
            f"24/7 Crypto-Gold Perpetual Basis ({source_used} vs. CAPITALCOM:GOLD Spot):\n"
            f"- Market Regime: {regime} | {source_used}: ${crypto_price:.2f} vs. Spot: ${spot_price:.2f} (Spread: ${spread_usd:+.2f} / {spread_pct:+.2f}% -> {signal})"
        )
        print(f"24/7 Gold Perp loaded ({source_used}): ${crypto_price:.2f} | Spread vs Spot (${spot_price:.2f}): ${spread_usd:+.2f}")
        return prompt_str, {
            "crypto_price": round(crypto_price, 2),
            "oanda_price": round(spot_price, 2),
            "spread_usd": spread_usd,
            "spread_pct": spread_pct,
            "source": source_used,
            "is_weekend": is_weekend,
            "regime": "Weekend Gap" if is_weekend else "Live Perp Basis",
            "signal": signal
        }
    return "24/7 XAU/USDT Basis: Unavailable.", {}

def fetch_yahoo_futures_zones(reference_price, zone_width=10.0):
    comex_zones = {}
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=5d&interval=1h"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if res.status_code != 200:
            print(f"Yahoo COMEX Status Code: {res.status_code}")
            return comex_zones
        result = res.json().get("chart", {}).get("result", [])
        if not result:
            return comex_zones
        q = result[0].get("indicators", {}).get("quote", [{}])[0]
        closes, highs, lows, volumes = q.get("close", []), q.get("high", []), q.get("low", []), q.get("volume", [])

        candles = []
        for c, h, l, v in zip(closes, highs, lows, volumes):
            if c is not None and h is not None and l is not None and v is not None and v > 0:
                candles.append({"c": float(c), "h": float(h), "l": float(l), "v": float(v)})

        if not candles:
            return comex_zones
        if reference_price <= 0:
            reference_price = candles[-1]["c"]

        raw_entries = []
        total_adj_vol = 0.0
        n = len(candles)

        for idx, cndl in enumerate(candles):
            p = cndl["c"]
            if abs(p - reference_price) / reference_price > MAX_ZONE_RADIUS_PCT:
                continue
            mitigated = False
            if idx < n - 2:
                for later in candles[idx + 2:]:
                    if later["l"] <= p <= later["h"]:
                        mitigated = True
                        break
            recency_factor = 0.5 + 0.5 * (idx / max(1, n - 1))
            adj_v = cndl["v"] * recency_factor * (0.25 if mitigated else 1.0)
            raw_entries.append((p, adj_v))
            total_adj_vol += adj_v

        if total_adj_vol <= 0:
            return comex_zones

        for p, adj_v in raw_entries:
            zone_key = round(p / zone_width) * zone_width
            if zone_key not in comex_zones:
                comex_zones[zone_key] = {"total": 0.0, "weighted_price_sum": 0.0}
            vol_pct = (adj_v / total_adj_vol) * 100.0
            comex_zones[zone_key]["total"] += vol_pct
            comex_zones[zone_key]["weighted_price_sum"] += p * vol_pct
    except Exception as e:
        print(f"Note: Yahoo Finance Futures unreachable ({e})")
    return comex_zones

def fetch_multi_positioning(oanda_sentiment):
    headers = {"User-Agent": "Mozilla/5.0"}
    okx_long_pct = 50.0
    cot_mm_long_pct = 0.0
    cot_report_date = "N/A"

    for inst in ["XAU-USDT-SWAP", "XAUT-USDT-SWAP"]:
        try:
            url = f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader?instId={inst}&period=1H"
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                data = res.json().get("data", [])
                if data and len(data[0]) >= 2:
                    ratio = float(data[0][1])
                    okx_long_pct = round((ratio / (1.0 + ratio)) * 100.0, 1)
                    break
        except Exception:
            pass

    try:
        cot_url = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json?$where=cftc_contract_market_code='088691'&$order=report_date_as_yyyy_mm_dd%20DESC&$limit=1"
        res_cot = requests.get(cot_url, headers=headers, timeout=6)
        if res_cot.status_code == 200:
            rows = res_cot.json()
            if rows:
                r = rows[0]
                mm_l = float(r.get("m_money_positions_long_all", 0))
                mm_s = float(r.get("m_money_positions_short_all", 0))
                if (mm_l + mm_s) > 0:
                    cot_mm_long_pct = round((mm_l / (mm_l + mm_s)) * 100.0, 1)
                    cot_report_date = str(r.get("report_date_as_yyyy_mm_dd", ""))[:10]
    except Exception:
        pass

    return {
        "oanda_long_pct": oanda_sentiment.get("long_pct", 50.0),
        "oanda_short_pct": oanda_sentiment.get("short_pct", 50.0),
        "okx_top_long_pct": okx_long_pct,
        "okx_top_short_pct": round(100.0 - okx_long_pct, 1),
        "cot_mm_long_pct": cot_mm_long_pct,
        "cot_mm_short_pct": round(100.0 - cot_mm_long_pct, 1) if cot_mm_long_pct > 0 else 0.0,
        "cot_date": cot_report_date
    }

def get_clustered_oanda_zones(zone_width=10.0, fallback_price=0.0):
    current_price = float(fallback_price) if fallback_price else 0.0
    combined_buckets = {}
    total_pos_long, total_pos_short = 0.0, 0.0
    
    if OANDA_API_KEY:
        headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
        url_ob = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/orderBook"
        url_pb = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/positionBook"
        
        for attempt in range(2):
            try:
                res_ob = requests.get(url_ob, headers=headers, timeout=12)
                if res_ob.status_code == 200:
                    ob_data = res_ob.json().get("orderBook", {})
                    ob_price = float(ob_data.get("price", 0))
                    if current_price <= 0 and ob_price > 0:
                        current_price = ob_price
                    for b in ob_data.get("buckets", []):
                        try:
                            p = float(b.get("price", 0))
                            if current_price > 0 and abs(p - current_price) / current_price <= MAX_ZONE_RADIUS_PCT:
                                combined_buckets[p] = {
                                    "long": float(b.get("longCountPercent", 0)),
                                    "short": float(b.get("shortCountPercent", 0))
                                }
                        except ValueError:
                            continue
                    print(f"OANDA OrderBook loaded: {len(combined_buckets)} price buckets within 2.5% radius of ${current_price:.2f}.")
                    break
                else:
                    print(f"OANDA orderBook status {res_ob.status_code} (Attempt {attempt+1}/2): {res_ob.text[:120]}")
                    time.sleep(2)
            except Exception as e:
                print(f"OANDA orderBook error (Attempt {attempt+1}/2): {e}")
                time.sleep(2)

        for attempt in range(2):
            try:
                res_pb = requests.get(url_pb, headers=headers, timeout=12)
                if res_pb.status_code == 200:
                    pb_data = res_pb.json().get("positionBook", {})
                    pb_price = float(pb_data.get("price", 0))
                    if current_price <= 0 and pb_price > 0:
                        current_price = pb_price
                    for b in pb_data.get("buckets", []):
                        try:
                            p = float(b.get("price", 0))
                            l_pct = float(b.get("longCountPercent", 0))
                            s_pct = float(b.get("shortCountPercent", 0))
                            total_pos_long += l_pct
                            total_pos_short += s_pct
                            if current_price > 0 and abs(p - current_price) / current_price <= MAX_ZONE_RADIUS_PCT:
                                if p not in combined_buckets:
                                    combined_buckets[p] = {"long": 0.0, "short": 0.0}
                                combined_buckets[p]["long"] += l_pct
                                combined_buckets[p]["short"] += s_pct
                        except ValueError:
                            continue
                    print(f"OANDA PositionBook loaded (Total active near-spot buckets: {len(combined_buckets)}).")
                    break
                else:
                    print(f"OANDA positionBook status {res_pb.status_code} (Attempt {attempt+1}/2): {res_pb.text[:120]}")
                    time.sleep(2)
            except Exception as e:
                print(f"OANDA positionBook error (Attempt {attempt+1}/2): {e}")
                time.sleep(2)

    pos_sum = total_pos_long + total_pos_short
    sentiment = {
        "long_pct": round((total_pos_long / pos_sum) * 100, 1) if pos_sum > 0 else 50.0,
        "short_pct": round((total_pos_short / pos_sum) * 100, 1) if pos_sum > 0 else 50.0
    }

    zones = {}
    for p, vols in combined_buckets.items():
        zone_key = round(p / zone_width) * zone_width
        if zone_key not in zones:
            zones[zone_key] = {
                "long": 0.0, "short": 0.0, "total": 0.0, "weighted_price_sum": 0.0,
                "oanda_vol": 0.0, "comex_vol": 0.0
            }
        tot = vols["long"] + vols["short"]
        zones[zone_key]["long"] += vols["long"]
        zones[zone_key]["short"] += vols["short"]
        zones[zone_key]["total"] += tot
        zones[zone_key]["oanda_vol"] += tot
        zones[zone_key]["weighted_price_sum"] += p * tot

    comex_zones = fetch_yahoo_futures_zones(current_price, zone_width=zone_width)
    for z_key, f_data in comex_zones.items():
        if current_price > 0 and abs(z_key - current_price) / current_price > MAX_ZONE_RADIUS_PCT:
            continue
        weight = 0.25 if combined_buckets else 1.0
        scaled_tot = f_data["total"] * weight
        if z_key not in zones:
            zones[z_key] = {
                "long": 0.0, "short": 0.0, "total": 0.0, "weighted_price_sum": 0.0,
                "oanda_vol": 0.0, "comex_vol": 0.0
            }
        zones[z_key]["long"] += scaled_tot / 2.0
        zones[z_key]["short"] += scaled_tot / 2.0
        zones[z_key]["total"] += scaled_tot
        zones[z_key]["comex_vol"] += f_data["total"]
        zones[z_key]["weighted_price_sum"] += f_data["weighted_price_sum"] * weight

    if not zones or current_price == 0:
        return current_price, [], sentiment, {}

    above_zones, below_zones = [], []
    all_zone_objects = {}
    for z_key, z_data in zones.items():
        if z_data["total"] > 0:
            vwap_price = round(z_data["weighted_price_sum"] / z_data["total"], 2)
            net_diff = round(z_data["long"] - z_data["short"], 2)
            if z_data["oanda_vol"] == 0:
                dominance = "SELL" if vwap_price >= current_price else "BUY"
            else:
                dominance = "BUY" if net_diff >= 0 else "SELL"
            
            active_src = []
            if z_data["oanda_vol"] >= 0.4:
                active_src.append("OANDA")
            if z_data["comex_vol"] >= 0.5:
                active_src.append("COMEX")
            is_overlap = len(active_src) >= 2
            sources_label = "+".join(active_src) if active_src else "OANDA"

            zone_obj = {
                "price": vwap_price,
                "zone_center": z_key,
                "long": round(z_data["long"], 2),
                "short": round(z_data["short"], 2),
                "volume": round(z_data["total"], 2),
                "oanda_vol": round(z_data["oanda_vol"], 2),
                "comex_vol": round(z_data["comex_vol"], 2),
                "net": net_diff,
                "dominance": dominance,
                "overlap": is_overlap,
                "sources": sources_label
            }
            all_zone_objects[int(z_key)] = zone_obj
            if vwap_price >= current_price:
                above_zones.append(zone_obj)
            else:
                below_zones.append(zone_obj)

    above_zones.sort(key=lambda x: x["volume"], reverse=True)
    below_zones.sort(key=lambda x: x["volume"], reverse=True)
    selected_zones = above_zones[:3] + below_zones[:3]
    selected_zones.sort(key=lambda x: x["price"], reverse=True)

    return current_price, selected_zones, sentiment, all_zone_objects

# ==============================================================================
# 3D SURFACE ANALYTICS ENGINE (MODELS 1, 2 & 3)
# ==============================================================================
def build_3d_surfaces(current_price, all_zones_dict, h1_candles, vol_metrics, sentiment, multi_pos, old_ob_history=None):
    spot = current_price if current_price > 0 else (h1_candles[-1]["c"] if h1_candles else 3850.0)
    center_10 = int(round(spot / 10.0) * 10)
    price_grid = [center_10 + offset for offset in range(-50, 60, 10)]

    # MODEL 2: ORDERBOOK TOPOGRAPHY OVER TIME
    history_list = list(old_ob_history) if isinstance(old_ob_history, list) else []
    now_str = datetime.utcnow().strftime("%H:%M")

    curr_snapshot_levels = {}
    for p_lvl in price_grid:
        z_item = all_zones_dict.get(p_lvl)
        if z_item:
            sign = 1.0 if z_item["dominance"] == "BUY" else -1.0
            curr_snapshot_levels[str(p_lvl)] = {
                "vol": round(z_item["volume"], 2),
                "signed_vol": round(z_item["volume"] * sign, 2),
                "dom": z_item["dominance"]
            }
        else:
            curr_snapshot_levels[str(p_lvl)] = {
                "vol": 0.4,
                "signed_vol": 0.2 if p_lvl < spot else -0.2,
                "dom": "BUY" if p_lvl < spot else "SELL"
            }

    if len(history_list) < 10 and h1_candles:
        seed_candles = h1_candles[-12:]
        seeded = []
        for idx, c in enumerate(seed_candles[:-1]):
            t_lbl = c["dt_str"][-5:]
            c_levels = {}
            for p_lvl in price_grid:
                base_z = curr_snapshot_levels[str(p_lvl)]["vol"]
                dist_to_c = abs(p_lvl - c["c"])
                swept = (c["l"] <= p_lvl <= c["h"])
                mod = 0.65 if swept else (1.15 if dist_to_c <= 20 else 0.92)
                wave = 1.0 + 0.12 * math.sin(idx * 0.7 + (p_lvl / 20.0))
                v_val = round(max(0.2, base_z * mod * wave), 2)
                dom = "BUY" if p_lvl < c["c"] else "SELL"
                s_val = round(v_val if dom == "BUY" else -v_val, 2)
                c_levels[str(p_lvl)] = {"vol": v_val, "signed_vol": s_val}
            seeded.append({"time": t_lbl, "spot": round(c["c"], 2), "levels": c_levels})
        history_list = seeded + history_list

    if not history_list or history_list[-1].get("time") != now_str:
        history_list.append({
            "time": now_str,
            "spot": round(spot, 2),
            "levels": curr_snapshot_levels
        })
    else:
        history_list[-1]["spot"] = round(spot, 2)
        history_list[-1]["levels"] = curr_snapshot_levels

    history_list = history_list[-24:]

    ob_timestamps = [h["time"] for h in history_list]
    ob_vol_matrix = []
    ob_signed_matrix = []
    for p_lvl in price_grid:
        row_vol = []
        row_signed = []
        for h in history_list:
            cell = h.get("levels", {}).get(str(p_lvl))
            if not cell:
                cell = curr_snapshot_levels.get(str(p_lvl), {"vol": 0.5, "signed_vol": 0.0})
            row_vol.append(cell.get("vol", 0.5))
            row_signed.append(cell.get("signed_vol", 0.0))
        ob_vol_matrix.append(row_vol)
        ob_signed_matrix.append(row_signed)

    ob_surface = {
        "timestamps": ob_timestamps,
        "price_levels": price_grid,
        "vol_matrix": ob_vol_matrix,
        "signed_matrix": ob_signed_matrix,
        "spot_price": round(spot, 2)
    }

    # MODEL 1: 3D IMPLIED VOLATILITY & SKEW SURFACE
    daily_atr = float((vol_metrics or {}).get("est_daily_atr", 38.0) or 38.0)
    atr_ratio = float((vol_metrics or {}).get("move_to_daily_atr_ratio", 0.7) or 0.7)
    base_iv = round(max(11.0, min(42.0, (daily_atr / max(1.0, spot)) * math.sqrt(252) * 100.0)), 2)

    chg_24h = float((vol_metrics or {}).get("change_24h", 0.0) or 0.0)
    cot_long = float((multi_pos or {}).get("cot_mm_long_pct", 55.0) or 55.0)
    call_put_skew = round(((chg_24h / max(10.0, daily_atr)) * 1.4) + ((cot_long - 50.0) * 0.06), 2)

    strike_offsets = [-80, -60, -40, -20, 0, 20, 40, 60, 80]
    strikes = [int(center_10 + off) for off in strike_offsets]
    tenors = ["1D (Event)", "3D (Swing)", "1W (Weekly)", "2W (Bi-Weekly)", "1M (Monthly)"]
    tenor_days = [1, 3, 7, 14, 30]

    iv_matrix = []
    for d_days in tenor_days:
        row_iv = []
        term_mult = 1.0 + (atr_ratio - 0.75) * (0.28 / math.sqrt(max(1.0, d_days / 2.0)))
        for s_off in strike_offsets:
            moneyness = s_off / max(10.0, daily_atr * 2.0)
            smile = 2.4 * (moneyness ** 2) * (1.0 / math.sqrt(max(1.0, d_days / 3.0)))
            skew_shift = call_put_skew * moneyness * (1.1 if s_off > 0 else 0.7)
            iv_val = round(max(9.5, base_iv * term_mult + smile + skew_shift), 2)
            row_iv.append(iv_val)
        iv_matrix.append(row_iv)

    skew_regime = "BULLISH CALL SKEW (Upside Breakout Demand)" if call_put_skew >= 0.5 else ("BEARISH PUT SKEW (Downside Hedge Demand)" if call_put_skew <= -0.5 else "SYMMETRIC SMILE (Balanced Vol Regime)")
    vol_surface = {
        "strikes": strikes,
        "tenors": tenors,
        "iv_matrix": iv_matrix,
        "base_iv_pct": base_iv,
        "skew_score": call_put_skew,
        "skew_regime": skew_regime
    }

    # MODEL 3: 3D SESSION & VOLATILITY PROFILE
    days_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    hours_labels = [f"{h:02d}:00" for h in range(24)]

    range_sums = [[0.0 for _ in range(24)] for _ in range(5)]
    bias_sums = [[0.0 for _ in range(24)] for _ in range(5)]
    counts = [[0 for _ in range(24)] for _ in range(5)]

    for c in (h1_candles or []):
        dt_obj = datetime.utcfromtimestamp(c["ts"])
        w_day = dt_obj.weekday()
        if w_day > 4:
            continue
        hr = dt_obj.hour
        rng = max(0.2, c["h"] - c["l"])
        net_move = c["c"] - c["o"]
        range_sums[w_day][hr] += rng
        bias_sums[w_day][hr] += net_move
        counts[w_day][hr] += 1

    range_matrix = []
    bias_matrix = []
    best_asian_hr = "01:00 UTC"
    best_asian_val = -999.0

    for d_i in range(5):
        r_row = []
        b_row = []
        for h_i in range(24):
            cnt = counts[d_i][h_i]
            if cnt > 0:
                avg_r = round(range_sums[d_i][h_i] / cnt, 2)
                avg_b = round(bias_sums[d_i][h_i] / cnt, 2)
            else:
                avg_r = round(daily_atr / 5.5, 2)
                avg_b = 0.0
            r_row.append(avg_r)
            b_row.append(avg_b)
        range_matrix.append(r_row)
        bias_matrix.append(b_row)

    for h_i in range(0, 9):
        avg_hr_range = sum(range_matrix[d][h_i] for d in range(5)) / 5.0
        if avg_hr_range > best_asian_val:
            best_asian_val = avg_hr_range
            best_asian_hr = f"{h_i:02d}:00 UTC (Avg ${avg_hr_range:.2f}/h)"

    session_surface = {
        "hours": hours_labels,
        "days": days_labels,
        "range_matrix": range_matrix,
        "bias_matrix": bias_matrix,
        "best_asian_window": best_asian_hr
    }

    return ob_surface, vol_surface, session_surface, history_list

def fetch_oanda_orderbook(top_zones_cache=None, multi_pos=None):
    if top_zones_cache is not None:
        current_price, top_zones, sentiment = top_zones_cache[:3]
    else:
        current_price, top_zones, sentiment, _ = get_clustered_oanda_zones(zone_width=10.0)
    if not top_zones:
        return "No orderbook zones available."

    pos_line = f"OANDA Retail: {sentiment['long_pct']}% Long"
    if multi_pos:
        pos_line += f" | OKX Top Traders: {multi_pos.get('okx_top_long_pct', 50)}% Long"
        if multi_pos.get("cot_mm_long_pct", 0) > 0:
            pos_line += f" | CFTC Managed Money (Hedge Funds): {multi_pos['cot_mm_long_pct']}% Net Long (as of {multi_pos['cot_date']})"

    liquidity_info = (
        f"Current CAPITALCOM:GOLD / XAUUSD Price: {current_price}\n"
        f"Multi-Source Positioning: {pos_line}\n"
        f"Liquidity Zones (OANDA Resting + Unmitigated COMEX Volume within 2.5% radius):\n"
    )
    for z in top_zones:
        pos_label = "ABOVE price" if z["price"] >= current_price else "BELOW price"
        overlap_tag = f"CONFLUENCE [{z['sources']}]" if z.get("overlap") else f"[{z.get('sources', 'OANDA')}]"
        liquidity_info += (
            f"- Zone at ${z['price']} ({pos_label}) {overlap_tag} | Combined Vol: {z['volume']}% "
            f"[OANDA: {z['oanda_vol']}% | COMEX: {z['comex_vol']}%] (Net {z['dominance']})\n"
        )
    return liquidity_info

def fetch_aggressive_taker_flow(h1_candles, oanda_sentiment=None):
    headers = {"User-Agent": "Mozilla/5.0"}
    taker_buy_vol, taker_sell_vol, funding_rate_pct = 0.0, 0.0, 0.0

    for inst in ["XAU-USDT-SWAP", "XAUT-USDT-SWAP"]:
        try:
            res = requests.get(f"https://www.okx.com/api/v5/market/trades?instId={inst}&limit=100", headers=headers, timeout=6)
            if res.status_code == 200:
                for t in res.json().get("data", []):
                    sz = float(t.get("sz", 0))
                    if t.get("side") == "buy":
                        taker_buy_vol += sz
                    elif t.get("side") == "sell":
                        taker_sell_vol += sz
                if (taker_buy_vol + taker_sell_vol) > 0:
                    break
        except Exception:
            pass

    try:
        res_fr = requests.get("https://www.okx.com/api/v5/public/funding-rate?instId=XAU-USDT-SWAP", headers=headers, timeout=5)
        if res_fr.status_code == 200:
            fr_data = res_fr.json().get("data", [])
            if fr_data and fr_data[0].get("fundingRate"):
                funding_rate_pct = round(float(fr_data[0]["fundingRate"]) * 100.0, 4)
    except Exception:
        pass

    candle_buy_pressure = 50.0
    wick_note = "Balanced Intrabar Flow"
    if h1_candles and len(h1_candles) >= 4:
        buy_pts, sell_pts, lower_wick_sum, upper_wick_sum = 0.0, 0.0, 0.0, 0.0
        for c in h1_candles[-4:]:
            rng = max(0.01, c["h"] - c["l"])
            buy_pts += (c["c"] - c["l"]) / rng
            sell_pts += (c["h"] - c["c"]) / rng
            lower_wick_sum += (min(c["o"], c["c"]) - c["l"]) / rng
            upper_wick_sum += (c["h"] - max(c["o"], c["c"])) / rng
        if (buy_pts + sell_pts) > 0:
            candle_buy_pressure = round((buy_pts / (buy_pts + sell_pts)) * 100.0, 1)
        if lower_wick_sum > upper_wick_sum * 1.35:
            wick_note = "Strong Lower-Wick Dip Buying (Sell Walls Absorbed)"
        elif upper_wick_sum > lower_wick_sum * 1.35:
            wick_note = "Strong Upper-Wick Rejection (Selling at Highs)"

    tot_taker = taker_buy_vol + taker_sell_vol
    raw_taker_buy_pct = round((taker_buy_vol / tot_taker) * 100.0, 1) if tot_taker > 0 else candle_buy_pressure
    composite_buy_pct = round(0.60 * raw_taker_buy_pct + 0.40 * candle_buy_pressure, 1)
    composite_sell_pct = round(100.0 - composite_buy_pct, 1)

    retail_long = (oanda_sentiment or {}).get("long_pct", 50.0)
    if composite_buy_pct >= 58.0:
        state_label = "AGGRESSIVE MARKET BUYING (Ask Absorption)"
    elif composite_buy_pct <= 42.0:
        state_label = "AGGRESSIVE MARKET SELLING (Bid Sweep)"
    else:
        state_label = "TWO-WAY BALANCED EXECUTION"

    if retail_long >= 62.0 and composite_buy_pct <= 45.0:
        squeeze_warning = "HIGH LONG-STOP CASCADE RISK"
    elif retail_long <= 38.0 and composite_buy_pct >= 55.0:
        squeeze_warning = "HIGH SHORT-SQUEEZE RISK"
    else:
        squeeze_warning = "Moderate Stop-Run Risk"

    flow_obj = {
        "taker_buy_pct": composite_buy_pct,
        "taker_sell_pct": composite_sell_pct,
        "raw_tape_buy_pct": raw_taker_buy_pct,
        "candle_buy_pressure_pct": candle_buy_pressure,
        "funding_rate_pct": funding_rate_pct,
        "state_label": state_label,
        "wick_note": wick_note,
        "squeeze_warning": squeeze_warning
    }
    print(f"Aggressive Taker Flow loaded: {composite_buy_pct}% Buy / {composite_sell_pct}% Sell ({state_label})")
    prompt_str = f"Aggressive Taker Flow: {composite_buy_pct}% Market Buys vs. {composite_sell_pct}% Sells ({state_label}, {wick_note}, {squeeze_warning})"
    return prompt_str, flow_obj

def compute_dual_horizon_drivers(valid_clusters, stats_history=None):
    def build_ranked_list(items_list):
        driver_map = {}
        total_weight = 0.0
        for item in items_list:
            n_name = item["narrative"]
            w = item["weight"]
            total_weight += w
            if n_name not in driver_map:
                driver_map[n_name] = {
                    "narrative": n_name,
                    "category": item["category"],
                    "clusters_count": 0,
                    "sources_count": 0,
                    "weight": 0.0,
                    "bull_score": 0.0,
                    "bear_score": 0.0,
                    "prob_sum": 0
                }
            d = driver_map[n_name]
            d["clusters_count"] += 1
            d["sources_count"] += item["sources"]
            d["weight"] += w
            d["prob_sum"] += item["prob"]
            if "Bullish" in item["impact"]:
                d["bull_score"] += w
            elif "Bearish" in item["impact"]:
                d["bear_score"] += w

        out = []
        for d in driver_map.values():
            share_pct = round((d["weight"] / total_weight) * 100.0, 1) if total_weight > 0 else 0.0
            avg_prob = round(d["prob_sum"] / max(1, d["clusters_count"]))
            net_bias = "Bullish" if d["bull_score"] > d["bear_score"] * 1.15 else ("Bearish" if d["bear_score"] > d["bull_score"] * 1.15 else "Mixed")
            out.append({
                "narrative": d["narrative"],
                "category": d["category"],
                "share_pct": share_pct,
                "clusters_count": d["clusters_count"],
                "sources_count": d["sources_count"],
                "net_bias": net_bias,
                "avg_prob": avg_prob
            })
        out.sort(key=lambda x: x["share_pct"], reverse=True)
        return out[:4]

    items_24h = []
    for c in (valid_clusters or []):
        s = c.get("summary") if isinstance(c.get("summary"), dict) else {}
        if s.get("Is_Gold_Relevant") is False:
            continue
        arts = c.get("articles", [])
        if not arts:
            continue
        text = " ".join([f"{a.get('title', '')} {a.get('description', '')}" for a in arts[:3]])
        n_name, b_cat = detect_specific_narrative(text)
        prob = int(s.get("Probability_Percent", 50) or 50)
        imp = str(s.get("XAUUSD_Impact", "Neutral"))
        w = len(arts) * (0.6 + (prob / 100.0))
        items_24h.append({"narrative": n_name, "category": b_cat, "sources": len(arts), "prob": prob, "impact": imp, "weight": w})

    drivers_24h = build_ranked_list(items_24h)

    items_weekly = []
    cutoff_14d = datetime.now().timestamp() - (14 * 86400)
    seen_ids = set()

    for c in (valid_clusters or []):
        s = c.get("summary") if isinstance(c.get("summary"), dict) else {}
        if s.get("Is_Gold_Relevant") is False:
            continue
        arts = c.get("articles", [])
        if not arts:
            continue
        cid = arts[0].get("link", "")
        seen_ids.add(cid)
        text = " ".join([f"{a.get('title', '')} {a.get('description', '')}" for a in arts[:3]])
        n_name, b_cat = detect_specific_narrative(text)
        prob = int(s.get("Probability_Percent", 50) or 50)
        imp = str(s.get("XAUUSD_Impact", "Neutral"))
        tf = str(s.get("Timeframe", "Hours/Days"))
        tf_mult = 3.0 if "Weeks" in tf and "Months" in tf else (2.0 if "Weeks" in tf else 0.7)
        w = len(arts) * (0.6 + (prob / 100.0)) * tf_mult
        items_weekly.append({"narrative": n_name, "category": b_cat, "sources": len(arts), "prob": prob, "impact": imp, "weight": w})

    if isinstance(stats_history, dict):
        for sig_id, rec in stats_history.items():
            if sig_id in seen_ids:
                continue
            ts = float(rec.get("created_ts", 0))
            if ts < cutoff_14d:
                continue
            hl = rec.get("headline", "")
            n_name, b_cat = detect_specific_narrative(hl)
            prob = int(rec.get("prob", 60) or 60)
            imp = str(rec.get("impact", "Bullish"))
            age_days = max(0.0, (datetime.now().timestamp() - ts) / 86400.0)
            decay = max(0.35, 1.0 - (age_days / 18.0))
            w = 2.2 * (0.6 + (prob / 100.0)) * decay
            items_weekly.append({"narrative": n_name, "category": b_cat, "sources": 2, "prob": prob, "impact": imp, "weight": w})

    drivers_weekly = build_ranked_list(items_weekly) if items_weekly else drivers_24h

    prompt_lines = ["Dual-Horizon Market Narratives:"]
    prompt_lines.append("- Structural Weekly Regime (7-14d): " + " | ".join([f"#{i+1} {d['narrative']} ({d['share_pct']}%, {d['net_bias']})" for i, d in enumerate(drivers_weekly)]))
    prompt_lines.append("- Dominant 24h Intraday Pulse: " + " | ".join([f"#{i+1} {d['narrative']} ({d['share_pct']}%, {d['net_bias']})" for i, d in enumerate(drivers_24h)]))
    return "\n".join(prompt_lines), drivers_24h, drivers_weekly

def fetch_oanda_price_and_volatility(tv_live_price=0.0):
    parsed = []
    if OANDA_API_KEY:
        url = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/candles?count=500&granularity=H1&price=M"
        headers = {"Authorization": f"Bearer {OANDA_API_KEY}", "Accept-Datetime-Format": "UNIX"}
        for attempt in range(2):
            try:
                response = requests.get(url, headers=headers, timeout=12)
                if response.status_code == 200:
                    raw_candles = response.json().get("candles", [])
                    for c in raw_candles:
                        ts = float(c.get("time", 0))
                        parsed.append({
                            "ts": ts,
                            "dt_str": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M"),
                            "o": float(c["mid"]["o"]),
                            "h": float(c["mid"]["h"]),
                            "l": float(c["mid"]["l"]),
                            "c": float(c["mid"]["c"])
                        })
                    break
                else:
                    print(f"OANDA H1 Candle API Status {response.status_code} (Attempt {attempt+1}/2): {response.text[:120]}")
                    time.sleep(2)
            except Exception as e:
                print(f"OANDA H1 Candle Error (Attempt {attempt+1}/2): {e}")
                time.sleep(2)

    if len(parsed) < 15:
        eff_p = tv_live_price if tv_live_price > 0 else 3850.0
        return f"XAU/USD Spot (CAPITALCOM:GOLD): ${eff_p:.2f}", parsed, {
            "current_price": eff_p,
            "atr_h1": 9.5,
            "est_daily_atr": 38.0,
            "change_4h": 0.0,
            "change_24h": 0.0,
            "move_to_daily_atr_ratio": 0.5
        }

    if tv_live_price > 0:
        parsed[-1]["c"] = tv_live_price
        parsed[-1]["h"] = max(parsed[-1]["h"], tv_live_price)
        parsed[-1]["l"] = min(parsed[-1]["l"], tv_live_price)

    trs = [max(parsed[i]["h"] - parsed[i]["l"], abs(parsed[i]["h"] - parsed[i-1]["c"]), abs(parsed[i]["l"] - parsed[i-1]["c"])) for i in range(1, len(parsed))]
    atr_14_h1 = sum(trs[-14:]) / 14.0
    est_daily_atr = round(atr_14_h1 * 3.8, 2)
    curr_c = parsed[-1]["c"]
    change_4h = round(curr_c - parsed[-5]["c"], 2) if len(parsed) >= 5 else 0.0
    change_24h = round(curr_c - parsed[-25]["c"], 2) if len(parsed) >= 25 else 0.0
    ratio_4h = round(abs(change_4h) / est_daily_atr, 2) if est_daily_atr > 0 else 0.0

    vol_metrics = {
        "current_price": round(curr_c, 2),
        "atr_h1": round(atr_14_h1, 2),
        "est_daily_atr": est_daily_atr,
        "change_4h": change_4h,
        "change_24h": change_24h,
        "move_to_daily_atr_ratio": ratio_4h
    }
    history_str = f"XAU/USD Volatility (CAPITALCOM:GOLD Synced): Spot ${curr_c:.2f} | 4h: ${change_4h:+.2f} | 24h: ${change_24h:+.2f} | Daily ATR: ${est_daily_atr:.2f} (4h Ratio: {ratio_4h}x)"
    return history_str, parsed, vol_metrics

def fetch_macro_context():
    if not OANDA_API_KEY:
        return "No macro data.", {}
    instruments = {
        "EUR_USD": "EUR/USD",
        "USB10Y_USD": "US 10Y Bond",
        "NAS100_USD": "Nasdaq 100",
        "BCO_USD": "Brent Oil"
    }
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    macro_lines = ["Cross-Asset Macro (24h Change):"]
    macro_summary = {}
    for instr, label in instruments.items():
        try:
            url = f"https://api-fxpractice.oanda.com/v3/instruments/{instr}/candles?count=24&granularity=H1&price=M"
            res = requests.get(url, headers=headers, timeout=8)
            if res.status_code == 200:
                candles = res.json().get("candles", [])
                if len(candles) >= 2:
                    first_c = float(candles[0]["mid"]["c"])
                    last_c = float(candles[-1]["mid"]["c"])
                    pct = round(((last_c - first_c) / first_c) * 100, 2) if first_c > 0 else 0.0
                    macro_summary[instr] = {"price": last_c, "change_24h_pct": pct}
                    macro_lines.append(f"- {label}: {last_c} ({pct:+.2f}%)")
        except Exception:
            pass
    return "\n".join(macro_lines), macro_summary

def compute_regime_matrix(history_dict):
    def init_b():
        return {"hits": 0, "misses": 0, "total": 0, "win_rate": 0.0, "dd_sum": 0.0, "dd_count": 0, "avg_dd": 0.0}

    matrix = {
        "by_direction": {"Bullish": init_b(), "Bearish": init_b()},
        "by_atr_regime": {"Fresh (<0.8x ATR)": init_b(), "Moderate (0.8x-1.2x ATR)": init_b(), "Exhausted (>=1.2x ATR)": init_b(), "Unrecorded (Legacy)": init_b()},
        "by_macro_alignment": {"With USD Tailwind": init_b(), "Against USD Headwind": init_b(), "USD Neutral / Legacy": init_b()},
        "by_category": {"Geopolitics & War": init_b(), "Fed, Rates & Macro": init_b(), "Tech, AI & Equities": init_b(), "Commodities & Energy": init_b()},
        "by_confidence": {"Moderate (51-64%)": init_b(), "High (65-74%)": init_b(), "Extreme (>=75%)": init_b()}
    }

    def add_b(b, is_hit, dd):
        b["hits" if is_hit else "misses"] += 1
        b["total"] += 1
        if dd and dd > 0:
            b["dd_sum"] += dd
            b["dd_count"] += 1

    for sig_id, rec in history_dict.items():
        if rec.get("status") not in ("HIT", "MISS"):
            continue
        is_hit = (rec.get("status") == "HIT")
        imp = "Bullish" if "Bullish" in str(rec.get("impact", "Bullish")) else "Bearish"
        dd = float(rec.get("max_dd_usd", 0.0) or 0.0)

        add_b(matrix["by_direction"][imp], is_hit, dd)
        atr_r = rec.get("atr_ratio_at_signal")
        atr_k = "Unrecorded (Legacy)" if atr_r is None else ("Fresh (<0.8x ATR)" if float(atr_r) < 0.8 else ("Moderate (0.8x-1.2x ATR)" if float(atr_r) < 1.2 else "Exhausted (>=1.2x ATR)"))
        add_b(matrix["by_atr_regime"][atr_k], is_hit, dd)

        eur_chg = rec.get("eur_chg_at_signal")
        if eur_chg is not None:
            mac_k = "With USD Tailwind" if ((imp == "Bullish" and eur_chg >= 0.12) or (imp == "Bearish" and eur_chg <= -0.12)) else ("Against USD Headwind" if ((imp == "Bullish" and eur_chg <= -0.12) or (imp == "Bearish" and eur_chg >= 0.12)) else "USD Neutral / Legacy")
        else:
            mac_k = "USD Neutral / Legacy"
        add_b(matrix["by_macro_alignment"][mac_k], is_hit, dd)

        cat = rec.get("category")
        if not cat or cat not in matrix["by_category"]:
            cat = classify_news_category(rec.get("headline", sig_id), rec.get("source", ""))
        add_b(matrix["by_category"][cat], is_hit, dd)

        prob = int(rec.get("prob", 67) or 67)
        conf_k = "Moderate (51-64%)" if prob < 65 else ("High (65-74%)" if prob < 75 else "Extreme (>=75%)")
        add_b(matrix["by_confidence"][conf_k], is_hit, dd)

    for group in matrix.values():
        for b in group.values():
            if b["total"] > 0:
                b["win_rate"] = round((b["hits"] / b["total"]) * 100.0, 1)
            if b["dd_count"] > 0:
                b["avg_dd"] = round(b["dd_sum"] / b["dd_count"], 2)
            del b["dd_sum"], b["dd_count"]
    return matrix

def evaluate_clusters(all_clusters, h1_candles, vol_metrics=None, macro_summary=None):
    stats = {"total_evaluated": 0, "hits": 0, "misses": 0, "win_rate": 0.0, "active_signals": 0, "regime_matrix": {}, "history": {}}
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    stats.update(loaded)
        except Exception:
            pass
    if not h1_candles:
        return stats

    now_ts = datetime.now().timestamp()
    curr_price = (vol_metrics or {}).get("current_price") or h1_candles[-1]["c"]
    est_daily_atr = (vol_metrics or {}).get("est_daily_atr") or 35.0
    atr_ratio_now = (vol_metrics or {}).get("move_to_daily_atr_ratio", 0.0)
    eur_chg = (macro_summary or {}).get("EUR_USD", {}).get("change_24h_pct", 0.0)
    nas_chg = (macro_summary or {}).get("NAS100_USD", {}).get("change_24h_pct", 0.0)

    cluster_map = {}
    for cluster in all_clusters:
        summary = cluster.get("summary")
        if not isinstance(summary, dict) or summary.get("Is_Gold_Relevant") is False:
            continue
        impact = summary.get("XAUUSD_Impact", "")
        prob = int(summary.get("Probability_Percent", 0) or 0)
        target_raw = str(summary.get("Orderflow_Price_Target", "")).replace("$", "").replace(",", "").strip()
        try:
            target_price = float(target_raw)
        except ValueError:
            continue
        if prob < MIN_EVAL_PROBABILITY or not ("Bullish" in impact or "Bearish" in impact):
            continue

        cluster_id = cluster["articles"][0]["link"] if cluster.get("articles") else str(cluster.get("_timestamp"))
        cluster_map[cluster_id] = cluster
        cluster_ts = float(cluster.get("_timestamp", now_ts))
        matching_candles = [c for c in h1_candles if c["ts"] <= cluster_ts]
        entry_price = matching_candles[-1]["c"] if matching_candles else curr_price

        if cluster_id not in stats["history"]:
            first_art = cluster["articles"][0] if cluster.get("articles") else {}
            hl_str = first_art.get("title", "Unknown")[:90]
            src_str = first_art.get("source", "News")
            stats["history"][cluster_id] = {
                "status": "ACTIVE",
                "headline": hl_str,
                "source": src_str,
                "category": classify_news_category(hl_str, src_str),
                "impact": "Bullish" if "Bullish" in impact else "Bearish",
                "prob": prob,
                "entry_price": round(entry_price, 2),
                "target": round(target_price, 2),
                "max_dd_usd": 0.0,
                "atr_ratio_at_signal": atr_ratio_now,
                "daily_atr_at_signal": est_daily_atr,
                "eur_chg_at_signal": eur_chg,
                "nas_chg_at_signal": nas_chg,
                "created_ts": cluster_ts,
                "date": cluster.get("main_date", ""),
                "fail_reason": ""
            }
        elif stats["history"][cluster_id].get("status") == "ACTIVE":
            stats["history"][cluster_id]["target"] = round(target_price, 2)
            stats["history"][cluster_id]["prob"] = prob

    for sig_id, rec in stats["history"].items():
        status = rec.get("status", "ACTIVE")
        if status in ("HIT", "MISS"):
            if sig_id in cluster_map:
                cluster_map[sig_id]["eval_status"] = status
                cluster_map[sig_id]["max_dd_usd"] = rec.get("max_dd_usd", 0.0)
            continue

        sig_ts = float(rec.get("created_ts", now_ts))
        entry_p = float(rec.get("entry_price", curr_price))
        target_p = float(rec.get("target", 0))
        imp = rec.get("impact", "Bullish")
        sig_atr = float(rec.get("daily_atr_at_signal", est_daily_atr)) or 35.0
        max_allowed_dd = round(sig_atr * MAX_DRAWDOWN_ATR_MULT, 2)

        hit_achieved, stopped_out = False, False
        worst_dd = float(rec.get("max_dd_usd", 0.0))

        for c in [c for c in h1_candles if c["ts"] + 3600 >= sig_ts]:
            if imp == "Bullish":
                dd = round(max(0.0, entry_p - c["l"]), 2)
                worst_dd = max(worst_dd, dd)
                if dd >= max_allowed_dd:
                    stopped_out = True
                    break
                if c["h"] >= target_p:
                    hit_achieved = True
                    break
            else:
                dd = round(max(0.0, c["h"] - entry_p), 2)
                worst_dd = max(worst_dd, dd)
                if dd >= max_allowed_dd:
                    stopped_out = True
                    break
                if c["l"] <= target_p:
                    hit_achieved = True
                    break

        rec["max_dd_usd"] = worst_dd
        if hit_achieved:
            rec["status"] = "HIT"
        elif stopped_out or ((now_ts - sig_ts) / 3600.0 >= EVAL_HORIZON_HOURS):
            rec["status"] = "MISS"
        else:
            rec["status"] = "ACTIVE"

        if sig_id in cluster_map:
            cluster_map[sig_id]["eval_status"] = rec["status"]
            cluster_map[sig_id]["max_dd_usd"] = worst_dd

    hits = sum(1 for v in stats["history"].values() if v.get("status") == "HIT")
    misses = sum(1 for v in stats["history"].values() if v.get("status") == "MISS")
    total_eval = hits + misses
    stats["hits"] = hits
    stats["misses"] = misses
    stats["total_evaluated"] = total_eval
    stats["win_rate"] = round((hits / total_eval) * 100, 1) if total_eval > 0 else 0.0
    stats["active_signals"] = sum(1 for v in stats["history"].values() if v.get("status") == "ACTIVE")
    stats["regime_matrix"] = compute_regime_matrix(stats["history"])

    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
    except Exception:
        pass
    return stats

def format_performance_memory(stats):
    if not stats or stats.get("total_evaluated", 0) == 0:
        return "Historical Track Record: Calibrating."
    rm = stats.get("regime_matrix") or compute_regime_matrix(stats.get("history", {}))
    def fmt_g(g):
        return " | ".join([f"{k}: {b['win_rate']}% ({b['hits']}/{b['total']})" for k, b in g.items() if b["total"] > 0])
    return (
        f"=== EMPIRICAL REGIME MATRIX ({stats['total_evaluated']} EVALUATED SETUPS, {stats['win_rate']}% WIN RATE) ===\n"
        f"- By Direction: {fmt_g(rm.get('by_direction', {}))}\n"
        f"- By Volatility / ATR: {fmt_g(rm.get('by_atr_regime', {}))}\n"
        f"- By USD Alignment: {fmt_g(rm.get('by_macro_alignment', {}))}\n"
        f"- By Category: {fmt_g(rm.get('by_category', {}))}\n"
    )

# ==============================================================================
# TAB [4] MASTER AI TRADE JOURNAL & SELF-LEARNING ENGINE
# ==============================================================================
def load_trade_journal():
    default_journal = {
        "active_trade": None,
        "last_decision": None,
        "stats": {
            "total_closed": 0,
            "wins": 0,
            "losses": 0,
            "expired": 0,
            "invalidated": 0,
            "win_rate": 0.0,
            "net_pnl_usd": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0
        },
        "archive": []
    }
    if os.path.exists(TRADE_JOURNAL_FILE):
        try:
            with open(TRADE_JOURNAL_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    default_journal.update(loaded)
        except Exception:
            pass
    return default_journal

def save_trade_journal(journal):
    try:
        with open(TRADE_JOURNAL_FILE, "w", encoding="utf-8") as f:
            json.dump(journal, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving {TRADE_JOURNAL_FILE}: {e}")

def recompute_trade_journal_stats(journal):
    archive = journal.get("archive", [])
    wins = [t for t in archive if t.get("status") == "WIN"]
    losses = [t for t in archive if t.get("status") == "LOSS"]
    expired = [t for t in archive if t.get("status") == "EXPIRED"]
    invalidated = [t for t in archive if t.get("status") == "INVALIDATED"]

    decisive_count = len(wins) + len(losses)
    total_closed = len(archive)
    net_pnl = round(sum(float(t.get("pnl_usd", 0.0)) for t in archive), 2)
    avg_win = round(sum(float(t.get("pnl_usd", 0.0)) for t in wins) / len(wins), 2) if wins else 0.0
    avg_loss = round(sum(float(t.get("pnl_usd", 0.0)) for t in losses) / len(losses), 2) if losses else 0.0

    journal["stats"] = {
        "total_closed": total_closed,
        "wins": len(wins),
        "losses": len(losses),
        "expired": len(expired),
        "invalidated": len(invalidated),
        "win_rate": round((len(wins) / decisive_count) * 100.0, 1) if decisive_count > 0 else 0.0,
        "net_pnl_usd": net_pnl,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss
    }

def format_trade_journal_lessons(journal):
    st = journal.get("stats", {})
    archive = journal.get("archive", [])
    if not archive:
        return "=== MASTER TRADE ARCHIVE LESSONS ===\nNo closed master trades yet. Be highly selective: only issue LONG or SHORT if Chart Orderbook, News Catalysts, and 3D Surfaces align with >= $10.00 target distance."

    lines = [
        f"=== MASTER TRADE ARCHIVE LESSONS ({st.get('total_closed', 0)} Closed | Winrate: {st.get('win_rate', 0)}% | Net PnL: ${st.get('net_pnl_usd', 0):+.2f} | Avg Win: +${st.get('avg_win_usd', 0):.2f} | Avg Loss: ${st.get('avg_loss_usd', 0):.2f}) ===",
        "Recent Closed Master Trades (Learn from wins, stop-outs, and expirations):"
    ]
    for t in archive[:8]:
        lines.append(
            f"- [{t.get('status')}] {t.get('direction')} @ ${t.get('entry_price')} -> TP ${t.get('tp_price')} | SL ${t.get('sl_price')} | "
            f"Exit @ ${t.get('exit_price')} (PnL: ${t.get('pnl_usd', 0):+.2f}, MaxDD: -${t.get('max_dd_usd', 0):.2f}) | "
            f"Setup: {t.get('setup_title')} | Outcome Note: {t.get('close_reason')}"
        )
    return "\n".join(lines)

def call_gemini_structured(prompt_text, response_schema):
    if not client:
        return None
    active_models = [m for m in FALLBACK_MODELS if m not in DEAD_MODELS] or ["gemini-3.5-flash-lite"]
    for model_name in active_models:
        for attempt in range(2):
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema
                    )
                )
                raw = res.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                return json.loads(raw)
            except Exception as e:
                err_str = str(e).splitlines()[0][:100]
                if "404" in err_str or "NOT_FOUND" in err_str:
                    DEAD_MODELS.add(model_name)
                    break
                time.sleep(3)
    return None

def manage_master_trade(current_spot, h1_candles, vol_metrics, oanda_zones_cache, macro_summary, crypto_basis_obj, taker_flow_obj, valid_clusters, drivers_24h, drivers_weekly, ob_surf, vol_surf, sess_surf, cluster_stats):
    """
    Evaluates any active Master Trade (TP, 1.2x ATR SL, 20h expiry, and AI thesis validity).
    If no active trade remains, asks the AI to synthesize Tab [1], [2], and [3] plus historical archive lessons
    into a high-conviction trade (min $10 TP distance) or NO_TRADE.
    """
    journal = load_trade_journal()
    now_ts = datetime.now().timestamp()
    now_utc_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    daily_atr = float((vol_metrics or {}).get("est_daily_atr", 38.0) or 38.0)

    # Top relevant news summary for prompt context
    rel_clusters = [
        c for c in valid_clusters
        if isinstance(c.get("summary"), dict) and c["summary"].get("Is_Gold_Relevant") is not False
    ][:10]
    news_briefs = []
    for idx, c in enumerate(rel_clusters):
        s = c["summary"]
        arts = sorted(c.get("articles", []), key=lambda x: x.get("_timestamp", 0), reverse=True)
        lead_art = arts[0] if arts else {}
        news_briefs.append(
            f"News #{idx+1} [{s.get('XAUUSD_Impact')} {s.get('Probability_Percent')}% | Driver: {s.get('Dominant_Driver')} | {c.get('main_date')}]: "
            f"{s.get('Facts')} | Macro: {s.get('Macro_Context')} | Source: {lead_art.get('source', 'News')} ({lead_art.get('link', '')})"
        )
    news_context_block = "\n".join(news_briefs) if news_briefs else "No major breaking news clusters in the last 24h."

    # 1. CHECK & UPDATE EXISTING ACTIVE TRADE
    active = journal.get("active_trade")
    if isinstance(active, dict) and active.get("status") == "ACTIVE":
        entry_ts = float(active.get("entry_ts", now_ts))
        entry_p = float(active.get("entry_price", current_spot))
        tp_p = float(active.get("tp_price", 0.0))
        sl_p = float(active.get("sl_price", 0.0))
        direction = active.get("direction", "LONG")
        elapsed_h = round((now_ts - entry_ts) / 3600.0, 2)

        hit_tp, hit_sl = False, False
        worst_dd = float(active.get("max_dd_usd", 0.0))
        best_fav = float(active.get("max_favorable_usd", 0.0))

        # Check H1 candles since trade opened + live spot price
        relevant_bars = [c for c in (h1_candles or []) if c["ts"] + 3600 >= entry_ts]
        for c in relevant_bars:
            if direction == "LONG":
                dd = round(max(0.0, entry_p - c["l"]), 2)
                fav = round(max(0.0, c["h"] - entry_p), 2)
                worst_dd = max(worst_dd, dd)
                best_fav = max(best_fav, fav)
                if c["l"] <= sl_p:
                    hit_sl = True
                    break
                if c["h"] >= tp_p:
                    hit_tp = True
                    break
            else:
                dd = round(max(0.0, c["h"] - entry_p), 2)
                fav = round(max(0.0, entry_p - c["l"]), 2)
                worst_dd = max(worst_dd, dd)
                best_fav = max(best_fav, fav)
                if c["h"] >= sl_p:
                    hit_sl = True
                    break
                if c["l"] <= tp_p:
                    hit_tp = True
                    break

        # Also check current live spot price
        if not hit_tp and not hit_sl and current_spot > 0:
            if direction == "LONG":
                worst_dd = max(worst_dd, round(max(0.0, entry_p - current_spot), 2))
                best_fav = max(best_fav, round(max(0.0, current_spot - entry_p), 2))
                if current_spot <= sl_p:
                    hit_sl = True
                elif current_spot >= tp_p:
                    hit_tp = True
            else:
                worst_dd = max(worst_dd, round(max(0.0, current_spot - entry_p), 2))
                best_fav = max(best_fav, round(max(0.0, entry_p - current_spot), 2))
                if current_spot >= sl_p:
                    hit_sl = True
                elif current_spot <= tp_p:
                    hit_tp = True

        active["current_price"] = round(current_spot, 2)
        active["elapsed_hours"] = elapsed_h
        active["remaining_hours"] = round(max(0.0, TRADE_EXPIRY_HOURS - elapsed_h), 2)
        active["max_dd_usd"] = worst_dd
        active["max_favorable_usd"] = best_fav
        active["unrealized_pnl_usd"] = round((current_spot - entry_p) if direction == "LONG" else (entry_p - current_spot), 2)

        close_status = None
        exit_price = current_spot
        close_reason = ""

        if hit_sl:
            close_status = "LOSS"
            exit_price = sl_p
            close_reason = f"Stop-Loss (1.2x ATR) triggered at ${sl_p:.2f} after {elapsed_h}h."
        elif hit_tp:
            close_status = "WIN"
            exit_price = tp_p
            close_reason = f"Take-Profit target ${tp_p:.2f} reached after {elapsed_h}h (Max Drawdown: -${worst_dd:.2f})."
        elif elapsed_h >= TRADE_EXPIRY_HOURS:
            close_status = "EXPIRED"
            exit_price = current_spot
            close_reason = f"20-hour time limit expired at ${current_spot:.2f} (Max Favorable: +${best_fav:.2f}, Max DD: -${worst_dd:.2f})."
        else:
            # Ask AI if active trade thesis is still valid given latest news & orderbook
            review_prompt = f"""You are a disciplined Gold (XAU/USD) Risk Manager.
            Review this ACTIVE {direction} trade opened at ${entry_p:.2f} ({elapsed_h}h ago, {active['remaining_hours']}h remaining):
            - Target TP: ${tp_p:.2f} | Hard SL (1.2x ATR): ${sl_p:.2f} | Current Spot: ${current_spot:.2f}
            - Original Thesis: {active.get('detailed_rationale')}
            - Latest Breaking News:
            {news_context_block}

            Question: Does this trade thesis STILL make fundamental and technical sense, or has a major opposing shock invalidated it?
            Set Still_Valid = true unless a strong contrary catalyst clearly invalidates the setup."""
            rev_res = call_gemini_structured(review_prompt, ACTIVE_TRADE_REVIEW_SCHEMA)
            if isinstance(rev_res, dict):
                active["last_check_utc"] = now_utc_str
                active["last_check_note"] = rev_res.get("Status_Note", "Thesis verified intact.")
                if rev_res.get("Still_Valid") is False:
                    close_status = "INVALIDATED"
                    exit_price = current_spot
                    close_reason = f"Invalidated by AI Risk Check at ${current_spot:.2f}: {rev_res.get('Status_Note', 'Thesis broken.')}"

        if close_status:
            pnl_usd = round((exit_price - entry_p) if direction == "LONG" else (entry_p - exit_price), 2)
            active["status"] = close_status
            active["exit_price"] = round(exit_price, 2)
            active["closed_utc"] = now_utc_str
            active["closed_ts"] = now_ts
            active["pnl_usd"] = pnl_usd
            active["close_reason"] = close_reason
            journal["archive"].insert(0, active)
            journal["archive"] = journal["archive"][:50]
            journal["active_trade"] = None
            recompute_trade_journal_stats(journal)
            save_trade_journal(journal)
            print(f"Master Trade Closed [{close_status}]: {direction} Entry ${entry_p} -> Exit ${exit_price} (PnL: ${pnl_usd:+.2f})")
        else:
            save_trade_journal(journal)
            print(f"Active Master Trade maintained: {direction} @ ${entry_p} -> TP ${tp_p} ({active['remaining_hours']}h left)")
            return journal

    # 2. NO ACTIVE TRADE: GENERATE NEW MASTER TRADE PROPOSAL (OR NO_TRADE)
    if current_spot <= 0:
        return journal

    sl_dist_usd = round(daily_atr * TRADE_SL_ATR_MULT, 2)
    ob_str = fetch_oanda_orderbook(top_zones_cache=oanda_zones_cache)
    archive_lessons = format_trade_journal_lessons(journal)
    perf_str = format_performance_memory(cluster_stats)

    # Summarize 3D Surface metrics for the AI
    vol_s_summary = f"Model 1 (IV & Skew): Base ATM IV = {vol_surf.get('base_iv_pct')}% | Skew Score = {vol_surf.get('skew_score'):+.2f} ({vol_surf.get('skew_regime')})"
    sess_s_summary = f"Model 3 (Session Profile): Best Asian Window = {sess_surf.get('best_asian_window')} | Current UTC Hour = {datetime.utcnow().strftime('%H:00 UTC')}"
    ob_s_summary = f"Model 2 (Liquidity Canyon): Spot Ref = ${ob_surf.get('spot_price')} across 24 UTC snapshots"

    master_prompt = f"""You are the Chief Quantitative & Macro Strategist for GMIT // XAU (Gold XAU/USD).
    Synthesize ALL data across the entire terminal ([1] Chart & Orderbook, [2] News Intelligence, [3] 3D Surfaces, and the Historical Trade Archive) to decide whether to open a new XAU/USD trade right now.

    STRICT MANDATORY RULES (ZERO EXCEPTIONS):
    1. **Minimum Target Distance >= $10.00:** If you choose 'LONG', Take_Profit_Price MUST be at least ${current_spot + TRADE_MIN_TP_DISTANCE_USD:.2f} (+$10.00 above current spot ${current_spot:.2f}). If you choose 'SHORT', Take_Profit_Price MUST be at most ${current_spot - TRADE_MIN_TP_DISTANCE_USD:.2f} (-$10.00 below current spot ${current_spot:.2f}).
    2. **Selective High-Conviction Only ('NO_TRADE' when unclear):** Do NOT force a trade! If signals conflict, if ATR is already exhausted (>= 1.2x Daily ATR), or if no clean >= $10.00 setup exists, set Decision = 'NO_TRADE' and Take_Profit_Price = 0.
    3. **Fixed Risk Parameters:** Every trade has a 20-hour expiration limit and an automatic Stop-Loss at 1.2x Daily ATR (${sl_dist_usd:.2f} distance -> Long SL: ${current_spot - sl_dist_usd:.2f} | Short SL: ${current_spot + sl_dist_usd:.2f}). Ensure your TP aligns with real OANDA/COMEX liquidity walls.
    4. **Concrete Source Attribution (German Language Output):** Write all explanations in clear, institutional German. Explicitly cite:
       - Concrete news sources & URLs from the feed below (type='NEWS_LINK')
       - Exact 3D surface metrics (type='GRAPH_DATA')
       - Exact OANDA/COMEX orderbook levels & percentages (type='ORDERBOOK')
       - Concrete lessons from the historical winrate / trade archive (type='WINRATE_STATS')

    Current Time: {now_utc_str}
    Current Live Spot Price (CAPITALCOM:GOLD): ${current_spot:.2f}
    Volatility Telemetry: 4h Move ${vol_metrics.get('change_4h', 0):+.2f} | 24h Move: ${vol_metrics.get('change_24h', 0):+.2f} | Daily ATR: ${daily_atr:.2f} (4h Ratio: {vol_metrics.get('move_to_daily_atr_ratio', 0)}x) | 1.2x ATR SL Distance: ${sl_dist_usd:.2f}
    Taker & Perp Flow: {taker_flow_obj.get('state_label', 'N/A')} ({taker_flow_obj.get('taker_buy_pct', 50)}% Buy) | 24/7 Perp Basis: ${crypto_basis_obj.get('spread_usd', 0):+.2f} ({crypto_basis_obj.get('regime', 'N/A')})

    === [1] CHART & ORDERBOOK ZONES ===
    {ob_str}

    === [2] VERIFIED GOLD NEWS CATALYSTS ===
    {news_context_block}

    === [3] 3D VOLATILITY, SKEW & SESSION SURFACES ===
    - {vol_s_summary}
    - {ob_s_summary}
    - {sess_s_summary}

    === [4] HISTORICAL EXPERIENCE & EMPIRICAL WINRATES ===
    {archive_lessons}
    {perf_str}"""

    ai_decision = call_gemini_structured(master_prompt, MASTER_TRADE_SCHEMA)
    if not isinstance(ai_decision, dict):
        return journal

    raw_dec = str(ai_decision.get("Decision", "NO_TRADE")).strip().upper()
    tp_price = round(float(ai_decision.get("Take_Profit_Price", 0.0) or 0.0), 2)
    tp_dist = abs(tp_price - current_spot)

    # Hard enforcement of user rules: valid direction + minimum $10.00 distance to TP
    is_valid_long = (raw_dec == "LONG" and (tp_price - current_spot) >= TRADE_MIN_TP_DISTANCE_USD)
    is_valid_short = (raw_dec == "SHORT" and (current_spot - tp_price) >= TRADE_MIN_TP_DISTANCE_USD)

    decision_record = {
        "timestamp_utc": now_utc_str,
        "spot_at_eval": round(current_spot, 2),
        "decision": raw_dec if (is_valid_long or is_valid_short) else "NO_TRADE",
        "raw_ai_decision": raw_dec,
        "confidence_pct": int(ai_decision.get("Confidence_Percent", 0) or 0),
        "setup_title": ai_decision.get("Setup_Title", "Marktanalyse"),
        "detailed_rationale": ai_decision.get("Detailed_Rationale", ""),
        "chart_orderbook_evidence": ai_decision.get("Chart_Orderbook_Evidence", ""),
        "news_catalyst_evidence": ai_decision.get("News_Catalyst_Evidence", ""),
        "surface_3d_evidence": ai_decision.get("Surface_3D_Evidence", ""),
        "experience_adjustment": ai_decision.get("Experience_Adjustment", ""),
        "invalidation_note": ai_decision.get("Invalidation_Note", ""),
        "cited_sources": ai_decision.get("Cited_Sources", [])
    }

    if raw_dec in ("LONG", "SHORT") and not (is_valid_long or is_valid_short):
        decision_record["filter_rejection_note"] = (
            f"Prognose verworfen: Zielabstand betrug nur ${tp_dist:.2f} (Mindestabstand >= $10.00 erforderlich)."
        )
        print(f"Master Trade rejected by $10 minimum rule: {raw_dec} TP ${tp_price} vs Spot ${current_spot}")

    journal["last_decision"] = decision_record

    if is_valid_long or is_valid_short:
        sl_price = round(current_spot - sl_dist_usd, 2) if is_valid_long else round(current_spot + sl_dist_usd, 2)
        expiry_dt = datetime.utcnow() + timedelta(hours=TRADE_EXPIRY_HOURS)
        new_trade = {
            "id": f"TRD-{int(now_ts)}",
            "status": "ACTIVE",
            "direction": raw_dec,
            "entry_price": round(current_spot, 2),
            "current_price": round(current_spot, 2),
            "tp_price": tp_price,
            "tp_distance_usd": round(tp_dist, 2),
            "sl_price": sl_price,
            "sl_distance_usd": sl_dist_usd,
            "atr_at_entry": daily_atr,
            "risk_reward_ratio": round(tp_dist / max(1.0, sl_dist_usd), 2),
            "confidence_pct": decision_record["confidence_pct"],
            "opened_utc": now_utc_str,
            "entry_ts": now_ts,
            "expires_utc": expiry_dt.strftime("%Y-%m-%d %H:%M UTC"),
            "elapsed_hours": 0.0,
            "remaining_hours": TRADE_EXPIRY_HOURS,
            "max_dd_usd": 0.0,
            "max_favorable_usd": 0.0,
            "unrealized_pnl_usd": 0.0,
            "setup_title": decision_record["setup_title"],
            "detailed_rationale": decision_record["detailed_rationale"],
            "chart_orderbook_evidence": decision_record["chart_orderbook_evidence"],
            "news_catalyst_evidence": decision_record["news_catalyst_evidence"],
            "surface_3d_evidence": decision_record["surface_3d_evidence"],
            "experience_adjustment": decision_record["experience_adjustment"],
            "invalidation_note": decision_record["invalidation_note"],
            "cited_sources": decision_record["cited_sources"]
        }
        journal["active_trade"] = new_trade
        print(f"NEW MASTER TRADE OPENED: {raw_dec} @ ${current_spot:.2f} | TP: ${tp_price:.2f} (+${tp_dist:.2f}) | SL: ${sl_price:.2f} (-${sl_dist_usd:.2f})")
    else:
        print(f"Master Trade Status: NO_TRADE ({decision_record['setup_title']})")

    save_trade_journal(journal)
    return journal

def build_local_fallback_summary(cluster, oanda_zones_cache, vol_metrics, macro_summary, crypto_basis_obj=None, taker_flow_obj=None):
    articles = sorted(cluster.get("articles", []), key=lambda x: x.get("_timestamp", 0))
    oldest = articles[0] if articles else {}
    newest = articles[-1] if articles else {}
    age_h = round((datetime.now().timestamp() - float(oldest.get("_timestamp", datetime.now().timestamp()))) / 3600.0, 1)

    def extract_body_only(art):
        desc = art.get("description", "").strip()
        if desc and desc.lower() != art.get("title", "").strip().lower():
            return desc
        return f"Report from {art.get('source', 'feed')} covering {art.get('title', '').rstrip('?')}."

    facts_list = []
    if len(articles) > 1:
        facts_list.append(f"[Update {newest.get('date', '')} UTC via {newest.get('source', '')}]: {extract_body_only(newest)}")
    for a in articles[:3]:
        if len(articles) > 1 and a["link"] == newest["link"]:
            continue
        facts_list.append(extract_body_only(a))
    facts_str = " | ".join(facts_list)

    full_text = " ".join([f"{a.get('title','')} {a.get('description','')}" for a in articles]).lower()
    narrative_name, _ = detect_specific_narrative(full_text)
    bull_hits = sum(1 for k in ["attack", "strike", "missile", "escalat", "war", "inflation", "recession", "cut", "crisis", "selloff"] if k in full_text)
    bear_hits = sum(1 for k in ["truce", "ceasefire", "peace", "deal", "strong dollar", "rate hike", "rally"] if k in full_text)

    curr_price, top_zones, sentiment = oanda_zones_cache[:3] if oanda_zones_cache else (0.0, [], {"long_pct": 50, "short_pct": 50})
    above_z = [z["price"] for z in top_zones if z["price"] > curr_price]
    below_z = [z["price"] for z in top_zones if z["price"] < curr_price]

    impact = "Bullish" if bull_hits > bear_hits else ("Bearish" if bear_hits > bull_hits else "Neutral")
    target = f"{min(above_z):.2f}" if (impact == "Bullish" and above_z) else (f"{max(below_z):.2f}" if (impact == "Bearish" and below_z) else "N/A")
    atr_ratio = vol_metrics.get("move_to_daily_atr_ratio", 0)

    return {
        "Is_Gold_Relevant": True,
        "Dominant_Driver": narrative_name,
        "Event_Timestamp": f"First reported: {oldest.get('date', 'Unknown')} UTC (~{age_h}h ago)",
        "Facts": facts_str,
        "Macro_Context": f"OANDA Retail: {sentiment['long_pct']}% Long | 4h Move: ${vol_metrics.get('change_4h', 0):+.2f} ({atr_ratio}x Daily ATR).",
        "Impact_Timing": "Past (Already Priced In)" if (age_h > 12 or atr_ratio >= 1.2) else "Present",
        "Timeframe": "Days/Weeks",
        "XAUUSD_Impact": impact,
        "Probability_Percent": 55 if impact != "Neutral" else 40,
        "Orderflow_Price_Target": target
    }

def generate_batch_summaries(clusters_batch, orderbook_data, price_history_data, macro_context_str, crypto_basis_str, taker_flow_str, dominant_drivers_str, perf_memory_str):
    if not client or not clusters_batch:
        return {}
    now_utc_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    clusters_input = []
    for idx, c in enumerate(clusters_batch):
        chronological_arts = sorted(c["articles"], key=lambda x: x.get("_timestamp", 0))
        combined_text = " ".join([f"{a.get('title', '')} {a.get('description', '')}" for a in chronological_arts])
        specific_drv, cat_label = detect_specific_narrative(combined_text)
        selected_arts = [chronological_arts[0]] + chronological_arts[-4:] if len(chronological_arts) > 5 else chronological_arts
        arts_lines = []
        for a_i, a in enumerate(selected_arts):
            role = "INITIAL REPORT" if a_i == 0 else f"NEW UPDATE #{a_i}"
            body = a.get("description", "").strip()
            content_block = f"Content: {body} (Context topic: {a['title']})" if body else f"Topic to synthesize into prose (do NOT copy title): {a['title']}"
            arts_lines.append(f"  * [{role} | {a['date']} UTC | {a['source']}]: {content_block}")
        clusters_input.append(f"=== CLUSTER_ID: {idx} | SOURCES: {len(chronological_arts)} | CATEGORY: {cat_label} | DRIVER: {specific_drv} ===\n" + "\n".join(arts_lines))

    prompt = f"""You are a senior Gold (XAU/USD) & Macro Intelligence Filter.
    Analyze these {len(clusters_batch)} News Clusters using the market context below.

    STAGE 2 RELEVANCE FILTER ('Is_Gold_Relevant' boolean):
    - Set 'Is_Gold_Relevant' = true ONLY if the event has a real transmission channel to Gold (XAU/USD) or broad market risk:
      * Direct Gold, commodities, oil/energy supply, central bank buying, BRICS/de-dollarization.
      * Geopolitics, war, Middle East escalation, Red Sea, Iran, tariffs, sanctions, US-China trade/tech war.
      * Fed policy, CPI/inflation, jobs/NFP, GDP, US Dollar, Treasury yields, sovereign debt.
      * SYSTEMIC Tech/AI/Nasdaq shocks: Mega-cap crises (e.g., Nvidia/Apple/Microsoft/TSMC leadership scandals, DOJ probes, export bans, AI bubble/selloff, or market-wide liquidity/margin shocks).
    - Set 'Is_Gold_Relevant' = false for trivial, isolated single-stock news, routine earnings beats/misses of minor companies, or local news with zero macro/market spillover.

    STRICT RULES FOR 'Facts' AND 'Macro_Context' (CONCISE & NO HEADLINES):
    1. NEVER copy or repeat raw article titles/headlines into 'Facts'.
    2. Write 'Facts' as a crisp, 2-3 sentence synthesized summary of what happened. If a cluster has >1 source ('NEW UPDATE #...'), start 'Facts' with '[Update YYYY-MM-DD HH:MM UTC via Source]: <1 sentence on the new development> | <1-2 sentences on core background>'.
    3. Keep 'Macro_Context' to 1-2 short, direct sentences explaining how this affects XAU/USD right now.

    Current Time: {now_utc_str}
    {perf_memory_str}
    {dominant_drivers_str}
    {taker_flow_str}
    {crypto_basis_str}
    {macro_context_str}
    {price_history_data}
    {orderbook_data}

    Clusters:
    {chr(10).join(clusters_input)}"""

    active_models = [m for m in FALLBACK_MODELS if m not in DEAD_MODELS]
    if not active_models:
        active_models = ["gemini-3.5-flash-lite"]

    backoff_delays = [3, 6, 10]
    for model_name in active_models:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=CLUSTER_SUMMARY_SCHEMA
                    )
                )
                raw_text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed_list = json.loads(raw_text)
                if isinstance(parsed_list, dict):
                    for v in parsed_list.values():
                        if isinstance(v, list):
                            parsed_list = v
                            break
                if isinstance(parsed_list, list) and parsed_list:
                    print(f" -> Batch ({len(parsed_list)} clusters) analyzed successfully via {model_name}!")
                    return {item.get("Cluster_ID", i): item for i, item in enumerate(parsed_list) if isinstance(item, dict)}
            except Exception as e:
                err_str = str(e).splitlines()[0][:100]
                if "404" in err_str or "NOT_FOUND" in err_str:
                    DEAD_MODELS.add(model_name)
                    print(f" -> {model_name} retired (404), skipping in future batches.")
                    break
                wait_s = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                print(f" -> {model_name} attempt {attempt+1}/3 temporary spike ({err_str}) - retrying in {wait_s}s...")
                time.sleep(wait_s)
    return {}

def fetch_fallback_h4_candles(target_spot_price):
    candles = []
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=1mo&interval=1h"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if res.status_code == 200:
            result = res.json().get("chart", {}).get("result", [])
            if result:
                timestamps = result[0].get("timestamp", [])
                q = result[0].get("indicators", {}).get("quote", [{}])[0]
                opens, highs, lows, closes = q.get("open", []), q.get("high", []), q.get("low", []), q.get("close", [])
                raw_1h = []
                for t, o, h, l, c in zip(timestamps, opens, highs, lows, closes):
                    if all(x is not None for x in (t, o, h, l, c)):
                        raw_1h.append({"time": int(t), "open": float(o), "high": float(h), "low": float(l), "close": float(c)})
                if raw_1h:
                    last_f_close = raw_1h[-1]["close"]
                    basis_offset = (target_spot_price - last_f_close) if target_spot_price > 0 else 0.0
                    for i in range(0, len(raw_1h), 4):
                        chunk = raw_1h[i:i+4]
                        candles.append({
                            "time": chunk[0]["time"],
                            "open": round(chunk[0]["open"] + basis_offset, 2),
                            "high": round(max(x["high"] for x in chunk) + basis_offset, 2),
                            "low": round(min(x["low"] for x in chunk) + basis_offset, 2),
                            "close": round(chunk[-1]["close"] + basis_offset, 2)
                        })
    except Exception as e:
        print(f"Fallback H4 candle note: {e}")
    return candles[-100:]

def update_market_data(top_zones_cache=None, macro_summary=None, vol_metrics=None, crypto_basis_obj=None, taker_flow_obj=None, drivers_24h=None, drivers_weekly=None, multi_pos=None, h1_candles=None, tv_live_price=0.0, tv_source="CAPITALCOM:GOLD"):
    old_md = {}
    if os.path.exists("market_data.json"):
        try:
            with open("market_data.json", "r", encoding="utf-8") as f:
                old_md = json.load(f)
        except Exception:
            pass

    loaded_h4_candles = []
    if OANDA_API_KEY:
        headers = {"Authorization": f"Bearer {OANDA_API_KEY}", "Accept-Datetime-Format": "UNIX"}
        for attempt in range(2):
            try:
                res_c = requests.get("https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/candles?count=100&granularity=H4&price=M", headers=headers, timeout=10)
                if res_c.status_code == 200:
                    for c in res_c.json().get("candles", []):
                        loaded_h4_candles.append({
                            "time": int(float(c["time"])),
                            "open": float(c["mid"]["o"]),
                            "high": float(c["mid"]["h"]),
                            "low": float(c["mid"]["l"]),
                            "close": float(c["mid"]["c"])
                        })
                    print(f"Successfully loaded {len(loaded_h4_candles)} OANDA H4 candles.")
                    break
                else:
                    print(f"OANDA H4 Candle API Error {res_c.status_code} (Attempt {attempt+1}/2): {res_c.text[:120]}")
                    time.sleep(2)
            except Exception as e:
                print(f"Error loading OANDA H4 candles (Attempt {attempt+1}/2): {e}")
                time.sleep(2)

    curr_p, top_zones, sentiment, all_zones_dict = top_zones_cache if top_zones_cache else get_clustered_oanda_zones(zone_width=10.0)
    eff_spot = tv_live_price if tv_live_price > 0 else (curr_p if curr_p > 0 else (vol_metrics or {}).get("current_price", 0.0))

    now_ts = int(time.time())
    if not loaded_h4_candles or (now_ts - loaded_h4_candles[-1]["time"] > 129600):
        fb_candles = fetch_fallback_h4_candles(eff_spot)
        if fb_candles:
            loaded_h4_candles = fb_candles
            print(f"Loaded {len(loaded_h4_candles)} calibrated H4 candles synced to {tv_source}.")
        elif old_md.get("candles"):
            loaded_h4_candles = old_md["candles"]

    if loaded_h4_candles and eff_spot > 0:
        last_c = loaded_h4_candles[-1]
        if now_ts - last_c["time"] > 14400:
            loaded_h4_candles.append({
                "time": (now_ts // 14400) * 14400,
                "open": last_c["close"],
                "high": max(last_c["close"], eff_spot),
                "low": min(last_c["close"], eff_spot),
                "close": round(eff_spot, 2)
            })
        else:
            last_c["close"] = round(eff_spot, 2)
            last_c["high"] = round(max(last_c["high"], eff_spot), 2)
            last_c["low"] = round(min(last_c["low"], eff_spot), 2)

    if eff_spot <= 0 and loaded_h4_candles:
        eff_spot = loaded_h4_candles[-1]["close"]

    if vol_metrics is not None and eff_spot > 0:
        vol_metrics["current_price"] = round(eff_spot, 2)
    if crypto_basis_obj and eff_spot > 0 and crypto_basis_obj.get("crypto_price"):
        crypto_basis_obj["oanda_price"] = round(eff_spot, 2)
        crypto_basis_obj["spread_usd"] = round(crypto_basis_obj["crypto_price"] - eff_spot, 2)

    ob_surf, vol_surf, sess_surf, updated_ob_hist = build_3d_surfaces(
        eff_spot, all_zones_dict, h1_candles, vol_metrics, sentiment, multi_pos, old_md.get("ob_history", [])
    )

    market_data = {
        "spot_price": round(eff_spot, 2),
        "spot_source": tv_source if tv_live_price > 0 else "OANDA:XAUUSD",
        "candles": loaded_h4_candles,
        "orderbook": [],
        "sentiment": sentiment,
        "multi_positioning": multi_pos or old_md.get("multi_positioning", {}),
        "macro": macro_summary or old_md.get("macro", {}),
        "volatility": vol_metrics or old_md.get("volatility", {}),
        "crypto_basis": crypto_basis_obj or old_md.get("crypto_basis", {}),
        "taker_flow": taker_flow_obj or old_md.get("taker_flow", {}),
        "dominant_drivers": drivers_24h if drivers_24h is not None else old_md.get("dominant_drivers", []),
        "weekly_drivers": drivers_weekly if drivers_weekly is not None else old_md.get("weekly_drivers", []),
        "ob_history": updated_ob_hist,
        "ob_surface": ob_surf,
        "vol_surface": vol_surf,
        "session_surface": sess_surf
    }

    for z in top_zones:
        if eff_spot > 0 and abs(z["price"] - eff_spot) / eff_spot > MAX_ZONE_RADIUS_PCT:
            continue
        market_data["orderbook"].append({
            "price": z["price"],
            "volume": z["volume"],
            "oanda_vol": z.get("oanda_vol", 0),
            "comex_vol": z.get("comex_vol", 0),
            "long": z["long"],
            "short": z["short"],
            "net": z["net"],
            "dominance": z["dominance"],
            "overlap": z.get("overlap", False),
            "sources": z.get("sources", "OANDA")
        })
    print(f"Successfully synced Spot (${eff_spot:.2f} via {market_data['spot_source']}), {len(market_data['orderbook'])} near-spot zones & 3D surfaces.")

    with open("market_data.json", "w", encoding="utf-8") as f:
        json.dump(market_data, f)

    return eff_spot, ob_surf, vol_surf, sess_surf

def main():
    print("Loading embeddings model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    existing_data = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = []

    print("Fetching Live CAPITALCOM:GOLD Price from TradingView Scanner...")
    tv_live_price, tv_source = fetch_capitalcom_tv_price()

    print("Loading 500 H1 Candles, Near-Spot OANDA + COMEX Zones & 3D Surface Telemetry...")
    price_history_data, h1_candles, vol_metrics = fetch_oanda_price_and_volatility(tv_live_price=tv_live_price)
    effective_spot_price = tv_live_price if tv_live_price > 0 else vol_metrics.get("current_price", 0.0)
    
    oanda_zones_cache = get_clustered_oanda_zones(zone_width=10.0, fallback_price=effective_spot_price)
    if effective_spot_price <= 0 and oanda_zones_cache:
        effective_spot_price = oanda_zones_cache[0]
    oanda_sentiment = oanda_zones_cache[2] if oanda_zones_cache else {"long_pct": 50.0, "short_pct": 50.0}
    
    multi_pos = fetch_multi_positioning(oanda_sentiment)
    crypto_basis_str, crypto_basis_obj = fetch_crypto_gold_basis(effective_spot_price)
    taker_flow_str, taker_flow_obj = fetch_aggressive_taker_flow(h1_candles, oanda_sentiment=oanda_sentiment)
    macro_context_str, macro_summary = fetch_macro_context()

    current_stats = evaluate_clusters(existing_data, h1_candles, vol_metrics=vol_metrics, macro_summary=macro_summary)
    perf_memory_str = format_performance_memory(current_stats)

    cutoff_timestamp = (datetime.now() - timedelta(hours=24)).timestamp()
    valid_clusters = [c for c in existing_data if c.get("_timestamp", 0) > cutoff_timestamp]

    for c in valid_clusters:
        s = c.get("summary")
        if isinstance(s, dict):
            facts_str = str(s.get("Facts", ""))
            if "Is_Gold_Relevant" not in s or "[INITIAL REPORT" in facts_str or " — " in facts_str:
                c["needs_update"] = True

    existing_urls = {a["link"] for c in valid_clusters for a in c.get("articles", [])}

    print("Fetching new articles in parallel (Stage 1 Macro/Geo/Systemic-Tech Filter active)...")
    new_articles = fetch_new_articles(existing_urls)
    if new_articles:
        new_articles.sort(key=lambda x: x.get("_timestamp", 0))
        print(f"Found {len(new_articles)} new articles passing Stage 1. Clustering (Threshold: {SIMILARITY_THRESHOLD})...")
        cluster_embeddings = [model.encode(" ".join([a["title"] for a in c["articles"]])) for c in valid_clusters]
        for article in new_articles:
            article_emb = model.encode(article["title"] + " " + article["description"])
            best_match_idx, highest_sim = -1, -1
            for i, cluster_emb in enumerate(cluster_embeddings):
                sim = util.cos_sim(article_emb, cluster_emb).item()
                if sim > highest_sim:
                    highest_sim, best_match_idx = sim, i
            if highest_sim > SIMILARITY_THRESHOLD and best_match_idx != -1:
                tc = valid_clusters[best_match_idx]
                tc["articles"].append(article)
                oldest_a = min(tc["articles"], key=lambda x: x["_timestamp"])
                newest_a = max(tc["articles"], key=lambda x: x["_timestamp"])
                tc["original_date"] = oldest_a["date"]
                tc["main_date"] = f"First Reported: {tc['original_date']} UTC | Last Update: {newest_a['date']} UTC" if newest_a["date"] != tc["original_date"] else f"First Reported: {newest_a['date']} UTC"
                tc["_timestamp"] = newest_a["_timestamp"]
                tc["needs_update"] = True
                cluster_embeddings[best_match_idx] = model.encode(" ".join([a["title"] for a in tc["articles"]]))
            else:
                valid_clusters.append({
                    "main_date": f"First Reported: {article['date']} UTC",
                    "original_date": article["date"],
                    "_timestamp": article["_timestamp"],
                    "articles": [article],
                    "needs_update": True,
                    "summary": None
                })
                cluster_embeddings.append(article_emb)
    else:
        print("No new articles found.")

    valid_clusters.sort(key=lambda x: float(x.get("_timestamp", 0)), reverse=True)
    dominant_drivers_str, _, _ = compute_dual_horizon_drivers(valid_clusters, current_stats.get("history", {}))

    pending = [c for c in valid_clusters if c.get("needs_update", False) or not isinstance(c.get("summary"), dict)]
    if pending:
        orderbook_data = fetch_oanda_orderbook(top_zones_cache=oanda_zones_cache, multi_pos=multi_pos)
        total_batches = min(MAX_BATCH_CALLS_PER_RUN, (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE)
        for b_idx in range(total_batches):
            batch_slice = pending[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
            print(f"Running AI Batch Analysis {b_idx + 1}/{total_batches} ({len(batch_slice)} clusters in 1 call)...")
            batch_results = generate_batch_summaries(batch_slice, orderbook_data, price_history_data, macro_context_str, crypto_basis_str, taker_flow_str, dominant_drivers_str, perf_memory_str)
            for idx_in_batch, cluster in enumerate(batch_slice):
                if idx_in_batch in batch_results:
                    cluster["summary"] = batch_results[idx_in_batch]
                    cluster["needs_update"] = False
            if b_idx < total_batches - 1:
                time.sleep(3)

    for c in valid_clusters:
        if not isinstance(c.get("summary"), dict):
            c["summary"] = build_local_fallback_summary(c, oanda_zones_cache, vol_metrics, macro_summary, crypto_basis_obj, taker_flow_obj)
            c["needs_update"] = True

    _, drivers_24h, drivers_weekly = compute_dual_horizon_drivers(valid_clusters, current_stats.get("history", {}))
    updated_stats = evaluate_clusters(valid_clusters, h1_candles, vol_metrics=vol_metrics, macro_summary=macro_summary)

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_clusters, f, indent=4, ensure_ascii=False)

    print("Updating Chart, CAPITALCOM:GOLD Spot Price & 3D Surfaces...")
    synced_spot, ob_surf, vol_surf, sess_surf = update_market_data(
        oanda_zones_cache, macro_summary, vol_metrics, crypto_basis_obj,
        taker_flow_obj, drivers_24h, drivers_weekly, multi_pos, h1_candles,
        tv_live_price=effective_spot_price, tv_source=tv_source
    )

    print("Running Tab [4] Master AI Trade Synthesis & Journal Evaluation...")
    manage_master_trade(
        synced_spot, h1_candles, vol_metrics, oanda_zones_cache, macro_summary,
        crypto_basis_obj, taker_flow_obj, valid_clusters, drivers_24h, drivers_weekly,
        ob_surf, vol_surf, sess_surf, updated_stats
    )
    print("Update completed.")

if __name__ == "__main__":
    main()