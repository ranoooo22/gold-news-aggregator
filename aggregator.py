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

# Simplified schema: Only pure objective summarization of facts, without trading opinions or predictions
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
            "Market_Context": types.Schema(type=types.Type.STRING),
        },
        required=[
            "Cluster_ID", "Is_Gold_Relevant", "Dominant_Driver", "Event_Timestamp",
            "Facts", "Market_Context"
        ]
    )
)

BATCH_SIZE = 6
MAX_BATCH_CALLS_PER_RUN = 4
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
SIMILARITY_THRESHOLD = 0.54

def fetch_capitalcom_tv_price():
    url = "https://scanner.tradingview.com/global/scan"
    payload = {
        "symbols": {"tickers": ["CAPITALCOM:GOLD", "OANDA:XAUUSD", "FOREXCOM:XAUUSD"]},
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
    return 0.0, "N/A"

def fetch_crypto_gold_basis(spot_price):
    crypto_price, source_used = fetch_live_perp_price()
    if crypto_price > 0 and spot_price <= 0:
        spot_price = crypto_price
    if crypto_price > 0 and spot_price > 0:
        spread_usd = round(crypto_price - spot_price, 2)
        spread_pct = round((spread_usd / spot_price) * 100.0, 2)
        return {
            "crypto_price": round(crypto_price, 2),
            "spot_price": round(spot_price, 2),
            "spread_usd": spread_usd,
            "spread_pct": spread_pct,
            "source": source_used
        }
    return {}

def fetch_yahoo_futures_zones(reference_price, zone_width=10.0):
    comex_zones = {}
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=5d&interval=1h"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if res.status_code != 200:
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
        "cot_mm_long_pct": cot_mm_long_pct,
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
        try:
            res_ob = requests.get(url_ob, headers=headers, timeout=10)
            if res_ob.status_code == 200:
                ob_data = res_ob.json().get("orderBook", {})
                ob_price = float(ob_data.get("price", 0))
                if current_price <= 0 and ob_price > 0:
                    current_price = ob_price
                for b in ob_data.get("buckets", []):
                    p = float(b.get("price", 0))
                    if current_price > 0 and abs(p - current_price) / current_price <= MAX_ZONE_RADIUS_PCT:
                        combined_buckets[p] = {"long": float(b.get("longCountPercent", 0)), "short": float(b.get("shortCountPercent", 0))}
        except Exception:
            pass
        try:
            res_pb = requests.get(url_pb, headers=headers, timeout=10)
            if res_pb.status_code == 200:
                pb_data = res_pb.json().get("positionBook", {})
                for b in pb_data.get("buckets", []):
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
        except Exception:
            pass

    pos_sum = total_pos_long + total_pos_short
    sentiment = {
        "long_pct": round((total_pos_long / pos_sum) * 100, 1) if pos_sum > 0 else 50.0,
        "short_pct": round((total_pos_short / pos_sum) * 100, 1) if pos_sum > 0 else 50.0
    }

    zones = {}
    for p, vols in combined_buckets.items():
        zone_key = round(p / zone_width) * zone_width
        if zone_key not in zones:
            zones[zone_key] = {"long": 0.0, "short": 0.0, "total": 0.0, "weighted_price_sum": 0.0, "oanda_vol": 0.0, "comex_vol": 0.0}
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
            zones[z_key] = {"long": 0.0, "short": 0.0, "total": 0.0, "weighted_price_sum": 0.0, "oanda_vol": 0.0, "comex_vol": 0.0}
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
            dominance = "BUY" if net_diff >= 0 else "SELL"
            active_src = []
            if z_data["oanda_vol"] >= 0.4: active_src.append("OANDA")
            if z_data["comex_vol"] >= 0.5: active_src.append("COMEX")
            is_overlap = len(active_src) >= 2
            sources_label = "+".join(active_src) if active_src else "OANDA"
            zone_obj = {
                "price": vwap_price,
                "zone_center": z_key,
                "volume": round(z_data["total"], 2),
                "oanda_vol": round(z_data["oanda_vol"], 2),
                "comex_vol": round(z_data["comex_vol"], 2),
                "net": net_diff,
                "dominance": dominance,
                "overlap": is_overlap,
                "sources": sources_label
            }
            all_zone_objects[int(z_key)] = zone_obj
            if vwap_price >= current_price: above_zones.append(zone_obj)
            else: below_zones.append(zone_obj)

    above_zones.sort(key=lambda x: x["volume"], reverse=True)
    below_zones.sort(key=lambda x: x["volume"], reverse=True)
    selected_zones = above_zones[:3] + below_zones[:3]
    selected_zones.sort(key=lambda x: x["price"], reverse=True)
    return current_price, selected_zones, sentiment, all_zone_objects

def build_3d_surfaces(current_price, all_zones_dict, h1_candles, vol_metrics, old_ob_history=None):
    spot = current_price if current_price > 0 else (h1_candles[-1]["c"] if h1_candles else 3850.0)
    center_10 = int(round(spot / 10.0) * 10)
    price_grid = [center_10 + offset for offset in range(-50, 60, 10)]

    history_list = list(old_ob_history) if isinstance(old_ob_history, list) else []
    now_str = datetime.utcnow().strftime("%H:%M")
    curr_snapshot_levels = {}
    for p_lvl in price_grid:
        z_item = all_zones_dict.get(p_lvl)
        if z_item:
            sign = 1.0 if z_item["dominance"] == "BUY" else -1.0
            curr_snapshot_levels[str(p_lvl)] = {
                "vol": round(z_item["volume"], 2),
                "signed_vol": round(z_item["net"], 2)
            }
        else:
            curr_snapshot_levels[str(p_lvl)] = {
                "vol": 0.4,
                "signed_vol": 0.2 if p_lvl < spot else -0.2
            }

    if not history_list or history_list[-1].get("time") != now_str:
        history_list.append({"time": now_str, "spot": round(spot, 2), "levels": curr_snapshot_levels})
    else:
        history_list[-1]["spot"] = round(spot, 2)
        history_list[-1]["levels"] = curr_snapshot_levels

    history_list = history_list[-24:]
    ob_timestamps = [h["time"] for h in history_list]
    ob_vol_matrix, ob_signed_matrix = [], []
    for p_lvl in price_grid:
        row_vol, row_signed = [], []
        for h in history_list:
            cell = h.get("levels", {}).get(str(p_lvl), {"vol": 0.5, "signed_vol": 0.0})
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

    # Model 3: 30-Day Session & Volatility Profile (Pure factual historical stats)
    days_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    hours_labels = [f"{h:02d}:00" for h in range(24)]
    range_sums = [[0.0 for _ in range(24)] for _ in range(5)]
    bias_sums = [[0.0 for _ in range(24)] for _ in range(5)]
    counts = [[0 for _ in range(24)] for _ in range(5)]

    for c in (h1_candles or []):
        dt_obj = datetime.utcfromtimestamp(c["ts"])
        w_day = dt_obj.weekday()
        if w_day > 4: continue
        hr = dt_obj.hour
        rng = max(0.2, c["h"] - c["l"])
        net_move = c["c"] - c["o"]
        range_sums[w_day][hr] += rng
        bias_sums[w_day][hr] += net_move
        counts[w_day][hr] += 1

    range_matrix, bias_matrix = [], []
    for d_i in range(5):
        r_row, b_row = [], []
        for h_i in range(24):
            cnt = counts[d_i][h_i]
            if cnt > 0:
                r_row.append(round(range_sums[d_i][h_i] / cnt, 2))
                b_row.append(round(bias_sums[d_i][h_i] / cnt, 2))
            else:
                r_row.append(10.0)
                b_row.append(0.0)
        range_matrix.append(r_row)
        bias_matrix.append(b_row)

    session_surface = {
        "hours": hours_labels,
        "days": days_labels,
        "range_matrix": range_matrix,
        "bias_matrix": bias_matrix
    }
    return ob_surface, session_surface, history_list

def fetch_oanda_price_and_volatility(tv_live_price=0.0):
    parsed = []
    if OANDA_API_KEY:
        url = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/candles?count=500&granularity=H1&price=M"
        headers = {"Authorization": f"Bearer {OANDA_API_KEY}", "Accept-Datetime-Format": "UNIX"}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                for c in response.json().get("candles", []):
                    ts = float(c.get("time", 0))
                    parsed.append({
                        "ts": ts,
                        "dt_str": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M"),
                        "o": float(c["mid"]["o"]),
                        "h": float(c["mid"]["h"]),
                        "l": float(c["mid"]["l"]),
                        "c": float(c["mid"]["c"])
                    })
        except Exception:
            pass

    if len(parsed) < 15:
        eff_p = tv_live_price if tv_live_price > 0 else 3850.0
        return parsed, {"current_price": eff_p, "atr_h1": 9.5, "est_daily_atr": 38.0, "change_4h": 0.0, "change_24h": 0.0, "move_to_daily_atr_ratio": 0.5}

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
    return parsed, vol_metrics

def fetch_macro_context():
    if not OANDA_API_KEY:
        return {}
    instruments = {"EUR_USD": "EUR/USD", "USB10Y_USD": "US 10Y Bond", "NAS100_USD": "Nasdaq 100", "BCO_USD": "Brent Oil"}
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    macro_summary = {}
    for instr in instruments.keys():
        try:
            url = f"https://api-fxpractice.oanda.com/v3/instruments/{instr}/candles?count=24&granularity=H1&price=M"
            res = requests.get(url, headers=headers, timeout=6)
            if res.status_code == 200:
                candles = res.json().get("candles", [])
                if len(candles) >= 2:
                    first_c = float(candles[0]["mid"]["c"])
                    last_c = float(candles[-1]["mid"]["c"])
                    pct = round(((last_c - first_c) / first_c) * 100, 2) if first_c > 0 else 0.0
                    macro_summary[instr] = {"price": last_c, "change_24h_pct": pct}
        except Exception:
            pass
    return macro_summary

def build_local_fallback_summary(cluster):
    articles = sorted(cluster.get("articles", []), key=lambda x: x.get("_timestamp", 0))
    oldest = articles[0] if articles else {}
    newest = articles[-1] if articles else {}
    facts_list = [f"[Update {newest.get('date', '')} UTC via {newest.get('source', '')}]: {newest.get('description', '')}" if len(articles) > 1 else oldest.get('description', '')]
    return {
        "Is_Gold_Relevant": True,
        "Dominant_Driver": "Gold Bullion & Macro",
        "Event_Timestamp": f"First reported: {oldest.get('date', 'Unknown')} UTC",
        "Facts": " | ".join(facts_list),
        "Market_Context": "Objective news feed item without speculative prediction."
    }

def generate_batch_summaries(clusters_batch):
    if not client or not clusters_batch:
        return {}
    clusters_input = []
    for idx, c in enumerate(clusters_batch):
        chronological_arts = sorted(c["articles"], key=lambda x: x.get("_timestamp", 0))
        specific_drv, _ = detect_specific_narrative(" ".join([a.get('title', '') for a in chronological_arts]))
        arts_lines = [f"  * [{a['date']} UTC | {a['source']}]: {a['title']} - {a.get('description', '')}" for a in chronological_arts[-4:]]
        clusters_input.append(f"=== CLUSTER_ID: {idx} | DRIVER: {specific_drv} ===\n" + "\n".join(arts_lines))

    prompt = f"""You are a neutral news summarizer for a financial terminal.
    Summarize these {len(clusters_batch)} news clusters objectively. DO NOT predict prices, DO NOT give trading advice, DO NOT assign bullish/bearish probabilities. Only summarize the core factual news events and their macro transmission channels.

    Clusters:
    {chr(10).join(clusters_input)}"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CLUSTER_SUMMARY_SCHEMA
            )
        )
        raw_text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed_list = json.loads(raw_text)
        if isinstance(parsed_list, list):
            return {item.get("Cluster_ID", i): item for i, item in enumerate(parsed_list) if isinstance(item, dict)}
    except Exception as e:
        print(f"AI batch error: {e}")
    return {}

def update_market_data(top_zones_cache=None, macro_summary=None, vol_metrics=None, crypto_basis_obj=None, h1_candles=None, tv_live_price=0.0, tv_source="CAPITALCOM:GOLD"):
    old_md = {}
    if os.path.exists("market_data.json"):
        try:
            with open("market_data.json", "r", encoding="utf-8") as f:
                old_md = json.load(f)
        except Exception:
            pass

    loaded_h4_candles = old_md.get("candles", [])
    curr_p, top_zones, sentiment, all_zones_dict = top_zones_cache if top_zones_cache else get_clustered_oanda_zones(zone_width=10.0)
    eff_spot = tv_live_price if tv_live_price > 0 else (curr_p if curr_p > 0 else (vol_metrics or {}).get("current_price", 0.0))

    ob_surf, sess_surf, updated_ob_hist = build_3d_surfaces(eff_spot, all_zones_dict, h1_candles, vol_metrics, old_md.get("ob_history", []))

    market_data = {
        "spot_price": round(eff_spot, 2),
        "spot_source": tv_source if tv_live_price > 0 else "OANDA:XAUUSD",
        "candles": loaded_h4_candles,
        "orderbook": [],
        "sentiment": sentiment,
        "macro": macro_summary or {},
        "volatility": vol_metrics or {},
        "crypto_basis": crypto_basis_obj or {},
        "ob_history": updated_ob_hist,
        "ob_surface": ob_surf,
        "session_surface": sess_surf
    }

    for z in top_zones:
        if eff_spot > 0 and abs(z["price"] - eff_spot) / eff_spot > MAX_ZONE_RADIUS_PCT:
            continue
        market_data["orderbook"].append({
            "price": z["price"],
            "volume": z["volume"],
            "oanda_vol": z["oanda_vol"],
            "comex_vol": z["comex_vol"],
            "net": z["net"],
            "dominance": z["dominance"],
            "overlap": z["overlap"],
            "sources": z["sources"]
        })
    with open("market_data.json", "w", encoding="utf-8") as f:
        json.dump(market_data, f, indent=4)

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

    print("Loading H1 Candles & Near-Spot OANDA + COMEX Zones...")
    h1_candles, vol_metrics = fetch_oanda_price_and_volatility(tv_live_price=tv_live_price)
    effective_spot_price = tv_live_price if tv_live_price > 0 else vol_metrics.get("current_price", 0.0)
    
    oanda_zones_cache = get_clustered_oanda_zones(zone_width=10.0, fallback_price=effective_spot_price)
    if effective_spot_price <= 0 and oanda_zones_cache:
        effective_spot_price = oanda_zones_cache[0]
    
    crypto_basis_obj = fetch_crypto_gold_basis(effective_spot_price)
    macro_summary = fetch_macro_context()

    # Clean up old data and keep strictly 24h window based on original date
    cutoff_timestamp = (datetime.now() - timedelta(hours=24)).timestamp()
    valid_clusters = [c for c in existing_data if float(c.get("_timestamp", 0)) > cutoff_timestamp]

    existing_urls = {a["link"] for c in valid_clusters for a in c.get("articles", [])}
    new_articles = fetch_new_articles(existing_urls)
    
    if new_articles:
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
                newest_a = max(tc["articles"], key=lambda x: x["_timestamp"])
                tc["main_date"] = f"First Reported: {tc['original_date']} UTC | Last Update: {newest_a['date']} UTC"
                tc["_timestamp"] = newest_a["_timestamp"]
                tc["needs_update"] = True
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

    pending = [c for c in valid_clusters if c.get("needs_update", False) or not isinstance(c.get("summary"), dict)]
    if pending:
        total_batches = min(MAX_BATCH_CALLS_PER_RUN, (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE)
        for b_idx in range(total_batches):
            batch_slice = pending[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
            batch_results = generate_batch_summaries(batch_slice)
            for idx_in_batch, cluster in enumerate(batch_slice):
                if idx_in_batch in batch_results:
                    cluster["summary"] = batch_results[idx_in_batch]
                    cluster["needs_update"] = False

    for c in valid_clusters:
        if not isinstance(c.get("summary"), dict):
            c["summary"] = build_local_fallback_summary(c)

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_clusters, f, indent=4, ensure_ascii=False)

    update_market_data(oanda_zones_cache, macro_summary, vol_metrics, crypto_basis_obj, h1_candles, tv_live_price=effective_spot_price, tv_source=tv_source)
    print("Update completed (Pure Data Terminal Mode).")

if __name__ == "__main__":
    main()
