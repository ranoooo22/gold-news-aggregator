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

# --- KONFIGURATION ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OANDA_API_KEY = os.environ.get("OANDA_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None
    print("WARNUNG: GEMINI_API_KEY nicht gefunden.")

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

# Auswertungs-Parameter für das Performance-Gedächtnis
MIN_EVAL_PROBABILITY = 51       # Prüft alle gerichteten Signale > 50% (ab 51%)
EVAL_HORIZON_HOURS = 20.0       # Zeitfenster für Zielerreichung
MAX_DRAWDOWN_ATR_MULT = 1.8     # Weites Notbremse-Limit (1.8x Tages-ATR Gegenbewegung)

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
        print(f"Timeout/Fehler bei {source}: {e}")
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

def fetch_yahoo_futures_zones(reference_price, zone_width=10.0):
    comex_zones = {}
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=5d&interval=1h"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code != 200:
            print(f"Yahoo COMEX Status-Code: {res.status_code}")
            return comex_zones
            
        data = res.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return comex_zones
            
        indicators = result[0].get("indicators", {}).get("quote", [{}])[0]
        closes = indicators.get("close", [])
        volumes = indicators.get("volume", [])
        
        valid_closes = [float(c) for c in closes if c is not None]
        if reference_price <= 0 and valid_closes:
            reference_price = valid_closes[-1]
            
        raw_entries = []
        total_vol = 0.0
        
        for c, v in zip(closes, volumes):
            if c is not None and v is not None and v > 0:
                if reference_price > 0 and abs(c - reference_price) / reference_price <= 0.20:
                    raw_entries.append((float(c), float(v)))
                    total_vol += float(v)
                    
        if total_vol <= 0:
            return comex_zones
            
        for p, v in raw_entries:
            zone_key = round(p / zone_width) * zone_width
            if zone_key not in comex_zones:
                comex_zones[zone_key] = {"total": 0.0, "weighted_price_sum": 0.0}
            
            vol_pct = (v / total_vol) * 100.0
            comex_zones[zone_key]["total"] += vol_pct
            comex_zones[zone_key]["weighted_price_sum"] += p * vol_pct
            
    except Exception as e:
        print(f"Hinweis: Yahoo Finance Futures nicht erreichbar ({e})")
        
    return comex_zones

def get_clustered_oanda_zones(zone_width=10.0, fallback_price=0.0):
    current_price = float(fallback_price) if fallback_price else 0.0
    combined_buckets = {}
    total_pos_long = 0.0
    total_pos_short = 0.0
    
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
                    if ob_price > 0:
                        current_price = ob_price
                    buckets = ob_data.get("buckets", [])
                    for b in buckets:
                        try:
                            p = float(b.get("price", 0))
                            if current_price > 0 and abs(p - current_price) / current_price <= 0.20:
                                combined_buckets[p] = {
                                    "long": float(b.get("longCountPercent", 0)),
                                    "short": float(b.get("shortCountPercent", 0))
                                }
                        except ValueError:
                            continue
                    print(f"OANDA OrderBook geladen: {len(combined_buckets)} Preis-Buckets im 20%-Radius.")
                    break
                else:
                    print(f"OANDA orderBook Status {res_ob.status_code} (Versuch {attempt+1}/2): {res_ob.text[:120]}")
                    time.sleep(2)
            except Exception as e:
                print(f"Fehler bei OANDA orderBook (Versuch {attempt+1}/2): {e}")
                time.sleep(2)

        for attempt in range(2):
            try:
                res_pb = requests.get(url_pb, headers=headers, timeout=12)
                if res_pb.status_code == 200:
                    pb_data = res_pb.json().get("positionBook", {})
                    pb_price = float(pb_data.get("price", 0))
                    if current_price == 0 and pb_price > 0:
                        current_price = pb_price
                    buckets = pb_data.get("buckets", [])
                    for b in buckets:
                        try:
                            p = float(b.get("price", 0))
                            l_pct = float(b.get("longCountPercent", 0))
                            s_pct = float(b.get("shortCountPercent", 0))
                            total_pos_long += l_pct
                            total_pos_short += s_pct
                            
                            if current_price > 0 and abs(p - current_price) / current_price <= 0.20:
                                if p not in combined_buckets:
                                    combined_buckets[p] = {"long": 0.0, "short": 0.0}
                                combined_buckets[p]["long"] += l_pct
                                combined_buckets[p]["short"] += s_pct
                        except ValueError:
                            continue
                    print(f"OANDA PositionBook geladen (Gesamt-Buckets: {len(combined_buckets)}).")
                    break
                else:
                    print(f"OANDA positionBook Status {res_pb.status_code} (Versuch {attempt+1}/2): {res_pb.text[:120]}")
                    time.sleep(2)
            except Exception as e:
                print(f"Fehler bei OANDA positionBook (Versuch {attempt+1}/2): {e}")
                time.sleep(2)

    pos_sum = total_pos_long + total_pos_short
    if pos_sum > 0:
        sentiment = {
            "long_pct": round((total_pos_long / pos_sum) * 100, 1),
            "short_pct": round((total_pos_short / pos_sum) * 100, 1)
        }
    else:
        sentiment = {"long_pct": 50.0, "short_pct": 50.0}

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
        weight = 0.20 if combined_buckets else 1.0
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
        return current_price, [], sentiment

    above_zones = []
    below_zones = []

    for z_key, z_data in zones.items():
        if z_data["total"] > 0:
            vwap_price = round(z_data["weighted_price_sum"] / z_data["total"], 2)
            net_diff = round(z_data["long"] - z_data["short"], 2)
            
            if z_data["oanda_vol"] == 0:
                dominance = "SELL" if vwap_price >= current_price else "BUY"
            else:
                dominance = "BUY" if net_diff >= 0 else "SELL"
            
            has_oanda = z_data["oanda_vol"] >= 0.5
            has_comex = z_data["comex_vol"] >= 0.5
            is_overlap = has_oanda and has_comex

            if is_overlap:
                sources_label = "OANDA+COMEX"
            elif has_comex and not has_oanda:
                sources_label = "COMEX Futures"
            else:
                sources_label = "OANDA"

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
            if vwap_price >= current_price:
                above_zones.append(zone_obj)
            else:
                below_zones.append(zone_obj)

    above_zones.sort(key=lambda x: x["volume"], reverse=True)
    below_zones.sort(key=lambda x: x["volume"], reverse=True)

    selected_zones = above_zones[:3] + below_zones[:3]
    selected_zones.sort(key=lambda x: x["price"], reverse=True)

    return current_price, selected_zones, sentiment

def fetch_oanda_orderbook(top_zones_cache=None):
    if top_zones_cache is not None:
        current_price, top_zones, sentiment = top_zones_cache
    else:
        current_price, top_zones, sentiment = get_clustered_oanda_zones(zone_width=10.0)
        
    if not top_zones:
        return "Keine Orderbuch-Daten verfügbar."
        
    liquidity_info = (
        f"Current XAU/USD Price: {current_price}\n"
        f"Global OANDA Retail Positioning: {sentiment['long_pct']}% Long vs. {sentiment['short_pct']}% Short\n"
        f"Combined Multi-Source Liquidity Zones ($10 Clustered OANDA + Yahoo COMEX Futures):\n"
    )
    for z in top_zones:
        pos_label = "ABOVE price" if z["price"] >= current_price else "BELOW price"
        role_hint = "Buy-Stop/Long Magnet" if z["dominance"] == "BUY" else "Sell-Limit/Short Wall"
        overlap_tag = "OVERLAP [OANDA+COMEX]" if z.get("overlap") else f"[{z.get('sources', 'OANDA')}]"
        liquidity_info += (
            f"- Zone at {z['price']} ({pos_label}) {overlap_tag} | Vol: {z['volume']}% "
            f"(Net {z['dominance']} [{role_hint}])\n"
        )
    return liquidity_info

def fetch_oanda_price_and_volatility():
    if not OANDA_API_KEY:
        return "Keine Preisdaten verfügbar.", [], {}
    url = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/candles?count=48&granularity=H1&price=M"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}", "Accept-Datetime-Format": "UNIX"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return "OANDA Candle API Fehler", [], {}
            
        raw_candles = response.json().get("candles", [])
        parsed = []
        for c in raw_candles:
            if c.get("complete") or c == raw_candles[-1]:
                ts = float(c.get("time", 0))
                dt_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                parsed.append({
                    "ts": ts,
                    "dt_str": dt_str,
                    "o": float(c["mid"]["o"]),
                    "h": float(c["mid"]["h"]),
                    "l": float(c["mid"]["l"]),
                    "c": float(c["mid"]["c"])
                })
        
        if len(parsed) < 15:
            return "Zu wenige H1-Kerzen verfügbar.", parsed, {}

        trs = []
        for i in range(1, len(parsed)):
            h = parsed[i]["h"]
            l = parsed[i]["l"]
            prev_c = parsed[i - 1]["c"]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)
        
        atr_14_h1 = sum(trs[-14:]) / 14.0
        est_daily_atr = round(atr_14_h1 * 3.8, 2)

        curr_c = parsed[-1]["c"]
        c_4h_ago = parsed[-5]["c"] if len(parsed) >= 5 else parsed[0]["c"]
        c_24h_ago = parsed[-25]["c"] if len(parsed) >= 25 else parsed[0]["c"]

        change_4h = round(curr_c - c_4h_ago, 2)
        change_24h = round(curr_c - c_24h_ago, 2)
        ratio_4h = round(abs(change_4h) / est_daily_atr, 2) if est_daily_atr > 0 else 0.0

        vol_metrics = {
            "current_price": curr_c,
            "atr_h1": round(atr_14_h1, 2),
            "est_daily_atr": est_daily_atr,
            "change_4h": change_4h,
            "change_24h": change_24h,
            "move_to_daily_atr_ratio": ratio_4h
        }

        history_str = (
            f"XAU/USD Volatility & Priced-In Metrics:\n"
            f"- Current Price: ${curr_c} | 4h Change: ${change_4h:+.2f} | 24h Change: ${change_24h:+.2f}\n"
            f"- H1 ATR(14): ${atr_14_h1:.2f} | Est. Daily ATR: ${est_daily_atr:.2f} | 4h Move/Daily ATR Ratio: {ratio_4h}x\n"
        )
        return history_str, parsed, vol_metrics
    except Exception as e:
        return f"OANDA Candle Fehler: {e}", [], {}

def fetch_macro_context():
    if not OANDA_API_KEY:
        return "Keine Makro-Korrelationsdaten verfügbar.", {}
        
    instruments = {
        "EUR_USD": "EUR/USD (Inverse USD Proxy)",
        "USB10Y_USD": "US 10Y Bond Price (Inverse Yield Proxy)",
        "NAS100_USD": "Nasdaq 100 Tech/AI Index",
        "BCO_USD": "Brent Crude Oil (USD)"
    }
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    macro_lines = ["Cross-Asset Macro & Tech Context (24h Change):"]
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
                    pct_change = round(((last_c - first_c) / first_c) * 100, 2) if first_c > 0 else 0.0
                    macro_summary[instr] = {"price": last_c, "change_24h_pct": pct_change}
                    macro_lines.append(f"- {label}: {last_c} ({pct_change:+.2f}%)")
        except Exception as e:
            print(f"Makro-Abruf Fehler bei {instr}: {e}")

    return "\n".join(macro_lines), macro_summary

def evaluate_clusters(all_clusters, h1_candles, vol_metrics=None, macro_summary=None):
    """
    Prüft alle Signale mit Probability > 50% (ab 51%):
    - Punkt A: Speichert Headline, Quelle, ATR-Ratio & Makro-Kontext dauerhaft mit.
    - Punkt B: Trackt den maximalen Drawdown (USD) und nutzt ein weites Notbremse-Limit (1.8x Tages-ATR).
    - Punkt C: Evaluiert direkt über stats['history'], sodass Signale auch nach >24h nicht verloren gehen.
    """
    stats = {
        "total_evaluated": 0,
        "hits": 0,
        "misses": 0,
        "win_rate": 0.0,
        "active_signals": 0,
        "history": {}
    }
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
    macro_snap_str = f"EUR {eur_chg:+.2f}%, NAS100 {nas_chg:+.2f}%"

    # 1. Neue oder aktualisierte Cluster (>50% Wahrscheinlichkeit) in stats["history"] registrieren
    cluster_map = {}
    for cluster in all_clusters:
        summary = cluster.get("summary")
        if not isinstance(summary, dict):
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

        # Startpreis zum Zeitpunkt der Meldung aus den Kerzen bestimmen
        matching_candles = [c for c in h1_candles if c["ts"] <= cluster_ts]
        entry_price = matching_candles[-1]["c"] if matching_candles else curr_price

        if cluster_id not in stats["history"]:
            first_art = cluster["articles"][0] if cluster.get("articles") else {}
            stats["history"][cluster_id] = {
                "status": "ACTIVE",
                "headline": first_art.get("title", "Unbekannt")[:90],
                "source": first_art.get("source", "News"),
                "impact": "Bullish" if "Bullish" in impact else "Bearish",
                "prob": prob,
                "entry_price": round(entry_price, 2),
                "target": round(target_price, 2),
                "max_dd_usd": 0.0,
                "atr_ratio_at_signal": atr_ratio_now,
                "daily_atr_at_signal": est_daily_atr,
                "macro_snapshot": macro_snap_str,
                "created_ts": cluster_ts,
                "date": cluster.get("main_date", ""),
                "fail_reason": ""
            }
        else:
            # Falls das Cluster durch ein KI-Update ein neues Target bekommen hat, solange es noch ACTIVE ist
            existing_rec = stats["history"][cluster_id]
            if existing_rec.get("status") == "ACTIVE":
                existing_rec["target"] = round(target_price, 2)
                existing_rec["prob"] = prob

    # 2. Alle in stats["history"] gespeicherten Signale anhand der H1-Kerzen auswerten (unabhängig von data.json!)
    for sig_id, rec in stats["history"].items():
        status = rec.get("status", "ACTIVE")
        
        # Bereits abgeschlossene Signale nur an das aktuelle Cluster zurückmelden
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

        relevant_candles = [c for c in h1_candles if c["ts"] + 3600 >= sig_ts]
        hit_achieved = False
        stopped_out = False
        worst_dd = float(rec.get("max_dd_usd", 0.0))

        for c in relevant_candles:
            if imp == "Bullish":
                dd = round(max(0.0, entry_p - c["l"]), 2)
                if dd > worst_dd:
                    worst_dd = dd
                if dd >= max_allowed_dd:
                    stopped_out = True
                    break
                if c["h"] >= target_p:
                    hit_achieved = True
                    break
            elif imp == "Bearish":
                dd = round(max(0.0, c["h"] - entry_p), 2)
                if dd > worst_dd:
                    worst_dd = dd
                if dd >= max_allowed_dd:
                    stopped_out = True
                    break
                if c["l"] <= target_p:
                    hit_achieved = True
                    break

        rec["max_dd_usd"] = worst_dd

        if hit_achieved:
            rec["status"] = "HIT"
            rec["fail_reason"] = f"Target reached (Max Drawdown along the way: -${worst_dd:.2f})"
        elif stopped_out:
            rec["status"] = "MISS"
            rec["fail_reason"] = f"Exceeded catastrophe drawdown limit (-${worst_dd:.2f} >= {MAX_DRAWDOWN_ATR_MULT}x ATR) before target"
        else:
            age_hours = (now_ts - sig_ts) / 3600.0
            if age_hours >= EVAL_HORIZON_HOURS:
                rec["status"] = "MISS"
                rec["fail_reason"] = f"20h time limit expired without reaching ${target_p} (Max DD: -${worst_dd:.2f})"
            else:
                rec["status"] = "ACTIVE"

        if sig_id in cluster_map:
            cluster_map[sig_id]["eval_status"] = rec["status"]
            cluster_map[sig_id]["max_dd_usd"] = worst_dd

    hits = sum(1 for v in stats["history"].values() if v.get("status") == "HIT")
    misses = sum(1 for v in stats["history"].values() if v.get("status") == "MISS")
    active_count = sum(1 for v in stats["history"].values() if v.get("status") == "ACTIVE")
    total_eval = hits + misses

    stats["hits"] = hits
    stats["misses"] = misses
    stats["total_evaluated"] = total_eval
    stats["win_rate"] = round((hits / total_eval) * 100, 1) if total_eval > 0 else 0.0
    stats["active_signals"] = active_count

    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Fehler beim Speichern von {STATS_FILE}: {e}")

    return stats

def format_performance_memory(stats):
    """Baut aus den detaillierten Kontext-Einträgen (Punkt A + B + C) das Lern-Gedächtnis für den KI-Agenten."""
    if not stats or stats.get("total_evaluated", 0) == 0:
        return f"Historical Track Record (>50% setups): No completed 20h evaluations yet ({stats.get('active_signals', 0)} active setups currently tracked)."
    
    completed = [v for v in stats.get("history", {}).values() if v.get("status") in ("HIT", "MISS")]
    recent_items = completed[-6:]
    
    details_lines = []
    for item in recent_items:
        hl = item.get("headline", "News")[:55]
        imp = item.get("impact", "")
        prob = item.get("prob", 0)
        ent = item.get("entry_price", 0)
        trg = item.get("target", 0)
        st = item.get("status", "")
        dd = item.get("max_dd_usd", 0.0)
        atr_r = item.get("atr_ratio_at_signal", 0.0)
        mac = item.get("macro_snapshot", "")
        reason = item.get("fail_reason", "")
        details_lines.append(
            f"  * [{st}] '{hl}...' | {imp} ({prob}%) Entry ${ent} -> Target ${trg} | Max DD: -${dd} | Signal-ATR-Ratio: {atr_r}x | Macro: {mac} | Note: {reason}"
        )
    
    recent_block = "\n".join(details_lines)
    
    return (
        f"Historical Signal Track Record (All directional setups >50% probability):\n"
        f"- Total Evaluated: {stats['total_evaluated']} | Win Rate: {stats['win_rate']}% ({stats['hits']} Hits, {stats['misses']} Misses) | Currently Active: {stats.get('active_signals', 0)}\n"
        f"- Last {len(recent_items)} Completed Setups with Context & Drawdown:\n{recent_block}\n"
        f"- Self-Learning Rule: Analyze WHY past setups hit or missed (e.g., high Signal-ATR-Ratio >1.2x, opposing Macro, or overly distant targets) and explicitly reference these lessons in 'Historical_Calibration'."
    )

def build_local_fallback_summary(cluster, oanda_zones_cache, vol_metrics, macro_summary):
    articles = cluster.get("articles", [])
    oldest = min(articles, key=lambda x: x.get("_timestamp", 0)) if articles else {}
    age_h = round((datetime.now().timestamp() - float(oldest.get("_timestamp", datetime.now().timestamp()))) / 3600.0, 1)

    facts_list = []
    for a in articles[:4]:
        desc = a.get("description", "").strip()
        snippet = f"{a['title']} - {desc}" if desc and desc.lower() != a['title'].lower() else a['title']
        facts_list.append(f"[{a['source']}] {snippet[:260]}")
    facts_str = " | ".join(facts_list)

    full_text = " ".join([f"{a.get('title','')} {a.get('description','')}" for a in articles]).lower()
    bull_kws = ["attack", "strike", "missile", "escalat", "war", "inflation", "recession", "cut", "crisis", "selloff", "crash", "sanction", "houthi", "iran"]
    bear_kws = ["truce", "ceasefire", "peace", "deal", "strong dollar", "rate hike", "surge", "rally"]

    bull_hits = sum(1 for k in bull_kws if k in full_text)
    bear_hits = sum(1 for k in bear_kws if k in full_text)

    curr_price, top_zones, sentiment = oanda_zones_cache if oanda_zones_cache else (0.0, [], {"long_pct": 50, "short_pct": 50})
    above_z = [z["price"] for z in top_zones if z["price"] > curr_price]
    below_z = [z["price"] for z in top_zones if z["price"] < curr_price]

    if bull_hits > bear_hits:
        impact = "Bullish"
        target = f"{min(above_z):.2f}" if above_z else "N/A"
    elif bear_hits > bull_hits:
        impact = "Bearish"
        target = f"{max(below_z):.2f}" if below_z else "N/A"
    else:
        impact = "Neutral"
        target = "N/A"

    eur_chg = macro_summary.get("EUR_USD", {}).get("change_24h_pct", 0)
    nas_chg = macro_summary.get("NAS100_USD", {}).get("change_24h_pct", 0)
    bco_chg = macro_summary.get("BCO_USD", {}).get("change_24h_pct", 0)
    atr_ratio = vol_metrics.get("move_to_daily_atr_ratio", 0)
    chg_4h = vol_metrics.get("change_4h", 0)

    return {
        "Event_Timestamp": f"Erstmeldung: {oldest.get('date', 'Unbekannt')} UTC (vor ca. {age_h}h)",
        "Facts": facts_str,
        "Uncertainties": "Direkte Auswertung aus den Primaerquellen (KI-Tiefenanalyse wird im naechsten freien Zyklus ergaenzt).",
        "Macro_Context": f"24h Makro-Umfeld: EUR/USD {eur_chg:+.2f}%, Nasdaq-100 {nas_chg:+.2f}%, Brent Oil {bco_chg:+.2f}% | OANDA Sentiment: {sentiment['long_pct']}% Long / {sentiment['short_pct']}% Short.",
        "Bull_Case": f"Naechste OANDA/COMEX Liquiditaetszone auf der Oberseite liegt bei ${min(above_z):.2f}." if above_z else "Geopolitische/Makro-Risikopraemie beachten.",
        "Bear_Case": f"Naechste Unterstuetzungs-/Verkaufszone auf der Unterseite bei ${max(below_z):.2f}. 4h-ATR-Ratio steht bei {atr_ratio}x." if below_z else f"4h-ATR-Ratio steht bei {atr_ratio}x.",
        "Historical_Calibration": f"Lokale Echtzeit-Zusammenfassung aktiv. Ereignis-Alter: {age_h}h.",
        "Impact_Timing": "Past (Already Priced In)" if (age_h > 12 or atr_ratio >= 1.2) else "Present",
        "Timeframe": "Hours/Days",
        "XAUUSD_Impact": impact,
        "Probability_Percent": 55 if impact != "Neutral" else 40,
        "Past_Price_Action_Cause": f"4h XAU/USD Bewegung: ${chg_4h:+.2f} ({atr_ratio}x der geschaetzten Tages-ATR).",
        "Orderflow_Price_Target": target
    }

def generate_batch_summaries(clusters_batch, orderbook_data, price_history_data, macro_context_str, perf_memory_str):
    if not client or not clusters_batch:
        return {}
    
    now_utc_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    
    clusters_input = []
    for idx, c in enumerate(clusters_batch):
        recent_arts = sorted(c["articles"], key=lambda x: x.get("_timestamp", 0), reverse=True)[:4]
        arts_text = "\n".join([f"  * [{a['source']}] ({a['date']} UTC): {a['title']} — {a['description']}" for a in recent_arts])
        clusters_input.append(f"=== CLUSTER_ID: {idx} ===\n{arts_text}")
        
    all_clusters_str = "\n\n".join(clusters_input)
    
    prompt = f"""You are an objective Intelligence & Calibration Agent for a manual Gold (XAU/USD) & Macro trader.
    Analyze the following {len(clusters_batch)} News Clusters using the shared market, macro, orderbook, and historical track record context below.
    Do NOT hide or dismiss minor news—thoroughly contextualize every cluster!
    Note: Every directional forecast ('Bullish' or 'Bearish') with Probability_Percent > 50% will be automatically tracked and evaluated over 20 hours against actual OANDA candles!
    Output ONLY a valid JSON array of {len(clusters_batch)} objects (one object per CLUSTER_ID). Write all values in English!

    Current Reference Time: {now_utc_str}

    Shared Market, Orderbook & Self-Learning Memory:
    {perf_memory_str}
    {macro_context_str}
    {price_history_data}
    {orderbook_data}

    News Clusters to Analyze:
    {all_clusters_str}

    Required JSON output format (MUST be a JSON Array containing one object for each CLUSTER_ID from 0 to {len(clusters_batch)-1}):
    [
      {{
        "Cluster_ID": 0,
        "Event_Timestamp": "Exact or estimated date/time when the real-world event occurred and how many hours ago relative to {now_utc_str}.",
        "Facts": "Comprehensive, unfiltered summary of all core facts, numbers, and WHY it matters. Include [Source].",
        "Uncertainties": "What is unclear, speculative, or unconfirmed?",
        "Macro_Context": "How USD, 10Y Yields, Nasdaq 100 (Tech/AI), Brent Oil, and OANDA Retail Sentiment interact with this news.",
        "Bull_Case": "Arguments and catalysts supporting a Bullish XAU/USD reaction.",
        "Bear_Case": "Counter-arguments, priced-in risks, or macro headwinds supporting a Bearish/muted XAU/USD reaction.",
        "Historical_Calibration": "Candid calibration note referencing specific lessons from the Historical Signal Track Record (e.g., past drawdowns, ATR exhaustion, macro alignment).",
        "Impact_Timing": "Future, Present, or Past (Already Priced In)",
        "Timeframe": "Hours/Days, Days/Weeks, or Weeks/Months (Write 'N/A' if Past)",
        "XAUUSD_Impact": "Bullish, Bearish, or Neutral",
        "Probability_Percent": 60,
        "Past_Price_Action_Cause": "How the 4h/24h price change and ATR ratio align with this news.",
        "Orderflow_Price_Target": "Exact numerical Take-Profit target from the Liquidity Zones (strictly > current price for Bullish, < current price for Bearish, or 'N/A' for Neutral)."
      }}
    ]
    """
    
    for model_name in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()
                
            parsed_list = json.loads(raw_text)
            if isinstance(parsed_list, dict):
                for v in parsed_list.values():
                    if isinstance(v, list):
                        parsed_list = v
                        break
                        
            if isinstance(parsed_list, list) and len(parsed_list) > 0:
                result_map = {}
                for i, item in enumerate(parsed_list):
                    if isinstance(item, dict):
                        cid = item.get("Cluster_ID", i)
                        result_map[cid] = item
                print(f" -> Batch ({len(result_map)} Cluster) auf einen Schlag erfolgreich analysiert mit {model_name}!")
                return result_map
        except Exception as e:
            err_msg = str(e).splitlines()[0][:110]
            print(f" -> {model_name} nicht verfügbar ({err_msg}) - probiere nächstes Modell...")
            time.sleep(1.5)
            
    return {}

def update_market_data(top_zones_cache=None, macro_summary=None, vol_metrics=None):
    if not OANDA_API_KEY:
        print("WARNUNG: Kein OANDA_API_KEY vorhanden, überspringe Chart-Update.")
        return
    
    old_market_data = {}
    if os.path.exists("market_data.json"):
        try:
            with open("market_data.json", "r", encoding="utf-8") as f:
                old_market_data = json.load(f)
        except Exception:
            old_market_data = {}

    market_data = {
        "candles": [],
        "orderbook": [],
        "sentiment": {"long_pct": 50.0, "short_pct": 50.0},
        "macro": macro_summary or old_market_data.get("macro", {}),
        "volatility": vol_metrics or old_market_data.get("volatility", {})
    }
    
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Accept-Datetime-Format": "UNIX"
    }
    
    try:
        url_candles = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/candles?count=100&granularity=H4&price=M"
        res_c = requests.get(url_candles, headers=headers, timeout=10)
        if res_c.status_code == 200:
            for c in res_c.json().get("candles", []):
                market_data["candles"].append({
                    "time": int(float(c["time"])),
                    "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"])
                })
            print(f"Erfolgreich {len(market_data['candles'])} H4-Kerzen geladen.")
        else:
            print(f"OANDA Candle API Fehler: {res_c.status_code} - {res_c.text}")
    except Exception as e:
        print(f"Fehler beim Laden der H4-Kerzen: {e}")

    try:
        if top_zones_cache is not None:
            _, top_zones, sentiment = top_zones_cache
        else:
            _, top_zones, sentiment = get_clustered_oanda_zones(zone_width=10.0)
            
        market_data["sentiment"] = sentiment
        for z in top_zones:
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
        print(f"Erfolgreich {len(market_data['orderbook'])} kombinierte Order-Zonen geladen.")
    except Exception as e:
        print(f"Fehler beim Laden der Orderbuch-Zonen: {e}")

    if not market_data["candles"] and old_market_data.get("candles"):
        print("WARNUNG: Keine neuen Kerzen erhalten! Behalte bisherige Kerzen bei.")
        market_data["candles"] = old_market_data["candles"]

    if not market_data["orderbook"] and old_market_data.get("orderbook"):
        print("WARNUNG: Keine neuen Order-Zonen erhalten! Behalte bisherige Zonen bei.")
        market_data["orderbook"] = old_market_data["orderbook"]
        if market_data["sentiment"]["long_pct"] == 50.0 and old_market_data.get("sentiment"):
            market_data["sentiment"] = old_market_data["sentiment"]

    with open("market_data.json", "w", encoding="utf-8") as f:
        json.dump(market_data, f)

def main():
    print("Lade Embeddings-Modell...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    existing_data = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = []

    print("Lade Volatilität (ATR), Multi-Source Order-Zonen (OANDA + COMEX Futures) & Makro/Tech-Korrelationen...")
    price_history_data, h1_candles, vol_metrics = fetch_oanda_price_and_volatility()
    candle_fallback_price = vol_metrics.get("current_price", 0.0)
    
    oanda_zones_cache = get_clustered_oanda_zones(zone_width=10.0, fallback_price=candle_fallback_price)
    macro_context_str, macro_summary = fetch_macro_context()

    current_stats = evaluate_clusters(existing_data, h1_candles, vol_metrics=vol_metrics, macro_summary=macro_summary)
    perf_memory_str = format_performance_memory(current_stats)

    now = datetime.now()
    cutoff_timestamp = (now - timedelta(hours=24)).timestamp()
    
    valid_clusters = [c for c in existing_data if c.get("_timestamp", 0) > cutoff_timestamp]
    
    existing_urls = set()
    for c in valid_clusters:
        for a in c.get("articles", []):
            existing_urls.add(a["link"])

    print("Suche parallel nach neuen Artikeln (Geopolitik, Rohstoffe, Tech/AI & Weltwirtschaft)...")
    new_articles = fetch_new_articles(existing_urls)
    
    if new_articles:
        print(f"{len(new_articles)} neue Artikel gefunden. Clustere (Threshold: {SIMILARITY_THRESHOLD})...")
        
        cluster_embeddings = []
        for cluster in valid_clusters:
            cluster_text = " ".join([a["title"] for a in cluster["articles"]])
            cluster_embeddings.append(model.encode(cluster_text))
            
        for article in new_articles:
            article_emb = model.encode(article["title"] + " " + article["description"])
            best_match_idx = -1
            highest_sim = -1
            
            for i, cluster_emb in enumerate(cluster_embeddings):
                sim = util.cos_sim(article_emb, cluster_emb).item()
                if sim > highest_sim:
                    highest_sim = sim
                    best_match_idx = i
                    
            if highest_sim > SIMILARITY_THRESHOLD and best_match_idx != -1:
                target_cluster = valid_clusters[best_match_idx]
                target_cluster["articles"].append(article)
                
                oldest_article = min(target_cluster["articles"], key=lambda x: x["_timestamp"])
                newest_article = max(target_cluster["articles"], key=lambda x: x["_timestamp"])
                
                target_cluster["original_date"] = oldest_article["date"]
                
                if newest_article["date"] != target_cluster["original_date"]:
                    target_cluster["main_date"] = f"Erstmeldung: {target_cluster['original_date']} | Letztes Update: {newest_article['date']}"
                else:
                    target_cluster["main_date"] = f"Erstmeldung: {newest_article['date']}"

                target_cluster["_timestamp"] = newest_article["_timestamp"]
                target_cluster["needs_update"] = True
                
                updated_text = " ".join([a["title"] for a in target_cluster["articles"]])
                cluster_embeddings[best_match_idx] = model.encode(updated_text)
            else:
                new_cluster = {
                    "main_date": f"Erstmeldung: {article['date']}",
                    "original_date": article["date"],
                    "_timestamp": article["_timestamp"],
                    "articles": [article],
                    "needs_update": True,
                    "summary": None
                }
                valid_clusters.append(new_cluster)
                cluster_embeddings.append(article_emb)
    else:
        print("Keine neuen Artikel gefunden.")

    valid_clusters.sort(key=lambda x: float(x.get("_timestamp", 0)), reverse=True)
    
    pending_clusters = [
        c for c in valid_clusters 
        if c.get("needs_update", False) or not isinstance(c.get("summary"), dict)
    ]
    
    if pending_clusters:
        orderbook_data = fetch_oanda_orderbook(top_zones_cache=oanda_zones_cache)
        total_batches = min(MAX_BATCH_CALLS_PER_RUN, (len(pending_clusters) + BATCH_SIZE - 1) // BATCH_SIZE)
        
        for b_idx in range(total_batches):
            batch_slice = pending_clusters[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
            print(f"Starte KI-Batch-Analyse {b_idx + 1}/{total_batches} ({len(batch_slice)} Cluster gleichzeitig in 1 Aufruf)...")
            
            batch_results = generate_batch_summaries(
                batch_slice,
                orderbook_data,
                price_history_data,
                macro_context_str,
                perf_memory_str
            )
            
            for idx_in_batch, cluster in enumerate(batch_slice):
                if idx_in_batch in batch_results:
                    cluster["summary"] = batch_results[idx_in_batch]
                    cluster["needs_update"] = False
            
            if b_idx < total_batches - 1 and batch_results:
                time.sleep(3)

    fallback_count = 0
    for c in valid_clusters:
        if not isinstance(c.get("summary"), dict):
            c["summary"] = build_local_fallback_summary(c, oanda_zones_cache, vol_metrics, macro_summary)
            c["needs_update"] = True
            fallback_count += 1
            
    if fallback_count > 0:
        print(f"Info: {fallback_count} Cluster wurden sofort mit der lokalen Echtzeit-Zusammenfassung gefüllt (kein Informationsverlust).")

    evaluate_clusters(valid_clusters, h1_candles, vol_metrics=vol_metrics, macro_summary=macro_summary)
            
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_clusters, f, indent=4, ensure_ascii=False)
        
    print("Aktualisiere Chart- & Makro-Daten...")
    update_market_data(top_zones_cache=oanda_zones_cache, macro_summary=macro_summary, vol_metrics=vol_metrics)
    print("Update abgeschlossen.")

if __name__ == "__main__":
    main()
