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

# --- KONFIGURATION ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OANDA_API_KEY = os.environ.get("OANDA_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None
    print("WARNUNG: GEMINI_API_KEY nicht gefunden.")

MODEL_NAME = "gemini-3.5-flash-lite"

# Globale & Regionale RSS-Feeds
FEEDS = {
    # Westliche Leitmedien
    "CNBC Commodities": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000118",
    "Investing.com": "https://www.investing.com/rss/news_11.rss",
    "Yahoo Finance Gold": "https://finance.yahoo.com/rss/headline?s=GC=F",
    "Yahoo Finance Oil": "https://finance.yahoo.com/rss/headline?s=CL=F",
    
    # Regional Finance & Energie (Golfregion)
    "Zawya": "https://www.zawya.com/en/rss",
    "Arab News": "https://www.arabnews.com/cat/4/rss.xml",
    
    # Vor-Ort Geopolitik (Nahost, Iran, Israel, Jemen)
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
    "irgc", "escalat", "retaliat", "war", "truce", "ceasefire", "fed", 
    "inflation", "rate", "central bank", "brics", "dollar", "treasury", "tariff", "saudi"
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
                "description": clean_desc,
                "_timestamp": pub_dt.timestamp()
            })
    except Exception as e:
        print(f"Timeout/Fehler bei {source}: {e}")
    return articles

def fetch_new_articles(existing_urls):
    articles = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        results = executor.map(fetch_single_feed, FEEDS.items())
        for feed_articles in results:
            for a in feed_articles:
                if a["link"] not in existing_urls:
                    articles.append(a)
    return articles

def fetch_binance_paxg_zones(oanda_price, zone_width=10.0):
    """Ruft ohne Login das tiefe PAXG/USDT Orderbuch von Binance ab, normiert es auf den XAU/USD Kurs und clustert in 10$-Zonen."""
    paxg_zones = {}
    try:
        url = "https://api.binance.com/api/v3/depth?symbol=PAXGUSDT&limit=1000"
        res = requests.get(url, timeout=8)
        if res.status_code != 200:
            return paxg_zones
            
        data = res.json()
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if not bids or not asks:
            return paxg_zones

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        paxg_mid = (best_bid + best_ask) / 2.0

        # Preisdifferenz (Basis) zwischen PAXG und OANDA Spot-Gold ausgleichen
        basis_offset = (oanda_price - paxg_mid) if oanda_price > 0 else 0.0
        ref_price = oanda_price if oanda_price > 0 else paxg_mid

        raw_entries = []
        total_oz = 0.0

        for p_str, q_str in bids:
            adj_p = float(p_str) + basis_offset
            q = float(q_str)
            if abs(adj_p - ref_price) / ref_price <= 0.20:
                raw_entries.append((adj_p, q, 0.0))
                total_oz += q

        for p_str, q_str in asks:
            adj_p = float(p_str) + basis_offset
            q = float(q_str)
            if abs(adj_p - ref_price) / ref_price <= 0.20:
                raw_entries.append((adj_p, 0.0, q))
                total_oz += q

        if total_oz <= 0:
            return paxg_zones

        # Auf 100% Gesamtvolumen normieren, damit die Gewichtung zu OANDA passt
        for adj_p, b_oz, a_oz in raw_entries:
            zone_key = round(adj_p / zone_width) * zone_width
            if zone_key not in paxg_zones:
                paxg_zones[zone_key] = {"long": 0.0, "short": 0.0, "total": 0.0, "weighted_price_sum": 0.0}
            
            b_pct = (b_oz / total_oz) * 100.0
            a_pct = (a_oz / total_oz) * 100.0
            tot_pct = b_pct + a_pct

            paxg_zones[zone_key]["long"] += b_pct
            paxg_zones[zone_key]["short"] += a_pct
            paxg_zones[zone_key]["total"] += tot_pct
            paxg_zones[zone_key]["weighted_price_sum"] += adj_p * tot_pct

    except Exception as e:
        print(f"Hinweis: Binance PAXG Orderbuch nicht erreichbar ({e})")

    return paxg_zones

def get_clustered_oanda_zones(zone_width=10.0):
    """Kombiniert OANDA (OrderBook + PositionBook) und Binance (PAXG Depth) im 20%-Radius und markiert Overlaps."""
    if not OANDA_API_KEY:
        return 0.0, [], {"long_pct": 50.0, "short_pct": 50.0}
        
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    url_ob = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/orderBook"
    url_pb = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/positionBook"
    
    current_price = 0.0
    combined_buckets = {}
    total_pos_long = 0.0
    total_pos_short = 0.0
    
    try:
        res_ob = requests.get(url_ob, headers=headers, timeout=10)
        if res_ob.status_code == 200:
            ob_data = res_ob.json().get("orderBook", {})
            current_price = float(ob_data.get("price", 0))
            for b in ob_data.get("buckets", []):
                try:
                    p = float(b.get("price", 0))
                    if current_price > 0 and abs(p - current_price) / current_price <= 0.20:
                        combined_buckets[p] = {
                            "long": float(b.get("longCountPercent", 0)),
                            "short": float(b.get("shortCountPercent", 0))
                        }
                except ValueError:
                    continue
    except Exception as e:
        print(f"Fehler bei OANDA orderBook: {e}")

    try:
        res_pb = requests.get(url_pb, headers=headers, timeout=10)
        if res_pb.status_code == 200:
            pb_data = res_pb.json().get("positionBook", {})
            if current_price == 0:
                current_price = float(pb_data.get("price", 0))
            for b in pb_data.get("buckets", []):
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
    except Exception as e:
        print(f"Fehler bei OANDA positionBook: {e}")

    pos_sum = total_pos_long + total_pos_short
    if pos_sum > 0:
        sentiment = {
            "long_pct": round((total_pos_long / pos_sum) * 100, 1),
            "short_pct": round((total_pos_short / pos_sum) * 100, 1)
        }
    else:
        sentiment = {"long_pct": 50.0, "short_pct": 50.0}

    if not combined_buckets or current_price == 0:
        return 0.0, [], sentiment

    # OANDA in 10$-Zonen clustern
    zones = {}
    for p, vols in combined_buckets.items():
        zone_key = round(p / zone_width) * zone_width
        if zone_key not in zones:
            zones[zone_key] = {
                "long": 0.0, "short": 0.0, "total": 0.0, "weighted_price_sum": 0.0,
                "oanda_vol": 0.0, "paxg_vol": 0.0
            }
        
        tot = vols["long"] + vols["short"]
        zones[zone_key]["long"] += vols["long"]
        zones[zone_key]["short"] += vols["short"]
        zones[zone_key]["total"] += tot
        zones[zone_key]["oanda_vol"] += tot
        zones[zone_key]["weighted_price_sum"] += p * tot

    # Zweite Quelle: Binance PAXG Orderbuch abrufen und mit OANDA-Zonen verschmelzen
    paxg_zones = fetch_binance_paxg_zones(current_price, zone_width=zone_width)
    for z_key, p_data in paxg_zones.items():
        # PAXG mit Faktor 0.25 gewichten, damit einzelne Krypto-Maker-Walls OANDA nicht komplett verzerren
        weight = 0.25
        scaled_long = p_data["long"] * weight
        scaled_short = p_data["short"] * weight
        scaled_tot = p_data["total"] * weight

        if z_key not in zones:
            zones[z_key] = {
                "long": 0.0, "short": 0.0, "total": 0.0, "weighted_price_sum": 0.0,
                "oanda_vol": 0.0, "paxg_vol": 0.0
            }

        zones[z_key]["long"] += scaled_long
        zones[z_key]["short"] += scaled_short
        zones[z_key]["total"] += scaled_tot
        zones[z_key]["paxg_vol"] += p_data["total"]
        zones[z_key]["weighted_price_sum"] += p_data["weighted_price_sum"] * weight

    above_zones = []
    below_zones = []

    for z_key, z_data in zones.items():
        if z_data["total"] > 0:
            vwap_price = round(z_data["weighted_price_sum"] / z_data["total"], 2)
            net_diff = round(z_data["long"] - z_data["short"], 2)
            dominance = "BUY" if net_diff >= 0 else "SELL"
            
            has_oanda = z_data["oanda_vol"] >= 0.5
            has_paxg = z_data["paxg_vol"] >= 0.8
            is_overlap = has_oanda and has_paxg

            if is_overlap:
                sources_label = "OANDA+PAXG"
            elif has_paxg and not has_oanda:
                sources_label = "PAXG"
            else:
                sources_label = "OANDA"

            zone_obj = {
                "price": vwap_price,
                "zone_center": z_key,
                "long": round(z_data["long"], 2),
                "short": round(z_data["short"], 2),
                "volume": round(z_data["total"], 2),
                "oanda_vol": round(z_data["oanda_vol"], 2),
                "paxg_vol": round(z_data["paxg_vol"], 2),
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
        f"Global OANDA Retail Positioning (Contrarian Indicator): {sentiment['long_pct']}% Long vs. {sentiment['short_pct']}% Short\n"
        f"Combined Multi-Source Liquidity Zones ($10 Clustered OANDA + Binance PAXG within 20% radius):\n"
    )
    for z in top_zones:
        pos_label = "ABOVE price" if z["price"] >= current_price else "BELOW price"
        role_hint = "Buy-Stop/Long Magnet" if z["dominance"] == "BUY" else "Sell-Limit/Short Wall"
        overlap_tag = "OVERLAP [OANDA+PAXG]" if z.get("overlap") else f"[{z.get('sources', 'OANDA')}]"
        liquidity_info += (
            f"- Zone at {z['price']} ({pos_label}) {overlap_tag} | Combined Vol: {z['volume']}% "
            f"(OANDA: {z.get('oanda_vol', 0)}%, PAXG: {z.get('paxg_vol', 0)}% | Net {z['dominance']} Dominance [{role_hint}])\n"
        )
    return liquidity_info

def fetch_oanda_price_and_volatility():
    if not OANDA_API_KEY:
        return "Keine Preisdaten verfügbar.", [], {}
    url = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/candles?count=30&granularity=H1&price=M"
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
            f"- Current Price: ${curr_c}\n"
            f"- 4h Price Change: ${change_4h:+.2f} | 24h Price Change: ${change_24h:+.2f}\n"
            f"- H1 ATR(14): ${atr_14_h1:.2f} | Estimated Daily ATR: ${est_daily_atr:.2f}\n"
            f"- 4h Move vs. Daily ATR Ratio: {ratio_4h}x (WARNING: If >= 1.2x in the direction of the news, the move is likely ALREADY PRICED IN!)\n\n"
            f"XAU/USD H1 Price History (Last 24h):\n"
        )
        for c in parsed[-24:]:
            history_str += f"- {c['dt_str']} UTC: Open {c['o']}, High {c['h']}, Low {c['l']}, Close {c['c']}\n"

        return history_str, parsed, vol_metrics
    except Exception as e:
        return f"OANDA Candle Fehler: {e}", [], {}

def fetch_macro_context():
    if not OANDA_API_KEY:
        return "Keine Makro-Korrelationsdaten verfügbar.", {}
        
    instruments = {
        "EUR_USD": "EUR/USD (Inverse USD Proxy)",
        "USB10Y_USD": "US 10Y Treasury Bond Price (Inverse Yield Proxy)",
        "BCO_USD": "Brent Crude Oil (USD)"
    }
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    macro_lines = ["Cross-Asset Macro Context (24h Change):"]
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
                    
                    if instr == "EUR_USD":
                        dxy_dir = "USD Strengthening (Headwind for Gold)" if pct_change < -0.15 else ("USD Weakening (Tailwind for Gold)" if pct_change > 0.15 else "USD Flat")
                        macro_lines.append(f"- {label}: {last_c} ({pct_change:+.2f}%) -> {dxy_dir}")
                    elif instr == "USB10Y_USD":
                        yield_dir = "Bond Prices Down / Yields Rising (Headwind for Gold)" if pct_change < -0.20 else ("Bond Prices Up / Yields Falling (Tailwind for Gold)" if pct_change > 0.20 else "Yields Stable")
                        macro_lines.append(f"- {label}: {last_c} ({pct_change:+.2f}%) -> {yield_dir}")
                    else:
                        macro_lines.append(f"- {label}: {last_c} ({pct_change:+.2f}%)")
        except Exception as e:
            print(f"Makro-Abruf Fehler bei {instr}: {e}")

    return "\n".join(macro_lines), macro_summary

def evaluate_clusters(all_clusters, h1_candles):
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

        if prob < 67 or not ("Bullish" in impact or "Bearish" in impact):
            continue

        cluster_id = cluster["articles"][0]["link"] if cluster.get("articles") else str(cluster.get("_timestamp"))
        cluster_ts = float(cluster.get("_timestamp", 0))

        if cluster_id in stats["history"] and stats["history"][cluster_id]["status"] == "HIT":
            cluster["eval_status"] = "HIT"
            continue

        relevant_candles = [c for c in h1_candles if c["ts"] + 3600 >= cluster_ts]
        hit_achieved = False

        for c in relevant_candles:
            if "Bullish" in impact and c["h"] >= target_price:
                hit_achieved = True
                break
            elif "Bearish" in impact and c["l"] <= target_price:
                hit_achieved = True
                break

        if hit_achieved:
            cluster["eval_status"] = "HIT"
            stats["history"][cluster_id] = {
                "status": "HIT",
                "impact": impact,
                "target": target_price,
                "date": cluster.get("main_date", "")
            }
        else:
            age_hours = (now_ts - cluster_ts) / 3600.0
            if age_hours >= 20.0:
                cluster["eval_status"] = "MISS"
                stats["history"][cluster_id] = {
                    "status": "MISS",
                    "impact": impact,
                    "target": target_price,
                    "date": cluster.get("main_date", "")
                }
            else:
                cluster["eval_status"] = "ACTIVE"

    hits = sum(1 for v in stats["history"].values() if v.get("status") == "HIT")
    misses = sum(1 for v in stats["history"].values() if v.get("status") == "MISS")
    total_eval = hits + misses
    active_count = sum(1 for c in all_clusters if c.get("eval_status") == "ACTIVE")

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

def update_market_data(top_zones_cache=None, macro_summary=None, vol_metrics=None):
    if not OANDA_API_KEY:
        print("WARNUNG: Kein OANDA_API_KEY vorhanden, überspringe Chart-Update.")
        return
    
    market_data = {
        "candles": [],
        "orderbook": [],
        "sentiment": {"long_pct": 50.0, "short_pct": 50.0},
        "macro": macro_summary or {},
        "volatility": vol_metrics or {}
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
                "paxg_vol": z.get("paxg_vol", 0),
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

    if not market_data["candles"] and os.path.exists("market_data.json"):
        print("WARNUNG: Keine neuen Kerzen erhalten! Behalte bisherige Kerzen bei.")
        try:
            with open("market_data.json", "r", encoding="utf-8") as f:
                old_data = json.load(f)
                market_data["candles"] = old_data.get("candles", [])
        except Exception:
            pass

    with open("market_data.json", "w", encoding="utf-8") as f:
        json.dump(market_data, f)

def generate_summary(articles_in_cluster, orderbook_data, price_history_data, macro_context_str, max_retries=2):
    if not client:
        return None
    
    news_data = "\n".join([f"Source: [{a['source']}]\nDate: {a['date']}\nHeadline: {a['title']}\nDetails: {a['description']}\n" for a in articles_in_cluster])
    
    prompt = f"""You are a senior quantitative gold & macro strategist. Analyze these news reports, the 24h price & ATR volatility metrics, cross-asset macro correlations (USD, 10Y Bonds, Brent Oil), and multi-source orderflow liquidity (OANDA + Binance PAXG). Output ONLY a valid JSON object without markdown blocks. Write all values in English!
    
    CRITICAL RULE 1 - 'PRICED-IN' EXHAUSTION CHECK:
    - Check the '4h Move vs. Daily ATR Ratio' in the Volatility Metrics below.
    - If XAU/USD has ALREADY moved by >= 1.2x of the Estimated Daily ATR in the direction of the news over the last 4-12 hours, you MUST downgrade 'XAUUSD_Impact' to 'Neutral' (or lower 'Probability_Percent' below 55) and set 'Impact_Timing' to 'Past (Already Priced In)'. Never chase an exhausted spike!

    CRITICAL RULE 2 - MACRO HEADWIND CHECK:
    - Compare the news with the 'Cross-Asset Macro Context' (EUR/USD as inverse USD proxy and USB10Y_USD as inverse Treasury Yield proxy).
    - If USD is strengthening sharply AND Treasury yields are rising (bond prices falling), reduce Bullish Gold probabilities by 15-20% unless the geopolitical shock is severe.

    CRITICAL RULE 3 - 'Orderflow_Price_Target' & MULTI-SOURCE CONFLUENCE:
    - Look at the 'Current XAU/USD Price' and the 'Combined Multi-Source Liquidity Zones' below. Prefer zones marked with 'OVERLAP [OANDA+PAXG]' as primary price targets.
    - If 'XAUUSD_Impact' is 'Bullish', the 'Orderflow_Price_Target' (Take Profit) MUST be a numerical price strictly HIGHER than the Current XAU/USD Price. Never output a level below current price for a Bullish signal!
    - If 'XAUUSD_Impact' is 'Bearish', the 'Orderflow_Price_Target' (Take Profit) MUST be a numerical price strictly LOWER than the Current XAU/USD Price. Never output a level above current price for a Bearish signal!
    - If 'XAUUSD_Impact' is 'Neutral', write 'N/A'.
    - Format the target strictly as a clean number (e.g., '4035.50').

    Required JSON structure:
    {{
        "Facts": "Detailed summary of core facts and WHY it happened. Include [Source].",
        "Uncertainties": "What is unclear, speculative, or unconfirmed?",
        "Macro_Context": "Explain how USD, 10Y Yields, Brent Oil, and OANDA Retail Sentiment interact with this news.",
        "Impact_Timing": "Future, Present, or Past (Already Priced In)",
        "Timeframe": "Hours/Days, Days/Weeks, or Weeks/Months (Write 'N/A' if Past)",
        "XAUUSD_Impact": "Bullish, Bearish, or Neutral",
        "Probability_Percent": <Integer 0 to 100>,
        "Past_Price_Action_Cause": "Explain how the 4h/24h price change and ATR ratio align with this news and whether it is already priced in.",
        "Orderflow_Price_Target": "Exact numerical Take-Profit target following CRITICAL RULE 3."
    }}

    News Reports:
    {news_data}
    
    Cross-Asset Macro Context:
    {macro_context_str}

    24h Price History & ATR Volatility:
    {price_history_data}

    Current Order Book & Retail Sentiment:
    {orderbook_data}
    """
    
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()
                
            return json.loads(raw_text)
        except Exception as e:
            print(f"Gemini API Versuch {attempt}/{max_retries} ({MODEL_NAME}) fehlgeschlagen: {e}")
            if attempt < max_retries:
                time.sleep(4)
    return None

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

    print("Lade Multi-Source Order-Zonen (OANDA + PAXG), Volatilität (ATR) & Makro-Korrelationen...")
    oanda_zones_cache = get_clustered_oanda_zones(zone_width=10.0)
    price_history_data, h1_candles, vol_metrics = fetch_oanda_price_and_volatility()
    macro_context_str, macro_summary = fetch_macro_context()

    evaluate_clusters(existing_data, h1_candles)

    now = datetime.now()
    cutoff_timestamp = (now - timedelta(hours=24)).timestamp()
    
    valid_clusters = [c for c in existing_data if c.get("_timestamp", 0) > cutoff_timestamp]
    
    existing_urls = set()
    for c in valid_clusters:
        for a in c.get("articles", []):
            existing_urls.add(a["link"])

    print("Suche parallel nach neuen Artikeln...")
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
                
                newest_article = max(target_cluster["articles"], key=lambda x: x["_timestamp"])
                
                if "original_date" not in target_cluster:
                    target_cluster["original_date"] = target_cluster.get("main_date", newest_article["date"])
                
                if newest_article["date"][:10] != target_cluster["original_date"][:10]:
                    target_cluster["main_date"] = f"{newest_article['date']} (updated)"
                else:
                    target_cluster["main_date"] = newest_article["date"]

                target_cluster["_timestamp"] = newest_article["_timestamp"]
                target_cluster["needs_update"] = True
                
                updated_text = " ".join([a["title"] for a in target_cluster["articles"]])
                cluster_embeddings[best_match_idx] = model.encode(updated_text)
            else:
                new_cluster = {
                    "main_date": article["date"],
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
    
    pending_clusters = [c for c in valid_clusters if c.get("needs_update", False)]
    if pending_clusters:
        orderbook_data = fetch_oanda_orderbook(top_zones_cache=oanda_zones_cache)
        
        for idx, cluster in enumerate(pending_clusters, 1):
            print(f"Erstelle analytische Zusammenfassung ({idx}/{len(pending_clusters)}): {cluster['articles'][0]['title'][:60]}...")
            summary = generate_summary(cluster["articles"], orderbook_data, price_history_data, macro_context_str)
            
            if summary:
                cluster["summary"] = summary
                cluster["needs_update"] = False
                time.sleep(4)

    evaluate_clusters(valid_clusters, h1_candles)
            
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_clusters, f, indent=4, ensure_ascii=False)
        
    print("Aktualisiere Chart- & Makro-Daten...")
    update_market_data(top_zones_cache=oanda_zones_cache, macro_summary=macro_summary, vol_metrics=vol_metrics)
    print("Update abgeschlossen.")

if __name__ == "__main__":
    main()
