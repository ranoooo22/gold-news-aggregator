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
SIMILARITY_THRESHOLD = 0.35

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

def get_clustered_oanda_zones(zone_width=10.0):
    """Kombiniert OANDA OrderBook & PositionBook und bündelt sie in 10$-Liquiditätszonen."""
    if not OANDA_API_KEY:
        return 0.0, []
        
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    url_ob = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/orderBook"
    url_pb = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/positionBook"
    
    current_price = 0.0
    combined_buckets = {}
    
    try:
        res_ob = requests.get(url_ob, headers=headers, timeout=10)
        if res_ob.status_code == 200:
            ob_data = res_ob.json().get("orderBook", {})
            current_price = float(ob_data.get("price", 0))
            for b in ob_data.get("buckets", []):
                try:
                    p = float(b.get("price", 0))
                    if current_price > 0 and abs(p - current_price) / current_price <= 0.04:
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
                    if current_price > 0 and abs(p - current_price) / current_price <= 0.04:
                        if p not in combined_buckets:
                            combined_buckets[p] = {"long": 0.0, "short": 0.0}
                        combined_buckets[p]["long"] += float(b.get("longCountPercent", 0))
                        combined_buckets[p]["short"] += float(b.get("shortCountPercent", 0))
                except ValueError:
                    continue
    except Exception as e:
        print(f"Fehler bei OANDA positionBook: {e}")

    if not combined_buckets or current_price == 0:
        return 0.0, []

    zones = {}
    for p, vols in combined_buckets.items():
        zone_key = round(p / zone_width) * zone_width
        if zone_key not in zones:
            zones[zone_key] = {"long": 0.0, "short": 0.0, "total": 0.0, "weighted_price_sum": 0.0}
        
        tot = vols["long"] + vols["short"]
        zones[zone_key]["long"] += vols["long"]
        zones[zone_key]["short"] += vols["short"]
        zones[zone_key]["total"] += tot
        zones[zone_key]["weighted_price_sum"] += p * tot

    clustered_list = []
    for z_key, z_data in zones.items():
        if z_data["total"] > 0:
            vwap_price = round(z_data["weighted_price_sum"] / z_data["total"], 2)
            clustered_list.append({
                "price": vwap_price,
                "zone_center": z_key,
                "long": round(z_data["long"], 2),
                "short": round(z_data["short"], 2),
                "volume": round(z_data["total"], 2)
            })

    clustered_list.sort(key=lambda x: x["volume"], reverse=True)
    return current_price, clustered_list[:5]

def fetch_oanda_orderbook(top_zones_cache=None):
    if top_zones_cache is not None:
        current_price, top_zones = top_zones_cache
    else:
        current_price, top_zones = get_clustered_oanda_zones(zone_width=10.0)
        
    if not top_zones:
        return "Keine OANDA API-Daten verfügbar."
        
    liquidity_info = f"Current XAU/USD Price: {current_price}\nCombined Liquidity Zones ($10 Clustered OrderBook + PositionBook):\n"
    for z in top_zones:
        liquidity_info += f"- Zone around {z['price']} | Combined Volume: {z['volume']}% (Buy/Long: {z['long']}% | Sell/Short: {z['short']}%)\n"
    return liquidity_info

def fetch_oanda_price_history():
    if not OANDA_API_KEY:
        return "Keine Preisdaten verfügbar."
    url = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/candles?count=24&granularity=H1"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            candles = response.json().get("candles", [])
            history = "XAU/USD H1 Price History (Last 24h):\n"
            for c in candles:
                if c.get("complete"):
                    time_str = c.get("time")[:16].replace("T", " ")
                    close_price = c.get("mid", {}).get("c")
                    history += f"- {time_str} UTC: {close_price}\n"
            return history
        return "OANDA Candle API Fehler"
    except Exception as e:
        return f"OANDA Candle Fehler: {e}"

def update_market_data(top_zones_cache=None):
    if not OANDA_API_KEY:
        print("WARNUNG: Kein OANDA_API_KEY vorhanden, überspringe Chart-Update.")
        return
    
    market_data = {"candles": [], "orderbook": []}
    
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
            _, top_zones = top_zones_cache
        else:
            _, top_zones = get_clustered_oanda_zones(zone_width=10.0)
            
        for z in top_zones:
            market_data["orderbook"].append({
                "price": z["price"],
                "volume": z["volume"]
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

def generate_summary(articles_in_cluster, orderbook_data, price_history_data, max_retries=2):
    if not client:
        return None
    
    news_data = "\n".join([f"Source: [{a['source']}]\nDate: {a['date']}\nHeadline: {a['title']}\nDetails: {a['description']}\n" for a in articles_in_cluster])
    
    prompt = f"""You are a quantitative financial analyst. Analyze these news reports, the past 24h price action, and current liquidity. Output ONLY a valid JSON object without markdown blocks. Write all values in English!
    
    CRITICAL RULE FOR 'Orderflow_Price_Target':
    - Look at the 'Current XAU/USD Price' in the Order Book data below.
    - If 'XAUUSD_Impact' is 'Bullish', the 'Orderflow_Price_Target' (Take Profit) MUST be a numerical price strictly HIGHER than the Current XAU/USD Price. Never output a support level below current price for a Bullish signal!
    - If 'XAUUSD_Impact' is 'Bearish', the 'Orderflow_Price_Target' (Take Profit) MUST be a numerical price strictly LOWER than the Current XAU/USD Price. Never output a resistance level above current price for a Bearish signal!
    - If 'XAUUSD_Impact' is 'Neutral', write 'N/A'.
    - Format the target strictly as a clean number (e.g., '4035.50').

    Required JSON structure:
    {{
        "Facts": "Detailed summary of core facts and WHY it happened. Include [Source].",
        "Uncertainties": "What is unclear, speculative, or unconfirmed?",
        "Macro_Context": "Macro-economic context.",
        "Impact_Timing": "Past, Present, or Future",
        "Timeframe": "Hours/Days, Days/Weeks, or Weeks/Months (Write 'N/A' if Past)",
        "XAUUSD_Impact": "Bullish, Bearish, or Neutral",
        "Probability_Percent": <Integer 0 to 100>,
        "Past_Price_Action_Cause": "Explain briefly if and how these specific news events align with the provided 24h price action. Write 'N/A' if no clear correlation.",
        "Orderflow_Price_Target": "Exact numerical Take-Profit target following the CRITICAL RULE above."
    }}

    News Reports:
    {news_data}
    
    24h Price History:
    {price_history_data}

    Current Order Book:
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

    now = datetime.now()
    cutoff_timestamp = (now - timedelta(hours=24)).timestamp()
    
    valid_clusters = [c for c in existing_data if c.get("_timestamp", 0) > cutoff_timestamp]
    
    existing_urls = set()
    for c in valid_clusters:
        for a in c.get("articles", []):
            existing_urls.add(a["link"])

    print("Suche parallel nach neuen Artikeln...")
    new_articles = fetch_new_articles(existing_urls)
    
    print("Lade OANDA Order-Zonen & Preishistorie...")
    oanda_zones_cache = get_clustered_oanda_zones(zone_width=10.0)
    
    if new_articles:
        print(f"{len(new_articles)} neue Artikel gefunden. Clustere...")
        
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
                # Artikel zu bestehendem Cluster hinzufügen
                target_cluster = valid_clusters[best_match_idx]
                target_cluster["articles"].append(article)
                
                # Neuesten Artikel im Cluster bestimmen
                newest_article = max(target_cluster["articles"], key=lambda x: x["_timestamp"])
                
                # Ursprungsdatum einmalig festhalten (Fallback für alte Cluster ohne original_date)
                if "original_date" not in target_cluster:
                    target_cluster["original_date"] = target_cluster.get("main_date", newest_article["date"])
                
                # Anderer Tag als Ursprung -> (updated) markieren
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

    # Hauptsortierung: nach dem neuesten _timestamp absteigend
    valid_clusters.sort(key=lambda x: float(x.get("_timestamp", 0)), reverse=True)
    
    pending_clusters = [c for c in valid_clusters if c.get("needs_update", False)]
    if pending_clusters:
        price_history_data = fetch_oanda_price_history()
        orderbook_data = fetch_oanda_orderbook(top_zones_cache=oanda_zones_cache)
        
        for idx, cluster in enumerate(pending_clusters, 1):
            print(f"Erstelle analytische Zusammenfassung ({idx}/{len(pending_clusters)}): {cluster['articles'][0]['title'][:60]}...")
            summary = generate_summary(cluster["articles"], orderbook_data, price_history_data)
            
            if summary:
                cluster["summary"] = summary
                cluster["needs_update"] = False
                # 4 Sekunden Pause hält uns sicher unter dem Free-Tier-Limit von 15 Anfragen pro Minute (RPM)
                time.sleep(4)
            
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_clusters, f, indent=4, ensure_ascii=False)
        
    print("Aktualisiere Chart-Daten...")
    update_market_data(top_zones_cache=oanda_zones_cache)
    print("Update abgeschlossen.")

if __name__ == "__main__":
    main()
