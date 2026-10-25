import os
import json
import time
import requests
import feedparser
import re
from datetime import datetime, timedelta
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

FEEDS = {
    "CNBC Commodities": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000118",
    "Investing.com": "https://www.investing.com/rss/news_11.rss",
    "Yahoo Finance Gold": "https://finance.yahoo.com/rss/headline?s=GC=F",
    "Yahoo Finance Oil": "https://finance.yahoo.com/rss/headline?s=CL=F"
}

DATA_FILE = "data.json"
SIMILARITY_THRESHOLD = 0.35

def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)

def fetch_new_articles(existing_urls):
    """Holt neue Artikel und protokolliert unsichtbare Server-Blockaden."""
    articles = []
    now = datetime.now()
    cutoff_time = now - timedelta(hours=24)

    for source, url in FEEDS.items():
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if response.status_code != 200:
                print(f"Blockiert bei {source}: HTTP Status {response.status_code}")
                continue
                
            feed = feedparser.parse(response.content)
            if not feed.entries:
                print(f"Warnung bei {source}: Seite erreichbar (HTTP 200), aber Feed ist leer.")
                continue
            
            for entry in feed.entries[:20]:
                link = entry.link
                if link in existing_urls:
                    continue 
                
                pub_dt = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        pub_dt = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    except Exception:
                        pass
                
                if not pub_dt:
                    pub_dt = now

                if pub_dt < cutoff_time:
                    continue

                pub_date_str = pub_dt.strftime("%Y-%m-%d %H:%M")
                raw_desc = entry.get("summary", entry.get("description", ""))
                clean_desc = clean_html(raw_desc).strip()
                
                articles.append({
                    "title": entry.title,
                    "link": link,
                    "source": source,
                    "date": pub_date_str,
                    "description": clean_desc,
                    "_timestamp": pub_dt.timestamp()
                })
        except Exception as e:
            print(f"Technischer Fehler bei {source}: {e}")
            
    return articles

def fetch_oanda_orderbook():
    """Holt die 5 grössten Order-Cluster für XAU/USD von OANDA."""
    if not OANDA_API_KEY:
        return "Keine OANDA API-Daten verfügbar (Key fehlt)."
    url = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/orderBook"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            buckets = data.get("orderBook", {}).get("buckets", [])
            buckets.sort(key=lambda x: float(x.get("longCountPercent", 0)) + float(x.get("shortCountPercent", 0)), reverse=True)
            
            liquidity_info = "XAU/USD Retail Liquidity Zones (Top 5 Clusters):\n"
            for b in buckets[:5]:
                liquidity_info += f"- Price: {b.get('price')} | Buy Orders: {b.get('longCountPercent')}% | Sell Orders: {b.get('shortCountPercent')}%\n"
            return liquidity_info
        return f"OANDA API Fehler: {response.status_code}"
    except Exception as e:
        return f"OANDA Verbindungsfehler: {e}"

def generate_summary(articles_in_cluster, orderbook_data, max_retries=3):
    """Führt eine tiefgehende Analyse durch und erzwingt eine JSON-Ausgabe."""
    if not client:
        return None
    
    news_data = "\n".join([f"Source: [{a['source']}]\nHeadline: {a['title']}\nDetails: {a['description']}\n" for a in articles_in_cluster])
    
    prompt = f"""You are a quantitative financial analyst. Analyze these news reports and the current retail order book liquidity. Output ONLY a valid JSON object without markdown blocks. Write all values in English!
    
    Required JSON structure:
    {{
        "Facts": "Detailed summary of core facts and WHY it happened. Include [Source].",
        "Uncertainties": "What is unclear, speculative, or unconfirmed?",
        "Macro_Context": "Macro-economic context (e.g., monetary policy, oil prices).",
        "Impact_Timing": "Past, Present, or Future",
        "Timeframe": "Hours/Days, Days/Weeks, or Weeks/Months (Write 'N/A' if Impact_Timing is Past)",
        "XAUUSD_Impact": "Bullish, Bearish, or Neutral",
        "Probability_Percent": <Integer 0 to 100>,
        "Orderflow_Price_Target": "Based on the News Impact and the provided Liquidity Zones, state the exact logical price target for XAUUSD (e.g., '2750.50' or 'N/A') and one brief sentence explaining why it hits that liquidity cluster."
    }}

    News Reports:
    {news_data}
    
    Current Order Book:
    {orderbook_data}
    """
    
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=prompt
            )
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()
                
            return json.loads(raw_text)
        except Exception as e:
            print(f"Gemini API Versuch {attempt}/{max_retries} fehlgeschlagen: {e}")
            if attempt < max_retries:
                time.sleep(5 * attempt)
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
    
    # Filtere alte Cluster raus
    valid_clusters = [c for c in existing_data if c.get("_timestamp", 0) > cutoff_timestamp]
    
    existing_urls = set()
    for c in valid_clusters:
        for a in c.get("articles", []):
            existing_urls.add(a["link"])

    print("Suche nach neuen Artikeln...")
    new_articles = fetch_new_articles(existing_urls)
    
    if not new_articles:
        print("Keine neuen Artikel gefunden.")
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(valid_clusters, f, indent=4, ensure_ascii=False)
        return

    print(f"{len(new_articles)} neue Artikel gefunden. Clustere...")
    
    for article in new_articles:
        article_emb = model.encode(article["title"] + " " + article["description"])
        best_match_idx = -1
        highest_sim = -1
        
        for i, cluster in enumerate(valid_clusters):
            cluster_text = " ".join([a["title"] for a in cluster["articles"]])
            cluster_emb = model.encode(cluster_text)
            sim = util.cos_sim(article_emb, cluster_emb).item()
            if sim > highest_sim:
                highest_sim = sim
                best_match_idx = i
                
        if highest_sim > SIMILARITY_THRESHOLD:
            valid_clusters[best_match_idx]["articles"].append(article)
            valid_clusters[best_match_idx]["_timestamp"] = max(valid_clusters[best_match_idx].get("_timestamp", 0), article["_timestamp"])
            valid_clusters[best_match_idx]["needs_update"] = True
        else:
            valid_clusters.append({
                "main_date": article["date"],
                "_timestamp": article["_timestamp"],
                "articles": [article],
                "needs_update": True,
                "summary": None
            })
    
    # Sortiere chronologisch (neueste zuerst)
    valid_clusters.sort(key=lambda x: x.get("_timestamp", 0), reverse=True)
    
    for cluster in valid_clusters:
        if cluster.get("needs_update", False):
            print(f"Lade OANDA Orderflow-Daten für: {cluster['articles'][0]['title']}...")
            orderbook_data = fetch_oanda_orderbook()
            
            print(f"Erstelle analytische Zusammenfassung...")
            summary = generate_summary(cluster["articles"], orderbook_data)
            
            if summary:
                cluster["summary"] = summary
                cluster["needs_update"] = False
                print("Erfolgreich. Warte für Rate-Limits...")
                time.sleep(5)
            
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_clusters, f, indent=4, ensure_ascii=False)
    print("Update abgeschlossen.")

if __name__ == "__main__":
    main()
