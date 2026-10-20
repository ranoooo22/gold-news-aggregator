import os
import json
import time
import re
from datetime import datetime, timedelta
import requests
import feedparser
from sentence_transformers import SentenceTransformer, util
from google import genai

# 1. KI EINRICHTEN
API_KEY = os.environ.get("GEMINI_API_KEY")
client = None
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print(f"Fehler bei KI-Initialisierung: {e}")

# 2. QUELLEN & TARNKAPPE
FEEDS = {
    "Kitco": "https://www.kitco.com/news/rss",
    "CNBC Commodities": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000118",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Investing.com": "https://de.investing.com/rss/news_11.rss"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

print("Lade Embeddings-Modell...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_html(raw_html):
    """Entfernt HTML-Tags aus RSS-Beschreibungen."""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html) if raw_html else ""

def load_history():
    """Lädt data.json, behält nur valide Zusammenfassungen der letzten 24h."""
    now = datetime.now()
    cutoff_time = now - timedelta(hours=24)
    
    try:
        with open("data.json", "r", encoding="utf-8") as f:
            history = json.load(f)
            
            clean_history = []
            for cluster in history:
                summary = cluster.get("summary", "")
                if not summary or "could not be generated" in summary or "konnte nicht generiert werden" in summary:
                    continue
                
                try:
                    cluster_date = datetime.strptime(cluster.get("main_date", ""), "%Y-%m-%d %H:%M")
                    if cluster_date >= cutoff_time:
                        clean_history.append(cluster)
                except ValueError:
                    continue
            
            print(f"Historie geladen: {len(clean_history)} gültige Cluster aus den letzten 24h.")
            return clean_history
            
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def fetch_new_articles(existing_urls):
    """Holt neue Artikel inkl. Snippets, maximal 24 Stunden alt."""
    articles = []
    now = datetime.now()
    cutoff_time = now - timedelta(hours=24)

    for source, url in FEEDS.items():
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            feed = feedparser.parse(response.content)
            
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
                
                # Extrahiere detaillierten Text für die KI-Analyse
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
            print(f"Fehler bei {source}: {e}")
    return articles

def generate_summary(articles_in_cluster, max_retries=3):
    """Führt eine tiefgehende Analyse mit Quellenangaben und XAUUSD-Prognose durch."""
    if not client:
        return None
    
    # Bereite den Kontext aus verschiedenen Quellen vor
    news_data = []
    for a in articles_in_cluster:
        news_data.append(f"Source: [{a['source']}]\nHeadline: {a['title']}\nDetails: {a['description']}\n")
    
    context_text = "\n".join(news_data)
    
    prompt = f"""You are an expert financial analyst and commodities trader. Analyze these news reports about the same event.
    
    Task 1: Synthesize a single, objective summary (3-4 sentences). Compare the reports and combine the facts.
    Task 2: Use inline citations strictly matching the provided source names to indicate where specific facts came from (e.g., "The Fed paused rates [CNBC Commodities], causing dollar weakness [Kitco].").
    Task 3: Based ONLY on this specific news event, predict the immediate short-term impact on the XAUUSD (Gold) price.
    
    Format your response EXACTLY like this:
    [Your synthesized summary with inline citations here.]
    
    XAUUSD Outlook: [Bullish / Bearish / Neutral] - [1 sentence explaining why based on the text]

    News Reports to Analyze:
    {context_text}
    """
    
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=prompt
            )
            # HTML-Zeilenumbrüche einfügen, damit die Website die Absätze sauber trennt
            return response.text.strip().replace('\n', '<br>')
        except Exception as e:
            print(f"Gemini API Versuch {attempt}/{max_retries} fehlgeschlagen: {e}")
            if attempt < max_retries:
                time.sleep(5 * attempt)
            else:
                return None

def process_news():
    history = load_history()
    
    existing_urls = set()
    for cluster in history:
        for article in cluster.get("articles", []):
            existing_urls.add(article["link"])

    new_articles = fetch_new_articles(existing_urls)
    if not new_articles:
        print("Keine neuen tagesaktuellen Artikel gefunden.")
        return history

    print(f"{len(new_articles)} neue Artikel aus den letzten 24h gefunden. Clustere...")
    titles = [a["title"] for a in new_articles]
    embeddings = embed_model.encode(titles)
    
    new_clusters = []
    for idx, article in enumerate(new_articles):
        added = False
        for cluster in new_clusters:
            sim = util.cos_sim(embeddings[idx], cluster["main_embedding"]).item()
            if sim > 0.65:
                sources_in_cluster = [a["source"] for a in cluster["articles"]]
                if article["source"] not in sources_in_cluster:
                    cluster["articles"].append(article)
                added = True
                break
        if not added:
            new_clusters.append({
                "cluster_title": article["title"],
                "main_date": article["date"],
                "_timestamp": article["_timestamp"],
                "main_embedding": embeddings[idx],
                "articles": [article]
            })

    final_new_clusters = []
    # Verarbeite maximal 8 neue Cluster pro Durchlauf (schützt vor dem 5-Requests-Pro-Minute Limit)
    for cluster in new_clusters[:8]:
        del cluster["main_embedding"]
        print(f"Erstelle analytische Zusammenfassung für: {cluster['cluster_title']}")
        summary = generate_summary(cluster["articles"])
        
        if summary:
            cluster["summary"] = summary
            final_new_clusters.append(cluster)
            print("Erfolgreich. Warte 13s für Rate-Limits...")
            time.sleep(13) 
        else:
            print(f"Überspringe Cluster ohne Zusammenfassung: {cluster['cluster_title']}")

    all_clusters = final_new_clusters + history
    
    for c in all_clusters:
        if "_timestamp" not in c:
            try:
                c["_timestamp"] = datetime.strptime(c["main_date"], "%Y-%m-%d %H:%M").timestamp()
            except Exception:
                c["_timestamp"] = 0

    all_clusters.sort(key=lambda x: x["_timestamp"], reverse=True)

    for c in all_clusters:
        c.pop("_timestamp", None)
        for a in c.get("articles", []):
            a.pop("_timestamp", None)
            a.pop("description", None) # Halte die JSON-Datei schlank

    return all_clusters[:50]

if __name__ == "__main__":
    final_data = process_news()
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
