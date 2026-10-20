import os
import json
import time
from datetime import datetime
import requests
import feedparser
from sentence_transformers import SentenceTransformer, util
from google import genai

# 1. KI EINRICHTEN (NEUES SDK)
API_KEY = os.environ.get("GEMINI_API_KEY")
client = None
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print(f"Fehler bei KI-Initialisierung: {e}")

# 2. QUELLEN & TARNKAPPE (User-Agent)
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

def load_history():
    """Lädt die alte data.json, um eine Historie aufzubauen."""
    try:
        with open("data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def fetch_new_articles(existing_urls):
    """Holt nur neue Artikel, die noch nicht in der Historie sind."""
    articles = []
    for source, url in FEEDS.items():
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            feed = feedparser.parse(response.content)
            
            for entry in feed.entries[:15]:
                link = entry.link
                if link in existing_urls:
                    continue 
                
                try:
                    pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed)).strftime("%Y-%m-%d %H:%M")
                except:
                    pub_date = datetime.now().strftime("%Y-%m-%d %H:%M")
                
                articles.append({"title": entry.title, "link": link, "source": source, "date": pub_date})
        except Exception as e:
            print(f"Fehler bei {source}: {e}")
    return articles

def generate_summary(articles_in_cluster):
    """Generates an English summary using the new Google GenAI SDK."""
    if not client:
        return "No AI summary available (Missing API Key)."
    
    titles = [a["title"] for a in articles_in_cluster]
    prompt = f"You are a neutral financial analyst. Summarize these headlines into a single event in 2 to 3 objective, unbiased sentences in English. Avoid sensationalism. Headlines: {'; '.join(titles)}"
    
    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt
        )
        return response.text.replace('\n', ' ').strip()
    except Exception as e:
        print(f"Gemini API Fehler: {e}")
        return "Summary could not be generated."

def process_news():
    history = load_history()
    
    existing_urls = set()
    for cluster in history:
        for article in cluster["articles"]:
            existing_urls.add(article["link"])

    new_articles = fetch_new_articles(existing_urls)
    if not new_articles:
        print("Keine neuen Artikel gefunden.")
        return history

    print(f"{len(new_articles)} neue Artikel gefunden. Clustere...")
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
                "main_embedding": embeddings[idx],
                "articles": [article]
            })

    final_new_clusters = []
    for cluster in new_clusters:
        del cluster["main_embedding"]
        print(f"Erstelle KI-Zusammenfassung für: {cluster['cluster_title']}")
        cluster["summary"] = generate_summary(cluster["articles"])
        final_new_clusters.append(cluster)

    updated_history = final_new_clusters + history
    return updated_history[:100]

if __name__ == "__main__":
    final_data = process_news()
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
