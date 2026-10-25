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
