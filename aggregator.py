import feedparser
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# 1. QUELLEN DEFINIEREN (Gratis RSS-Feeds mit starkem Rohstoff/Gold-Bezug)
FEEDS = {
    "Kitco": "https://www.kitco.com/news/rss",
    "CNBC Commodities": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000118",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Investing.com": "https://de.investing.com/rss/news_11.rss" # Deutsche Rohstoff-News (Beispiel)
}

# 2. KI-MODELL LADEN (Dieses Modell ist klein, schnell und gratis)
print("Lade KI-Modell für Textvergleich...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_news():
    print("Sammle Nachrichten von den Quellen...")
    articles = []
    
    for source, url in FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]: # Holen wir die neuesten 15 Artikel pro Quelle
                # Datum parsen (Falls vorhanden, ansonsten aktuelle Zeit)
                try:
                    pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed)).strftime("%Y-%m-%d %H:%M")
                except:
                    pub_date = datetime.now().strftime("%Y-%m-%d %H:%M")
                
                articles.append({
                    "title": entry.title,
                    "link": entry.link,
                    "source": source,
                    "date": pub_date
                })
        except Exception as e:
            print(f"Fehler beim Abrufen von {source}: {e}")
            
    print(f"Insgesamt {len(articles)} Artikel gesammelt.")
    return articles

def cluster_articles(articles, threshold=0.65):
    print("Dedupliziere und gruppiere Artikel...")
    if not articles:
        return []

    # Alle Titel in Vektoren (Zahlenreihen) umwandeln
    titles = [article["title"] for article in articles]
    embeddings = model.encode(titles)

    clusters = []
    
    for idx, article in enumerate(articles):
        added = False
        # Vergleiche den aktuellen Artikel mit bestehenden Clustern
        for cluster in clusters:
            # Wir vergleichen ihn einfach mit dem ersten (Haupt-)Artikel des Clusters
            main_embedding = cluster["main_embedding"]
            similarity = util.cos_sim(embeddings[idx], main_embedding).item()
            
            # Wenn die Ähnlichkeit über unserem Schwellenwert (z.B. 65%) liegt
            if similarity > threshold:
                # Prüfen, ob die Quelle nicht schon im Cluster ist (wir wollen keine zwei Reuters-Artikel im selben Cluster)
                sources_in_cluster = [a["source"] for a in cluster["articles"]]
                if article["source"] not in sources_in_cluster:
                    cluster["articles"].append(article)
                added = True
                break
        
        # Wenn er zu keinem bestehenden Cluster passt, mach ein neues auf
        if not added:
            clusters.append({
                "cluster_title": article["title"], # Der erste Artikel gibt dem Cluster den Namen
                "main_date": article["date"],
                "main_embedding": embeddings[idx], # Wird nur intern für den Vergleich gespeichert
                "articles": [article]
            })

    # Wir bereinigen das Ganze: Wir brauchen die Vektoren nicht in der JSON-Datei
    final_clusters = []
    for cluster in clusters:
        del cluster["main_embedding"]
        # Wir sortieren nach Clustern, die von mehr als einer Quelle gemeldet wurden (Je mehr Quellen, desto relevanter)
        final_clusters.append(cluster)

    final_clusters.sort(key=lambda x: len(x["articles"]), reverse=True)
    
    print(f"Es wurden {len(final_clusters)} einzigartige News-Cluster erstellt.")
    return final_clusters

if __name__ == "__main__":
    # Hauptprogramm
    raw_articles = fetch_news()
    clustered_news = cluster_articles(raw_articles)
    
    # Als JSON speichern
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(clustered_news, f, indent=4, ensure_ascii=False)
        
    print("Fertig! data.json wurde erstellt.")
