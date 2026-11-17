# Gold Terminal

A static site (GitHub Pages) with two tabs:

- **Market**: live spot gold chart with the combined order books of several venues drawn on it, a depth ladder, broker reference quotes, OANDA's client order book, and the TradingView Capital.com chart.
- **News**: gold-relevant news grouped into stories with short factual AI summaries. Search, filter, sort, and pin stories (pins are saved in your browser).

No analysis, signals or predictions anywhere.

## Data sources

| What | Source | Updated |
|---|---|---|
| Spot price & candles | OKX XAU/USD index (composite of spot quotes) | live, every 2 s, in the browser |
| Order books | OKX XAU/USDT perp, OKX PAXG/XAUT, Kraken PAXG/XAUT, Binance PAXG, Bybit XAUT | live, every 3 s, in the browser |
| Reference quotes | OANDA, Forex.com, Pepperstone, ICE, TVC via TradingView's scanner | every 5 s, in the browser |
| OANDA client orders | OANDA v20 API (`oanda_book.json`) | each workflow run |
| News | Publisher RSS feeds and Google News searches (`data.json`) | each workflow run |

Order books are fetched directly by the visitor's browser, so a venue that blocks the visitor's country shows as "blocked" in the venue table.

## How the news update works

`.github/workflows/update_news.yml` runs `aggregator.py` (started every 15 min by cron-job.org, or manually):

1. Fetch the feeds in `FEEDS`; keep articles matching `KEYWORDS`, drop SEO/forecast filler (`FILLER_RE`). Google News results are only accepted from `GOOGLE_NEWS_PUBLISHERS`.
2. Group articles about the same event (MiniLM sentence embeddings, `SIM_THRESHOLD`).
3. Gemini writes a factual headline, summary and key points for new or updated stories and marks off-topic ones (hidden by default on the site).
4. Commit `data.json` and `oanda_book.json`.

Secrets: `GEMINI_API_KEY`, `OANDA_API_KEY` (practice account is enough).

## Run locally

```bash
pip install -r requirements.txt
GEMINI_API_KEY=... OANDA_API_KEY=... python aggregator.py
python -m http.server 8000   # open http://localhost:8000
```
