# Gold Terminal

A static site (GitHub Pages) with three tabs:

- **Market**: live spot gold chart with the combined order books of several venues drawn on it, the largest levels up to ±$200 from spot, a depth ladder, broker reference quotes, OANDA's client order book, and the TradingView Capital.com chart.
- **Macro**: cross-asset dashboard (metals, oil, dollar, US yields, inflation, equities), Polymarket odds for Fed and geopolitical events, shipping through Hormuz / Bab el-Mandeb / Suez, an earnings calendar, and the CFTC Commitments of Traders report.
- **News**: gold-relevant news grouped into stories with short factual AI summaries. Search, filter, sort, and pin stories (pins are saved in your browser).

No analysis, signals or predictions anywhere.

## Data sources

| What | Source | Updated |
|---|---|---|
| Spot price & candles | OKX XAU/USD index (composite of spot quotes) | live, every 2 s, in the browser |
| Order books | OKX XAU/USDT perp, OKX PAXG/XAUT, Kraken PAXG/XAUT, Binance PAXG, Bybit XAUT | live, every 6 s, in the browser |
| Reference quotes | OANDA, Forex.com, Pepperstone, ICE, TVC via TradingView's scanner | every 5 s, in the browser |
| OANDA client orders | OANDA v20 API (`oanda_book.json`) | each workflow run |
| Cross-asset quotes | TradingView's scanner | every 30 s, in the browser |
| US 10Y real yield & breakeven | FRED (`macro.json`), needs `FRED_API_KEY` | daily |
| Prediction markets | Polymarket Gamma API | every 3 min, in the browser |
| Shipping chokepoints | IMF PortWatch (satellite AIS) | weekly, a few days' delay, in the browser |
| Earnings calendar | Alpha Vantage (`earnings.json`), needs `ALPHAVANTAGE_API_KEY` | daily |
| COT report | CFTC Public Reporting API | weekly (Fridays), in the browser |
| News | Publisher RSS feeds and Google News searches (`data.json`) | each workflow run |

Anything fetched by the visitor's browser can be blocked by a provider for the visitor's country; the page then shows that source as unavailable.

## How the update works

`.github/workflows/update_news.yml` runs `aggregator.py` (started every 15 min by cron-job.org, or manually):

1. Fetch the feeds in `FEEDS`; keep articles matching `KEYWORDS`, drop SEO/forecast filler (`FILLER_RE`). Google News results are only accepted from `GOOGLE_NEWS_PUBLISHERS`.
2. Group articles about the same event (MiniLM sentence embeddings, `SIM_THRESHOLD`).
3. Gemini writes a factual headline, summary and key points for new or updated stories and marks off-topic ones (hidden by default on the site).
4. Optional jobs in `extras.py` (each skipped when its secret is missing): FRED series, earnings calendar, push alerts.
5. Commit the data files.

## Secrets (Settings → Secrets and variables → Actions)

| Secret | Required | What for |
|---|---|---|
| `GEMINI_API_KEY` | yes | news summaries |
| `OANDA_API_KEY` | yes | OANDA client order book (practice account is enough) |
| `FRED_API_KEY` | optional | real yield and breakeven rows; free key at fred.stlouisfed.org (My Account → API Keys) |
| `ALPHAVANTAGE_API_KEY` | optional | earnings calendar; free key at alphavantage.co/support/#api-key |
| `NTFY_TOPIC` | optional | push alerts, see below |

## Push alerts

Alerts are sent with [ntfy](https://ntfy.sh) (free app for iOS/Android, no account needed):

1. Pick a long random topic name, e.g. `gold-terminal-7f3k9q2x` (anyone who knows the name can read it, so don't use a guessable one).
2. In the ntfy app, subscribe to that topic.
3. Add the same name as the `NTFY_TOPIC` secret.
4. Edit `alerts.json` in the repo:
   - `price_above` / `price_below`: gold price levels. Each fires once, and again only after price has moved back by `rearm_gap_usd`.
   - `news_min_sources`: notify when a story reaches this many sources.
   - `news_topics`: limit news alerts to these topics (as shown in the News tab); empty = all on-topic stories.
   - `site_url`: link opened when you tap a notification.

Checks run with the workflow (every 15 min). Price alerts use the high/low since the previous check, so a touch between runs is not missed. On the very first run, existing stories are only recorded, not sent.

## Run locally

```bash
pip install -r requirements.txt
GEMINI_API_KEY=... OANDA_API_KEY=... python aggregator.py
python -m http.server 8000   # open http://localhost:8000
```
