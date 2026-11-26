# Rano

A static site (GitHub Pages) with three tabs:

- **Market**: live spot gold chart with OANDA + COMEX order zones (bands, lines and a zone list), an optional volume profile, small charts below it (buy vs sell pressure, COT, real yield, Hormuz), broker quotes, aggressor flow, and the TradingView Capital.com chart.
- **Macro**: cross-asset dashboard (metals, oil, dollar, US yields, inflation, equities), shipping through Hormuz / Bab el-Mandeb / Suez, and the CFTC Commitments of Traders report.
- **News**: gold-relevant news grouped into stories with short factual AI summaries. Search, filter, sort, and pin stories (pins are saved in your browser). Gold market analysis from analyst publishers appears between the news as clearly labelled cards, as published (publisher teaser, no AI summary).

No AI analysis, signals or predictions. Human analysts' commentary is shown as such, clearly labelled and never summarised.

## Using the chart

- **Layers** (button above the chart): choose what is drawn on the gold chart (order zones, volume profile, volume zones) and which small charts appear below it (buy vs sell pressure, hedge fund positioning, US real yield, Hormuz tankers). Quick choices: Price only / Key levels / Everything.
- **Settings**: zone size and price range, COMEX period for the zones, volume profile period and source.
- Choices are saved in your browser. On phones the menus open from the bottom and the tabs sit at the bottom of the screen.

## Data sources

| What | Source | Updated |
|---|---|---|
| Spot price & candles | OKX XAU/USD index (composite of spot quotes) | live, every 2 s, in the browser |
| Order zones | OANDA client order book (`oanda_book.json`, each workflow run) + COMEX gold futures hourly volume (`comex_volume.json`) | OANDA every 20 min, COMEX hourly |
| Volume profile | COMEX gold futures (GC=F) hourly volume via Yahoo Finance (`comex_volume.json`), or OKX XAU/USDT perp candles; both aligned to the spot index | COMEX: each workflow run; OKX: in the browser |
| Aggressor flow | OKX hourly taker buy/sell volume of the XAU/USDT perp (`taker_volume.json`) | hourly |
| Reference quotes | OANDA, Forex.com, Pepperstone, ICE, TVC via TradingView's scanner | every 5 s, in the browser |
| Cross-asset quotes | TradingView's scanner | every 30 s, in the browser |
| US 10Y real yield & breakeven | U.S. Treasury daily yield curves (`macro.json`) | daily |
| Shipping chokepoints | IMF PortWatch (satellite AIS) | weekly, a few days' delay, in the browser |
| COT report | CFTC Public Reporting API | weekly (Fridays), in the browser |
| News | Publisher RSS feeds and Google News searches (`data.json`) | each workflow run |
| Analysis & commentary | Kitco (pages), ING Think, FXEmpire and Investing.com (RSS), Google News for FXStreet, StoneX, Saxo, WGC and others (`analysis.json`) | each workflow run |

Anything fetched by the visitor's browser can be blocked by a provider for the visitor's country; the page then shows that source as unavailable.

## How the update works

`.github/workflows/update_news.yml` runs `aggregator.py` (started every 15 min by cron-job.org, or manually):

1. Fetch the feeds in `FEEDS`; keep articles matching `KEYWORDS`, drop SEO/forecast filler (`FILLER_RE`). Google News results are only accepted from `GOOGLE_NEWS_PUBLISHERS`.
2. Group articles about the same event (MiniLM sentence embeddings, `SIM_THRESHOLD`).
3. Gemini writes a factual headline, summary and key points for new or updated stories and marks off-topic ones (hidden by default on the site).
4. Jobs in `extras.py`: US Treasury real yields (daily) and OKX hourly taker buy/sell volume for the volume profile (`taker_volume.json`).
5. Commit the data files.

## Secrets (Settings → Secrets and variables → Actions)

| Secret | Required | What for |
|---|---|---|
| `GEMINI_API_KEY` | yes | news summaries |
| `OANDA_API_KEY` | yes | OANDA client order book (practice account is enough) |

## Run locally

```bash
pip install -r requirements.txt
GEMINI_API_KEY=... OANDA_API_KEY=... python aggregator.py
python -m http.server 8000   # open http://localhost:8000
```
