"""Optional data jobs run after the news update. Each one is skipped when its secret is
missing, and a failure in one never stops the others or the news update.

- FRED (FRED_API_KEY): US 10Y real yield (TIPS) and 10Y breakeven inflation -> macro.json
- Alpha Vantage (ALPHAVANTAGE_API_KEY): upcoming earnings for a watchlist -> earnings.json
- Alerts (NTFY_TOPIC): push notifications for gold price levels and big news stories,
  configured in alerts.json, state kept in alerts_state.json
"""
import csv
import io
import json
import os
import time
from datetime import datetime, timedelta, timezone

import requests

MACRO_FILE = "macro.json"
EARNINGS_FILE = "earnings.json"
ALERTS_CONFIG = "alerts.json"
ALERTS_STATE = "alerts_state.json"

FRED_SERIES = {
    "DFII10": {"label": "US 10Y real yield (TIPS)", "unit": "%"},
    "T10YIE": {"label": "US 10Y breakeven inflation", "unit": "%"},
}
FRED_REFRESH_HOURS = 6        # daily series; no need to fetch every 15 min
FRED_HISTORY_DAYS = 400

EARNINGS_REFRESH_HOURS = 20   # free Alpha Vantage tier allows 25 calls/day; one is enough
EARNINGS_HORIZON_DAYS = 90
EARNINGS_WATCHLIST = {
    "NVDA": "Nvidia", "MSFT": "Microsoft", "AAPL": "Apple", "GOOGL": "Alphabet", "AMZN": "Amazon",
    "META": "Meta", "TSLA": "Tesla", "AVGO": "Broadcom", "TSM": "TSMC", "AMD": "AMD", "ORCL": "Oracle",
    "JPM": "JPMorgan", "NEM": "Newmont", "AEM": "Agnico Eagle",
}

HTTP_TIMEOUT = 20


def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def data_age_hours(path):
    """Age from the file's own generated_utc field (git checkouts reset file timestamps)."""
    try:
        stamp = datetime.strptime(read_json(path, {})["generated_utc"], "%Y-%m-%d %H:%M")
        return (datetime.now(timezone.utc) - stamp.replace(tzinfo=timezone.utc)).total_seconds() / 3600
    except (KeyError, TypeError, ValueError):
        return float("inf")


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------- FRED

def update_fred():
    key = os.environ.get("FRED_API_KEY")
    if not key:
        print("FRED: no FRED_API_KEY, skipped")
        return
    existing = read_json(MACRO_FILE, {})
    if existing.get("fred") and data_age_hours(MACRO_FILE) < FRED_REFRESH_HOURS:
        return
    start = (datetime.now(timezone.utc) - timedelta(days=FRED_HISTORY_DAYS)).strftime("%Y-%m-%d")
    out = {}
    for series_id, meta in FRED_SERIES.items():
        try:
            res = requests.get("https://api.stlouisfed.org/fred/series/observations", params={
                "series_id": series_id, "api_key": key, "file_type": "json", "observation_start": start,
            }, timeout=HTTP_TIMEOUT)
            res.raise_for_status()
            points = [[o["date"], float(o["value"])] for o in res.json().get("observations", [])
                      if o.get("value") not in (None, "", ".")]
            if points:
                out[series_id] = {**meta, "points": points}
        except Exception as ex:
            print(f"FRED {series_id} failed: {type(ex).__name__}")
    if out:
        write_json(MACRO_FILE, {"generated_utc": utc_now_str(), "fred": out})
        latest = ", ".join("%s %s" % (k, v["points"][-1][1]) for k, v in out.items())
        print(f"FRED: {latest}")


# ---------------------------------------------------------------- earnings

def parse_earnings_csv(text, today=None):
    today = today or datetime.now(timezone.utc).date()
    horizon = today + timedelta(days=EARNINGS_HORIZON_DAYS)
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        symbol = (r.get("symbol") or "").strip()
        if symbol not in EARNINGS_WATCHLIST:
            continue
        try:
            date = datetime.strptime(r["reportDate"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        if not today <= date <= horizon:
            continue
        rows.append({
            "symbol": symbol,
            "name": EARNINGS_WATCHLIST[symbol],
            "date": date.isoformat(),
            "time": (r.get("timeOfTheDay") or "").strip(),   # pre-market / post-market / ""
            "estimate": float(r["estimate"]) if (r.get("estimate") or "").strip() else None,
            "currency": (r.get("currency") or "").strip(),
            "fiscal_period_end": (r.get("fiscalDateEnding") or "").strip(),
        })
    rows.sort(key=lambda x: (x["date"], x["symbol"]))
    return rows


def update_earnings():
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        print("Earnings: no ALPHAVANTAGE_API_KEY, skipped")
        return
    if data_age_hours(EARNINGS_FILE) < EARNINGS_REFRESH_HOURS:
        return
    try:
        res = requests.get("https://www.alphavantage.co/query", params={
            "function": "EARNINGS_CALENDAR", "horizon": "3month", "apikey": key,
        }, timeout=30)
        res.raise_for_status()
        if not res.text.startswith("symbol,"):
            raise ValueError(res.text[:120])  # rate limit or key message comes back as JSON/text
        rows = parse_earnings_csv(res.text)
        write_json(EARNINGS_FILE, {"generated_utc": utc_now_str(), "source": "Alpha Vantage", "rows": rows})
        print(f"Earnings: {len(rows)} upcoming reports in watchlist")
    except Exception as ex:
        print(f"Earnings failed: {ex}")


# ---------------------------------------------------------------- alerts

DEFAULT_ALERTS = {
    "price_above": [],
    "price_below": [],
    "rearm_gap_usd": 10,
    "news_min_sources": 6,
    "news_topics": [],
    "site_url": "",
    "ntfy_server": "https://ntfy.sh",
}
MAX_NEWS_ALERTS_PER_RUN = 3


def send_ntfy(server, topic, title, message, click="", tags=""):
    body = {"topic": topic, "title": title, "message": message}
    if click:
        body["click"] = click
    if tags:
        body["tags"] = [tags]
    res = requests.post(server.rstrip("/"), json=body, timeout=HTTP_TIMEOUT)
    res.raise_for_status()


def fetch_gold_range(since_ts):
    """High and low of the OKX XAU/USD spot index since `since_ts`, plus the latest close."""
    res = requests.get("https://www.okx.com/api/v5/market/index-candles",
                       params={"instId": "XAU-USD", "bar": "15m", "limit": "100"}, timeout=HTTP_TIMEOUT)
    res.raise_for_status()
    rows = sorted(res.json()["data"], key=lambda r: int(r[0]))
    # include the candle that was open at the last check
    recent = [r for r in rows if int(r[0]) / 1000 >= since_ts - 15 * 60] or rows[-1:]
    high = max(float(r[2]) for r in recent)
    low = min(float(r[3]) for r in recent)
    return high, low, float(rows[-1][4])


def check_price_levels(cfg, state, high, low, close, notify):
    levels = state.setdefault("levels", {})
    gap = float(cfg.get("rearm_gap_usd", 10))
    wanted = set()
    for side, values in (("above", cfg.get("price_above", [])), ("below", cfg.get("price_below", []))):
        for level in values:
            level = float(level)
            key = f"{side}:{level:g}"
            wanted.add(key)
            if key not in levels:
                # a new level is only armed once price is on the other side of it
                levels[key] = {"armed": close < level if side == "above" else close > level}
                continue
            entry = levels[key]
            if side == "above":
                if entry["armed"] and high >= level:
                    notify(f"Gold above ${level:,.0f}", f"Spot gold reached ${high:,.2f} (now ${close:,.2f}).", "chart_with_upwards_trend")
                    entry["armed"] = False
                elif not entry["armed"] and close < level - gap:
                    entry["armed"] = True
            else:
                if entry["armed"] and low <= level:
                    notify(f"Gold below ${level:,.0f}", f"Spot gold fell to ${low:,.2f} (now ${close:,.2f}).", "chart_with_downwards_trend")
                    entry["armed"] = False
                elif not entry["armed"] and close > level + gap:
                    entry["armed"] = True
    for key in list(levels):
        if key not in wanted:
            del levels[key]


def check_news(cfg, state, clusters, notify, first_run):
    notified = set(state.get("notified_clusters", []))
    topics = set(cfg.get("news_topics") or [])
    min_sources = int(cfg.get("news_min_sources", 6))
    sent = 0
    for c in sorted(clusters, key=lambda x: -len(x["articles"])):
        s = c.get("summary") or {}
        if c["id"] in notified or not s.get("on_topic") or len(c["articles"]) < min_sources:
            continue
        if topics and s.get("topic") not in topics:
            continue
        notified.add(c["id"])
        if first_run or sent >= MAX_NEWS_ALERTS_PER_RUN:
            continue  # first run: remember existing stories without flooding the phone
        notify(s.get("headline") or "Major news story",
               f"{s.get('summary', '')}\n\n{len(c['articles'])} sources · {s.get('topic', '')}", "newspaper")
        sent += 1
    live_ids = {c["id"] for c in clusters}
    state["notified_clusters"] = sorted(notified & live_ids)


def run_alerts(clusters):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print("Alerts: no NTFY_TOPIC, skipped")
        return
    cfg = {**DEFAULT_ALERTS, **read_json(ALERTS_CONFIG, {})}
    state = read_json(ALERTS_STATE, {})
    first_run = "last_check_ts" not in state
    now = time.time()
    sent = []

    def notify(title, message, tags=""):
        try:
            send_ntfy(cfg["ntfy_server"], topic, title, message, cfg.get("site_url", ""), tags)
            sent.append(title)
        except Exception as ex:
            print(f"Alerts: ntfy failed: {type(ex).__name__}")

    try:
        high, low, close = fetch_gold_range(state.get("last_check_ts", now))
        check_price_levels(cfg, state, high, low, close, notify)
    except Exception as ex:
        print(f"Alerts: price check failed: {type(ex).__name__}")
    check_news(cfg, state, clusters, notify, first_run)
    state["last_check_ts"] = now
    write_json(ALERTS_STATE, state)
    print(f"Alerts: {len(sent)} sent" + (f" ({'; '.join(sent)})" if sent else ""))


def run_all(clusters):
    for job in (update_fred, update_earnings):
        try:
            job()
        except Exception as ex:
            print(f"{job.__name__} failed: {ex}")
    try:
        run_alerts(clusters)
    except Exception as ex:
        print(f"run_alerts failed: {ex}")
