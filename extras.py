"""Data jobs run after the news update. A failure in one never stops the other
or the news update.

- US Treasury: 10Y real yield (TIPS) and 10Y breakeven inflation -> macro.json
- OKX: hourly aggressive (taker) buy and sell volume of the XAU/USDT perp -> taker_volume.json
  (OKX keeps 60 days; the file keeps building up history beyond that)
- Yahoo Finance: hourly COMEX gold futures (GC=F) candles with volume, aligned to the spot
  index -> comex_volume.json, for the volume profile on the chart
- Google News: gold market analysis and commentary from analyst publishers -> analysis.json
  (listed as published: no clustering, no AI summary)
"""
import calendar
import csv
import html
import io
import json
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import feedparser

import requests

MACRO_FILE = "macro.json"
TAKER_FILE = "taker_volume.json"

TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                "daily-treasury-rates.csv/{year}/all?type={kind}&field_tdr_date_value={year}&page&_format=csv")
MACRO_REFRESH_HOURS = 6       # daily data; no need to fetch every 15 min
MACRO_HISTORY_DAYS = 400

TAKER_URL = "https://www.okx.com/api/v5/rubik/stat/taker-volume-contract"
TAKER_INST = "XAU-USDT-SWAP"
TAKER_KEEP_DAYS = 100         # enough for the 3-month volume profile
TAKER_MAX_PAGES = 16          # 100 hours per page; OKX serves at most ~60 days

COMEX_FILE = "comex_volume.json"
COMEX_URL = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
COMEX_KEEP_DAYS = 100         # enough for the 3-month volume profile
OKX_INDEX_URL = "https://www.okx.com/api/v5/market/history-index-candles"

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0"}


def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def data_age_hours(path):
    """Age from the file's own generated_utc field (git checkouts reset file timestamps)."""
    try:
        stamp = datetime.strptime(read_json(path, {})["generated_utc"], "%Y-%m-%d %H:%M")
        return (datetime.now(timezone.utc) - stamp.replace(tzinfo=timezone.utc)).total_seconds() / 3600
    except (KeyError, TypeError, ValueError):
        return float("inf")


# ---------------------------------------------------------------- US Treasury yields

def fetch_treasury_column(kind, column, years):
    """{iso_date: value} for one column of a Treasury daily rates CSV."""
    out = {}
    for year in years:
        res = requests.get(TREASURY_URL.format(year=year, kind=kind), headers=HTTP_HEADERS, timeout=40)
        res.raise_for_status()
        for row in csv.DictReader(io.StringIO(res.text)):
            value = (row.get(column) or "").strip()
            if not value:
                continue
            date = datetime.strptime(row["Date"], "%m/%d/%Y").date().isoformat()
            out[date] = float(value)
    return out


def update_macro():
    if read_json(MACRO_FILE, {}).get("series") and data_age_hours(MACRO_FILE) < MACRO_REFRESH_HOURS:
        return
    now = datetime.now(timezone.utc)
    years = sorted({now.year, (now - timedelta(days=MACRO_HISTORY_DAYS)).year})
    real = fetch_treasury_column("daily_treasury_real_yield_curve", "10 YR", years)
    nominal = fetch_treasury_column("daily_treasury_yield_curve", "10 Yr", years)
    start = (now - timedelta(days=MACRO_HISTORY_DAYS)).date().isoformat()
    dates = sorted(d for d in real if d >= start)
    series = {
        "REAL10": {"label": "US 10Y real yield (TIPS)", "unit": "%",
                   "points": [[d, real[d]] for d in dates]},
        # breakeven = nominal 10Y yield minus real 10Y yield, same day
        "BE10": {"label": "US 10Y breakeven inflation", "unit": "%",
                 "points": [[d, round(nominal[d] - real[d], 2)] for d in dates if d in nominal]},
    }
    if not series["REAL10"]["points"]:
        raise ValueError("no Treasury real yield data")
    write_json(MACRO_FILE, {"generated_utc": utc_now_str(), "source": "U.S. Department of the Treasury",
                            "series": series})
    last = series["REAL10"]["points"][-1]
    print(f"Treasury: 10Y real yield {last[1]}% on {last[0]}")


# ---------------------------------------------------------------- OKX taker buy/sell volume

def write_taker_file(rows):
    """One row per line so each hourly update is a small git diff."""
    lines = ",\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
    with open(TAKER_FILE, "w", encoding="utf-8") as f:
        f.write('{"inst":"%s","unit":"contracts","columns":["ts_ms","taker_sell","taker_buy"],"rows":[\n%s\n]}\n'
                % (TAKER_INST, lines))


def update_taker_volume():
    stored = {r[0]: r for r in read_json(TAKER_FILE, {}).get("rows", [])}
    newest_stored = max(stored) if stored else 0
    # only completed hours: the current hour is still changing
    current_hour = int(time.time() // 3600 * 3600 * 1000)
    fetched, end = {}, ""
    for _ in range(TAKER_MAX_PAGES):
        params = {"instId": TAKER_INST, "period": "1H", "limit": "100"}
        if end:
            params["end"] = end
        res = requests.get(TAKER_URL, params=params, headers=HTTP_HEADERS, timeout=20)
        res.raise_for_status()
        body = res.json()
        if body.get("code") != "0":
            raise ValueError(body.get("msg") or "OKX error")
        page = body.get("data") or []
        if not page:
            break
        for ts, sell, buy in page:
            ts = int(ts)
            if ts < current_hour:
                fetched[ts] = [ts, round(float(sell)), round(float(buy))]
        end = page[-1][0]
        if int(end) <= newest_stored:
            break  # reached hours we already have
        time.sleep(0.45)  # OKX allows 5 requests per 2 s on this endpoint
    merged = {**stored, **fetched}
    cutoff = (time.time() - TAKER_KEEP_DAYS * 86400) * 1000
    rows = sorted(r for ts, r in merged.items() if ts >= cutoff)
    if rows != sorted(stored.values()):
        write_taker_file(rows)
        print(f"Taker volume: {len(rows)} hours stored ({len(rows) - len(stored)} new)")


# ---------------------------------------------------------------- COMEX gold futures volume

def fetch_index_closes(since_ms):
    """{hour_ms: close} of the OKX XAU/USD spot index for completed hours since since_ms."""
    closes, after = {}, ""
    for _ in range(30):  # 100 hours per page, ~3 months at most
        params = {"instId": "XAU-USD", "bar": "1H", "limit": "100"}
        if after:
            params["after"] = after
        res = requests.get(OKX_INDEX_URL, params=params, headers=HTTP_HEADERS, timeout=20)
        res.raise_for_status()
        page = res.json().get("data") or []
        if not page:
            break
        for r in page:
            closes[int(r[0])] = float(r[4])
        after = page[-1][0]
        if int(after) <= since_ms:
            break
        time.sleep(0.12)
    return closes


def write_comex_file(rows):
    """One row per line so each update is a small git diff."""
    lines = ",\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
    with open(COMEX_FILE, "w", encoding="utf-8") as f:
        f.write('{"symbol":"GC=F","source":"Yahoo Finance","unit":"contracts (100 oz)",'
                '"note":"low/high shifted by each hour\'s futures-minus-spot gap",'
                '"columns":["ts_ms","low_spot","high_spot","volume"],"rows":[\n%s\n]}\n' % lines)


def update_comex():
    stored = {r[0]: r for r in read_json(COMEX_FILE, {}).get("rows", [])}
    newest = max(stored) if stored else 0
    # small request once history exists; 3 months on the first run
    rng = "5d" if newest and time.time() * 1000 - newest < 4 * 86400000 else "3mo"
    res = requests.get(COMEX_URL, params={"range": rng, "interval": "1h"}, headers=HTTP_HEADERS, timeout=30)
    res.raise_for_status()
    result = res.json()["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    now_ms = time.time() * 1000
    candles = []
    for ts, high, low, close, vol in zip(result["timestamp"], quote["high"], quote["low"],
                                         quote["close"], quote["volume"]):
        ts_ms = ts * 1000
        # completed hourly candles only, with volume, and not already stored
        if ts % 3600 or ts_ms + 3600000 > now_ms or not vol or None in (high, low, close) or ts_ms in stored:
            continue
        candles.append((ts_ms, high, low, close, vol))
    if not candles:
        return
    index = fetch_index_closes(min(c[0] for c in candles))
    for ts_ms, high, low, close, vol in candles:
        spot = index.get(ts_ms)
        if spot is None:
            continue
        basis = close - spot  # futures premium over spot at that hour
        stored[ts_ms] = [ts_ms, round(low - basis, 2), round(high - basis, 2), int(vol)]
    cutoff = now_ms - COMEX_KEEP_DAYS * 86400000
    rows = sorted(r for ts, r in stored.items() if ts >= cutoff)
    write_comex_file(rows)
    print(f"COMEX volume: {len(rows)} hours stored, latest {datetime.fromtimestamp(rows[-1][0] / 1000, timezone.utc):%Y-%m-%d %H:%M} UTC")


# ---------------------------------------------------------------- analysis & commentary

ANALYSIS_FILE = "analysis.json"
ANALYSIS_KEEP_DAYS = 7
ANALYSIS_QUERIES = [
    "gold (site:kitco.com OR site:investing.com)",
    "gold (site:fxstreet.com OR site:fxempire.com)",
    "(gold OR commodities OR metals) (site:think.ing.com OR site:home.saxo OR site:gold.org OR site:stonex.com)",
    'gold (outlook OR forecast OR analysis OR "week ahead" OR weekly)',
]
# publisher names as Google News reports them -> name shown on the site
ANALYSIS_PUBLISHERS = {
    "kitco": "Kitco", "investing.com": "Investing.com", "fxstreet": "FXStreet", "fxempire": "FXEmpire",
    "ing think": "ING Think", "saxo": "Saxo", "world gold council": "World Gold Council",
    "stonex": "StoneX", "investinglive": "investingLive", "reuters": "Reuters", "bloomberg": "Bloomberg",
    "cnbc": "CNBC", "marketwatch": "MarketWatch", "barron's": "Barron's", "financial times": "Financial Times",
    "wsj": "WSJ", "the wall street journal": "WSJ", "mining.com": "Mining.com",
}
ANALYSIS_TOPIC_RE = re.compile(r"\b(gold|bullion|xau|precious metals?|metals?|commodit\w*)\b", re.IGNORECASE)
# landing pages, ads and SEO price pages are not analysis
ANALYSIS_JUNK_RE = re.compile(r"trade gold on|news, analysis & commentary|methodology|price today|prices today"
                              r"|current price of|live price|price chart|how to buy|best way to buy"
                              r"|^form \d|\|.*\|"                                # filings, navigation pages
                              r"|\b(ftse|tsx|asx|nikkei|dax|stoxx|sensex|nifty)\b",  # equity market wraps
                              re.IGNORECASE)


def analysis_publisher(name):
    key = re.sub(r"\s+(uk|india|south africa|nigeria|australia|canada)$", "", name.strip().lower())
    return ANALYSIS_PUBLISHERS.get(key)


def update_analysis():
    # re-apply the junk filter to stored items too, so filter changes take effect immediately
    stored = {i["key"]: i for i in read_json(ANALYSIS_FILE, {}).get("items", [])
              if not ANALYSIS_JUNK_RE.search(i["title"])}
    found = 0
    for q in ANALYSIS_QUERIES:
        url = ("https://news.google.com/rss/search?q=" + quote_plus(f"{q} when:2d")
               + "&hl=en-US&gl=US&ceid=US:en")
        res = requests.get(url, headers=HTTP_HEADERS, timeout=20)
        if res.status_code == 429:
            time.sleep(8)
            res = requests.get(url, headers=HTTP_HEADERS, timeout=20)
        res.raise_for_status()
        for e in feedparser.parse(res.content).entries:
            raw_source = (e.get("source") or {}).get("title") or ""
            source = analysis_publisher(raw_source)
            title = html.unescape(e.get("title", "")).strip()
            if title.endswith(f" - {raw_source}"):
                title = title[: -len(raw_source) - 3].strip()
            if not source or not title or ANALYSIS_JUNK_RE.search(title) or not ANALYSIS_TOPIC_RE.search(title):
                continue
            parsed = e.get("published_parsed")
            ts = min(calendar.timegm(parsed), time.time()) if parsed else time.time()
            key = re.sub(r"[^a-z0-9]", "", title.lower())[:120]  # same story via several queries
            if key not in stored:
                stored[key] = {"key": key, "title": title, "source": source, "link": e.get("link", ""), "ts": int(ts)}
                found += 1
        time.sleep(1.5)  # Google News rate-limits bursts
    cutoff = time.time() - ANALYSIS_KEEP_DAYS * 86400
    items = sorted((i for i in stored.values() if i["ts"] >= cutoff), key=lambda i: -i["ts"])
    write_json(ANALYSIS_FILE, {"generated_utc": utc_now_str(), "items": items})
    print(f"Analysis: {len(items)} articles in the last {ANALYSIS_KEEP_DAYS} days ({found} new)")


def run_all():
    for job in (update_macro, update_taker_volume, update_comex, update_analysis):
        try:
            job()
        except Exception as ex:
            print(f"{job.__name__} failed: {ex}")
