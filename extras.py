"""Daily data jobs run after the news update. A failure in one never stops the other
or the news update.

- US Treasury (no key): 10Y real yield (TIPS) and 10Y breakeven inflation -> macro.json
- Alpha Vantage (ALPHAVANTAGE_API_KEY): upcoming earnings for a watchlist -> earnings.json
"""
import csv
import io
import json
import os
from datetime import datetime, timedelta, timezone

import requests

MACRO_FILE = "macro.json"
EARNINGS_FILE = "earnings.json"

TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                "daily-treasury-rates.csv/{year}/all?type={kind}&field_tdr_date_value={year}&page&_format=csv")
MACRO_REFRESH_HOURS = 6       # daily data; no need to fetch every 15 min
MACRO_HISTORY_DAYS = 400

EARNINGS_REFRESH_HOURS = 20   # free Alpha Vantage tier allows 25 calls/day; one is enough
EARNINGS_HORIZON_DAYS = 90
EARNINGS_WATCHLIST = {
    "NVDA": "Nvidia", "MSFT": "Microsoft", "AAPL": "Apple", "GOOGL": "Alphabet", "AMZN": "Amazon",
    "META": "Meta", "TSLA": "Tesla", "AVGO": "Broadcom", "TSM": "TSMC", "AMD": "AMD", "ORCL": "Oracle",
    "JPM": "JPMorgan", "NEM": "Newmont", "AEM": "Agnico Eagle",
}

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
    res = requests.get("https://www.alphavantage.co/query", params={
        "function": "EARNINGS_CALENDAR", "horizon": "3month", "apikey": key,
    }, timeout=30)
    res.raise_for_status()
    if not res.text.startswith("symbol,"):
        raise ValueError(res.text[:120])  # rate limit or key message comes back as JSON/text
    rows = parse_earnings_csv(res.text)
    write_json(EARNINGS_FILE, {"generated_utc": utc_now_str(), "source": "Alpha Vantage", "rows": rows})
    print(f"Earnings: {len(rows)} upcoming reports in watchlist")


def run_all():
    for job in (update_macro, update_earnings):
        try:
            job()
        except Exception as ex:
            print(f"{job.__name__} failed: {ex}")
