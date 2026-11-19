"""Data jobs run after the news update. A failure in one never stops the other
or the news update.

- US Treasury: 10Y real yield (TIPS) and 10Y breakeven inflation -> macro.json
- OKX: hourly aggressive (taker) buy and sell volume of the XAU/USDT perp -> taker_volume.json
  (OKX keeps 60 days; the file keeps building up history beyond that)
"""
import csv
import io
import json
import time
from datetime import datetime, timedelta, timezone

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


def run_all():
    for job in (update_macro, update_taker_volume):
        try:
            job()
        except Exception as ex:
            print(f"{job.__name__} failed: {ex}")
