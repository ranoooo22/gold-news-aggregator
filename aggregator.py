def update_market_data():
    if not OANDA_API_KEY:
        return
    
    market_data = {"candles": [], "orderbook": []}
    
    # WICHTIG: UNIX-Format erzwingen, damit das Chart-Frontend die Zeit fehlerfrei liest
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Accept-Datetime-Format": "UNIX"
    }
    
    try:
        url_candles = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/candles?count=100&granularity=H4&price=M"
        res_c = requests.get(url_candles, headers=headers, timeout=10)
        if res_c.status_code == 200:
            for c in res_c.json().get("candles", []):
                if c.get("complete"):
                    market_data["candles"].append({
                        "time": int(float(c["time"])),  # Direkte und fehlerfreie UNIX-Übernahme
                        "open": float(c["mid"]["o"]),
                        "high": float(c["mid"]["h"]),
                        "low": float(c["mid"]["l"]),
                        "close": float(c["mid"]["c"])
                    })
        else:
            print(f"API Fehler (Candles): {res_c.status_code} - {res_c.text}")
    except Exception as e:
        print(f"Fehler beim Laden der H4-Kerzen: {e}")

    try:
        url_ob = "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/orderBook"
        res_ob = requests.get(url_ob, headers=headers, timeout=10)
        if res_ob.status_code == 200:
            orderbook = res_ob.json().get("orderBook", {})
            current_price = float(orderbook.get("price", "0"))
            buckets = orderbook.get("buckets", [])
            valid_buckets = []
            
            for b in buckets:
                try:
                    p = float(b.get("price", 0))
                    if current_price > 0 and abs(p - current_price) / current_price <= 0.03:
                        valid_buckets.append(b)
                except ValueError:
                    continue
            
            valid_buckets.sort(key=lambda x: float(x.get("longCountPercent", 0)) + float(x.get("shortCountPercent", 0)), reverse=True)
            
            for b in valid_buckets[:5]:
                market_data["orderbook"].append({
                    "price": float(b.get("price")),
                    "volume": float(b.get("longCountPercent", 0)) + float(b.get("shortCountPercent", 0))
                })
        else:
            print(f"API Fehler (Orderbuch): {res_ob.status_code} - {res_ob.text}")
    except Exception as e:
        print(f"Fehler beim Laden der Orderbuch-Level: {e}")

    with open("market_data.json", "w", encoding="utf-8") as f:
        json.dump(market_data, f)
    print(f"market_data.json aktualisiert. (Kerzen: {len(market_data['candles'])})")
