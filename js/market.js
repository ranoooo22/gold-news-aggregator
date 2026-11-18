'use strict';

/*
 * Market tab: everything here is fetched live by the browser from public APIs.
 *  - Spot price & candles: OKX XAU/USD index (composite of spot gold quotes).
 *  - Order books: public books of gold perps and gold-backed tokens on several exchanges.
 *  - Reference quotes: broker prices via TradingView's public scanner.
 *  - OANDA client orders: snapshot saved by the GitHub workflow (oanda_book.json).
 */

const OKX = 'https://www.okx.com/api/v5';
const SPOT_INST = 'XAU-USD';

const TIMEFRAMES = { '1m': '1m', '5m': '5m', '15m': '15m', '1H': '1H', '4H': '4H', '1D': '1Dutc' };
const TF_SECONDS = { '1m': 60, '5m': 300, '15m': 900, '1H': 3600, '4H': 14400, '1D': 86400 };

const SPOT_POLL_MS = 2000;
const BOOK_POLL_MS = 6000;      // deep books are ~80 KB each; with smoothing faster polling adds nothing
const REF_POLL_MS = 5000;
const VENUE_RETRY_MS = 30000;   // wait this long after a venue errors
const MAX_VENUE_DEVIATION = 0.03; // ignore a venue whose mid is >3% away from spot (bad data)

function okxBook(json, ozPerUnit) {
    const book = json.data && json.data[0];
    if (!book) throw new Error(json.msg || 'no data');
    const conv = (rows) => rows.map((r) => [Number(r[0]), Number(r[1]) * ozPerUnit]);
    return { bids: conv(book.bids), asks: conv(book.asks) };
}

function krakenBook(json) {
    if (json.error && json.error.length) throw new Error(json.error[0]);
    const book = Object.values(json.result || {})[0];
    if (!book) throw new Error('no data');
    const conv = (rows) => rows.map((r) => [Number(r[0]), Number(r[1])]);
    return { bids: conv(book.bids), asks: conv(book.asks) };
}

// Sizes are converted to troy ounces. PAXG and XAUT are each backed by 1 troy oz.
const VENUES = [
    {
        id: 'okx-perp', exchange: 'OKX', instrument: 'XAU/USDT perpetual', color: '#f5a524',
        url: `${OKX}/market/books-full?instId=XAU-USDT-SWAP&sz=2000`,
        parse: (j) => okxBook(j, okxContractOz),
    },
    {
        id: 'okx-paxg', exchange: 'OKX', instrument: 'PAXG/USDT', color: '#ffd166',
        url: `${OKX}/market/books-full?instId=PAXG-USDT&sz=2000`, parse: (j) => okxBook(j, 1),
    },
    {
        id: 'okx-xaut', exchange: 'OKX', instrument: 'XAUT/USDT', color: '#c9a227',
        url: `${OKX}/market/books-full?instId=XAUT-USDT&sz=2000`, parse: (j) => okxBook(j, 1),
    },
    {
        id: 'kraken-paxg', exchange: 'Kraken', instrument: 'PAXG/USD', color: '#8b7cf6',
        url: 'https://api.kraken.com/0/public/Depth?pair=PAXGUSD&count=500', parse: krakenBook,
    },
    {
        id: 'kraken-xaut', exchange: 'Kraken', instrument: 'XAUT/USD', color: '#b4a9ff',
        url: 'https://api.kraken.com/0/public/Depth?pair=XAUTUSD&count=500', parse: krakenBook,
    },
    {
        id: 'binance-paxg', exchange: 'Binance', instrument: 'PAXG/USDT', color: '#2ec4b6',
        url: 'https://api.binance.com/api/v3/depth?symbol=PAXGUSDT&limit=1000',
        parse: (j) => ({
            bids: j.bids.map((r) => [Number(r[0]), Number(r[1])]),
            asks: j.asks.map((r) => [Number(r[0]), Number(r[1])]),
        }),
    },
    {
        id: 'bybit-xaut', exchange: 'Bybit', instrument: 'XAUT/USDT', color: '#4cc9f0',
        url: 'https://api.bybit.com/v5/market/orderbook?category=spot&symbol=XAUTUSDT&limit=200',
        parse: (j) => {
            if (j.retCode !== 0 || !j.result) throw new Error(j.retMsg || 'no data');
            return {
                bids: j.result.b.map((r) => [Number(r[0]), Number(r[1])]),
                asks: j.result.a.map((r) => [Number(r[0]), Number(r[1])]),
            };
        },
    },
];

const REFERENCE_TICKERS = [
    ['OANDA:XAUUSD', 'OANDA'],
    ['FOREXCOM:XAUUSD', 'Forex.com'],
    ['PEPPERSTONE:XAUUSD', 'Pepperstone'],
    ['FX_IDC:XAUUSD', 'ICE (FX_IDC)'],
    ['TVC:GOLD', 'TVC composite'],
];

let okxContractOz = 0.001; // replaced by the live contract spec on start

const settings = Object.assign({
    chartMode: 'spot',   // 'spot' | 'tradingview'
    tf: '15m',
    step: 'auto',        // $ bucket size for depth, or 'auto' (depends on the range)
    rangeUsd: 100,       // show depth within ±$ of spot
    align: true,         // shift each venue by its offset to spot
    walls: true,         // draw largest levels as lines
    smooth: 5,           // minutes of time-averaging for the depth (0 = live)
    levels: 'larger',    // 'all' | 'larger' | 'largest': which levels are drawn on the chart
    enabled: {},         // venue id -> false when switched off
}, store.get('gt_market_settings', {}));
// settings saved before the range became a $ amount: switch to the new defaults
if (settings.rangePct !== undefined) {
    settings.rangePct = undefined;
    settings.rangeUsd = 100;
    settings.step = 'auto';
    store.set('gt_market_settings', settings);
}

// Liquidity on these books is always densest right at spot. The "largest levels" table and
// the marked lines skip this zone (the ladder shows it) so that big orders further out surface.
const NEAR_ZONE_USD = 10;
const isFarLevel = (b) => Math.abs(b.price - state.spot) >= NEAR_ZONE_USD;

const AUTO_STEPS = [0.5, 1, 2, 5, 10];
// 'auto' picks the smallest bucket that keeps the ladder at roughly 80 rows or fewer
function effectiveStep() {
    if (settings.step !== 'auto') return Number(settings.step);
    const raw = (settings.rangeUsd * 2) / 80;
    return AUTO_STEPS.find((x) => x >= raw) || AUTO_STEPS[AUTO_STEPS.length - 1];
}

function saveSettings() { store.set('gt_market_settings', settings); }

const state = {
    spot: null, open24h: null, spotTs: 0,
    candles: [], venues: {}, buckets: [], maxOz: 0,
    smoothed: new Map(), lastAggregate: 0, threshold: 0, wallKeys: '',
    chart: null, series: null, wallLines: [], overlay: null,
    tvLoaded: false, oanda: null,
};
VENUES.forEach((v) => { state.venues[v.id] = { status: 'loading', nextTry: 0 }; });

// ---------------------------------------------------------------- spot & candles

async function loadContractSpec() {
    try {
        const j = await getJSON(`${OKX}/public/instruments?instType=SWAP&instId=XAU-USDT-SWAP`);
        const spec = j.data && j.data[0];
        if (spec && Number(spec.ctVal) > 0) okxContractOz = Number(spec.ctVal);
    } catch (e) { /* keep default */ }
}

async function fetchCandles(tf) {
    const bar = TIMEFRAMES[tf];
    let rows = (await getJSON(`${OKX}/market/index-candles?instId=${SPOT_INST}&bar=${bar}&limit=100`)).data || [];
    for (let i = 0; i < 2 && rows.length; i++) {
        const oldest = rows[rows.length - 1][0];
        const more = (await getJSON(`${OKX}/market/history-index-candles?instId=${SPOT_INST}&bar=${bar}&limit=100&after=${oldest}`)).data || [];
        if (!more.length) break;
        rows = rows.concat(more);
    }
    const byTime = new Map();
    rows.forEach((r) => byTime.set(Math.floor(Number(r[0]) / 1000), {
        time: Math.floor(Number(r[0]) / 1000), open: +r[1], high: +r[2], low: +r[3], close: +r[4],
    }));
    return [...byTime.values()].sort((a, b) => a.time - b.time);
}

async function loadCandles() {
    try {
        state.candles = await fetchCandles(settings.tf);
        if (state.series) state.series.setData(state.candles);
        $('chart-status').textContent = '';
    } catch (e) {
        $('chart-status').textContent = `Candles unavailable (${e.message})`;
    }
}

async function pollSpot() {
    try {
        const j = await getJSON(`${OKX}/market/index-tickers?instId=${SPOT_INST}`);
        const t = j.data && j.data[0];
        if (!t) return;
        state.spot = Number(t.idxPx);
        state.open24h = Number(t.open24h);
        state.spotTs = Number(t.ts) / 1000;
        renderSpotHeader();
        applySpotToCandles();
    } catch (e) {
        $('hdr-spot-time').textContent = `spot feed: ${e.message}`;
    }
}

function applySpotToCandles() {
    if (!state.series || !state.candles.length || !state.spot) return;
    const secs = TF_SECONDS[settings.tf];
    const barTime = Math.floor(state.spotTs / secs) * secs;
    const last = state.candles[state.candles.length - 1];
    if (barTime > last.time) {
        const bar = { time: barTime, open: state.spot, high: state.spot, low: state.spot, close: state.spot };
        state.candles.push(bar);
        state.series.update(bar);
    } else if (barTime === last.time) {
        last.close = state.spot;
        last.high = Math.max(last.high, state.spot);
        last.low = Math.min(last.low, state.spot);
        state.series.update(last);
    }
}

function renderSpotHeader() {
    $('hdr-spot').textContent = fmt(state.spot);
    const chg = state.spot - state.open24h;
    const pct = state.open24h ? (chg / state.open24h) * 100 : 0;
    const el = $('hdr-change');
    el.textContent = `${fmtSigned(chg)} (${fmtSigned(pct)}%) 24h`;
    el.className = 'hdr-change ' + (chg >= 0 ? 'up' : 'down');
    $('hdr-spot-time').textContent = `${utcTime(state.spotTs)}:${new Date(state.spotTs * 1000).toISOString().slice(17, 19)} UTC`;
}

// ---------------------------------------------------------------- order books

async function pollVenue(v) {
    const vs = state.venues[v.id];
    if (settings.enabled[v.id] === false) { vs.status = 'off'; return; }
    if (Date.now() < vs.nextTry) return;
    try {
        const book = v.parse(await getJSON(v.url, {}, 5000));
        book.bids.sort((a, b) => b[0] - a[0]);
        book.asks.sort((a, b) => a[0] - b[0]);
        if (!book.bids.length || !book.asks.length) throw new Error('empty book');
        const mid = (book.bids[0][0] + book.asks[0][0]) / 2;
        if (state.spot && Math.abs(mid - state.spot) / state.spot > MAX_VENUE_DEVIATION) {
            throw new Error(`price ${fmt(mid)} too far from spot`);
        }
        Object.assign(vs, { status: 'ok', error: null, bids: book.bids, asks: book.asks, mid, ts: Date.now() / 1000 });
    } catch (e) {
        const msg = e.message === 'Failed to fetch' ? 'blocked or offline' : e.message;
        Object.assign(vs, { status: 'error', error: msg, bids: null, asks: null, nextTry: Date.now() + VENUE_RETRY_MS });
    }
}

async function pollBooks() {
    await Promise.allSettled(VENUES.map(pollVenue));
    aggregateBooks();
    renderLargest();
    renderLadder();
    renderVenueTable();
    renderWalls();
}

// Builds the combined book from the latest snapshots, then time-averages it.
// Each price level (per venue, per side) is an exponential moving average with a time
// constant of `settings.smooth` minutes: an order has to rest for a few minutes to show
// at full size, and orders that flash in and out fade away instead of flickering.
function aggregateBooks() {
    const spot = state.spot;
    if (!spot) { state.buckets = []; state.maxOz = 0; return; }
    const lo = spot - settings.rangeUsd;
    const hi = spot + settings.rangeUsd;
    const step = effectiveStep();
    if (state.lastStep !== step) { resetDepthHistory(); state.lastStep = step; }

    const now = Date.now() / 1000;
    const dt = state.lastAggregate ? Math.min(now - state.lastAggregate, 60) : 0;
    state.lastAggregate = now;
    const tau = settings.smooth * 60;
    const alpha = tau > 0 && dt > 0 ? 1 - Math.exp(-dt / tau) : 1;
    // venue-to-spot gaps are always smoothed a little, otherwise aligned levels jump between buckets
    const offsetAlpha = dt > 0 ? 1 - Math.exp(-dt / Math.max(tau, 60)) : 1;

    const raw = new Map();
    VENUES.forEach((v) => {
        const vs = state.venues[v.id];
        if (vs.status !== 'ok') return;
        vs.offset = vs.mid - spot;
        vs.offsetSmooth = isFinite(vs.offsetSmooth) ? vs.offsetSmooth + offsetAlpha * (vs.offset - vs.offsetSmooth) : vs.offset;
        const shift = settings.align ? vs.offsetSmooth : 0;
        vs.bidOz = 0; vs.askOz = 0;
        // how far this venue's book reaches from spot (the shallower side counts)
        vs.reach = Math.min(spot - (vs.bids[vs.bids.length - 1][0] - shift), (vs.asks[vs.asks.length - 1][0] - shift) - spot);
        const add = (side, rows) => {
            for (const [px, oz] of rows) {
                const p = px - shift;
                if (p < lo || p > hi) continue;
                const key = Math.round(p / step) * step;
                let b = raw.get(key);
                if (!b) { b = { bidBy: {}, askBy: {} }; raw.set(key, b); }
                b[side + 'By'][v.id] = (b[side + 'By'][v.id] || 0) + oz;
                vs[side + 'Oz'] += oz;
            }
        };
        add('bid', vs.bids);
        add('ask', vs.asks);
    });

    const smoothed = state.smoothed;
    const seed = smoothed.size === 0;
    for (const key of new Set([...smoothed.keys(), ...raw.keys()])) {
        if (key < lo || key > hi) { smoothed.delete(key); continue; }
        const r = raw.get(key) || { bidBy: {}, askBy: {} };
        let s = smoothed.get(key);
        if (!s) { s = { price: key, bidBy: {}, askBy: {} }; smoothed.set(key, s); }
        for (const side of ['bidBy', 'askBy']) {
            for (const id of new Set([...Object.keys(s[side]), ...Object.keys(r[side])])) {
                const target = r[side][id] || 0;
                const next = seed ? target : (s[side][id] || 0) + alpha * (target - (s[side][id] || 0));
                if (next < 0.005) delete s[side][id]; else s[side][id] = next;
            }
        }
        s.bid = Object.values(s.bidBy).reduce((a, x) => a + x, 0);
        s.ask = Object.values(s.askBy).reduce((a, x) => a + x, 0);
        if (!s.bid && !s.ask) smoothed.delete(key);
    }

    state.buckets = [...smoothed.values()].sort((a, b) => b.price - a.price);
    state.maxOz = 0;
    const sizes = [];
    state.buckets.forEach((b) => {
        state.maxOz = Math.max(state.maxOz, b.bid, b.ask);
        if (b.bid) sizes.push(b.bid);
        if (b.ask) sizes.push(b.ask);
    });
    // "significant" = clearly larger than a typical level in the current range
    sizes.sort((a, b) => a - b);
    const median = sizes.length ? sizes[Math.floor(sizes.length / 2)] : 0;
    state.threshold = (LEVEL_FILTERS[settings.levels] || 0) * median;
}

const LEVEL_FILTERS = { all: 0, larger: 1.5, largest: 3 };

function isSignificant(size) {
    return size > 0 && size >= state.threshold;
}

function resetDepthHistory() {
    state.smoothed.clear();
    state.lastAggregate = 0;
}

function stackedBar(by, total, side) {
    if (!total) return '';
    const segs = VENUES.filter((v) => by[v.id]).map((v) => {
        const w = (by[v.id] / state.maxOz) * 100;
        return `<span style="width:${w.toFixed(2)}%;background:${v.color}"></span>`;
    }).join('');
    return `<div class="dom-bar ${side}">${segs}</div>`;
}

function venueBreakdown(by) {
    return VENUES.filter((v) => by[v.id]).map((v) => `${v.exchange} ${v.instrument}: ${fmt(by[v.id], 2)} oz`).join('\n');
}

let ladderCentered = false;

// The 5 biggest bid and ask levels in the range, with their distance from spot.
function renderLargest() {
    const el = $('largest-levels');
    if (!state.buckets.length) { el.innerHTML = ''; return; }
    const decimals = effectiveStep() < 1 ? 1 : 0;
    const top = (side) => state.buckets.filter((b) => b[side] > 0 && isFarLevel(b)).sort((a, b) => b[side] - a[side]).slice(0, 5);
    const row = (b, side) => `<tr class="${side}">
        <td>${side === 'bid' ? 'Bid' : 'Ask'}</td>
        <td class="num">${fmt(b.price, decimals)}</td>
        <td class="num"><b>${fmt(b[side], 0)} oz</b></td>
        <td class="num muted">${fmtSigned(b.price - state.spot, 0)}</td>
    </tr>`;
    const asks = top('ask').sort((a, b) => b.price - a.price).map((b) => row(b, 'ask'));
    const bids = top('bid').sort((a, b) => b.price - a.price).map((b) => row(b, 'bid'));
    el.innerHTML = `<table class="data-table largest">
        <thead><tr><th></th><th class="num">Price</th><th class="num">Size</th><th class="num">vs spot</th></tr></thead>
        <tbody>${asks.join('')}<tr class="spot-sep"><td colspan="4">Spot ${fmt(state.spot)}</td></tr>${bids.join('')}</tbody>
    </table>`;
    $('largest-title').textContent = `Largest levels $${NEAR_ZONE_USD}–$${settings.rangeUsd} from spot`;
}

function renderLadder() {
    const el = $('dom-ladder');
    if (!state.buckets.length) {
        el.innerHTML = '<div class="empty">No order book data yet.</div>';
        return;
    }
    const decimals = effectiveStep() < 1 ? 1 : 0;
    let spotInserted = false;
    const rows = [];
    let totalBid = 0, totalAsk = 0;
    for (const b of state.buckets) {
        totalBid += b.bid; totalAsk += b.ask;
        if (!spotInserted && b.price < state.spot) {
            rows.push(`<div class="dom-spot" id="dom-spot-row"><span>SPOT</span><span>${fmt(state.spot)}</span></div>`);
            spotInserted = true;
        }
        const tipBid = escapeHtml(venueBreakdown(b.bidBy));
        const tipAsk = escapeHtml(venueBreakdown(b.askBy));
        const faintBid = isSignificant(b.bid) ? '' : ' faint';
        const faintAsk = isSignificant(b.ask) ? '' : ' faint';
        rows.push(`<div class="dom-row">
            <div class="dom-side bid${faintBid}" title="${tipBid}">${stackedBar(b.bidBy, b.bid, 'bid')}<span class="dom-qty">${b.bid >= 0.05 ? fmt(b.bid, 1) : ''}</span></div>
            <div class="dom-px">${fmt(b.price, decimals)}</div>
            <div class="dom-side ask${faintAsk}" title="${tipAsk}">${stackedBar(b.askBy, b.ask, 'ask')}<span class="dom-qty">${b.ask >= 0.05 ? fmt(b.ask, 1) : ''}</span></div>
        </div>`);
    }
    if (!spotInserted) rows.push(`<div class="dom-spot" id="dom-spot-row"><span>SPOT</span><span>${fmt(state.spot)}</span></div>`);
    const prevScroll = el.scrollTop;
    el.innerHTML = rows.join('');
    $('dom-totals').innerHTML =
        `Bids <b class="up">${fmt(totalBid, 0)} oz</b> · Asks <b class="down">${fmt(totalAsk, 0)} oz</b> within ±$${settings.rangeUsd}`;
    if (!ladderCentered) {
        const spotRow = $('dom-spot-row');
        if (spotRow) el.scrollTop = spotRow.offsetTop - el.clientHeight / 2;
        ladderCentered = true;
    } else {
        el.scrollTop = prevScroll;
    }
}

function renderVenueTable() {
    const rows = VENUES.map((v) => {
        const vs = state.venues[v.id];
        const on = settings.enabled[v.id] !== false;
        let status;
        if (!on) status = '<span class="muted">off</span>';
        else if (vs.status === 'ok') status = '<span class="up">live</span>';
        else if (vs.status === 'loading') status = '<span class="muted">loading</span>';
        else status = `<span class="down" title="${escapeHtml(vs.error)}">${escapeHtml(vs.error)}</span>`;
        const ok = on && vs.status === 'ok';
        return `<tr>
            <td><label class="venue-toggle"><input type="checkbox" data-venue="${v.id}" ${on ? 'checked' : ''}>
                <span class="swatch" style="background:${v.color}"></span>${v.exchange}</label></td>
            <td>${v.instrument}</td>
            <td>${status}</td>
            <td class="num">${ok ? fmt(vs.mid) : '–'}</td>
            <td class="num">${ok ? fmtSigned(vs.offset) : '–'}</td>
            <td class="num">${ok ? fmt(vs.bidOz, 1) : '–'}</td>
            <td class="num">${ok ? fmt(vs.askOz, 1) : '–'}</td>
            <td class="num ${ok && vs.reach < settings.rangeUsd ? 'warn' : ''}" title="${ok && vs.reach < settings.rangeUsd ? 'This book does not reach the full range' : ''}">${ok ? '±$' + fmt(Math.min(vs.reach, 999), 0) : '–'}</td>
        </tr>`;
    }).join('');
    $('venue-table-body').innerHTML = rows;
}

// ---------------------------------------------------------------- chart

function initChart() {
    const container = $('spot-chart');
    state.chart = LightweightCharts.createChart(container, {
        autoSize: true,
        layout: { background: { type: 'solid', color: 'transparent' }, textColor: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' },
        grid: { vertLines: { color: '#161d2a' }, horzLines: { color: '#161d2a' } },
        timeScale: { timeVisible: true, secondsVisible: false, borderColor: '#21293a', rightOffset: 6 },
        rightPriceScale: { borderColor: '#21293a' },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        localization: { priceFormatter: (p) => fmt(p) },
    });
    state.series = state.chart.addCandlestickSeries({
        upColor: '#00c26e', downColor: '#ff4d4f', borderVisible: false,
        wickUpColor: '#00c26e', wickDownColor: '#ff4d4f',
        priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
        autoscaleInfoProvider: (original) => {
            const res = original();
            const prices = [...lastWallSet.bid, ...lastWallSet.ask];
            if (!res || !settings.walls || settings.chartMode !== 'spot' || !prices.length) return res;
            const pad = effectiveStep();
            return {
                priceRange: {
                    minValue: Math.min(res.priceRange.minValue, ...prices) - pad,
                    maxValue: Math.max(res.priceRange.maxValue, ...prices) + pad,
                },
                margins: res.margins,
            };
        },
    });
    state.overlay = $('depth-overlay');
    requestAnimationFrame(drawOverlayLoop);
}

let lastOverlayDraw = 0;
function drawOverlayLoop(t) {
    if (t - lastOverlayDraw > 150 && settings.chartMode === 'spot' && document.visibilityState === 'visible') {
        drawDepthOverlay();
        lastOverlayDraw = t;
    }
    requestAnimationFrame(drawOverlayLoop);
}

// Draws the aggregated book as horizontal bars on the right edge of the price chart.
function drawDepthOverlay() {
    const canvas = state.overlay;
    if (!canvas || !state.series) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
        canvas.width = Math.round(rect.width * dpr);
        canvas.height = Math.round(rect.height * dpr);
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!state.buckets.length || !state.maxOz) return;

    const paneWidth = state.chart.timeScale().width();
    const paneHeight = rect.height - state.chart.timeScale().height();
    const maxLen = Math.min(220, paneWidth * 0.28);
    const half = effectiveStep() / 2;
    for (const b of state.buckets) {
        const yTop = state.series.priceToCoordinate(b.price + half);
        const yBot = state.series.priceToCoordinate(b.price - half);
        if (yTop === null || yBot === null || yBot < 0 || yTop > paneHeight) continue;
        const h = Math.max(1, Math.abs(yBot - yTop) - 1);
        const y = Math.min(yTop, yBot);
        // bids and asks for the same bucket are drawn stacked (bid first, then ask)
        const bidLen = isSignificant(b.bid) ? (b.bid / state.maxOz) * maxLen : 0;
        const askLen = isSignificant(b.ask) ? (b.ask / state.maxOz) * maxLen : 0;
        if (bidLen) {
            ctx.fillStyle = 'rgba(0, 194, 110, 0.45)';
            ctx.fillRect(paneWidth - bidLen - askLen, y, bidLen, h);
        }
        if (askLen) {
            ctx.fillStyle = 'rgba(255, 77, 79, 0.45)';
            ctx.fillRect(paneWidth - askLen, y, askLen, h);
        }
    }
}

// Marks the 3 largest bid and ask levels. A marked level keeps its line while it stays
// among the 5 largest, so near-equal levels don't make the lines hop back and forth.
const WALL_COUNT = 3;
const WALL_KEEP_RANK = 5;
let lastWallSet = { bid: [], ask: [] };
let lastWallDraw = 0;

function pickWalls(side) {
    const ranked = state.buckets.filter((b) => isSignificant(b[side]) && isFarLevel(b)).sort((a, b) => b[side] - a[side]);
    const keepable = new Set(ranked.slice(0, WALL_KEEP_RANK).map((b) => b.price));
    const chosen = lastWallSet[side].filter((p) => keepable.has(p)).slice(0, WALL_COUNT);
    for (const b of ranked) {
        if (chosen.length >= WALL_COUNT) break;
        if (!chosen.includes(b.price)) chosen.push(b.price);
    }
    return chosen;
}

function renderWalls(force = false) {
    if (!state.series) return;
    const show = settings.walls && settings.chartMode === 'spot';
    const next = show ? { bid: pickWalls('bid'), ask: pickWalls('ask') } : { bid: [], ask: [] };
    const key = JSON.stringify(next);
    // redraw when the marked levels change, otherwise only every 30 s to refresh the oz labels
    if (!force && key === state.wallKeys && Date.now() - lastWallDraw < 30000) return;
    state.wallKeys = key;
    lastWallSet = next;
    lastWallDraw = Date.now();
    state.wallLines.forEach((l) => state.series.removePriceLine(l));
    state.wallLines = [];
    const byPrice = new Map(state.buckets.map((b) => [b.price, b]));
    for (const [side, color] of [['bid', '#00c26e'], ['ask', '#ff4d4f']]) {
        next[side].forEach((price) => {
            const b = byPrice.get(price);
            if (!b) return;
            state.wallLines.push(state.series.createPriceLine({
                price, color, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: true, title: `${side} ${fmt(b[side], 0)} oz`,
            }));
        });
    }
}

// ---------------------------------------------------------------- TradingView widget (Capital.com feed)

function loadTradingView() {
    if (state.tvLoaded) return;
    state.tvLoaded = true;
    const s = document.createElement('script');
    s.src = 'https://s3.tradingview.com/tv.js';
    s.onload = () => {
        new window.TradingView.widget({
            container_id: 'tv-widget', symbol: 'CAPITALCOM:GOLD', interval: '15',
            timezone: 'Etc/UTC', theme: 'dark', style: '1', locale: 'en', autosize: true,
            allow_symbol_change: true, hide_side_toolbar: false, backgroundColor: '#0d1117',
        });
    };
    s.onerror = () => { $('tv-widget').innerHTML = '<div class="empty">TradingView could not be loaded.</div>'; };
    document.head.appendChild(s);
}

// ---------------------------------------------------------------- reference quotes & OANDA

async function pollReference() {
    const el = $('ref-quotes');
    try {
        const j = await getJSON('https://scanner.tradingview.com/global/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'text/plain' },
            body: JSON.stringify({ symbols: { tickers: REFERENCE_TICKERS.map((t) => t[0]) }, columns: ['close'] }),
        });
        const prices = new Map((j.data || []).map((r) => [r.s, Number(r.d[0])]));
        el.innerHTML = REFERENCE_TICKERS.map(([sym, label]) => {
            const p = prices.get(sym);
            const diff = (p && state.spot) ? p - state.spot : null;
            return `<tr><td>${label}</td><td class="num">${p ? fmt(p) : '–'}</td>
                <td class="num muted">${diff !== null ? fmtSigned(diff) : '–'}</td></tr>`;
        }).join('');
    } catch (e) {
        el.innerHTML = `<tr><td colspan="3" class="muted">Unavailable (${escapeHtml(e.message)})</td></tr>`;
    }
}

async function loadOanda() {
    const el = $('oanda-book');
    try {
        const ob = await getJSON(`oanda_book.json?t=${Date.now()}`);
        const width = Math.max(ob.bucket_width, effectiveStep());
        const groups = new Map();
        ob.buckets.forEach((b) => {
            if (Math.abs(b.price - ob.price) > settings.rangeUsd) return;
            const key = Math.round(b.price / width) * width;
            const g = groups.get(key) || { price: key, buy: 0, sell: 0 };
            g.buy += b.long_pct; g.sell += b.short_pct;
            groups.set(key, g);
        });
        const rows = [...groups.values()].sort((a, b) => b.price - a.price);
        const max = Math.max(...rows.map((r) => Math.max(r.buy, r.sell)), 0.01);
        $('oanda-time').textContent = `snapshot ${escapeHtml(ob.time.slice(11, 16))} UTC · price then ${fmt(ob.price)}`;
        el.innerHTML = rows.map((r) => `<div class="dom-row small">
            <div class="dom-side bid"><div class="dom-bar bid"><span style="width:${(r.buy / max * 100).toFixed(1)}%;background:#00c26e"></span></div><span class="dom-qty">${fmt(r.buy, 2)}%</span></div>
            <div class="dom-px">${fmt(r.price, 0)}</div>
            <div class="dom-side ask"><div class="dom-bar ask"><span style="width:${(r.sell / max * 100).toFixed(1)}%;background:#ff4d4f"></span></div><span class="dom-qty">${fmt(r.sell, 2)}%</span></div>
        </div>`).join('');
    } catch (e) {
        el.innerHTML = '<div class="empty">No OANDA snapshot available.</div>';
    }
}

// ---------------------------------------------------------------- controls

function syncControls() {
    document.querySelectorAll('[data-chart-mode]').forEach((b) => b.classList.toggle('active', b.dataset.chartMode === settings.chartMode));
    document.querySelectorAll('[data-tf]').forEach((b) => b.classList.toggle('active', b.dataset.tf === settings.tf));
    $('opt-step').value = String(settings.step);
    $('opt-range').value = String(settings.rangeUsd);
    $('opt-align').checked = settings.align;
    $('opt-smooth').value = String(settings.smooth);
    $('opt-levels').value = settings.levels;
    $('opt-walls').checked = settings.walls;
    const spotMode = settings.chartMode === 'spot';
    $('spot-chart-wrap').hidden = !spotMode;
    $('tv-widget').hidden = spotMode;
    $('tf-group').hidden = !spotMode;
    $('chart-caption').innerHTML = spotMode
        ? `Candles: OKX XAU/USD spot index. Bars on the right: combined order books of the venues below (green bids, red asks)${settings.smooth ? `, averaged over ${settings.smooth} min` : ', live'}.`
        : 'TradingView chart, Capital.com GOLD feed. Order book levels cannot be drawn on this widget; see the ladder.';
}

function bindControls() {
    document.querySelectorAll('[data-chart-mode]').forEach((b) => b.addEventListener('click', () => {
        settings.chartMode = b.dataset.chartMode; saveSettings(); syncControls();
        if (settings.chartMode === 'tradingview') loadTradingView();
        renderWalls(true);
    }));
    document.querySelectorAll('[data-tf]').forEach((b) => b.addEventListener('click', () => {
        settings.tf = b.dataset.tf; saveSettings(); syncControls(); loadCandles();
    }));
    $('opt-step').addEventListener('change', (e) => { settings.step = e.target.value === 'auto' ? 'auto' : Number(e.target.value); saveSettings(); resetDepthHistory(); refreshDepthViews(); });
    $('opt-range').addEventListener('change', (e) => { settings.rangeUsd = Number(e.target.value); saveSettings(); ladderCentered = false; refreshDepthViews(); });
    $('opt-align').addEventListener('change', (e) => { settings.align = e.target.checked; saveSettings(); resetDepthHistory(); refreshDepthViews(); });
    $('opt-smooth').addEventListener('change', (e) => { settings.smooth = Number(e.target.value); saveSettings(); syncControls(); refreshDepthViews(); });
    $('opt-levels').addEventListener('change', (e) => { settings.levels = e.target.value; saveSettings(); refreshDepthViews(); });
    $('opt-walls').addEventListener('change', (e) => { settings.walls = e.target.checked; saveSettings(); renderWalls(true); });
    $('venue-table-body').addEventListener('change', (e) => {
        const id = e.target.dataset.venue;
        if (!id) return;
        settings.enabled[id] = e.target.checked;
        state.venues[id].nextTry = 0;
        saveSettings();
        pollBooks();
    });
    $('dom-recenter').addEventListener('click', () => { ladderCentered = false; renderLadder(); });
}

function refreshDepthViews() {
    aggregateBooks(); renderLargest(); renderLadder(); renderVenueTable(); renderWalls(true);
}

// ---------------------------------------------------------------- start

function every(ms, fn) {
    fn();
    setInterval(() => { if (document.visibilityState === 'visible') fn(); }, ms);
}

async function startMarket() {
    syncControls();
    bindControls();
    initChart();
    if (settings.chartMode === 'tradingview') loadTradingView();
    await loadContractSpec();
    await pollSpot();
    await loadCandles();
    every(SPOT_POLL_MS, pollSpot);
    every(BOOK_POLL_MS, pollBooks);
    every(REF_POLL_MS, pollReference);
    every(5 * 60 * 1000, loadOanda);
    setInterval(() => { if (document.visibilityState === 'visible') loadCandles(); }, 60 * 1000);
}
