'use strict';

/*
 * Market tab: live spot gold chart.
 *  - Spot price & candles: OKX XAU/USD index (composite of spot quotes), live in the browser.
 *  - Drawn on the chart (Layers menu): OANDA + COMEX order zones (zones.js), volume profile (volprofile.js).
 *  - Reference quotes: broker prices via TradingView's public scanner.
 *  - The TradingView widget shows the Capital.com GOLD feed as an alternative chart.
 */

const OKX = 'https://www.okx.com/api/v5';
const SPOT_INST = 'XAU-USD';

const TIMEFRAMES = { '1m': '1m', '5m': '5m', '15m': '15m', '1H': '1H', '4H': '4H', '1D': '1Dutc' };
const TF_SECONDS = { '1m': 60, '5m': 300, '15m': 900, '1H': 3600, '4H': 14400, '1D': 86400 };

const SPOT_POLL_MS = 2000;
const REF_POLL_MS = 5000;

const REFERENCE_TICKERS = [
    ['OANDA:XAUUSD', 'OANDA'],
    ['FOREXCOM:XAUUSD', 'Forex.com'],
    ['PEPPERSTONE:XAUUSD', 'Pepperstone'],
    ['FX_IDC:XAUUSD', 'ICE (FX_IDC)'],
    ['TVC:GOLD', 'TVC composite'],
];

let okxContractOz = 0.001; // OKX gold perp contract size in oz (used by flow.js); replaced by the live spec

const DEFAULT_LAYERS = { zones: true, vp: false, vpLines: false };
const settings = Object.assign({
    chartMode: 'spot',   // 'spot' | 'tradingview'
    tf: '15m',
    // what is drawn on the gold chart, and which charts are shown below it (Layers menu)
    layers: { ...DEFAULT_LAYERS },
    panes: { flow: false, cot: false, real: false, hormuz: false },
    zoneWidth: 10,       // $ per order zone
    zoneRange: 100,      // zones within ±$ of spot
    zoneDays: 5,         // days of COMEX volume counted
    vp: '1M',            // volume profile window: '1W' | '1M' | '3M'
    vpSource: 'comex',   // volume profile source: 'comex' (COMEX futures) | 'okx' (OKX perp)
}, store.get('gt_market_settings', {}));
// settings saved before the order zones replaced the crypto order book: start from the new defaults
if (!settings.layers || settings.layers.zones === undefined) settings.layers = { ...DEFAULT_LAYERS };
settings.panes = Object.assign({ flow: false, cot: false, real: false, hormuz: false }, settings.panes);
['step', 'rangeUsd', 'align', 'smooth', 'levels', 'enabled', 'walls', 'rangePct'].forEach((k) => { settings[k] = undefined; });
if (!['1W', '1M', '3M'].includes(settings.vp)) settings.vp = '1M';

function saveSettings() { store.set('gt_market_settings', settings); }
saveSettings();

const state = {
    spot: null, open24h: null, spotTs: 0,
    candles: [], chart: null, series: null, overlay: null, tvLoaded: false,
};

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
        // zoom out far enough that the order zones and volume zones are in view
        autoscaleInfoProvider: (original) => {
            const res = original();
            const prices = [];
            if (typeof zonesAutoscalePrices === 'function') prices.push(...zonesAutoscalePrices());
            if (typeof vpAutoscalePrices === 'function') prices.push(...vpAutoscalePrices());
            if (!res || settings.chartMode !== 'spot' || !prices.length) return res;
            return {
                priceRange: {
                    minValue: Math.min(res.priceRange.minValue, ...prices) - 3,
                    maxValue: Math.max(res.priceRange.maxValue, ...prices) + 3,
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
        drawOverlay();
        lastOverlayDraw = t;
    }
    requestAnimationFrame(drawOverlayLoop);
}

// Canvas on top of the chart for the zone bands and the volume profile bars.
function drawOverlay() {
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
    const paneWidth = state.chart.timeScale().width();
    const paneHeight = rect.height - state.chart.timeScale().height();
    if (typeof drawZones === 'function') drawZones(ctx, paneWidth, paneHeight);
    if (typeof drawVolumeProfile === 'function') drawVolumeProfile(ctx, paneWidth, paneHeight);
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

// ---------------------------------------------------------------- reference quotes

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

// ---------------------------------------------------------------- controls

function syncControls() {
    document.querySelectorAll('[data-chart-mode]').forEach((b) => b.classList.toggle('active', b.dataset.chartMode === settings.chartMode));
    document.querySelectorAll('[data-tf]').forEach((b) => b.classList.toggle('active', b.dataset.tf === settings.tf));
    const spotMode = settings.chartMode === 'spot';
    $('spot-chart-wrap').hidden = !spotMode;
    $('tv-widget').hidden = spotMode;
    $('tf-group').hidden = !spotMode;
    if (!spotMode) {
        $('chart-caption').textContent = 'TradingView chart, Capital.com GOLD feed. Layers cannot be drawn on this widget; the order zones are listed on the right.';
    } else if (typeof updateLegend === 'function') {
        updateLegend();
    }
}

function bindControls() {
    document.querySelectorAll('[data-chart-mode]').forEach((b) => b.addEventListener('click', () => {
        settings.chartMode = b.dataset.chartMode; saveSettings(); syncControls();
        if (settings.chartMode === 'tradingview') loadTradingView();
        if (typeof updateZoneLines === 'function') updateZoneLines();
        if (typeof updatePocLine === 'function') updatePocLine();
    }));
    document.querySelectorAll('[data-tf]').forEach((b) => b.addEventListener('click', () => {
        settings.tf = b.dataset.tf; saveSettings(); syncControls(); loadCandles();
    }));
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
    every(REF_POLL_MS, pollReference);
    setInterval(() => { if (document.visibilityState === 'visible') loadCandles(); }, 60 * 1000);
}
