'use strict';

/*
 * Volume profile on the spot chart: how much gold traded at each price over a past window.
 * Two sources:
 *  - COMEX gold futures (GC=F via Yahoo Finance), the main gold futures market. Saved hourly by
 *    the workflow (comex_volume.json, Yahoo can't be called from a browser), already shifted by
 *    each hour's futures-minus-spot gap so it lines up with the spot chart.
 *  - OKX XAU/USDT perpetual, live in the browser (details below).
 * Volume comes from OKX's XAU/USDT perpetual (candles with volume in troy oz). Each candle is
 * shifted by the perp's gap to the OKX XAU/USD spot index at that hour, so the profile lines up
 * with the spot candles. A check against COMEX gold futures over one month gave a 0.97
 * correlation between the two profiles (same value area and high-volume zones).
 *
 * Shown as bars on the left edge of the chart, with:
 *  - POC (point of control): the price bucket with the most volume
 *  - value area: the range around the POC that holds 70% of the volume
 */

const VP_WINDOWS = {
    '1W': { bar: '15m', candles: 672, step: 2, days: 7, label: '1 week' },
    '1M': { bar: '1H', candles: 720, step: 5, days: 30, label: '1 month' },
    '3M': { bar: '4H', candles: 540, step: 10, days: 91, label: '3 months' },
};
const VP_SOURCES = {
    comex: { label: 'COMEX GC futures via Yahoo, hourly' },
    okx: { label: 'OKX perp, live' },
};
const COMEX_OZ_PER_CONTRACT = 100;
// volume zones within this distance of spot are drawn as lines and kept in view on the chart
const VP_ZOOM_RANGE_USD = 200;
const VP_VALUE_AREA = 0.7;
const VP_REFRESH_MS = 15 * 60 * 1000;

const vpState = { profile: null, step: 5, poc: null, val: null, vah: null, peaks: [], lines: [], max: 0, loadId: 0 };

async function fetchOkxHistory(path, inst, bar, count) {
    const rows = [];
    let after = '';
    while (rows.length < count) {
        const url = `${OKX}/market/${path}?instId=${inst}&bar=${bar}&limit=100${after ? `&after=${after}` : ''}`;
        const data = (await getJSON(url, {}, 10000)).data || [];
        if (!data.length) break;
        rows.push(...data);
        after = data[data.length - 1][0];
    }
    return rows.slice(0, count);
}

// value area: grow outwards from the POC, always taking the bigger neighbour, until 70% is covered
function valueArea(keys, prof, pocIdx, total) {
    let i = pocIdx, j = pocIdx, acc = prof.get(keys[pocIdx]);
    while (acc < VP_VALUE_AREA * total && (i > 0 || j < keys.length - 1)) {
        const left = i > 0 ? prof.get(keys[i - 1]) : -1;
        const right = j < keys.length - 1 ? prof.get(keys[j + 1]) : -1;
        if (right >= left) { j++; acc += right; } else { i--; acc += left; }
    }
    return [keys[i], keys[j]];
}

// high-volume zones: local peaks, biggest first, at least 5 buckets apart
function findPeaks(keys, prof, step, n = 4) {
    const peaks = keys.filter((k, idx) => {
        const v = prof.get(k);
        for (let d = -2; d <= 2; d++) {
            const nb = keys[idx + d];
            if (d && nb !== undefined && prof.get(nb) > v) return false;
        }
        return true;
    }).sort((a, b) => prof.get(b) - prof.get(a));
    const chosen = [];
    for (const p of peaks) {
        if (chosen.every((c) => Math.abs(c - p) >= 5 * step)) chosen.push(p);
        if (chosen.length >= n) break;
    }
    return chosen;
}

// OKX perp candles, each shifted by the perp's gap to the spot index at that time
async function okxCandles(win) {
    const [perp, index] = await Promise.all([
        fetchOkxHistory('history-candles', 'XAU-USDT-SWAP', win.bar, win.candles),
        fetchOkxHistory('history-index-candles', SPOT_INST, win.bar, win.candles),
    ]);
    const indexClose = new Map(index.map((r) => [r[0], Number(r[4])]));
    const out = [];
    for (const r of perp) {
        const idxClose = indexClose.get(r[0]);
        const oz = Number(r[6]);
        if (idxClose === undefined || !(oz > 0)) continue;
        const basis = Number(r[4]) - idxClose;
        out.push({ lo: Number(r[3]) - basis, hi: Number(r[2]) - basis, oz });
    }
    out.latest = perp.length ? Number(perp[0][0]) : null;
    return out;
}

// COMEX hourly candles saved by the workflow, already aligned to spot
async function comexCandles(win) {
    const data = await getJSON(`comex_volume.json?t=${Date.now()}`, {}, 15000);
    const rows = data.rows || [];
    if (!rows.length) throw new Error('no COMEX data yet');
    const since = Date.now() - win.days * 86400000;
    const out = rows.filter((r) => r[0] >= since)
        .map((r) => ({ lo: r[1], hi: r[2], oz: r[3] * COMEX_OZ_PER_CONTRACT }));
    out.latest = rows[rows.length - 1][0];
    return out;
}

async function loadVolumeProfile() {
    const id = ++vpState.loadId;
    const win = VP_WINDOWS[settings.vp];
    if (!win) {
        Object.assign(vpState, { profile: null, peaks: [] });
        updatePocLine();
        renderVpSummary();
        return;
    }
    $('vp-summary').textContent = `Loading volume profile (${win.label})…`;
    try {
        const source = VP_SOURCES[settings.vpSource] ? settings.vpSource : 'comex';
        const candles = source === 'comex' ? await comexCandles(win) : await okxCandles(win);
        if (id !== vpState.loadId) return; // a newer request replaced this one
        const step = win.step;
        const prof = new Map();
        for (const c of candles) {
            // spread the candle's volume evenly over the buckets it traded through
            const b0 = Math.floor(c.lo / step), b1 = Math.floor(c.hi / step);
            const share = c.oz / (b1 - b0 + 1);
            for (let b = b0; b <= b1; b++) prof.set(b * step, (prof.get(b * step) || 0) + share);
        }
        if (!prof.size) throw new Error('no volume data');
        const keys = [...prof.keys()].sort((a, b) => a - b);
        const total = keys.reduce((a, k) => a + prof.get(k), 0);
        const pocIdx = keys.reduce((best, k, i) => (prof.get(k) > prof.get(keys[best]) ? i : best), 0);
        const [val, vah] = valueArea(keys, prof, pocIdx, total);
        Object.assign(vpState, {
            profile: prof, step, poc: keys[pocIdx], val, vah: vah + step, max: prof.get(keys[pocIdx]),
            peaks: findPeaks(keys, prof, step), total, source, latest: candles.latest,
        });
        updatePocLine();
        renderVpSummary();
    } catch (e) {
        if (id === vpState.loadId) $('vp-summary').textContent = `Volume profile unavailable (${e.message})`;
    }
}

// POC and high-volume zones as lines on the chart (the POC is always drawn, it is usually a peak too)
function vpLevels() {
    if (!vpState.profile) return [];
    const mid = (k) => k + vpState.step / 2;
    const levels = [{ price: mid(vpState.poc), title: 'VOL POC', poc: true }];
    vpState.peaks.forEach((p) => { if (p !== vpState.poc) levels.push({ price: mid(p), title: 'VOL zone' }); });
    return levels;
}

// levels close enough to spot to keep in view (used by the chart's autoscale in market.js)
function vpAutoscalePrices() {
    if (!vpState.profile || settings.chartMode !== 'spot' || !state.spot) return [];
    return vpLevels().filter((l) => Math.abs(l.price - state.spot) <= VP_ZOOM_RANGE_USD).map((l) => l.price);
}

function updatePocLine() {
    if (!state.series) return;
    (vpState.lines || []).forEach((l) => state.series.removePriceLine(l));
    vpState.lines = [];
    if (!vpState.profile || settings.chartMode !== 'spot') return;
    vpLevels().forEach((l) => {
        vpState.lines.push(state.series.createPriceLine({
            price: l.price, color: l.poc ? '#ffa31a' : '#38bdf8', lineWidth: l.poc ? 2 : 1,
            lineStyle: LightweightCharts.LineStyle.Dotted, axisLabelVisible: true, title: l.title,
        }));
    });
}

function renderVpSummary() {
    const el = $('vp-summary');
    if (!vpState.profile) { el.textContent = ''; return; }
    if (!state.spot) setTimeout(renderVpSummary, 2000); // distances follow once the first spot price is in
    const win = VP_WINDOWS[settings.vp];
    const mid = (k) => k + vpState.step / 2;
    const dist = (p) => (state.spot ? ` (${fmtSigned(p - state.spot, 0)})` : '');
    const zones = vpState.peaks.map((p) => `${fmt(mid(p), 0)}${dist(mid(p))}`).join(' · ');
    const through = vpState.source === 'comex' && vpState.latest
        ? `, through ${new Date(vpState.latest + 3600000).toISOString().slice(11, 16)} UTC` : '';
    el.innerHTML = `<b>Volume profile, ${win.label}</b> (${VP_SOURCES[vpState.source].label}${through}, ${fmt(vpState.total, 0)} oz):
        POC ${fmt(mid(vpState.poc), 0)}${dist(mid(vpState.poc))} ·
        value area ${fmt(vpState.val, 0)}–${fmt(vpState.vah, 0)} ·
        high-volume zones: ${zones}`;
}

// Called by the chart overlay loop in market.js.
function drawVolumeProfile(ctx, paneWidth, paneHeight) {
    if (!vpState.profile || !state.series) return;
    const maxLen = Math.min(240, paneWidth * 0.25);
    for (const [price, oz] of vpState.profile) {
        const yTop = state.series.priceToCoordinate(price + vpState.step);
        const yBot = state.series.priceToCoordinate(price);
        if (yTop === null || yBot === null || yBot < 0 || yTop > paneHeight) continue;
        const h = Math.max(1, Math.abs(yBot - yTop) - 1);
        const len = (oz / vpState.max) * maxLen;
        const inValueArea = price >= vpState.val && price < vpState.vah;
        ctx.fillStyle = price === vpState.poc ? 'rgba(255, 163, 26, 0.7)'
            : inValueArea ? 'rgba(56, 189, 248, 0.42)' : 'rgba(56, 189, 248, 0.2)';
        ctx.fillRect(0, Math.min(yTop, yBot), len, h);
    }
}

function startVolumeProfile() {
    const sel = $('opt-vp');
    sel.value = settings.vp;
    sel.addEventListener('change', () => { settings.vp = sel.value; saveSettings(); loadVolumeProfile(); });
    const src = $('opt-vp-src');
    src.value = VP_SOURCES[settings.vpSource] ? settings.vpSource : 'comex';
    src.addEventListener('change', () => { settings.vpSource = src.value; saveSettings(); loadVolumeProfile(); });
    loadVolumeProfile();
    setInterval(() => { if (document.visibilityState === 'visible') loadVolumeProfile(); }, VP_REFRESH_MS);
    // keep the "distance from spot" figures current
    setInterval(() => { if (vpState.profile) renderVpSummary(); }, 30000);
}
