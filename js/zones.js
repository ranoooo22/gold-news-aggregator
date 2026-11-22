'use strict';

/*
 * Order zones: the 3 most important $10 price zones above and below spot, from two sources:
 *  - OANDA: share of OANDA clients' open orders in the zone, and how many of them are buy vs
 *    sell orders (oanda_book.json, snapshot saved by the workflow; OANDA refreshes every 20 min).
 *  - COMEX: share of the gold futures volume of the last few days traded in the zone
 *    (comex_volume.json, hourly, already aligned to spot).
 * The two are different measures, so they are shown side by side, never added up. A zone's
 * importance is the average of its two shares; "OANDA + COMEX" marks zones where both are
 * clearly above average.
 */

const ZONES_PER_SIDE = 3;
const ZONE_STRONG = 1.25;       // "clearly above average" for the OANDA + COMEX mark
const ZONE_REFRESH_MS = 5 * 60 * 1000;

const zonesState = { oanda: null, comex: null, zones: [], lines: [], maxScore: 0 };

async function loadZoneData() {
    const [oanda, comex] = await Promise.all([
        getJSON(`oanda_book.json?t=${Date.now()}`, {}, 15000).catch(() => null),
        getJSON(`comex_volume.json?t=${Date.now()}`, {}, 15000).catch(() => null),
    ]);
    zonesState.oanda = oanda;
    zonesState.comex = comex && comex.rows && comex.rows.length ? comex : null;
    computeZones();
}

function computeZones() {
    const spot = state.spot || (zonesState.oanda && zonesState.oanda.price);
    if (!spot) return;
    const W = settings.zoneWidth;
    const lo = spot - settings.zoneRange, hi = spot + settings.zoneRange;
    const zones = new Map();
    const zoneAt = (price) => {
        const key = Math.floor(price / W) * W;
        if (!zones.has(key)) zones.set(key, { lo: key, hi: key + W, buy: 0, sell: 0, vol: 0 });
        return zones.get(key);
    };

    // OANDA: percentage of all open orders per price bucket
    if (zonesState.oanda) {
        for (const b of zonesState.oanda.buckets) {
            if (b.price < lo || b.price >= hi) continue;
            const z = zoneAt(b.price);
            z.buy += b.long_pct;
            z.sell += b.short_pct;
        }
    }
    // COMEX: hourly volume spread over each candle's range, last N days
    if (zonesState.comex) {
        const since = Date.now() - settings.zoneDays * 86400000;
        for (const [ts, low, high, vol] of zonesState.comex.rows) {
            if (ts < since || high < lo || low >= hi) continue;
            if (high - low < 0.01) { zoneAt(low).vol += vol; continue; }
            for (let k = Math.floor(Math.max(low, lo) / W) * W; k < Math.min(high, hi); k += W) {
                const overlap = Math.min(high, k + W) - Math.max(low, k);
                if (overlap > 0) zoneAt(k).vol += vol * (overlap / (high - low));
            }
        }
    }

    const list = [...zones.values()].filter((z) => z.lo >= lo - W && z.hi <= hi + W);
    const oTotal = list.reduce((a, z) => a + z.buy + z.sell, 0);
    const cTotal = list.reduce((a, z) => a + z.vol, 0);
    list.forEach((z) => {
        z.oandaShare = oTotal ? ((z.buy + z.sell) / oTotal) * 100 : null;
        z.comexShare = cTotal ? (z.vol / cTotal) * 100 : null;
        const parts = [z.oandaShare, z.comexShare].filter((x) => x !== null);
        z.score = parts.length ? parts.reduce((a, x) => a + x, 0) / parts.length : 0;
        const orders = z.buy + z.sell;
        z.buyPct = orders > 0 ? (z.buy / orders) * 100 : null;
        z.dir = z.buyPct === null ? 'none' : z.buyPct >= 55 ? 'buy' : z.buyPct <= 45 ? 'sell' : 'mixed';
        z.mid = z.lo + W / 2;
    });
    const avgO = list.length && oTotal ? 100 / list.length : null;
    const avgC = list.length && cTotal ? 100 / list.length : null;
    list.forEach((z) => {
        z.overlap = avgO !== null && avgC !== null && z.oandaShare >= ZONE_STRONG * avgO && z.comexShare >= ZONE_STRONG * avgC;
    });

    const pick = (side) => list.filter((z) => (side === 'above' ? z.mid > spot : z.mid <= spot))
        .sort((a, b) => b.score - a.score).slice(0, ZONES_PER_SIDE);
    zonesState.zones = [...pick('above'), ...pick('below')].sort((a, b) => b.lo - a.lo);
    zonesState.maxScore = Math.max(0, ...zonesState.zones.map((z) => z.score));
    renderZonesPanel(spot);
    updateZoneLines();
    if (state.series) state.series.applyOptions({}); // re-run autoscale
}

const DIR_LABEL = { buy: 'Buy orders', sell: 'Sell orders', mixed: 'Mixed', none: '' };
const DIR_COLOR = { buy: '0, 194, 110', sell: '255, 77, 79', mixed: '255, 163, 26', none: '56, 189, 248' };

function renderZonesPanel(spot) {
    const el = $('zones-list');
    if (!zonesState.zones.length) {
        el.innerHTML = '<div class="empty">No zone data yet.</div>';
        return;
    }
    const rows = [];
    let spotShown = false;
    const spotRow = `<div class="zone-spot">Spot ${fmt(spot)}</div>`;
    for (const z of zonesState.zones) {
        if (!spotShown && z.mid <= spot) { rows.push(spotRow); spotShown = true; }
        const dist = z.mid - spot;
        const oanda = z.oandaShare === null ? '<span class="muted">no OANDA data</span>'
            : `<span class="zone-bar"><span class="zb-buy" style="width:${(z.buyPct || 0).toFixed(0)}%"></span><span class="zb-sell" style="width:${(100 - (z.buyPct || 0)).toFixed(0)}%"></span></span>
               <span class="zone-val" title="${z.buyPct === null ? '' : `${fmt(z.buyPct, 0)}% buy orders, ${fmt(100 - z.buyPct, 0)}% sell orders`}">${z.buyPct === null ? '–' : `${fmt(z.buyPct, 0)}% buy`}</span>
               <span class="zone-share">${fmt(z.oandaShare, 1)}% of orders</span>`;
        const comex = z.comexShare === null ? '<span class="muted">no COMEX data</span>'
            : `<span class="zone-bar"><span class="zb-vol" style="width:${Math.min(100, (z.comexShare / Math.max(...zonesState.zones.map((x) => x.comexShare || 0))) * 100).toFixed(0)}%"></span></span>
               <span class="zone-val"></span>
               <span class="zone-share">${fmt(z.comexShare, 1)}% of volume</span>`;
        rows.push(`<div class="zone zone-${z.dir}">
            <div class="zone-head">
                <span class="zone-price">${fmt(z.lo, 0)} – ${fmt(z.hi, 0)}</span>
                <span class="zone-dist ${dist >= 0 ? 'up' : 'down'}">${fmtSigned(dist, 0)} $</span>
                ${z.dir !== 'none' ? `<span class="badge zone-tag ${z.dir}">${DIR_LABEL[z.dir]}</span>` : ''}
                ${z.overlap ? '<span class="badge zone-both" title="Both OANDA orders and COMEX volume are clearly above average here">OANDA + COMEX</span>' : ''}
            </div>
            <div class="zone-line"><span class="zone-src">OANDA</span>${oanda}</div>
            <div class="zone-line"><span class="zone-src">COMEX</span>${comex}</div>
        </div>`);
    }
    if (!spotShown) rows.push(spotRow);
    el.innerHTML = rows.join('');

    const o = zonesState.oanda, c = zonesState.comex;
    $('zones-asof').textContent = [
        o ? `OANDA ${o.time.slice(11, 16)} UTC` : 'OANDA –',
        c ? `COMEX through ${new Date(c.rows[c.rows.length - 1][0] + 3600000).toISOString().slice(11, 16)} UTC` : 'COMEX –',
    ].join(' · ');
    $('zones-sub').textContent = `$${settings.zoneWidth} zones within ±$${settings.zoneRange} · COMEX volume of the last ${settings.zoneDays} days`;
}

function updateZoneLines() {
    if (!state.series) return;
    zonesState.lines.forEach((l) => state.series.removePriceLine(l));
    zonesState.lines = [];
    if (!settings.layers.zones || settings.chartMode !== 'spot') return;
    const narrow = window.innerWidth < 680;
    for (const z of zonesState.zones) {
        const label = z.dir === 'buy' ? 'BUY' : z.dir === 'sell' ? 'SELL' : 'ZONE';
        zonesState.lines.push(state.series.createPriceLine({
            price: z.mid, color: `rgb(${DIR_COLOR[z.dir]})`, lineWidth: z.overlap ? 2 : 1,
            lineStyle: z.overlap ? LightweightCharts.LineStyle.Solid : LightweightCharts.LineStyle.Dashed,
            axisLabelVisible: true, title: narrow ? label : `${label}${z.overlap ? ' · OANDA+COMEX' : ''}`,
        }));
    }
}

// prices to keep in view (used by the chart's autoscale in market.js)
function zonesAutoscalePrices() {
    if (!settings.layers.zones || settings.chartMode !== 'spot') return [];
    return zonesState.zones.flatMap((z) => [z.lo, z.hi]);
}

// Called by the chart overlay loop in market.js: a translucent band for each zone.
function drawZones(ctx, paneWidth, paneHeight) {
    if (!settings.layers.zones || !zonesState.zones.length || !state.series) return;
    for (const z of zonesState.zones) {
        const yTop = state.series.priceToCoordinate(z.hi);
        const yBot = state.series.priceToCoordinate(z.lo);
        if (yTop === null || yBot === null || yBot < 0 || yTop > paneHeight) continue;
        const alpha = 0.07 + 0.13 * (zonesState.maxScore ? z.score / zonesState.maxScore : 0);
        ctx.fillStyle = `rgba(${DIR_COLOR[z.dir]}, ${alpha.toFixed(3)})`;
        ctx.fillRect(0, Math.min(yTop, yBot), paneWidth, Math.max(1, Math.abs(yBot - yTop)));
    }
}

function startZones() {
    const bind = (id, key) => {
        const el = $(id);
        el.value = String(settings[key]);
        el.addEventListener('change', () => { settings[key] = Number(el.value); saveSettings(); computeZones(); });
    };
    bind('opt-zone-width', 'zoneWidth');
    bind('opt-zone-range', 'zoneRange');
    bind('opt-zone-days', 'zoneDays');
    loadZoneData();
    setInterval(() => { if (document.visibilityState === 'visible') loadZoneData(); }, ZONE_REFRESH_MS);
    // zones are chosen relative to the current price: re-pick as it moves
    setInterval(() => { if (document.visibilityState === 'visible') computeZones(); }, 30000);
    // first pick as soon as the live price is in
    const wait = setInterval(() => { if (state.spot) { computeZones(); clearInterval(wait); } }, 500);
}
