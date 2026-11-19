'use strict';

/*
 * Shipping chokepoints: daily vessel transits from IMF PortWatch (satellite AIS data),
 * fetched by the browser from PortWatch's public ArcGIS service. Data arrives with a
 * delay of a few days. Ships that switch off their AIS transponder are not counted.
 */

const PW_URL = 'https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Chokepoints_Data/FeatureServer/0/query';
const PW_STRAITS = [
    { name: 'Strait of Hormuz', color: '#ffa31a' },
    { name: 'Bab el-Mandeb Strait', color: '#8b7cf6' },
    { name: 'Suez Canal', color: '#38bdf8' },
];
const PW_HISTORY_DAYS = 480;
const PW_AVG_DAYS = 7;

const shipState = { chart: null, series: {} };

async function fetchPortwatch() {
    const since = new Date(Date.now() - PW_HISTORY_DAYS * 86400000).toISOString().slice(0, 10);
    const names = PW_STRAITS.map((s) => `'${s.name}'`).join(',');
    let rows = [];
    for (let offset = 0; offset < 20000;) {
        const params = new URLSearchParams({
            where: `portname IN (${names}) AND date >= DATE '${since}'`,
            outFields: 'date,portname,n_tanker,n_total',
            orderByFields: 'date ASC',
            resultOffset: String(offset),
            resultRecordCount: '2000',
            f: 'json',
        });
        const j = await getJSON(`${PW_URL}?${params}`, {}, 20000);
        if (j.error) throw new Error(j.error.message || 'query failed');
        const batch = (j.features || []).map((f) => f.attributes);
        rows = rows.concat(batch);
        if (!j.exceededTransferLimit || !batch.length) break;
        offset += batch.length;
    }
    return rows;
}

// trailing N-day average for each date
function rollingAvg(points, n) {
    const out = [];
    let sum = 0;
    points.forEach((p, i) => {
        sum += p.v;
        if (i >= n) sum -= points[i - n].v;
        if (i >= n - 1) out.push({ time: p.time, value: Math.round((sum / n) * 10) / 10 });
    });
    return out;
}

function avgOver(points, endIdx, n) {
    const slice = points.slice(Math.max(0, endIdx - n + 1), endIdx + 1);
    return slice.length ? slice.reduce((a, p) => a + p.v, 0) / slice.length : null;
}

async function loadShipping() {
    try {
        const rows = await fetchPortwatch();
        const byStrait = {};
        PW_STRAITS.forEach((s) => { byStrait[s.name] = { tanker: [], total: [] }; });
        rows.forEach((r) => {
            const b = byStrait[r.portname];
            if (!b) return;
            b.tanker.push({ time: r.date, v: r.n_tanker });
            b.total.push({ time: r.date, v: r.n_total });
        });

        let latest = '';
        const tableRows = PW_STRAITS.map((s) => {
            const t = byStrait[s.name];
            if (!t.total.length) return `<tr><td>${s.name}</td><td colspan="5" class="muted">no data</td></tr>`;
            const last = t.total.length - 1;
            const lastDate = t.total[last].time;
            if (lastDate > latest) latest = lastDate;
            // same 7 days one year earlier
            const yearAgo = new Date(Date.parse(lastDate) - 364 * 86400000).toISOString().slice(0, 10);
            const yaIdx = t.total.findIndex((p) => p.time >= yearAgo);
            const cur = { total: avgOver(t.total, last, PW_AVG_DAYS), tanker: avgOver(t.tanker, last, PW_AVG_DAYS) };
            const prev = yaIdx >= PW_AVG_DAYS - 1
                ? { total: avgOver(t.total, yaIdx, PW_AVG_DAYS), tanker: avgOver(t.tanker, yaIdx, PW_AVG_DAYS) } : null;
            const chg = (a, b) => (b ? `${fmtSigned((a / b - 1) * 100, 0)}%` : '–');
            const cls = (a, b) => (b ? (a >= b ? 'up' : 'down') : 'muted');
            return `<tr>
                <td><span class="swatch" style="background:${s.color}"></span> ${s.name}</td>
                <td class="num"><b>${fmt(cur.total, 1)}</b></td>
                <td class="num">${prev ? fmt(prev.total, 1) : '–'}</td>
                <td class="num ${cls(cur.total, prev && prev.total)}">${chg(cur.total, prev && prev.total)}</td>
                <td class="num"><b>${fmt(cur.tanker, 1)}</b></td>
                <td class="num">${prev ? fmt(prev.tanker, 1) : '–'}</td>
                <td class="num ${cls(cur.tanker, prev && prev.tanker)}">${chg(cur.tanker, prev && prev.tanker)}</td>
            </tr>`;
        }).join('');

        $('ship-table').innerHTML = `<table class="data-table">
            <thead>
                <tr><th></th><th class="num" colspan="3">All vessels per day (7-day avg)</th><th class="num" colspan="3">Tankers per day (7-day avg)</th></tr>
                <tr><th>Chokepoint</th><th class="num">Now</th><th class="num">Year ago</th><th class="num">Change</th>
                    <th class="num">Now</th><th class="num">Year ago</th><th class="num">Change</th></tr>
            </thead><tbody>${tableRows}</tbody></table>`;
        $('ship-asof').textContent = latest ? `data through ${latest}` : '';
        renderShipChart(byStrait);
    } catch (e) {
        $('ship-table').innerHTML = `<div class="empty">PortWatch unavailable (${escapeHtml(e.message)})</div>`;
    }
}

function renderShipChart(byStrait) {
    if (!shipState.chart) {
        shipState.chart = LightweightCharts.createChart($('ship-chart'), {
            autoSize: true,
            layout: { background: { type: 'solid', color: 'transparent' }, textColor: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' },
            grid: { vertLines: { color: '#161d2a' }, horzLines: { color: '#161d2a' } },
            timeScale: { borderColor: '#21293a' },
            rightPriceScale: { borderColor: '#21293a' },
            handleScroll: false, handleScale: false,
        });
        PW_STRAITS.forEach((s) => {
            shipState.series[s.name] = shipState.chart.addLineSeries({ color: s.color, lineWidth: 2, priceLineVisible: false, title: '' });
        });
    }
    PW_STRAITS.forEach((s) => shipState.series[s.name].setData(rollingAvg(byStrait[s.name].tanker, PW_AVG_DAYS)));
    shipState.chart.timeScale().fitContent();
}

function startShipping() {
    loadShipping();
    // PortWatch updates about once a week; refresh every 6 hours for pages left open
    setInterval(() => { if (document.visibilityState === 'visible') loadShipping(); }, 6 * 3600 * 1000);
}
