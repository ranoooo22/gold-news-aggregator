'use strict';

/*
 * Small charts below the gold chart, chosen in the Layers menu. They reuse data the page already
 * loads for the detailed panels (no extra requests):
 *  - flow:   buy vs sell pressure, OKX gold perp (flow.js)
 *  - cot:    hedge funds' net position in COMEX gold futures (cot.js)
 *  - real:   US 10-year real yield (macro.js)
 *  - hormuz: tankers through the Strait of Hormuz (shipping.js)
 * Each has its own time range (they come at different frequencies); the header gives the latest
 * value in words.
 */

const PANE_DEFS = {
    flow: {
        title: 'Buy vs sell pressure',
        source: 'OKX gold perp, per 4 h',
        ready: () => typeof flowState !== 'undefined' && flowState.data,
        summary: () => {
            const rows = flowState.data.rows;
            const last = rows[rows.length - 1].ts;
            const day = rows.filter((r) => r.ts > last - 86400000);
            const buy = day.reduce((a, r) => a + r.buy, 0), sell = day.reduce((a, r) => a + r.sell, 0);
            const d = buy - sell;
            return `Last 24 h: <b class="${d >= 0 ? 'up' : 'down'}">${d >= 0 ? 'more buying' : 'more selling'}, ${fmtSigned(d, 0)} oz</b> (${fmt((sell / (buy + sell)) * 100, 0)}% aggressive sells)`;
        },
        draw: (chart) => {
            const bars = chart.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false });
            bars.setData(flowState.data.bars);
            const cum = chart.addLineSeries({ priceScaleId: 'left', color: THEME.accent, lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
            cum.setData(flowState.data.line);
            chart.applyOptions({ leftPriceScale: { visible: false } });
        },
        legend: 'bars: buys minus sells per 4 h · orange line: running total',
    },
    cot: {
        title: 'Hedge fund positioning',
        source: 'CFTC, COMEX gold, weekly',
        ready: () => typeof cotState !== 'undefined' && cotState.rows.length > 1,
        summary: () => {
            const rows = cotState.rows;
            const cur = rows[rows.length - 1].mm.net, prev = rows[rows.length - 2].mm.net;
            return `Net long <b>${fmt(cur, 0)} contracts</b> (${fmtSigned(cur - prev, 0)} vs prior week) · as of ${rows[rows.length - 1].date}`;
        },
        draw: (chart) => {
            const s = chart.addBaselineSeries({
                baseValue: { type: 'price', price: 0 }, lineWidth: 2, priceLineVisible: false,
                topLineColor: THEME.up, topFillColor1: rgba('up', 0.18), topFillColor2: rgba('up', 0.02),
                bottomLineColor: THEME.down, bottomFillColor1: rgba('down', 0.02), bottomFillColor2: rgba('down', 0.18),
            });
            s.setData(cotState.rows.slice(-52).map((r) => ({ time: r.date, value: r.mm.net })));
        },
        legend: 'managed money (hedge funds, CTAs): long minus short contracts, 52 weeks',
    },
    real: {
        title: 'US 10-year real yield',
        source: 'US Treasury (TIPS), daily',
        ready: () => typeof macroState !== 'undefined' && macroState.daily && macroState.daily.REAL10,
        summary: () => {
            const p = macroState.daily.REAL10.points;
            const last = p[p.length - 1], weekAgo = p[Math.max(0, p.length - 6)];
            return `<b>${fmt(last[1], 2)}%</b> on ${last[0]} (${fmtSigned((last[1] - weekAgo[1]) * 100, 0)} bp in a week)`;
        },
        draw: (chart) => {
            const s = chart.addLineSeries({ color: THEME.info, lineWidth: 2, priceLineVisible: false });
            s.setData(macroState.daily.REAL10.points.map(([time, value]) => ({ time, value })));
        },
        legend: 'yield of inflation-protected 10-year US bonds',
    },
    hormuz: {
        title: 'Strait of Hormuz tankers',
        source: 'IMF PortWatch, daily, a few days delayed',
        ready: () => typeof shipState !== 'undefined' && shipState.byStrait && shipState.byStrait['Strait of Hormuz'],
        summary: () => {
            const t = rollingAvg(shipState.byStrait['Strait of Hormuz'].tanker, PW_AVG_DAYS);
            const last = t[t.length - 1];
            const yearAgo = t.find((p) => p.time >= new Date(Date.parse(last.time) - 364 * 86400000).toISOString().slice(0, 10));
            return `<b>${fmt(last.value, 1)} tankers/day</b> (7-day avg, ${last.time})${yearAgo ? ` · a year ago ${fmt(yearAgo.value, 1)}` : ''}`;
        },
        draw: (chart) => {
            const s = chart.addAreaSeries({ lineColor: THEME.accent, topColor: rgba('accent', 0.2), bottomColor: rgba('accent', 0.02), lineWidth: 2, priceLineVisible: false });
            s.setData(rollingAvg(shipState.byStrait['Strait of Hormuz'].tanker, PW_AVG_DAYS));
        },
        legend: 'tankers passing per day, 7-day average (satellite AIS)',
    },
};
const PANE_ORDER = ['flow', 'cot', 'real', 'hormuz'];
const paneCharts = {};

function paneChartOptions() {
    return {
        autoSize: true,
        layout: chartLayout({ fontSize: 10 }),
        grid: chartGrid(false),
        timeScale: { borderColor: THEME.border },
        rightPriceScale: { borderColor: THEME.border },
        localization: { priceFormatter: (p) => (Math.abs(p) >= 1000 ? fmt(p, 0) : fmt(p, 2)) },
        handleScroll: false, handleScale: false,
    };
}

function refreshPanes() {
    const host = $('panes');
    if (!host) return;
    for (const key of PANE_ORDER) {
        const def = PANE_DEFS[key];
        const on = settings.panes[key];
        let el = host.querySelector(`[data-pane-box="${key}"]`);
        if (!on) {
            if (el) { paneCharts[key].remove(); delete paneCharts[key]; el.remove(); }
            continue;
        }
        if (!el) {
            el = document.createElement('div');
            el.className = 'pane';
            el.dataset.paneBox = key;
            el.innerHTML = `<div class="pane-head">
                    <span class="pane-title">${def.title}</span><span class="pane-src">${def.source}</span>
                    <button class="pane-close" title="Hide" aria-label="Hide ${def.title}">×</button>
                </div>
                <div class="pane-summary">Loading…</div>
                <div class="pane-chart"></div>
                <div class="pane-legend">${def.legend}</div>`;
            // keep the fixed order
            const next = PANE_ORDER.slice(PANE_ORDER.indexOf(key) + 1).map((k) => host.querySelector(`[data-pane-box="${k}"]`)).find(Boolean);
            host.insertBefore(el, next || null);
            el.querySelector('.pane-close').addEventListener('click', () => {
                settings.panes[key] = false;
                applyLayers();
            });
        }
        if (!def.ready()) continue;
        try {
            el.querySelector('.pane-summary').innerHTML = def.summary();
            if (paneCharts[key]) paneCharts[key].remove();
            const chart = LightweightCharts.createChart(el.querySelector('.pane-chart'), paneChartOptions());
            def.draw(chart);
            chart.timeScale().fitContent();
            paneCharts[key] = chart;
        } catch (e) {
            el.querySelector('.pane-summary').textContent = `Unavailable (${e.message})`;
        }
    }
}
