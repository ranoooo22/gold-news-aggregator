'use strict';

/*
 * Aggressor flow: buy vs sell pressure over time on OKX's XAU/USDT perpetual.
 * Every trade has a buyer and a seller; "aggressive buy" volume is where the buyer crossed the
 * spread (market buy lifting the offer), "aggressive sell" where the seller did. Delta = aggressive
 * buys minus aggressive sells; cumulative delta is its running total.
 * Data: OKX hourly taker volume, saved by the workflow (taker_volume.json) because the endpoint
 * can't be called from a browser. OKX keeps 60 days; the file builds up history beyond that.
 */

const FLOW_BAR_HOURS = 4;
const FLOW_REFRESH_MS = 15 * 60 * 1000;

const flowState = { chart: null, bars: null, cum: null };

async function loadFlow() {
    try {
        const data = await getJSON(`taker_volume.json?t=${Date.now()}`, {}, 15000);
        const oz = okxContractOz; // taker volume is in contracts; same spec as the order book
        const rows = (data.rows || []).map((r) => ({ ts: r[0], sell: r[1] * oz, buy: r[2] * oz }));
        if (!rows.length) throw new Error('no data yet');

        // group hours into FLOW_BAR_HOURS bars
        const size = FLOW_BAR_HOURS * 3600000;
        const groups = new Map();
        rows.forEach((r) => {
            const k = Math.floor(r.ts / size) * size;
            const g = groups.get(k) || { buy: 0, sell: 0 };
            g.buy += r.buy; g.sell += r.sell;
            groups.set(k, g);
        });
        const keys = [...groups.keys()].sort((a, b) => a - b);
        let cum = 0;
        const bars = [], line = [];
        keys.forEach((k) => {
            const g = groups.get(k);
            const delta = g.buy - g.sell;
            cum += delta;
            const time = k / 1000;
            bars.push({ time, value: Math.round(delta), color: delta >= 0 ? 'rgba(0, 194, 110, 0.6)' : 'rgba(255, 77, 79, 0.6)' });
            line.push({ time, value: Math.round(cum) });
        });
        renderFlowChart(bars, line);

        // totals for recent windows
        const lastTs = rows[rows.length - 1].ts;
        const sum = (hours) => rows.filter((r) => r.ts > lastTs - hours * 3600000)
            .reduce((a, r) => ({ buy: a.buy + r.buy, sell: a.sell + r.sell }), { buy: 0, sell: 0 });
        const cell = (hours, label) => {
            const s = sum(hours);
            const total = s.buy + s.sell;
            const d = s.buy - s.sell;
            const buyPct = total ? (s.buy / total) * 100 : 0;
            return `<tr><td>${label}</td>
                <td class="num up">${fmt(s.buy, 0)}</td>
                <td class="num down">${fmt(s.sell, 0)}</td>
                <td class="num ${d >= 0 ? 'up' : 'down'}"><b>${fmtSigned(d, 0)}</b></td>
                <td class="num">${fmt(buyPct, 1)}% / ${fmt(100 - buyPct, 1)}%</td></tr>`;
        };
        const days = Math.round((lastTs - rows[0].ts) / 86400000);
        $('flow-table').innerHTML = `<table class="data-table">
            <thead><tr><th></th><th class="num">Aggr. buys (oz)</th><th class="num">Aggr. sells (oz)</th>
                <th class="num">Delta (oz)</th><th class="num">Buy / sell share</th></tr></thead>
            <tbody>${cell(24, 'Last 24 h')}${cell(24 * 7, 'Last 7 days')}${cell(24 * 30, 'Last 30 days')}${cell(24 * 400, `All (${days} days)`)}</tbody>
        </table>`;
        $('flow-asof').textContent = `through ${new Date(lastTs + 3600000).toISOString().slice(0, 16).replace('T', ' ')} UTC`;
    } catch (e) {
        $('flow-table').innerHTML = `<div class="empty">Flow data unavailable (${escapeHtml(e.message)}). It appears after the next workflow run.</div>`;
    }
}

function renderFlowChart(bars, line) {
    if (!flowState.chart) {
        flowState.chart = LightweightCharts.createChart($('flow-chart'), {
            autoSize: true,
            layout: { background: { type: 'solid', color: 'transparent' }, textColor: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' },
            grid: { vertLines: { color: '#161d2a' }, horzLines: { color: '#161d2a' } },
            timeScale: { borderColor: '#21293a', timeVisible: true },
            rightPriceScale: { borderColor: '#21293a' },
            leftPriceScale: { visible: true, borderColor: '#21293a' },
            localization: { priceFormatter: (p) => fmt(p, 0) },
        });
        registerFitOnShow(flowState.chart);
        flowState.bars = flowState.chart.addHistogramSeries({ priceScaleId: 'right', priceLineVisible: false });
        flowState.cum = flowState.chart.addLineSeries({ priceScaleId: 'left', color: '#ffa31a', lineWidth: 2, priceLineVisible: false });
    }
    flowState.bars.setData(bars);
    flowState.cum.setData(line);
    flowState.chart.timeScale().fitContent();
}

function startFlow() {
    loadFlow();
    setInterval(() => { if (document.visibilityState === 'visible') loadFlow(); }, FLOW_REFRESH_MS);
}
