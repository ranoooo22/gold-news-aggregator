'use strict';

/*
 * CFTC Commitments of Traders, COMEX gold futures (Disaggregated report, futures only).
 * Positions are as of Tuesday and published by the CFTC on Friday afternoon (US time).
 * Source: CFTC Public Reporting Environment, fetched directly by the browser.
 */

const COT_URL = 'https://publicreporting.cftc.gov/resource/72hh-3qpy.json';
const COT_GOLD_CODE = '088691';
const COT_WEEKS = 53;            // one year of history plus the week before it
const OZ_PER_CONTRACT = 100;
const OZ_PER_TONNE = 32150.7466;

const COT_GROUPS = [
    { key: 'mm', label: 'Managed money', hint: 'Hedge funds, CTAs, commodity pools', long: 'm_money_positions_long_all', short: 'm_money_positions_short_all' },
    { key: 'pm', label: 'Producers & merchants', hint: 'Miners, refiners, fabricators, bullion dealers hedging', long: 'prod_merc_positions_long', short: 'prod_merc_positions_short' },
    { key: 'sd', label: 'Swap dealers', hint: 'Banks hedging OTC swaps', long: 'swap_positions_long_all', short: 'swap__positions_short_all' },
    { key: 'or', label: 'Other reportables', hint: 'Other large traders', long: 'other_rept_positions_long', short: 'other_rept_positions_short' },
    { key: 'nr', label: 'Non-reportable', hint: 'Positions below the reporting threshold (small traders)', long: 'nonrept_positions_long_all', short: 'nonrept_positions_short_all' },
];

const cotState = { rows: [], chart: null, series: null };

async function loadCot() {
    const fields = ['report_date_as_yyyy_mm_dd', 'open_interest_all',
        ...COT_GROUPS.flatMap((g) => [g.long, g.short])];
    const params = new URLSearchParams({
        cftc_contract_market_code: COT_GOLD_CODE,
        $select: fields.join(','),
        $order: 'report_date_as_yyyy_mm_dd DESC',
        $limit: String(COT_WEEKS),
    });
    try {
        const rows = await getJSON(`${COT_URL}?${params}`, {}, 15000);
        if (!rows.length) throw new Error('no data');
        cotState.rows = rows.map((r) => {
            const row = { date: r.report_date_as_yyyy_mm_dd.slice(0, 10), oi: Number(r.open_interest_all) };
            COT_GROUPS.forEach((g) => {
                row[g.key] = { long: Number(r[g.long]), short: Number(r[g.short]) };
                row[g.key].net = row[g.key].long - row[g.key].short;
            });
            return row;
        }).reverse(); // oldest first
        renderCotTable();
        renderCotChart();
    } catch (e) {
        $('cot-table').innerHTML = `<div class="empty">CFTC data unavailable (${escapeHtml(e.message)})</div>`;
    }
}

function fmtDate(iso) {
    const d = new Date(`${iso}T00:00:00Z`);
    return d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
}

function renderCotTable() {
    const rows = cotState.rows;
    const cur = rows[rows.length - 1];
    const prev = rows[rows.length - 2];
    const year = rows.slice(-52);
    $('cot-asof').textContent = `Positions as of ${fmtDate(cur.date)} · new report each Friday`;

    const body = COT_GROUPS.map((g) => {
        const c = cur[g.key];
        const chg = prev ? c.net - prev[g.key].net : null;
        const nets = year.map((r) => r[g.key].net);
        const lo = Math.min(...nets), hi = Math.max(...nets);
        const pos = hi > lo ? ((c.net - lo) / (hi - lo)) * 100 : 50;
        return `<tr title="${escapeHtml(g.hint)}">
            <td>${g.label}</td>
            <td class="num">${fmt(c.long, 0)}</td>
            <td class="num">${fmt(c.short, 0)}</td>
            <td class="num"><b class="${c.net >= 0 ? 'up' : 'down'}">${fmtSigned(c.net, 0)}</b></td>
            <td class="num ${chg >= 0 ? 'up' : 'down'}">${chg === null ? '–' : fmtSigned(chg, 0)}</td>
            <td class="num">${fmt((c.net * OZ_PER_CONTRACT) / OZ_PER_TONNE, 0)} t</td>
            <td class="num">${fmt((c.net / cur.oi) * 100, 1)}%</td>
            <td class="range-cell" title="52-week range of the net position: ${fmtSigned(lo, 0)} to ${fmtSigned(hi, 0)}">
                <div class="range-bar"><span style="left:${pos.toFixed(1)}%"></span></div>
            </td>
        </tr>`;
    }).join('');

    $('cot-table').innerHTML = `<table class="data-table">
        <thead><tr><th>Group</th><th class="num">Long</th><th class="num">Short</th><th class="num">Net</th>
            <th class="num">Net vs prior week</th><th class="num">Net (tonnes)</th><th class="num">Net % of OI</th>
            <th title="Where the current net position sits within its 52-week low–high">52-week range</th></tr></thead>
        <tbody>${body}</tbody>
    </table>
    <p class="note">Contracts (1 contract = 100 troy oz). Open interest ${fmt(cur.oi, 0)} contracts${prev ? ` (${fmtSigned(cur.oi - prev.oi, 0)} vs prior week)` : ''}.
        Hover a row for who is in the group.</p>`;
}

function renderCotChart() {
    const container = $('cot-chart');
    if (!cotState.chart) {
        cotState.chart = LightweightCharts.createChart(container, {
            autoSize: true,
            layout: { background: { type: 'solid', color: 'transparent' }, textColor: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' },
            grid: { vertLines: { color: '#161d2a' }, horzLines: { color: '#161d2a' } },
            timeScale: { borderColor: '#21293a' },
            rightPriceScale: { borderColor: '#21293a' },
            localization: { priceFormatter: (p) => fmt(p, 0) },
            handleScroll: false, handleScale: false,
        });
        cotState.series = cotState.chart.addBaselineSeries({
            baseValue: { type: 'price', price: 0 },
            topLineColor: '#00c26e', topFillColor1: 'rgba(0, 194, 110, 0.25)', topFillColor2: 'rgba(0, 194, 110, 0.02)',
            bottomLineColor: '#ff4d4f', bottomFillColor1: 'rgba(255, 77, 79, 0.02)', bottomFillColor2: 'rgba(255, 77, 79, 0.25)',
            lineWidth: 2,
        });
    }
    cotState.series.setData(cotState.rows.slice(-52).map((r) => ({ time: r.date, value: r.mm.net })));
    cotState.chart.timeScale().fitContent();
}

function startCot() {
    loadCot();
    // new reports appear weekly; re-check every 6 hours for pages left open
    setInterval(() => { if (document.visibilityState === 'visible') loadCot(); }, 6 * 3600 * 1000);
}
