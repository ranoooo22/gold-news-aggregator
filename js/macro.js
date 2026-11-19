'use strict';

/*
 * Macro tab: cross-asset dashboard.
 *  - Live quotes and 1D/1W/1M/3M/1Y changes: TradingView's public scanner (browser).
 *  - US 10Y real yield and breakeven: US Treasury, saved daily by the workflow (macro.json).
 */

const MACRO_POLL_MS = 30000;
const PERF_COLUMNS = ['change', 'Perf.W', 'Perf.1M', 'Perf.3M', 'Perf.Y'];
const PERIODS = ['1D', '1W', '1M', '3M', '1Y'];

// kind: 'price' -> % changes; 'yield' -> changes in basis points; 'level' -> no history available
const MACRO_ROWS = [
    { group: 'Metals', sym: 'OANDA:XAUUSD', label: 'Gold', kind: 'price', digits: 2 },
    { group: 'Metals', sym: 'TVC:SILVER', label: 'Silver', kind: 'price', digits: 2 },
    { group: 'Metals', id: 'gsr', label: 'Gold/silver ratio', kind: 'price', digits: 1,
        derive: (v) => v['OANDA:XAUUSD'] / v['TVC:SILVER'] },
    { group: 'Energy', sym: 'FX:UKOIL', label: 'Brent crude', kind: 'price', digits: 2 },
    { group: 'Energy', sym: 'FX:USOIL', label: 'WTI crude', kind: 'price', digits: 2 },
    { group: 'Dollar & rates', sym: 'TVC:DXY', label: 'US dollar index (DXY)', kind: 'price', digits: 2 },
    { group: 'Dollar & rates', sym: 'ECONOMICS:USINTR', label: 'Fed funds rate', kind: 'level', digits: 2, unit: '%' },
    { group: 'Dollar & rates', sym: 'TVC:US02Y', label: 'US 2Y yield', kind: 'yield', digits: 3, unit: '%' },
    { group: 'Dollar & rates', sym: 'TVC:US10Y', label: 'US 10Y yield', kind: 'yield', digits: 3, unit: '%' },
    { group: 'Dollar & rates', id: 'spread', label: '10Y − 2Y spread', kind: 'yield', digits: 3, unit: '%',
        derive: (v) => v['TVC:US10Y'] - v['TVC:US02Y'] },
    { group: 'Dollar & rates', daily: 'REAL10', label: 'US 10Y real yield (TIPS)', kind: 'yield', digits: 2, unit: '%' },
    { group: 'Inflation', daily: 'BE10', label: 'US 10Y breakeven inflation', kind: 'yield', digits: 2, unit: '%' },
    { group: 'Inflation', sym: 'ECONOMICS:USIRYY', label: 'US CPI inflation (YoY)', kind: 'level', digits: 1, unit: '%' },
    { group: 'Inflation', sym: 'ECONOMICS:USCIR', label: 'US core CPI inflation (YoY)', kind: 'level', digits: 1, unit: '%' },
    { group: 'Equities', sym: 'SP:SPX', label: 'S&P 500', kind: 'price', digits: 2 },
    { group: 'Equities', sym: 'NASDAQ:NDX', label: 'Nasdaq 100', kind: 'price', digits: 2 },
];

const macroState = { daily: null };

// Value `n` periods ago from a current value and a % performance figure.
const pastFromPerf = (last, perf) => (perf === null || perf === undefined ? null : last / (1 + perf / 100));

// daily history: latest value on or before a date
function dailyValueAt(points, isoDate) {
    for (let i = points.length - 1; i >= 0; i--) if (points[i][0] <= isoDate) return points[i][1];
    return null;
}

async function loadDailyMacro() {
    try {
        const data = await getJSON(`macro.json?t=${Date.now()}`);
        macroState.daily = data.series || null;
    } catch (e) {
        macroState.daily = null;
    }
}

async function pollMacro() {
    const tickers = MACRO_ROWS.filter((r) => r.sym).map((r) => r.sym);
    let quotes = new Map();
    try {
        const j = await getJSON('https://scanner.tradingview.com/global/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'text/plain' },
            body: JSON.stringify({ symbols: { tickers }, columns: ['close', ...PERF_COLUMNS] }),
        });
        quotes = new Map((j.data || []).map((r) => [r.s, r.d.map((x) => (x === null ? null : Number(x)))]));
    } catch (e) {
        $('macro-status').textContent = `Live quotes unavailable (${e.message})`;
    }
    renderMacro(quotes);
}

function renderMacro(quotes) {
    // current and past values for every symbol: [now, 1D ago, 1W ago, 1M ago, 3M ago, 1Y ago]
    const series = {};
    quotes.forEach((d, sym) => {
        const last = d[0];
        series[sym] = [last, ...d.slice(1).map((p) => pastFromPerf(last, p))];
    });
    const periodDays = [0, 1, 7, 30, 91, 365];

    let group = '';
    const rows = [];
    for (const r of MACRO_ROWS) {
        let vals = null;
        let note = '';
        if (r.sym) {
            vals = series[r.sym] || null;
        } else if (r.derive) {
            vals = periodDays.map((_, i) => {
                const v = {};
                for (const sym of Object.keys(series)) v[sym] = series[sym][i];
                const out = r.derive(v);
                return isFinite(out) ? out : null;
            });
        } else if (r.daily) {
            const s = macroState.daily && macroState.daily[r.daily];
            if (s && s.points.length) {
                const lastDate = s.points[s.points.length - 1][0];
                // changes measured back from the latest published day
                const base = Date.parse(lastDate);
                vals = periodDays.map((d) => dailyValueAt(s.points, new Date(base - d * 86400000).toISOString().slice(0, 10)));
                note = `daily · ${lastDate}`;
            } else {
                note = 'loading daily data';
            }
        }
        if (r.group !== group) {
            rows.push(`<tr class="group-row"><td colspan="7">${r.group}</td></tr>`);
            group = r.group;
        }
        const last = vals ? vals[0] : null;
        const cells = PERIODS.map((_, i) => {
            const past = vals ? vals[i + 1] : null;
            if (r.kind === 'level' || last === null || past === null || past === undefined) return '<td class="num muted">–</td>';
            if (r.kind === 'yield') {
                const bp = (last - past) * 100;
                return `<td class="num ${bp >= 0 ? 'up' : 'down'}">${fmtSigned(bp, 0)} bp</td>`;
            }
            const pct = (last / past - 1) * 100;
            return `<td class="num ${pct >= 0 ? 'up' : 'down'}">${fmtSigned(pct, 1)}%</td>`;
        }).join('');
        rows.push(`<tr>
            <td>${r.label}${note ? ` <span class="muted small">${note}</span>` : ''}</td>
            <td class="num"><b>${last === null ? '–' : fmt(last, r.digits) + (r.unit || '')}</b></td>
            ${cells}
        </tr>`);
    }
    $('macro-table').innerHTML = `<table class="data-table macro-table">
        <thead><tr><th></th><th class="num">Last</th>${PERIODS.map((p) => `<th class="num">${p}</th>`).join('')}</tr></thead>
        <tbody>${rows.join('')}</tbody></table>`;
    if (quotes.size) $('macro-status').textContent = `Live via TradingView · updated ${new Date().toISOString().slice(11, 19)} UTC`;
}

function startMacro() {
    loadDailyMacro().then(pollMacro);
    setInterval(() => { if (document.visibilityState === 'visible') pollMacro(); }, MACRO_POLL_MS);
    setInterval(() => {
        if (document.visibilityState === 'visible') loadDailyMacro();
    }, 60 * 60 * 1000);
}
