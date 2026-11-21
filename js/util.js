'use strict';

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function safeUrl(url) {
    try {
        const u = new URL(url);
        return (u.protocol === 'https:' || u.protocol === 'http:') ? u.href : '#';
    } catch (e) {
        return '#';
    }
}

function fmt(n, digits = 2) {
    if (n === null || n === undefined || !isFinite(n)) return '–';
    return Number(n).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function fmtSigned(n, digits = 2) {
    if (n === null || n === undefined || !isFinite(n)) return '–';
    return (n >= 0 ? '+' : '') + fmt(n, digits);
}

function utcTime(tsSec, withDate = false) {
    const d = new Date(tsSec * 1000);
    const hm = d.toISOString().slice(11, 16);
    if (!withDate) return hm;
    return `${d.getUTCDate()} ${d.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' })} ${hm}`;
}

function timeAgo(tsSec) {
    const s = Math.max(0, Date.now() / 1000 - tsSec);
    if (s < 60) return 'just now';
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m ago`;
    return `${Math.floor(s / 86400)}d ago`;
}

async function getJSON(url, options = {}, timeoutMs = 7000) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    try {
        const res = await fetch(url, { ...options, signal: ctl.signal });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (e) {
        if (e.name === 'AbortError') throw new Error('timeout');
        throw e;
    } finally {
        clearTimeout(timer);
    }
}

// Charts created inside a hidden tab get width 0, so their fitted time range is wrong once the
// tab is shown. Each chart registers here and is re-fitted whenever its tab becomes visible.
const fitOnShow = [];
function registerFitOnShow(chart) { fitOnShow.push(chart); }
function refitCharts() {
    // wait for the browser to lay out the newly shown tab and the charts to resize
    setTimeout(() => {
        const charts = [...fitOnShow, ...(typeof paneCharts !== 'undefined' ? Object.values(paneCharts) : [])];
        charts.forEach((c) => { try { c.timeScale().fitContent(); } catch (e) { /* ignore */ } });
    }, 60);
}

// localStorage can be unavailable (private mode, blocked storage): never let that break the page.
const store = {
    get(key, fallback) {
        try {
            const raw = localStorage.getItem(key);
            return raw === null ? fallback : JSON.parse(raw);
        } catch (e) {
            return fallback;
        }
    },
    set(key, value) {
        try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* ignore */ }
    },
};
