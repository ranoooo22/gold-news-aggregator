'use strict';

/*
 * Prediction markets: Polymarket odds for events that move gold (Fed, geopolitics, energy).
 * Fetched by the browser from Polymarket's public Gamma API. The probability shown is the
 * price of the "Yes" share (0.62 = 62%); the change is versus 24 hours ago.
 */

const PM_API = 'https://gamma-api.polymarket.com/events';
const PM_TAGS = ['fed-rates', 'geopolitics', 'iran', 'middle-east', 'economy', 'commodities'];
const PM_MAX_EVENTS = 12;
const PM_MIN_VOLUME_24H = 10000;      // ignore thin markets
const PM_MAX_OUTCOMES = 4;
// short-term coin flips and price-target ladders are not events
const PM_EXCLUDE = /up or down|what will .* hit|largest company|nominee|nomination|election|prime minister/i;
const PM_POLL_MS = 3 * 60 * 1000;

async function fetchPmTag(tag) {
    const params = new URLSearchParams({
        tag_slug: tag, closed: 'false', active: 'true', order: 'volume24hr', ascending: 'false', limit: '20',
    });
    return getJSON(`${PM_API}?${params}`, {}, 15000);
}

function yesPrice(m) {
    try { return Number(JSON.parse(m.outcomePrices)[0]); } catch (e) { return null; }
}

function pickOutcomes(event) {
    const now = Date.now();
    const open = (event.markets || []).filter((m) => !m.closed && Date.parse(m.endDate) > now && yesPrice(m) !== null);
    if (event.negRisk) {
        // mutually exclusive outcomes (e.g. Fed decision): most likely first
        return open.sort((a, b) => yesPrice(b) - yesPrice(a)).slice(0, PM_MAX_OUTCOMES);
    }
    // independent markets, usually a date ladder ("by Dec 31?", "by Mar 31?"): nearest dates first
    return open.sort((a, b) => Date.parse(a.endDate) - Date.parse(b.endDate)).slice(0, PM_MAX_OUTCOMES);
}

async function loadPolymarket() {
    const el = $('pm-list');
    try {
        const results = await Promise.allSettled(PM_TAGS.map(fetchPmTag));
        const events = new Map();
        results.forEach((r) => { if (r.status === 'fulfilled') r.value.forEach((e) => events.set(e.id, e)); });
        if (!events.size) throw new Error('no data');
        const now = Date.now();
        const chosen = [...events.values()]
            .filter((e) => !e.closed && Date.parse(e.endDate) > now && (e.volume24hr || 0) >= PM_MIN_VOLUME_24H && !PM_EXCLUDE.test(e.title))
            .sort((a, b) => (b.volume24hr || 0) - (a.volume24hr || 0))
            .slice(0, PM_MAX_EVENTS)
            .map((e) => ({ e, outcomes: pickOutcomes(e) }))
            .filter((x) => x.outcomes.length);

        el.innerHTML = chosen.map(({ e, outcomes }) => {
            const single = (e.markets || []).length === 1;
            const rows = outcomes.map((m) => {
                const p = yesPrice(m) * 100;
                const chg = m.oneDayPriceChange === null || m.oneDayPriceChange === undefined ? null : m.oneDayPriceChange * 100;
                const label = single ? 'Yes' : (m.groupItemTitle || m.question);
                return `<div class="pm-row">
                    <span class="pm-label">${escapeHtml(label)}</span>
                    <span class="pm-bar"><span style="width:${Math.max(0, Math.min(100, p)).toFixed(1)}%"></span></span>
                    <span class="pm-prob">${fmt(p, p < 1 || p > 99 ? 1 : 0)}%</span>
                    <span class="pm-chg ${chg > 0 ? 'up' : chg < 0 ? 'down' : 'muted'}">${chg === null ? '' : `${fmtSigned(chg, 1)}`}</span>
                </div>`;
            }).join('');
            return `<a class="pm-card" href="https://polymarket.com/event/${encodeURIComponent(e.slug)}" target="_blank" rel="noopener noreferrer">
                <div class="pm-title">${escapeHtml(e.title)}</div>
                ${rows}
                <div class="pm-meta">24h volume $${fmt(e.volume24hr, 0)} · resolves ${escapeHtml(new Date(e.endDate).toISOString().slice(0, 10))}</div>
            </a>`;
        }).join('') || '<div class="empty">No matching markets.</div>';
        $('pm-asof').textContent = `updated ${new Date().toISOString().slice(11, 16)} UTC`;
    } catch (e) {
        el.innerHTML = `<div class="empty">Polymarket unavailable (${escapeHtml(e.message)})</div>`;
    }
}

function startPolymarket() {
    loadPolymarket();
    setInterval(() => { if (document.visibilityState === 'visible') loadPolymarket(); }, PM_POLL_MS);
}
