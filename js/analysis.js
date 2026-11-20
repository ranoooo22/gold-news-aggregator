'use strict';

/*
 * Analysis & commentary: gold market analysis from analyst publishers (Kitco, Investing.com,
 * FXStreet, FXEmpire, ING Think, Saxo, StoneX, World Gold Council, …), collected by the workflow
 * via Google News (analysis.json). Shown exactly as published: title, publisher, time, link.
 * No clustering, no AI summary — these are the authors' own views, kept apart from the news.
 */

const ANALYSIS_SETTINGS_KEY = 'gt_analysis_settings';
const analysisState = { items: [], generated: '' };
const analysisSettings = Object.assign({ source: 'all', hours: 72 }, store.get(ANALYSIS_SETTINGS_KEY, {}));

async function loadAnalysis() {
    try {
        const data = await getJSON(`analysis.json?t=${Date.now()}`, {}, 15000);
        analysisState.items = data.items || [];
        analysisState.generated = data.generated_utc || '';
        renderAnalysisSources();
        renderAnalysis();
    } catch (e) {
        $('analysis-list').innerHTML = '<div class="empty">No analysis yet. It appears after the next workflow run.</div>';
    }
}

function renderAnalysisSources() {
    const counts = {};
    analysisState.items.forEach((i) => { counts[i.source] = (counts[i.source] || 0) + 1; });
    const sel = $('analysis-source');
    const current = analysisSettings.source;
    sel.innerHTML = '<option value="all">All publishers</option>' + Object.keys(counts).sort()
        .map((s) => `<option value="${escapeHtml(s)}">${escapeHtml(s)} (${counts[s]})</option>`).join('');
    sel.value = counts[current] ? current : 'all';
}

function renderAnalysis() {
    const query = $('analysis-search').value.trim().toLowerCase();
    const since = Date.now() / 1000 - analysisSettings.hours * 3600;
    const shown = analysisState.items.filter((i) => i.ts >= since
        && (analysisSettings.source === 'all' || i.source === analysisSettings.source)
        && (!query || `${i.title} ${i.source}`.toLowerCase().includes(query)));
    $('analysis-count').textContent = `${shown.length} article${shown.length === 1 ? '' : 's'} · updated ${analysisState.generated} UTC`;
    if (!shown.length) {
        $('analysis-list').innerHTML = '<div class="empty">No articles match these filters.</div>';
        return;
    }
    // group by UTC day, newest first
    let day = '';
    const html = [];
    for (const i of shown) {
        const d = new Date(i.ts * 1000).toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long', timeZone: 'UTC' });
        if (d !== day) { html.push(`<div class="analysis-day">${d}</div>`); day = d; }
        html.push(`<a class="analysis-item" href="${escapeHtml(safeUrl(i.link))}" target="_blank" rel="noopener noreferrer">
            <span class="src-time">${utcTime(i.ts)}</span>
            <span class="src-name">${escapeHtml(i.source)}</span>
            <span class="analysis-title">${escapeHtml(i.title)}</span>
        </a>`);
    }
    $('analysis-list').innerHTML = html.join('');
}

function setNewsMode(mode) {
    document.querySelectorAll('[data-news-mode]').forEach((b) => b.classList.toggle('active', b.dataset.newsMode === mode));
    $('news-view').hidden = mode !== 'news';
    $('analysis-view').hidden = mode !== 'analysis';
    store.set('gt_news_mode', mode);
}

function startAnalysis() {
    $('analysis-hours').value = String(analysisSettings.hours);
    const save = () => store.set(ANALYSIS_SETTINGS_KEY, analysisSettings);
    $('analysis-source').addEventListener('change', (e) => { analysisSettings.source = e.target.value; save(); renderAnalysis(); });
    $('analysis-hours').addEventListener('change', (e) => { analysisSettings.hours = Number(e.target.value); save(); renderAnalysis(); });
    $('analysis-search').addEventListener('input', renderAnalysis);
    document.querySelectorAll('[data-news-mode]').forEach((b) => b.addEventListener('click', () => setNewsMode(b.dataset.newsMode)));
    setNewsMode(store.get('gt_news_mode', 'news') === 'analysis' ? 'analysis' : 'news');
    loadAnalysis();
    setInterval(() => { if (document.visibilityState === 'visible') loadAnalysis(); }, 5 * 60 * 1000);
}
