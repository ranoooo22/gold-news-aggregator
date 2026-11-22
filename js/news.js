'use strict';

/*
 * News tab: reads data.json (written every 15 min by aggregator.py) and analysis.json
 * (analysts' commentary, shown as published between the news, clearly labelled).
 * Pins are stored in this browser together with a copy of the story,
 * so a pinned story stays visible after it drops out of data.json.
 */

const ANALYSIS_TOPIC = 'Analysis & commentary';

const PIN_KEY = 'gt_pins_v2';
const NEWS_SETTINGS_KEY = 'gt_news_settings';
const NEWS_REFRESH_MS = 3 * 60 * 1000;

const newsState = {
    clusters: [], analysis: [], feeds: [], generated: '',
    expanded: new Set(),
};

// An analysis article in the same shape as a news story, so filters, sorting and pins work for both.
function analysisAsStory(i) {
    return {
        id: `a:${i.key}`, type: 'analysis', first_ts: i.ts, last_ts: i.ts, item: i,
        articles: [{ title: i.title, link: i.link, source: i.source, ts: i.ts }],
    };
}

function liveStories() {
    return [...newsState.clusters, ...newsState.analysis.map(analysisAsStory)];
}

function storyTopic(c) {
    return c.type === 'analysis' ? ANALYSIS_TOPIC : (c.summary && c.summary.topic);
}
const newsSettings = Object.assign({
    topic: 'all', view: 'all', hours: 24, sort: 'updated', showFiltered: false,
}, store.get(NEWS_SETTINGS_KEY, {}));

function loadPins() { return store.get(PIN_KEY, {}); }
function savePins(p) { store.set(PIN_KEY, p); }

function clusterHeadline(c) {
    if (c.summary && c.summary.headline) return c.summary.headline;
    const newest = [...c.articles].sort((a, b) => b.ts - a.ts)[0];
    return newest ? newest.title : '(untitled)';
}

function togglePin(id) {
    const pins = loadPins();
    if (pins[id]) {
        delete pins[id];
    } else {
        const c = liveStories().find((x) => x.id === id);
        if (c) pins[id] = c;
    }
    savePins(pins);
    renderNews();
}

// Pinned stories: use the live version when it still exists (it may have new sources).
function allClustersWithPins() {
    const pins = loadPins();
    const live = new Map(liveStories().map((c) => [c.id, c]));
    let pinsChanged = false;
    for (const id of Object.keys(pins)) {
        if (live.has(id)) {
            pins[id] = live.get(id);
            pinsChanged = true;
        } else {
            live.set(id, { ...pins[id], archived: true });
        }
    }
    if (pinsChanged) savePins(pins);
    return { list: [...live.values()], pins };
}

function renderNews() {
    const container = $('news-list');
    const query = $('news-search').value.trim().toLowerCase();
    const { list, pins } = allClustersWithPins();
    const now = Date.now() / 1000;
    let filteredOut = 0;

    const shown = list.filter((c) => {
        const pinned = Boolean(pins[c.id]);
        const s = c.summary;
        if (!pinned && s && s.on_topic === false) {
            filteredOut++;
            if (!newsSettings.showFiltered) return false;
        }
        const isAnalysis = c.type === 'analysis';
        if (newsSettings.view === 'pinned' && !pinned) return false;
        if (newsSettings.view === 'multi' && c.articles.length < 2) return false;
        if (newsSettings.view === 'news' && isAnalysis) return false;
        if (!pinned && now - c.last_ts > newsSettings.hours * 3600) return false;
        if (newsSettings.topic !== 'all' && storyTopic(c) !== newsSettings.topic) return false;
        if (query) {
            const hay = [clusterHeadline(c), s?.summary, ...(s?.key_points || []),
                isAnalysis ? `${c.item.teaser || ''} ${c.item.author || ''}` : '',
                ...c.articles.map((a) => `${a.title} ${a.source}`)].join(' ').toLowerCase();
            if (!hay.includes(query)) return false;
        }
        return true;
    });

    const sorters = {
        updated: (a, b) => b.last_ts - a.last_ts,
        first: (a, b) => b.first_ts - a.first_ts,
        sources: (a, b) => (b.articles.length - a.articles.length) || (b.last_ts - a.last_ts),
        oldest: (a, b) => a.last_ts - b.last_ts,
    };
    shown.sort((a, b) => (Boolean(pins[b.id]) - Boolean(pins[a.id])) || sorters[newsSettings.sort](a, b));

    $('pin-count').textContent = Object.keys(pins).length;
    const nAnalysis = shown.filter((c) => c.type === 'analysis').length;
    const nNews = shown.length - nAnalysis;
    $('news-count').textContent = `${nNews} stor${nNews === 1 ? 'y' : 'ies'} · ${nAnalysis} analys${nAnalysis === 1 ? 'is' : 'es'}`;
    $('filtered-count').textContent = filteredOut;
    $('tab-news-count').textContent = shown.length;

    if (!shown.length) {
        container.innerHTML = '<div class="empty">No stories match these filters.</div>';
        return;
    }
    container.innerHTML = shown.map((c) => renderCard(c, Boolean(pins[c.id]))).join('');
}

// Analysis card: the author's own words (publisher teaser), never summarised.
function renderAnalysisCard(c, pinned) {
    const i = c.item;
    const label = i.kind === 'commentary' ? 'Commentary' : 'Analysis';
    const badges = [
        pinned ? `<span class="badge pin">${c.archived ? 'Pinned · expired' : 'Pinned'}</span>` : '',
        `<span class="badge analysis" title="Written by the publisher's analysts or columnists; their own views and forecasts">${label}</span>`,
        `<span class="badge src">${escapeHtml(i.source)}</span>`,
        i.author ? `<span class="card-author">by ${escapeHtml(i.author)}</span>` : '',
        i.ai ? '<span class="badge muted" title="The publisher marks this article as AI-generated or AI-assisted">AI-assisted (per publisher)</span>' : '',
    ].join('');
    return `<article class="news-card analysis-card ${pinned ? 'pinned' : ''}" data-id="${escapeHtml(c.id)}">
        <header class="card-head">
            <div class="card-meta">${badges}<span class="card-time">${utcTime(i.ts, true)} UTC (${timeAgo(i.ts)})</span></div>
            <button class="pin-btn ${pinned ? 'active' : ''}" data-action="pin" title="${pinned ? 'Unpin' : 'Pin to top'}">${pinned ? '★ Pinned' : '☆ Pin'}</button>
        </header>
        <h3 class="card-title"><a href="${escapeHtml(safeUrl(i.link))}" target="_blank" rel="noopener noreferrer">${escapeHtml(i.title)}</a></h3>
        ${i.teaser ? `<p class="analysis-teaser">${escapeHtml(i.teaser)}</p>` : ''}
        <a class="card-toggle" href="${escapeHtml(safeUrl(i.link))}" target="_blank" rel="noopener noreferrer">Read on ${escapeHtml(i.source)} ↗</a>
    </article>`;
}

function renderCard(c, pinned) {
    if (c.type === 'analysis') return renderAnalysisCard(c, pinned);
    const s = c.summary;
    const expanded = newsState.expanded.has(c.id);
    const arts = [...c.articles].sort((a, b) => b.ts - a.ts);
    const n = arts.length;
    const hasState = arts.some((a) => a.state_affiliated);
    const timeLine = n > 1
        ? `Updated ${utcTime(c.last_ts, true)} UTC (${timeAgo(c.last_ts)}) · first ${utcTime(c.first_ts, true)}`
        : `${utcTime(c.last_ts, true)} UTC (${timeAgo(c.last_ts)})`;

    const badges = [
        pinned ? `<span class="badge pin">${c.archived ? 'Pinned · expired' : 'Pinned'}</span>` : '',
        s ? `<span class="badge topic">${escapeHtml(s.topic)}</span>` : '<span class="badge pending">Summary pending</span>',
        `<span class="badge">${n} source${n > 1 ? 's' : ''}</span>`,
        hasState ? '<span class="badge state" title="Includes reports from government-controlled media">State media</span>' : '',
        s && s.on_topic === false ? '<span class="badge muted">Filtered: off-topic</span>' : '',
    ].join('');

    const summaryText = s ? s.summary : (arts[0].description || '');
    const points = s && s.key_points && s.key_points.length
        ? `<ul class="points">${s.key_points.map((p) => `<li>${escapeHtml(p)}</li>`).join('')}</ul>` : '';
    const sources = arts.map((a) => `
        <li>
            <span class="src-time">${utcTime(a.ts, true)}</span>
            <span class="src-name">${escapeHtml(a.source)}</span>
            ${a.state_affiliated ? '<span class="badge state">State media</span>' : ''}
            <a href="${escapeHtml(safeUrl(a.link))}" target="_blank" rel="noopener noreferrer">${escapeHtml(a.title)}</a>
        </li>`).join('');
    const staleNote = s && s.n_articles && s.n_articles < n
        ? `<div class="muted small">Summary covers ${s.n_articles} of ${n} sources; refresh pending.</div>` : '';

    return `<article class="news-card ${pinned ? 'pinned' : ''} ${expanded ? 'expanded' : ''}" data-id="${escapeHtml(c.id)}">
        <header class="card-head" data-action="expand">
            <div class="card-meta">${badges}<span class="card-time">${timeLine}</span></div>
            <button class="pin-btn ${pinned ? 'active' : ''}" data-action="pin" title="${pinned ? 'Unpin' : 'Pin to top'}">${pinned ? '★ Pinned' : '☆ Pin'}</button>
        </header>
        <h3 class="card-title" data-action="expand">${escapeHtml(clusterHeadline(c))}</h3>
        <p class="card-summary" data-action="expand">${escapeHtml(summaryText)}</p>
        <div class="card-body">
            ${staleNote}
            ${points}
            <div class="sources-label">Sources (newest first)</div>
            <ul class="sources">${sources}</ul>
        </div>
        <button class="card-toggle" data-action="expand">${expanded ? 'Hide details' : `Details & ${n} source${n > 1 ? 's' : ''}`}</button>
    </article>`;
}

function renderFeedHealth() {
    const ok = newsState.feeds.filter((f) => f.status === 'ok').length;
    $('feed-health-summary').textContent =
        `News updated ${newsState.generated} UTC · ${ok}/${newsState.feeds.length} feeds OK`;
    $('feed-health-list').innerHTML = newsState.feeds.map((f) =>
        `<li><span class="${f.status === 'ok' ? 'up' : 'down'}">${escapeHtml(f.status)}</span> ${escapeHtml(f.feed)} · ${f.kept} kept of ${f.items}</li>`).join('');
}

async function loadNews() {
    try {
        const [data, analysis] = await Promise.all([
            getJSON(`data.json?t=${Date.now()}`, {}, 15000),
            getJSON(`analysis.json?t=${Date.now()}`, {}, 15000).catch(() => ({ items: [] })),
        ]);
        if (!data || data.version !== 2) throw new Error('news data is in an old format; it will update on the next run');
        newsState.clusters = data.clusters || [];
        newsState.analysis = analysis.items || [];
        newsState.feeds = data.feeds || [];
        newsState.generated = data.generated_utc || '';
        renderFeedHealth();
        renderNews();
    } catch (e) {
        $('news-list').innerHTML = `<div class="empty">Could not load news: ${escapeHtml(e.message)}</div>`;
    }
}

function syncNewsControls() {
    $('news-topic').value = newsSettings.topic;
    $('news-view').value = newsSettings.view;
    $('news-hours').value = String(newsSettings.hours);
    $('news-sort').value = newsSettings.sort;
    $('news-show-filtered').checked = newsSettings.showFiltered;
}

function bindNewsControls() {
    const bind = (id, key, cast = (v) => v) => $(id).addEventListener('change', (e) => {
        newsSettings[key] = cast(e.target.type === 'checkbox' ? e.target.checked : e.target.value);
        store.set(NEWS_SETTINGS_KEY, newsSettings);
        renderNews();
    });
    bind('news-topic', 'topic');
    bind('news-view', 'view');
    bind('news-hours', 'hours', Number);
    bind('news-sort', 'sort');
    bind('news-show-filtered', 'showFiltered');
    $('news-search').addEventListener('input', renderNews);
    // phones: filters are folded away behind a button
    $('news-filter-toggle').addEventListener('click', (e) => {
        const open = document.querySelector('.news-toolbar').classList.toggle('filters-open');
        e.currentTarget.setAttribute('aria-expanded', String(open));
    });
    $('pin-filter').addEventListener('click', () => {
        newsSettings.view = newsSettings.view === 'pinned' ? 'all' : 'pinned';
        store.set(NEWS_SETTINGS_KEY, newsSettings);
        syncNewsControls();
        renderNews();
    });
    $('expand-all').addEventListener('click', () => {
        document.querySelectorAll('.news-card').forEach((el) => newsState.expanded.add(el.dataset.id));
        renderNews();
    });
    $('collapse-all').addEventListener('click', () => { newsState.expanded.clear(); renderNews(); });
    $('news-list').addEventListener('click', (e) => {
        if (e.target.closest('a')) return;
        const actionEl = e.target.closest('[data-action]');
        const card = e.target.closest('.news-card');
        if (!actionEl || !card) return;
        const id = card.dataset.id;
        if (actionEl.dataset.action === 'pin') {
            togglePin(id);
        } else {
            if (newsState.expanded.has(id)) newsState.expanded.delete(id); else newsState.expanded.add(id);
            card.classList.toggle('expanded');
            const n = card.querySelectorAll('.sources li').length;
            card.querySelector('.card-toggle').textContent = card.classList.contains('expanded')
                ? 'Hide details' : `Details & ${n} source${n > 1 ? 's' : ''}`;
        }
    });
}

function startNews() {
    syncNewsControls();
    bindNewsControls();
    loadNews();
    setInterval(() => { if (document.visibilityState === 'visible') loadNews(); }, NEWS_REFRESH_MS);
}
