'use strict';

/*
 * Layers and Settings menus for the gold chart.
 *  - Layers: what is drawn on the chart (order book, largest orders, volume profile, volume zones)
 *    and which small charts appear below it (buy/sell pressure, hedge fund positioning,
 *    real yield, Hormuz tankers). Presets switch several at once.
 *  - Settings: the detail controls (order book range, row size, smoothing, …).
 * On phones both open as a sheet from the bottom of the screen; on desktop as a side panel.
 * Choices are saved in this browser (settings in market.js).
 */

const LAYER_PRESETS = {
    clean: { layers: { book: false, walls: false, vp: false, vpLines: false } },
    levels: { layers: { book: false, walls: true, vp: false, vpLines: true } },
    all: { layers: { book: true, walls: true, vp: true, vpLines: true } },
};

function openSheet(id) {
    closeSheets();
    $(id).hidden = false;
    $('sheet-backdrop').hidden = false;
    document.body.classList.add('sheet-open');
}

function closeSheets() {
    document.querySelectorAll('.sheet').forEach((s) => { s.hidden = true; });
    $('sheet-backdrop').hidden = true;
    document.body.classList.remove('sheet-open');
}

function syncLayerControls() {
    document.querySelectorAll('[data-layer]').forEach((cb) => { cb.checked = Boolean(settings.layers[cb.dataset.layer]); });
    document.querySelectorAll('[data-pane]').forEach((cb) => { cb.checked = Boolean(settings.panes[cb.dataset.pane]); });
    const preset = Object.keys(LAYER_PRESETS).find((k) =>
        Object.entries(LAYER_PRESETS[k].layers).every(([key, v]) => settings.layers[key] === v));
    document.querySelectorAll('[data-preset]').forEach((b) => b.classList.toggle('active', b.dataset.preset === preset));
    const nOn = Object.values(settings.layers).filter(Boolean).length + Object.values(settings.panes).filter(Boolean).length;
    $('open-layers').querySelector('.count').textContent = nOn;
}

// Redraw everything that depends on the layer choices.
function applyLayers() {
    saveSettings();
    syncLayerControls();
    renderWalls(true);
    if (typeof updatePocLine === 'function') updatePocLine();
    if (typeof renderVpSummary === 'function') renderVpSummary();
    if (state.series) state.series.applyOptions({}); // re-run autoscale for the lines now shown
    updateLegend();
    if (typeof refreshPanes === 'function') refreshPanes();
}

// One line under the chart saying what is on it, in plain words.
function updateLegend() {
    if (settings.chartMode !== 'spot') return;
    const L = settings.layers;
    const items = [`<span class="lg"><i class="lg-candle"></i>Spot gold (OKX index)</span>`];
    if (L.book) items.push(`<span class="lg"><i class="lg-book"></i>Order book, right edge${settings.smooth ? ` (${settings.smooth}-min average)` : ''}</span>`);
    if (L.walls) items.push('<span class="lg"><i class="lg-walls"></i>Largest orders (bid / ask lines)</span>');
    if (L.vp) items.push(`<span class="lg"><i class="lg-vp"></i>Volume profile, left edge (${settings.vpSource === 'okx' ? 'OKX' : 'COMEX'}, ${({ '1W': '1 week', '1M': '1 month', '3M': '3 months' })[settings.vp] || ''})</span>`);
    if (L.vpLines) items.push('<span class="lg"><i class="lg-poc"></i>Volume zones (VOL lines)</span>');
    $('chart-caption').innerHTML = items.join('');
}

function startLayers() {
    $('open-layers').addEventListener('click', () => openSheet('layers-sheet'));
    $('open-settings').addEventListener('click', () => openSheet('settings-sheet'));
    $('sheet-backdrop').addEventListener('click', closeSheets);
    document.querySelectorAll('[data-close]').forEach((b) => b.addEventListener('click', closeSheets));
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSheets(); });

    document.querySelectorAll('[data-layer]').forEach((cb) => cb.addEventListener('change', () => {
        settings.layers[cb.dataset.layer] = cb.checked;
        applyLayers();
    }));
    document.querySelectorAll('[data-pane]').forEach((cb) => cb.addEventListener('change', () => {
        settings.panes[cb.dataset.pane] = cb.checked;
        applyLayers();
    }));
    document.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => {
        Object.assign(settings.layers, LAYER_PRESETS[b.dataset.preset].layers);
        applyLayers();
    }));
    // settings that change what the legend says
    ['opt-smooth', 'opt-vp', 'opt-vp-src'].forEach((id) => $(id).addEventListener('change', () => setTimeout(updateLegend, 0)));
    syncLayerControls();
    updateLegend();
}
