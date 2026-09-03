/* Favourites: the same grid restricted, with the same multi select gesture. */

import { api } from '../api.js';
import {
  el, clear, toast, spinner, empty, lazyImg, confirmSheet, selectionBar,
  moneyExact, pct,
} from '../ui.js';

let images = [];
let selectMode = false;
let selected = new Set();
let bar = null;
let grid = null;
let modeBtn = null;
let viewRef = null;

function setMode(on) {
  selectMode = on;
  if (!on) selected.clear();
  if (bar) bar.hidden = !on;
  if (modeBtn) modeBtn.textContent = on ? 'Salir de seleccion' : 'Seleccionar';
  syncTiles();
  refreshBar();
}

function refreshBar() {
  if (bar) bar.setCount(selected.size);
}

function syncTiles() {
  if (!grid) return;
  for (const tile of grid.children) {
    const id = tile.dataset.id;
    tile.setAttribute('aria-selected', String(selectMode && selected.has(id)));
    tile.classList.toggle('tile--selectable', selectMode);
  }
}

function toggle(id, tile) {
  if (selected.has(id)) selected.delete(id);
  else selected.add(id);
  tile.setAttribute('aria-selected', String(selected.has(id)));
  refreshBar();
}

async function removeSelected() {
  const ids = Array.from(selected);
  const ok = await confirmSheet(
    `Se quitaran ${ids.length} de favoritos. Las imagenes NO se borran, `
    + 'siguen en el album.',
    { title: 'Quitar de favoritos', confirmLabel: 'Quitar' });
  if (!ok) return;
  try {
    await api.post('/api/favorites/bulk', { image_ids: ids, favorite: false });
    toast(`${ids.length} quitada(s) de favoritos`, 'ok');
    setMode(false);
    await load();
    render();
  } catch (err) { toast(err.message, 'danger'); }
}

async function downloadSelected() {
  const ids = Array.from(selected);
  for (const id of ids) {
    const link = el('a', { href: `/api/album/${id}/download`, download: '' });
    document.body.appendChild(link);
    link.click();
    link.remove();
    await new Promise((r) => setTimeout(r, 350));
  }
  toast(`${ids.length} descarga(s) iniciada(s)`);
}

function tile(img) {
  let pressTimer = null;
  const node = el('div', {
    class: 'tile',
    dataset: { id: img.id },
    onClick: (event) => {
      if (selectMode) { event.preventDefault(); toggle(img.id, node); return; }
      window.open(img.url, '_blank', 'noopener');
    },
    onTouchstart: () => {
      pressTimer = setTimeout(() => {
        if (!selectMode) setMode(true);
        toggle(img.id, node);
      }, 500);
    },
    onTouchend: () => clearTimeout(pressTimer),
    onTouchmove: () => clearTimeout(pressTimer),
    onContextmenu: (event) => {
      event.preventDefault();
      if (!selectMode) setMode(true);
      toggle(img.id, node);
    },
  }, [
    lazyImg(img.thumb_url, ''),
    el('button', {
      class: 'tile__flag', type: 'button', 'aria-label': 'Quitar de favoritos',
      onClick: async (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (selectMode) { toggle(img.id, node); return; }
        try {
          await api.del(`/api/favorites/${img.id}`);
          toast('Quitada de favoritos');
          await load();
          render();
        } catch (err) { toast(err.message, 'danger'); }
      },
    }, '♥'),
    el('div', { class: 'tile__meta' }, [
      el('span', { text: pct(img.score) }),
      el('span', { text: img.cost_usd > 0 ? moneyExact(img.cost_usd) : 'gratis' }),
    ]),
  ]);
  return node;
}

async function load() {
  const data = await api.get('/api/favorites');
  images = data.images || [];
}

function render() {
  const view = viewRef;
  clear(view);

  modeBtn = el('button', {
    class: 'btn btn--secondary btn--sm', type: 'button',
    onClick: () => setMode(!selectMode),
  }, selectMode ? 'Salir de seleccion' : 'Seleccionar');

  view.appendChild(el('div', {
    style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      gap: '12px', marginBottom: '4px' },
  }, [
    el('h1', { text: 'Favoritos', style: { margin: 0 } }),
    images.length ? modeBtn : null,
  ]));
  view.appendChild(el('p', { class: 'view__sub',
    text: `${images.length} imagen(es) guardadas` }));

  if (!images.length) {
    view.appendChild(empty({
      icon: '♥', title: 'Sin favoritos todavia',
      text: 'Toca el corazon en cualquier imagen del album para guardarla aqui.',
      action: { href: '#/album', label: 'Ir al album' },
    }));
    return;
  }

  grid = el('div', { class: 'grid' });
  for (const img of images) grid.appendChild(tile(img));
  view.appendChild(grid);

  bar = selectionBar({
    onSelectAll: () => {
      for (const img of images) selected.add(img.id);
      syncTiles();
      refreshBar();
    },
    onClear: () => { selected.clear(); syncTiles(); refreshBar(); },
    onCancel: () => setMode(false),
    actions: [
      { label: 'Descargar', onClick: downloadSelected },
      { label: 'Quitar de favoritos', kind: 'danger', onClick: removeSelected },
    ],
  });
  bar.hidden = !selectMode;
  view.appendChild(bar);
  syncTiles();
  refreshBar();
}

export default {
  async mount(view) {
    viewRef = view;
    selectMode = false;
    selected.clear();
    view.appendChild(spinner());
    try {
      await load();
    } catch (err) {
      clear(view);
      view.appendChild(el('div', { class: 'note note--danger', text: err.message }));
      return;
    }
    render();
  },
  unmount() {
    bar = null;
    grid = null;
    modeBtn = null;
    viewRef = null;
  },
};
