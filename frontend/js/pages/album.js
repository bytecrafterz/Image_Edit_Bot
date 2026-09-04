/* The album: infinite grid, full screen viewer, and picking several at once. */

import { api } from '../api.js';
import {
  el, clear, note, toast, spinner, empty, lazyImg, confirmSheet, sheet,
  selectionBar, moneyExact, dateLabel, pct, kv,
} from '../ui.js';

const PAGE = 30;

let filter = 'all';
let offset = 0;
let total = 0;
let loading = false;
let images = [];
let selectMode = false;
let selected = new Set();
let observer = null;
let bar = null;
let grid = null;
let modeBtn = null;

const FILTERS = [
  { key: 'all', label: 'Todas', params: {} },
  { key: 'final', label: 'Finales', params: { kind: 'final' } },
  { key: 'preview', label: 'Previas', params: { kind: 'preview' } },
  { key: 'fav', label: 'Favoritas', params: { favorites: true } },
];

function currentParams() {
  return (FILTERS.find((f) => f.key === filter) || FILTERS[0]).params;
}

/* ---------------------------------------------------------------- selection */

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

/* -------------------------------------------------------------------- data */

async function loadPage(sentinel) {
  if (loading) return;
  loading = true;
  try {
    const data = await api.get(api.qs('/api/album',
      { ...currentParams(), limit: PAGE, offset }));
    total = data.total || 0;
    images = images.concat(data.images || []);
    for (const img of data.images || []) grid.appendChild(tile(img));
    offset += (data.images || []).length;
    syncTiles();
    if (offset >= total && observer) observer.unobserve(sentinel);
  } catch (err) {
    toast(err.message, 'danger');
  } finally {
    loading = false;
  }
}

function tile(img) {
  let pressTimer = null;
  const node = el('div', {
    class: 'tile',
    dataset: { id: img.id },
    onClick: () => {
      if (selectMode) { toggle(img.id, node); return; }
      openViewer(img);
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
    img.is_favorite ? el('div', { class: 'tile__flag', text: '♥' }) : null,
    el('div', { class: 'tile__meta' }, [
      el('span', { text: pct(img.score) }),
      el('span', { text: img.cost_usd > 0 ? moneyExact(img.cost_usd) : 'gratis' }),
    ]),
  ]);
  return node;
}

/* ------------------------------------------------------------------ viewer */

function openViewer(img) {
  let index = images.findIndex((i) => i.id === img.id);
  const picture = el('img', { src: images[index].url, alt: '' });
  const favBtn = el('button', { class: 'btn', type: 'button' },
    images[index].is_favorite ? 'Quitar favorito' : 'Favorito');

  const show = (next) => {
    if (next < 0 || next >= images.length) return;
    index = next;
    picture.src = images[index].url;
    favBtn.textContent = images[index].is_favorite ? 'Quitar favorito' : 'Favorito';
  };

  let startX = 0;
  const node = el('div', { class: 'viewer' }, [
    el('button', { class: 'viewer__close', type: 'button', 'aria-label': 'Cerrar',
      onClick: () => close() }, '×'),
    el('div', {
      class: 'viewer__img',
      onTouchstart: (e) => { startX = e.touches[0].clientX; },
      onTouchend: (e) => {
        const dx = e.changedTouches[0].clientX - startX;
        if (Math.abs(dx) > 60) show(index + (dx < 0 ? 1 : -1));
      },
    }, picture),
    el('div', { class: 'viewer__bar' }, [
      favBtn,
      el('a', { class: 'btn', href: `/api/album/${images[index].id}/download` },
        'Descargar'),
      el('button', { class: 'btn', type: 'button',
        onClick: () => showInfo(images[index]) }, 'Ficha'),
      el('button', { class: 'btn', type: 'button',
        onClick: async () => {
          const ok = await confirmSheet('Se eliminara esta imagen.',
            { title: 'Eliminar', confirmLabel: 'Eliminar', danger: true });
          if (!ok) return;
          try {
            await api.del(`/api/album/${images[index].id}`);
            toast('Eliminada');
            close();
            reload();
          } catch (err) { toast(err.message, 'danger'); }
        } }, 'Eliminar'),
    ]),
  ]);

  favBtn.addEventListener('click', async () => {
    const image = images[index];
    try {
      if (image.is_favorite) await api.del(`/api/favorites/${image.id}`);
      else await api.post(`/api/favorites/${image.id}`);
      image.is_favorite = !image.is_favorite;
      favBtn.textContent = image.is_favorite ? 'Quitar favorito' : 'Favorito';
    } catch (err) { toast(err.message, 'danger'); }
  });

  const onKey = (event) => {
    if (event.key === 'Escape') close();
    if (event.key === 'ArrowRight') show(index + 1);
    if (event.key === 'ArrowLeft') show(index - 1);
  };
  const close = () => {
    document.removeEventListener('keydown', onKey);
    node.remove();
  };
  document.addEventListener('keydown', onKey);
  document.body.appendChild(node);
}

function showInfo(img) {
  sheet({
    title: 'Ficha de la imagen',
    body: el('div', {}, [
      kv('Puntuacion', pct(img.score)),
      kv('Coste', img.cost_usd > 0 ? moneyExact(img.cost_usd) : 'gratis'),
      kv('Motor', `${img.provider || '-'} ${img.model || ''}`),
      kv('Creada', dateLabel(img.created_at)),
      img.summary ? el('p', { class: 'muted', style: { marginTop: '12px' },
        text: img.summary }) : null,
    ]),
    actions: [{ label: 'Cerrar', kind: 'secondary' }],
  });
}

/* ----------------------------------------------------------------- actions */

async function deleteSelected() {
  const ids = Array.from(selected);
  const ok = await confirmSheet(
    `Se eliminaran ${ids.length} imagen(es). No se puede deshacer.`,
    { title: 'Eliminar seleccionadas', confirmLabel: 'Eliminar', danger: true });
  if (!ok) return;
  try {
    const data = await api.post('/api/album/bulk-delete', { image_ids: ids });
    toast(`${data.deleted} eliminada(s)`, 'ok');
    setMode(false);
    reload();
  } catch (err) { toast(err.message, 'danger'); }
}

/* Move already-paid, approved images into "Finales" without generating them
   again.  The two images bought as this project's delivery sat in "Previas"
   while "Finales" read zero, and the only route into that tab re-rendered them
   at 0.04 USD each - paying twice for pixels that already exist.  The server
   refuses anything that did not pass its checks, so this button can only ever
   relabel work the robot already approved. */
async function markFinalSelected() {
  const ids = Array.from(selected);
  let done = 0;
  const failed = [];
  for (const id of ids) {
    try {
      await api.post(`/api/album/${id}/final`);
      done += 1;
    } catch (err) { failed.push(err.message); }
  }
  if (done) toast(`${done} en Finales, sin coste`, 'ok');
  if (failed.length) toast(failed[0], 'danger');
  setMode(false);
  reload();
}

async function favoriteSelected(on) {
  const ids = Array.from(selected);
  try {
    await api.post('/api/favorites/bulk', { image_ids: ids, favorite: on });
    toast(on ? 'Anadidas a favoritos' : 'Quitadas de favoritos', 'ok');
    setMode(false);
    reload();
  } catch (err) { toast(err.message, 'danger'); }
}

/* -------------------------------------------------------------------- page */

function reload() {
  offset = 0;
  images = [];
  selected.clear();
  render();
}

let viewRef = null;

async function render() {
  const view = viewRef;
  clear(view);

  modeBtn = el('button', {
    class: 'btn btn--secondary btn--sm', type: 'button',
    onClick: () => setMode(!selectMode),
  }, selectMode ? 'Salir de seleccion' : 'Seleccionar');

  view.appendChild(el('div', {
    style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      gap: '12px', marginBottom: '6px' },
  }, [el('h1', { text: 'Album', style: { margin: 0 } }), modeBtn]));

  const chips = el('div', { class: 'chips', style: { marginBottom: '14px' } });
  for (const item of FILTERS) {
    chips.appendChild(el('button', {
      class: 'chip' + (filter === item.key ? ' chip--on' : ''),
      type: 'button',
      onClick: () => { filter = item.key; reload(); },
    }, item.label));
  }
  view.appendChild(chips);

  grid = el('div', { class: 'grid' });
  const sentinel = el('div', { style: { height: '20px' } });
  view.appendChild(grid);
  view.appendChild(sentinel);

  bar = selectionBar({
    onSelectAll: () => {
      for (const img of images) selected.add(img.id);
      syncTiles();
      refreshBar();
    },
    onClear: () => { selected.clear(); syncTiles(); refreshBar(); },
    onCancel: () => setMode(false),
    actions: [
      { label: 'Marcar final', onClick: markFinalSelected },
      { label: 'Favoritos', onClick: () => favoriteSelected(true) },
      { label: 'Quitar favorito', onClick: () => favoriteSelected(false) },
      { label: 'Eliminar', kind: 'danger', onClick: deleteSelected },
    ],
  });
  bar.hidden = !selectMode;
  view.appendChild(bar);
  refreshBar();

  await loadPage(sentinel);

  if (!images.length) {
    clear(grid);
    bar.hidden = true;
    view.appendChild(empty({
      icon: '▦', title: 'Todavia no hay fotos',
      text: 'Cuando el robot genere imagenes apareceran aqui.',
      action: { href: '#/generate', label: 'Crear fotos' },
    }));
    return;
  }

  if (observer) observer.disconnect();
  observer = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting) && offset < total) loadPage(sentinel);
  }, { rootMargin: '400px' });
  observer.observe(sentinel);
}

export default {
  async mount(view) {
    viewRef = view;
    offset = 0;
    images = [];
    selected.clear();
    selectMode = false;
    await render();
  },
  unmount() {
    if (observer) { observer.disconnect(); observer = null; }
    bar = null;
    grid = null;
    modeBtn = null;
    viewRef = null;
  },
};
