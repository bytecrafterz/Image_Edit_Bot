/* DOM helpers.  Nodes are always built, never interpolated into innerHTML:
   filenames, prompts and reject reasons all come from outside. */

export function el(tag, props, children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = String(value);
    else if (key === 'html') node.innerHTML = value;      // only for our own markup
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
    else if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value === true) node.setAttribute(key, '');
    else node.setAttribute(key, String(value));
  }
  append(node, children);
  return node;
}

export function append(parent, children) {
  if (children === undefined || children === null) return parent;
  const list = Array.isArray(children) ? children : [children];
  for (const child of list) {
    if (child === undefined || child === null || child === false) continue;
    parent.appendChild(child instanceof Node ? child
      : document.createTextNode(String(child)));
  }
  return parent;
}

export function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

export function frag(children) { return append(document.createDocumentFragment(), children); }

/* ------------------------------------------------------------------ toast */

export function toast(message, kind) {
  const root = document.getElementById('toasts');
  if (!root) return;
  const node = el('div', { class: 'toast' + (kind ? ' toast--' + kind : ''), text: message });
  root.appendChild(node);
  setTimeout(() => {
    node.style.opacity = '0';
    node.style.transition = 'opacity .25s';
    setTimeout(() => node.remove(), 260);
  }, kind === 'danger' ? 5200 : 3200);
}

/* ------------------------------------------------------------------ sheet */

export function sheet({ title, subtitle, body, actions, onClose }) {
  const root = document.getElementById('sheet-root');
  const close = () => {
    backdrop.remove();
    document.body.style.overflow = '';
    if (onClose) onClose();
  };

  const panel = el('div', { class: 'sheet', role: 'dialog', 'aria-modal': 'true' }, [
    el('div', { class: 'sheet__grip' }),
    el('div', { class: 'sheet__head' }, [
      title ? el('h2', { text: title }) : null,
      subtitle ? el('p', { class: 'muted', text: subtitle, style: { margin: 0 } }) : null,
    ]),
    el('div', { class: 'sheet__body' }, body),
  ]);

  if (actions && actions.length) {
    panel.appendChild(el('div', { class: 'sheet__foot' }, actions.map((action) =>
      el('button', {
        class: 'btn' + (action.kind ? ' btn--' + action.kind : ''),
        type: 'button',
        // The action runs BEFORE the sheet closes.  Closing first fires
        // onClose, and confirmSheet treats that as "dismissed" and resolves
        // false - so every confirmation in the app answered cancel and the
        // delete, the unfavourite and the forget-originals all did nothing at
        // all, silently, however many times you pressed the red button.
        onClick: () => {
          if (action.onClick) action.onClick();
          if (!action.keepOpen) close();
        },
      }, action.label))));
  }

  const backdrop = el('div', { class: 'sheet-backdrop',
    onClick: (event) => { if (event.target === backdrop) close(); } }, panel);
  root.appendChild(backdrop);
  document.body.style.overflow = 'hidden';
  return { close, panel };
}

export function confirmSheet(message, { title, confirmLabel, danger } = {}) {
  return new Promise((resolve) => {
    let answered = false;
    sheet({
      title: title || 'Confirmar',
      body: el('p', { text: message }),
      actions: [
        { label: 'Cancelar', kind: 'secondary', onClick: () => { answered = true; resolve(false); } },
        { label: confirmLabel || 'Continuar', kind: danger ? 'danger' : undefined,
          onClick: () => { answered = true; resolve(true); } },
      ],
      onClose: () => { if (!answered) resolve(false); },
    });
  });
}

/* ---------------------------------------------------------------- widgets */

export function spinner(label) {
  return el('div', { style: { display: 'grid', placeItems: 'center', padding: '40px 0', gap: '12px' } }, [
    el('div', { class: 'spinner', role: 'status' }),
    label ? el('p', { class: 'muted', text: label, style: { margin: 0 } }) : null,
  ]);
}

export function progressBar(value) {
  const bar = el('div', { class: 'progress__bar' });
  bar.style.width = Math.round((value || 0) * 100) + '%';
  const wrap = el('div', { class: 'progress' }, bar);
  wrap.setValue = (v) => { bar.style.width = Math.round((v || 0) * 100) + '%'; };
  return wrap;
}

export function empty({ icon, title, text, action }) {
  return el('div', { class: 'empty' }, [
    el('div', { class: 'empty__icon', text: icon || '◌' }),
    el('h3', { text: title || '' }),
    text ? el('p', { class: 'muted', text }) : null,
    action ? el('a', { class: 'btn btn--sm', href: action.href, text: action.label,
      style: { marginTop: '14px', display: 'inline-flex' } }) : null,
  ]);
}

export function note(kind, title, text) {
  return el('div', { class: 'note note--' + kind }, [
    title ? el('strong', { text: title }) : null,
    text ? document.createTextNode(text) : null,
  ]);
}

export function kv(key, value) {
  return el('div', { class: 'kv' }, [
    el('span', { class: 'kv__k', text: key }),
    el('span', { class: 'kv__v', text: value }),
  ]);
}

export function field(labelText, input, hint) {
  return el('label', { class: 'field' }, [
    el('span', { text: labelText }),
    input,
    hint ? el('div', { class: 'field__hint', text: hint }) : null,
  ]);
}

/* ---------------------------------------------------------------- formats */

export function money(value) {
  const n = Number(value || 0);
  if (!isFinite(n)) return '-';
  if (n === 0) return 'gratis';
  if (n < 0.01) return '< 0,01 USD';
  return n.toFixed(2).replace('.', ',') + ' USD';
}

export function moneyExact(value) {
  const n = Number(value || 0);
  if (!isFinite(n)) return '-';
  return n.toFixed(n < 1 ? 4 : 2).replace('.', ',') + ' USD';
}

export function dateLabel(seconds) {
  if (!seconds) return '';
  const date = new Date(Number(seconds) * 1000);
  const diff = (Date.now() - date.getTime()) / 1000;
  if (diff < 60) return 'hace un momento';
  if (diff < 3600) return `hace ${Math.floor(diff / 60)} min`;
  if (diff < 86400) return `hace ${Math.floor(diff / 3600)} h`;
  if (diff < 172800) return 'ayer';
  return date.toLocaleDateString('es-ES', { day: 'numeric', month: 'short' });
}

export function lazyImg(src, alt) {
  return el('img', { src, alt: alt || '', loading: 'lazy', decoding: 'async' });
}

/* ------------------------------------------------------------- selection */

/**
 * The bar that appears while picking several images at once.
 *
 * Shared by the album and by favourites so the gesture is identical in both:
 * the client should never have to learn two ways of doing the same thing.
 * It sticks to the bottom above the tab bar, and it is the only thing on
 * screen that can act on the selection, so nothing destructive is ever one
 * stray tap away.
 */
export function selectionBar({ onSelectAll, onClear, onCancel, actions }) {
  const count = el('strong', { text: '0' });

  const buttons = (actions || []).map((action) =>
    el('button', {
      class: 'btn btn--sm' + (action.kind ? ' btn--' + action.kind : ' btn--secondary'),
      type: 'button',
      onClick: () => action.onClick(),
    }, action.label));

  // Placement lives in .selection-bar so it can measure itself against the
  // tabbar with env() and var(), which an inline style object cannot reach.
  const bar = el('div', { class: 'card selection-bar', hidden: true }, [
    el('div', { class: 'card__row', style: { flexWrap: 'wrap', gap: '10px' } }, [
      el('span', {}, [count, document.createTextNode(' seleccionadas')]),
      el('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap' } }, [
        el('button', { class: 'btn btn--ghost btn--sm', type: 'button',
          onClick: () => onSelectAll && onSelectAll() }, 'Todas'),
        el('button', { class: 'btn btn--ghost btn--sm', type: 'button',
          onClick: () => onClear && onClear() }, 'Ninguna'),
        el('button', { class: 'btn btn--ghost btn--sm', type: 'button',
          onClick: () => onCancel && onCancel() }, 'Cancelar'),
      ]),
    ]),
    el('div', { style: { display: 'flex', gap: '8px', marginTop: '10px' } }, buttons),
  ]);

  bar.setCount = (n) => {
    count.textContent = String(n);
    for (const button of buttons) button.disabled = n === 0;
  };
  bar.setCount(0);
  return bar;
}

/* --------------------------------------------------------- drag to scroll */

const DRAG_SLOP = 4;      // px of movement before it counts as a drag

/**
 * Grab-and-drag horizontal scrolling for a strip of cards.
 *
 * Touch is left alone: the browser's own inertial scrolling is better than
 * anything reimplemented here, and hijacking it breaks the phone. This is for
 * mouse and pen, where a horizontal overflow is otherwise only reachable by
 * hunting for a scrollbar or knowing about shift+wheel.
 *
 * The tricky part is that the strip is full of buttons. A drag ends with a
 * click event on whatever card is under the cursor, which would silently
 * select an option the user never meant to pick, so once the pointer has moved
 * past DRAG_SLOP the following click is swallowed in the capture phase.
 */
export function dragScroll(node) {
  if (!node || node.dataset.dragScroll === 'on') return node;
  node.dataset.dragScroll = 'on';

  let dragging = false;
  let startX = 0;
  let startLeft = 0;
  let moved = 0;
  // Set only for the click that immediately follows a drag, and cleared on the
  // next task whether or not that click arrives.  Deciding from `moved` alone
  // is wrong: a drag that ends outside the strip fires no click at all, the
  // stale distance survives, and the user's next real tap is eaten instead.
  let swallowNextClick = false;

  const end = () => {
    if (!dragging) return;
    dragging = false;
    node.classList.remove('scroller--grabbing');
    if (moved > DRAG_SLOP) {
      swallowNextClick = true;
      setTimeout(() => { swallowNextClick = false; }, 0);
    }
    moved = 0;
  };

  node.addEventListener('pointerdown', (event) => {
    if (event.pointerType === 'touch' || event.button !== 0) return;
    if (node.scrollWidth <= node.clientWidth) return;
    dragging = true;
    moved = 0;
    startX = event.clientX;
    startLeft = node.scrollLeft;
    // Deliberately no "grabbing" class here.  That class disables pointer
    // events on the children, and applying it on mere press makes the pointer
    // leave the card instantly: pointerup and click then retarget to the strip
    // and the card's own handler never runs, so a plain click selects nothing.
    // It goes on only once the pointer has actually travelled.
  });

  node.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    const dx = event.clientX - startX;
    if (Math.abs(dx) > moved) moved = Math.abs(dx);
    if (moved > DRAG_SLOP) {
      node.classList.add('scroller--grabbing');
      node.scrollLeft = startLeft - dx;
      event.preventDefault();
    }
  });

  node.addEventListener('pointerup', end);
  node.addEventListener('pointercancel', end);
  node.addEventListener('pointerleave', end);

  node.addEventListener('click', (event) => {
    if (swallowNextClick) {
      swallowNextClick = false;
      event.stopPropagation();
      event.preventDefault();
    }
  }, true);

  // A vertical wheel over a horizontal strip should move it sideways; without
  // this the page scrolls away underneath the cards the user is looking at.
  node.addEventListener('wheel', (event) => {
    if (node.scrollWidth <= node.clientWidth) return;
    if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
    node.scrollLeft += event.deltaY;
    event.preventDefault();
  }, { passive: false });

  return node;
}

export function pct(value) {
  return Math.round(Number(value || 0) * 100) + '%';
}
