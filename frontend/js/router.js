/* Hash router with a page lifecycle.  Every page returns unmount() so its
   polling timers are always cleared - a forgotten interval on a phone means a
   flat battery and a stream of pointless requests. */

import { store } from './store.js';

const pages = new Map();
let current = null;
let currentPath = '';

export const router = {
  register(path, page) { pages.set(path, page); },

  start() {
    window.addEventListener('hashchange', () => router.render());
    router.render();
  },

  go(path) {
    if (location.hash === path) router.render();
    else location.hash = path;
  },

  path() { return currentPath; },

  async render() {
    const raw = location.hash.replace(/^#/, '') || '/generate';
    const [path, query] = raw.split('?');
    const params = Object.fromEntries(new URLSearchParams(query || ''));
    const user = store.get('user');

    if (!user && path !== '/login') { location.hash = '#/login'; return; }
    if (user && path === '/login') { location.hash = '#/generate'; return; }
    if (path === '/admin' && !store.isAdmin()) { location.hash = '#/generate'; return; }

    const page = pages.get(path) || pages.get('/generate');
    if (!page) return;

    if (current && current.unmount) {
      try { current.unmount(); } catch (err) { console.error(err); }
    }

    const view = document.getElementById('view');
    view.textContent = '';
    currentPath = path;
    current = page;

    for (const tab of document.querySelectorAll('.tab')) {
      const active = tab.getAttribute('href') === '#' + path;
      if (active) tab.setAttribute('aria-current', 'page');
      else tab.removeAttribute('aria-current');
    }

    window.scrollTo(0, 0);
    try {
      await page.mount(view, params);
    } catch (err) {
      console.error(err);
      view.textContent = '';
      const box = document.createElement('div');
      box.className = 'note note--danger';
      box.textContent = err && err.message ? err.message : 'Algo ha fallado.';
      view.appendChild(box);
    }
  },
};
