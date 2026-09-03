/* Caches only the static shell.  Never API responses and never photographs:
   a stale balance or a cached private image would both be worse than a
   spinner. */

const VERSION = 'photorobot-v1';
const SHELL = [
  '/', '/index.html', '/css/app.css',
  '/js/app.js', '/js/api.js', '/js/ui.js', '/js/router.js', '/js/store.js',
  '/js/i18n.js',
  '/js/pages/login.js', '/js/pages/generate.js', '/js/pages/album.js',
  '/js/pages/favorites.js', '/js/pages/originals.js', '/js/pages/settings.js',
  '/js/pages/admin.js',
  '/icons/icon-192.png', '/icons/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(VERSION).then((cache) => cache.addAll(SHELL)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((key) => key !== VERSION).map((key) => caches.delete(key))
    ))
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== location.origin) return;
  if (url.pathname.startsWith('/api/')) return;   // always live

  const cacheable = ['/css/', '/js/', '/icons/', '/assets/']
    .some((prefix) => url.pathname.startsWith(prefix));

  if (cacheable) {
    event.respondWith(
      caches.match(event.request).then((hit) => hit || fetch(event.request)
        .then((response) => {
          const copy = response.clone();
          caches.open(VERSION).then((cache) => cache.put(event.request, copy));
          return response;
        }))
    );
    return;
  }

  event.respondWith(
    fetch(event.request).catch(() => caches.match('/index.html'))
  );
});
