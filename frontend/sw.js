/* Caches only the static shell.  Never API responses and never photographs:
   a stale balance or a cached private image would both be worse than a
   spinner.

   TWO THINGS THIS USED TO DO WRONG, both found on 2026-09-10 after the move
   to a public https origin (a service worker only registers over https, so
   neither could happen on localhost):

   1. It cached whatever came back for /js/ and /css/ without looking at the
      status.  While the server restarts, nginx answers those URLs with its
      Spanish 503 page - and that HTML was stored AS the JavaScript file,
      cache-first, under a version string that never changed.  The app then
      failed to start on her phone until the cache was cleared by hand.
   2. Cache-first plus a constant VERSION means a deploy is never seen: the
      phone keeps running the JS it cached the first day.  main.py serves the
      shell with Cache-Control: no-cache for exactly this reason, and the
      service worker sat in front of it and undid it.

   So code is NETWORK-FIRST (the cache is only the offline fallback) and only
   an ok response is ever stored.  Icons stay cache-first: they do not change
   and they are the only thing here worth saving a request for. */

const VERSION = 'photorobot-v2';
const SHELL = [
  '/', '/index.html', '/css/app.css',
  '/js/app.js', '/js/api.js', '/js/ui.js', '/js/router.js', '/js/store.js',
  '/js/i18n.js', '/js/onboarding.js',
  '/js/pages/login.js', '/js/pages/generate.js', '/js/pages/album.js',
  '/js/pages/favorites.js', '/js/pages/originals.js', '/js/pages/settings.js',
  '/js/pages/admin.js',
  '/icons/icon-192.png', '/icons/icon-512.png', '/icons/apple-touch-icon.png',
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

/* Store a response only when it is a real, successful one from our origin. */
function keep(request, response) {
  if (response && response.ok && response.type === 'basic') {
    const copy = response.clone();
    caches.open(VERSION).then((cache) => cache.put(request, copy)).catch(() => {});
  }
  return response;
}

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== location.origin) return;
  if (url.pathname.startsWith('/api/')) return;   // always live

  const isCode = ['/css/', '/js/'].some((prefix) => url.pathname.startsWith(prefix));
  const isIcon = url.pathname.startsWith('/icons/');

  if (isCode) {
    // Network first: a deploy is seen on the next load.  The cache only
    // answers when the network cannot, and never holds an error page.
    event.respondWith(
      fetch(event.request)
        .then((response) => keep(event.request, response))
        .catch(() => caches.match(event.request))
    );
    return;
  }

  if (isIcon) {
    event.respondWith(
      caches.match(event.request).then((hit) => hit || fetch(event.request)
        .then((response) => keep(event.request, response)))
    );
    return;
  }

  event.respondWith(
    fetch(event.request).catch(() => caches.match('/index.html'))
  );
});
