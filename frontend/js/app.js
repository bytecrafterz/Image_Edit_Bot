/* Boot: load the session, register the pages, start the router, and keep the
   balance chip honest.  The chip is the client's early warning system, so it
   changes colour before the money runs out rather than after. */

import { api } from './api.js';
import { router } from './router.js';
import { store } from './store.js';
import { el, clear, money, sheet, toast, dateLabel } from './ui.js';
import { setLocale } from './i18n.js';

import loginPage from './pages/login.js';
import generatePage from './pages/generate.js';
import albumPage from './pages/album.js';
import favoritesPage from './pages/favorites.js';
import originalsPage from './pages/originals.js';
import settingsPage from './pages/settings.js';
import adminPage from './pages/admin.js';

const ALERT_POLL_MS = 60000;
let alertTimer = null;

router.register('/login', loginPage);
router.register('/generate', generatePage);
router.register('/album', albumPage);
router.register('/favorites', favoritesPage);
router.register('/originals', originalsPage);
router.register('/settings', settingsPage);
router.register('/admin', adminPage);

function worstBalance(balances) {
  let worst = null;
  for (const [name, info] of Object.entries(balances || {})) {
    if (!info || info.balance === null || info.balance === undefined) continue;
    if (!worst || info.balance < worst.balance) worst = { name, ...info };
  }
  return worst;
}

function renderChrome() {
  const user = store.get('user');
  const bar = document.getElementById('appbar');
  const tabbar = document.getElementById('tabbar');
  const adminTab = document.getElementById('tab-admin');

  const signedIn = Boolean(user);
  bar.hidden = !signedIn;
  tabbar.hidden = !signedIn;
  adminTab.hidden = !store.isAdmin();

  const badge = document.getElementById('alert-badge');
  const unread = store.get('alertsUnread') || 0;
  badge.hidden = unread === 0;
  badge.textContent = String(unread);

  const chip = document.getElementById('balance-chip');
  const worst = worstBalance(store.get('balances'));
  if (!signedIn || !worst) {
    chip.hidden = true;
    return;
  }
  chip.hidden = false;
  const kind = worst.status === 'zero' || worst.status === 'critical' ? 'danger'
    : worst.status === 'low' ? 'warn' : 'ok';
  chip.className = 'chip chip--balance chip--' + kind;
  chip.textContent = worst.balance <= 0
    ? 'Sin saldo'
    : money(worst.balance);
  chip.title = `Saldo de ${worst.name}`;
}

async function refreshAlerts() {
  if (!store.get('user')) return;
  try {
    const data = await api.get('/api/settings/alerts?limit=20');
    store.set({ alertsUnread: data.unread || 0, alerts: data.alerts || [] });
  } catch { /* a failed poll must stay silent */ }
  refreshBalances();
}

/* The chip in the app bar used to be read once, at login, and then stood
   still through every run and every top-up: she watched 0,04 leave in the
   progress card while the bar kept the old figure.  Re-read whenever money
   could have moved - a run finishing, a recharge written down - and with
   the alerts poll for everything else. */
export async function refreshBalances() {
  if (!store.get('user')) return;
  try {
    const data = await api.get('/api/auth/me');
    store.set({ balances: data.balances || {} });
  } catch { /* stale is better than a spinner in the app bar */ }
}

function showAlerts() {
  const alerts = store.get('alerts') || [];
  sheet({
    title: 'Avisos',
    subtitle: alerts.length ? null : 'No tienes avisos.',
    body: alerts.length
      ? alerts.map((alert) => el('div', {
          class: 'note note--' + (alert.level === 'critical' ? 'danger'
            : alert.level === 'warning' ? 'warn' : 'info'),
        }, [
          el('strong', { text: alert.title || 'Aviso' }),
          document.createTextNode(alert.message || ''),
          el('div', { class: 'tiny', text: dateLabel(alert.created_at),
            style: { marginTop: '6px' } }),
        ]))
      : el('p', { class: 'muted', text: 'Cuando el saldo baje o algo requiera tu atencion, aparecera aqui.' }),
    actions: alerts.length ? [
      { label: 'Ir a Ajustes', kind: 'secondary',
        onClick: () => { location.hash = '#/settings'; } },
      { label: 'Marcar como leidas',
        onClick: async () => {
          try {
            await api.post('/api/settings/alerts/read-all');
            store.set({ alertsUnread: 0 });
            renderChrome();
          } catch (err) { toast(err.message, 'danger'); }
        } },
    ] : [{ label: 'Cerrar', kind: 'secondary' }],
  });
}

/* The server takes about ten seconds to come back after a deploy, and nginx
   answers 503 meanwhile.  This used to treat that exactly like "your session
   is invalid" and throw the token away, so every restart sent her back to the
   login screen - the one screen on which nothing explains why.  A 401 is the
   only answer that means the session is gone (api.js already clears the token
   on it); everything else - no network, a timeout, a 5xx - is waited out, and
   if it never comes back the token is KEPT so the next load logs her in. */
const SESSION_RETRIES = 4;
const SESSION_RETRY_MS = 3000;

export async function loadSession() {
  if (!api.isLoggedIn()) {
    store.set({ user: null, balances: {}, alertsUnread: 0 });
    return null;
  }
  for (let attempt = 0; ; attempt++) {
    try {
      const data = await api.get('/api/auth/me');
      store.set({
        user: data.user,
        balances: data.balances || {},
        alertsUnread: data.alerts_unread || 0,
        defaultProfile: data.default_profile || null,
      });
      if (data.user && data.user.locale) setLocale(data.user.locale);
      return data.user;
    } catch (err) {
      const status = err && typeof err.status === 'number' ? err.status : 0;
      if (status === 401 || status === 403) {
        api.setToken('');
        store.set({ user: null });
        return null;
      }
      if (attempt < SESSION_RETRIES) {
        await new Promise((resolve) => setTimeout(resolve, SESSION_RETRY_MS));
        continue;
      }
      store.set({ user: null });
      return null;
    }
  }
}

async function boot() {
  store.subscribe(renderChrome);
  document.getElementById('alert-btn').addEventListener('click', showAlerts);
  document.getElementById('balance-chip').addEventListener('click', () => {
    location.hash = '#/settings';
  });
  // Always in view, not only at the bottom of Ajustes: on a shared phone the
  // way out has to be one tap from any screen.  The server forgets the
  // session first, so a token left in the browser is worthless afterwards.
  document.getElementById('logout-btn').addEventListener('click', async () => {
    try { await api.post('/api/auth/logout'); } catch { /* already gone */ }
    api.setToken('');
    if (alertTimer) { clearInterval(alertTimer); alertTimer = null; }
    store.set({ user: null, balances: {}, alertsUnread: 0 });
    location.hash = '#/login';
  });

  await loadSession();
  renderChrome();
  router.start();

  if (store.get('user')) {
    refreshAlerts();
    alertTimer = setInterval(refreshAlerts, ALERT_POLL_MS);
  }

  if ('serviceWorker' in navigator && location.protocol === 'https:') {
    navigator.serviceWorker.register('/sw.js').catch(() => { /* optional */ });
  }
}

window.addEventListener('beforeunload', () => {
  if (alertTimer) clearInterval(alertTimer);
});

boot();
