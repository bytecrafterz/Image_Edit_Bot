/* A very small observable store: the app has one user, one balance and one
   active run, and that is not worth a framework. */

const state = {
  user: null,
  balances: {},
  alertsUnread: 0,
  activeRun: null,
  selection: {},
};

const listeners = new Set();

export const store = {
  get(key) { return key ? state[key] : state; },

  set(patch) {
    Object.assign(state, patch);
    for (const fn of listeners) {
      try { fn(state); } catch (err) { console.error(err); }
    }
  },

  subscribe(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },

  isAdmin() { return Boolean(state.user && state.user.role === 'admin'); },

  persist(key, value) {
    try { localStorage.setItem('pr_' + key, JSON.stringify(value)); } catch { /* ignore */ }
  },

  restore(key, fallback) {
    try {
      const raw = localStorage.getItem('pr_' + key);
      return raw ? JSON.parse(raw) : fallback;
    } catch { return fallback; }
  },
};
