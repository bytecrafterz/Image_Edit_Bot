/* Settings, arranged around the client's actual worry: what is this costing me
   and will it warn me before the money runs out. */

import { api } from '../api.js';
import { store } from '../store.js';
import {
  el, clear, note, toast, spinner, sheet, kv, field, money, moneyExact,
  dateLabel, confirmSheet,
} from '../ui.js';
import { loadSession } from '../app.js';

let data = null;
let usage = null;
let alerts = [];

const PROVIDER_ES = {
  anthropic: 'Anthropic (analisis y control de calidad)',
  fal: 'fal.ai (generacion de imagenes)',
  local: 'Motor local (gratuito)',
};

async function load() {
  const [settings, use, alertData] = await Promise.all([
    api.get('/api/settings'),
    api.get('/api/settings/usage?days=30'),
    api.get('/api/settings/alerts?limit=20'),
  ]);
  data = settings;
  usage = use;
  alerts = alertData.alerts || [];
}

function sectionTitle(text) {
  return el('div', { class: 'section__title', text });
}

/* --------------------------------------------------------------- balances */

function balancesSection(view) {
  const wrap = el('div', { class: 'section' }, sectionTitle('Saldo y avisos'));

  for (const [provider, info] of Object.entries(data.balances || {})) {
    if (provider === 'local') continue;
    const status = info.status || 'ok';
    const kind = status === 'zero' || status === 'critical' ? 'danger'
      : status === 'low' ? 'warn' : 'ok';
    wrap.appendChild(el('div', { class: 'card' }, [
      el('div', { class: 'card__row' }, [
        el('strong', { text: PROVIDER_ES[provider] || provider }),
        // "gratis" and "sin saldo" are opposite meanings and money() renders
        // zero as the first one; on a paid provider an empty balance must not
        // read as though everything were free.
        el('span', { class: 'chip chip--' + kind,
          text: info.balance === null ? 'sin coste'
            : (info.balance <= 0 ? 'Sin saldo' : money(info.balance)) }),
      ]),
      info.photos_left !== null && info.photos_left !== undefined
        ? kv('Alcanza para', info.photos_left > 0
            ? `unas ${info.photos_left} fotos`
            : 'nada: hay que recargar')
        : null,
      kv('Gastado hoy', moneyExact(info.spent_today || 0)),
      kv('Ultimos 30 dias', moneyExact(info.spent_30d || 0)),
      el('button', { class: 'btn btn--secondary', type: 'button',
        style: { marginTop: '10px' },
        onClick: () => rechargeSheet(view, provider) }, 'Registrar recarga'),
    ]));
  }

  wrap.appendChild(el('div', { class: 'card' }, [
    el('div', { class: 'switch' }, [
      el('span', {}, [
        el('div', { text: 'Avisarme cuando baje el saldo' }),
        el('div', { class: 'tiny', text: 'Recibiras un aviso antes de quedarte a cero.' }),
      ]),
      el('input', { type: 'checkbox', checked: data.settings.notify_low_balance,
        onChange: (e) => save({ notify_low_balance: e.target.checked }) }),
    ]),
    field('Avisar por debajo de (USD)',
      el('input', { type: 'number', step: '0.5', min: '0',
        value: data.settings.low_balance_threshold_usd,
        onChange: (e) => save({ low_balance_threshold_usd: Number(e.target.value) }) })),
  ]));

  if (alerts.length) {
    const list = el('div', { class: 'card' }, [
      el('div', { class: 'card__row' }, [
        el('strong', { text: 'Avisos recientes' }),
        el('button', { class: 'btn btn--ghost btn--sm', type: 'button',
          onClick: async () => {
            await api.post('/api/settings/alerts/read-all');
            store.set({ alertsUnread: 0 });
            toast('Marcados como leidos');
          } }, 'Marcar leidos'),
      ]),
      ...alerts.slice(0, 6).map((alert) => el('div', {
        class: 'note note--' + (alert.level === 'critical' ? 'danger'
          : alert.level === 'warning' ? 'warn' : 'info'),
      }, [
        el('strong', { text: alert.title || 'Aviso' }),
        document.createTextNode(alert.message || ''),
        el('div', { class: 'tiny', text: dateLabel(alert.created_at) }),
      ])),
    ]);
    wrap.appendChild(list);
  }

  return wrap;
}

function rechargeSheet(view, provider) {
  const amount = el('input', { type: 'number', step: '1', min: '1', value: '10' });
  sheet({
    title: 'Registrar una recarga',
    subtitle: PROVIDER_ES[provider] || provider,
    body: el('div', {}, [
      note('info', 'Esto no cobra nada',
        'Esta pantalla solo anota el saldo que ya has anadido en la web de '
        + (provider === 'fal' ? 'fal.ai' : 'Anthropic') + '. La aplicacion nunca '
        + 'cobra en tu tarjeta por su cuenta.'),
      field('Importe anadido (USD)', amount),
    ]),
    actions: [
      { label: 'Cancelar', kind: 'secondary' },
      { label: 'Anotar', onClick: async () => {
          try {
            const result = await api.post('/api/settings/recharge',
              { provider, amount_usd: Number(amount.value) });
            toast(`Saldo anotado: ${money(result.balance)}`, 'ok');
            await loadSession();
            await refresh(view);
          } catch (err) { toast(err.message, 'danger'); }
        } },
    ],
  });
}

/* ------------------------------------------------------------------ usage */

function usageSection() {
  const trend = usage.trend || [];
  const max = Math.max(1, ...trend.map((d) => d.count || 0));
  const bars = el('div', { class: 'bars' }, trend.map((day) => {
    const bar = el('div', { class: 'bars__bar', title: `${day.day}: ${day.count}` });
    bar.style.height = Math.max(2, ((day.count || 0) / max) * 66) + 'px';
    return bar;
  }));

  return el('div', { class: 'section' }, [
    sectionTitle('Uso y costes (30 dias)'),
    el('div', { class: 'card' }, [
      kv('Intentos', String(usage.attempts || 0)),
      kv('Fotos conseguidas', String(usage.accepted || 0)),
      kv('Descartadas por el robot', String(usage.rejected || 0)),
      kv('Reparadas sin regenerar', String(usage.repaired || 0)),
      el('div', { class: 'kv', style: { borderTop: '2px solid var(--line)',
        marginTop: '6px', paddingTop: '12px' } }, [
        el('span', { class: 'kv__k', text: 'Intentos por foto conseguida' }),
        el('span', { class: 'kv__v',
          text: usage.attempts_per_photo ? String(usage.attempts_per_photo) : '-' }),
      ]),
      kv('Gasto total', moneyExact(usage.total_usd || 0)),
      kv('Coste por foto', usage.cost_per_photo_usd
        ? moneyExact(usage.cost_per_photo_usd) : 'gratis'),
      trend.length ? bars : null,
      trend.length ? el('p', { class: 'tiny', style: { margin: '4px 0 0' },
        text: 'Imagenes por dia' }) : null,
    ]),
    (usage.reject_reasons || []).length ? el('div', { class: 'card' }, [
      el('div', { class: 'section__title', text: 'Por que se descartaron' }),
      ...usage.reject_reasons.map((r) => kv(r.reason, String(r.count))),
    ]) : null,
  ]);
}

/* ----------------------------------------------------------------- limits */

function limitsSection() {
  return el('div', { class: 'section' }, [
    sectionTitle('Limites de gasto'),
    el('div', { class: 'card' }, [
      field('Maximo por dia (USD)',
        el('input', { type: 'number', step: '0.5', min: '0',
          value: data.limits.daily_usd,
          onChange: (e) => save({ daily_budget_usd: Number(e.target.value) }) }),
        'Cuando lo alcances, el robot se detiene.'),
      field('Maximo por mes (USD)',
        el('input', { type: 'number', step: '1', min: '0',
          value: data.limits.monthly_usd,
          onChange: (e) => save({ monthly_budget_usd: Number(e.target.value) }) })),
      kv('Plan', data.plan === 'paid' ? 'De pago' : 'Gratuito'),
      kv('Generaciones gratis hoy',
        `${data.limits.free_used_today} de ${data.limits.free_quota_daily}`),
    ]),
  ]);
}

/* --------------------------------------------------------------- behaviour */

function behaviourSection() {
  const help = data.strictness_help || {};
  const strict = el('select', {
    onChange: (e) => save({ strictness: e.target.value }),
  }, ['suave', 'normal', 'estricto'].map((key) =>
    el('option', { value: key, selected: data.settings.strictness === key },
      key[0].toUpperCase() + key.slice(1))));

  const helpText = el('div', { class: 'field__hint',
    text: help[data.settings.strictness] || '' });
  strict.addEventListener('change', () => {
    helpText.textContent = help[strict.value] || '';
  });

  return el('div', { class: 'section' }, [
    sectionTitle('Calidad y comportamiento'),
    el('div', { class: 'card' }, [
      field('Vistas previas por defecto',
        el('input', { type: 'number', min: '1', max: '12',
          value: data.settings.default_n_previews,
          onChange: (e) => save({ default_n_previews: Number(e.target.value) }) })),
      field('Calidad por defecto',
        el('select', { onChange: (e) => save({ default_quality: e.target.value }) },
          ['preview', 'standard', 'high', 'max'].map((q) =>
            el('option', { value: q, selected: data.settings.default_quality === q },
              { preview: 'Vista previa', standard: 'Estandar', high: 'Alta',
                max: 'Maxima' }[q])))),
      el('div', { class: 'switch' }, [
        el('span', {}, [
          el('div', { text: 'Reparar sin regenerar' }),
          el('div', { class: 'tiny',
            text: 'Corrige solo la zona defectuosa. Ahorra dinero.' }),
        ]),
        el('input', { type: 'checkbox', checked: data.settings.autorepair,
          onChange: (e) => save({ autorepair: e.target.checked }) }),
      ]),
      field('Rondas de reparacion',
        el('input', { type: 'number', min: '0', max: '3',
          value: data.settings.max_repair_rounds,
          onChange: (e) => save({ max_repair_rounds: Number(e.target.value) }) })),
      field('Reintentos por imagen',
        el('input', { type: 'number', min: '0', max: '3',
          value: data.settings.max_retries,
          onChange: (e) => save({ max_retries: Number(e.target.value) }) })),
      el('label', { class: 'field' }, [
        el('span', { text: 'Estrictez del control de identidad' }),
        strict, helpText,
      ]),
    ]),
  ]);
}

/* ------------------------------------------------------------------- keys */

function keysSection(view) {
  const rows = [];
  for (const [provider, info] of Object.entries(data.keys || {})) {
    if (!['anthropic', 'fal'].includes(provider)) continue;
    const input = el('input', { type: 'password', placeholder: 'Pega aqui la clave' });
    rows.push(el('div', { class: 'card' }, [
      el('div', { class: 'card__row' }, [
        el('strong', { text: PROVIDER_ES[provider] || provider }),
        el('span', { class: 'chip chip--' + (info.present ? 'ok' : 'warn'),
          text: info.present ? 'Configurada' : 'Sin clave' }),
      ]),
      info.present
        ? kv('Clave', info.hint || 'oculta')
        : el('p', { class: 'tiny',
            text: 'Sin esta clave se usara el motor local gratuito.' }),
      info.from_env ? el('p', { class: 'tiny',
        text: 'Viene del entorno del servidor.' }) : null,
      store.isAdmin() ? el('div', { style: { marginTop: '10px' } }, [
        input,
        el('button', { class: 'btn btn--secondary', type: 'button',
          style: { marginTop: '8px' },
          onClick: async () => {
            try {
              const result = await api.post('/api/settings/keys',
                { provider, key: input.value.trim() });
              toast(result.message || 'Guardada', 'ok');
              input.value = '';
              await refresh(view);
            } catch (err) { toast(err.message, 'danger'); }
          } }, 'Guardar clave'),
      ]) : null,
    ]));
  }
  return el('div', { class: 'section' }, [
    sectionTitle('Claves de IA'),
    note('info', 'Para que sirven',
      'La clave de Anthropic se usa para leer tus fotos y revisar los resultados. '
      + 'La de fal.ai genera las imagenes. Sin claves el sistema funciona igual, '
      + 'con el motor local gratuito.'),
    ...rows,
  ]);
}

/* ---------------------------------------------------------------- account */

function accountSection(view) {
  const user = store.get('user') || {};
  const current = el('input', { type: 'password', placeholder: 'Contrasena actual' });
  const next = el('input', { type: 'password', placeholder: 'Nueva contrasena' });

  return el('div', { class: 'section' }, [
    sectionTitle('Cuenta'),
    el('div', { class: 'card' }, [
      kv('Nombre', user.display_name || '-'),
      kv('Correo', user.email || '-'),
      kv('Rol', user.role === 'admin' ? 'Administradora' : 'Usuaria'),
      el('button', { class: 'btn btn--secondary', type: 'button',
        style: { marginTop: '12px' },
        onClick: () => sheet({
          title: 'Cambiar contrasena',
          body: el('div', {}, [field('Actual', current), field('Nueva', next)]),
          actions: [
            { label: 'Cancelar', kind: 'secondary' },
            { label: 'Cambiar', onClick: async () => {
                try {
                  await api.post('/api/auth/password', {
                    current_password: current.value, new_password: next.value });
                  toast('Contrasena cambiada', 'ok');
                } catch (err) { toast(err.message, 'danger'); }
              } },
          ],
        }) }, 'Cambiar contrasena'),
      el('button', { class: 'btn btn--ghost', type: 'button',
        onClick: async () => {
          const ok = await confirmSheet('Se cerrara tu sesion.',
            { title: 'Salir', confirmLabel: 'Salir' });
          if (!ok) return;
          try { await api.post('/api/auth/logout'); } catch { /* ignore */ }
          api.setToken('');
          store.set({ user: null });
          location.hash = '#/login';
        } }, 'Cerrar sesion'),
    ]),
  ]);
}

async function save(patch) {
  try {
    const result = await api.put('/api/settings', patch);
    data.settings = result.settings;
    toast('Guardado', 'ok');
  } catch (err) { toast(err.message, 'danger'); }
}

async function refresh(view) {
  await load();
  render(view);
}

function render(view) {
  clear(view);
  view.appendChild(el('h1', { text: 'Ajustes' }));
  view.appendChild(balancesSection(view));
  view.appendChild(usageSection());
  view.appendChild(limitsSection());
  view.appendChild(behaviourSection());
  view.appendChild(keysSection(view));
  view.appendChild(accountSection(view));
}

export default {
  async mount(view) {
    view.appendChild(spinner());
    try { await load(); } catch (err) {
      clear(view);
      view.appendChild(note('danger', 'No se pudo cargar', err.message));
      return;
    }
    render(view);
  },
  unmount() {},
};
