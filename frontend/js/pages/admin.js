/* Administration console.  Accounts, platform numbers, providers, audit. */

import { api } from '../api.js';
import { store } from '../store.js';
import {
  el, clear, note, toast, spinner, sheet, kv, field, moneyExact, dateLabel,
  confirmSheet, empty,
} from '../ui.js';

let tab = 'users';

const TABS = [
  { key: 'users', label: 'Usuarios' },
  { key: 'stats', label: 'Estadisticas' },
  { key: 'providers', label: 'Proveedores' },
  { key: 'audit', label: 'Auditoria' },
  { key: 'maintenance', label: 'Mantenimiento' },
];

const STATUS_ES = { active: 'Activa', pending: 'Pendiente', suspended: 'Suspendida' };

async function render(view) {
  clear(view);
  view.appendChild(el('h1', { text: 'Administracion' }));

  const chips = el('div', { class: 'chips', style: { marginBottom: '16px' } });
  for (const item of TABS) {
    chips.appendChild(el('button', {
      class: 'chip' + (tab === item.key ? ' chip--on' : ''),
      type: 'button',
      onClick: () => { tab = item.key; render(view); },
    }, item.label));
  }
  view.appendChild(chips);

  const body = el('div');
  view.appendChild(body);
  body.appendChild(spinner());

  try {
    if (tab === 'users') await renderUsers(view, body);
    else if (tab === 'stats') await renderStats(body);
    else if (tab === 'providers') await renderProviders(body);
    else if (tab === 'audit') await renderAudit(body);
    else await renderMaintenance(view, body);
  } catch (err) {
    clear(body);
    if (err.status === 403) {
      body.appendChild(empty({ icon: '◇', title: 'No tienes permisos',
        text: 'Esta seccion es solo para administradores.' }));
    } else {
      body.appendChild(note('danger', 'No se pudo cargar', err.message));
    }
  }
}

/* ------------------------------------------------------------------ users */

async function renderUsers(view, body) {
  const data = await api.get('/api/admin/users');
  clear(body);

  const search = el('input', { type: 'text', placeholder: 'Buscar por correo o nombre',
    onInput: (event) => {
      const term = event.target.value.toLowerCase();
      for (const card of body.querySelectorAll('[data-email]')) {
        card.hidden = term && !card.dataset.email.includes(term)
          && !(card.dataset.name || '').includes(term);
      }
    } });
  body.appendChild(el('div', { style: { marginBottom: '14px' } }, search));

  if (!data.users.length) {
    body.appendChild(empty({ icon: '◇', title: 'No hay usuarios' }));
    return;
  }

  for (const user of data.users) {
    const badge = user.status === 'active' ? 'ok'
      : user.status === 'pending' ? 'warn' : 'danger';
    body.appendChild(el('div', {
      class: 'card',
      dataset: { email: (user.email || '').toLowerCase(),
        name: (user.display_name || '').toLowerCase() },
    }, [
      el('div', { class: 'card__row' }, [
        el('div', {}, [
          el('strong', { text: user.display_name || user.email }),
          el('div', { class: 'tiny', text: user.email }),
        ]),
        el('span', { class: 'chip chip--' + badge,
          text: STATUS_ES[user.status] || user.status }),
      ]),
      kv('Rol', user.role === 'admin' ? 'Administrador' : 'Usuario'),
      kv('Fotos originales', String(user.originals || 0)),
      kv('Imagenes', String(user.images || 0)),
      kv('Gasto 30 dias', moneyExact(user.spend_30d || 0)),
      kv('Ultimo acceso', user.last_login_at ? dateLabel(user.last_login_at) : 'nunca'),
      el('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap',
        marginTop: '12px' } }, [
        user.status === 'pending' ? el('button', {
          class: 'btn btn--sm', type: 'button',
          onClick: () => act(view, `/api/admin/users/${user.id}/approve`, 'Aprobada'),
        }, 'Aprobar') : null,
        user.status !== 'suspended' ? el('button', {
          class: 'btn btn--secondary btn--sm', type: 'button',
          onClick: () => act(view, `/api/admin/users/${user.id}/suspend`, 'Suspendida'),
        }, 'Suspender') : null,
        el('button', { class: 'btn btn--secondary btn--sm', type: 'button',
          onClick: () => editUser(view, user) }, 'Editar'),
        el('button', { class: 'btn btn--secondary btn--sm', type: 'button',
          onClick: () => resetPassword(user) }, 'Contrasena'),
        el('button', { class: 'btn btn--danger btn--sm', type: 'button',
          onClick: () => deleteUser(view, user) }, 'Eliminar'),
      ]),
    ]));
  }
}

async function act(view, path, message) {
  try {
    await api.post(path);
    toast(message, 'ok');
    render(view);
  } catch (err) { toast(err.message, 'danger'); }
}

function editUser(view, user) {
  const role = el('select', {}, ['user', 'admin'].map((r) =>
    el('option', { value: r, selected: user.role === r },
      r === 'admin' ? 'Administrador' : 'Usuario')));
  const plan = el('select', {}, ['free', 'paid'].map((p) =>
    el('option', { value: p, selected: user.plan === p },
      p === 'paid' ? 'De pago' : 'Gratuito')));
  const daily = el('input', { type: 'number', step: '0.5', min: '0',
    value: user.daily_limit_usd });
  const monthly = el('input', { type: 'number', step: '1', min: '0',
    value: user.monthly_limit_usd });
  const quota = el('input', { type: 'number', min: '0',
    value: user.free_quota_daily });

  sheet({
    title: 'Editar usuario',
    subtitle: user.email,
    body: el('div', {}, [
      field('Rol', role), field('Plan', plan),
      field('Limite diario (USD)', daily),
      field('Limite mensual (USD)', monthly),
      field('Generaciones gratis por dia', quota),
    ]),
    actions: [
      { label: 'Cancelar', kind: 'secondary' },
      { label: 'Guardar', onClick: async () => {
          try {
            await api.patch(`/api/admin/users/${user.id}`, {
              role: role.value, plan: plan.value,
              daily_limit_usd: Number(daily.value),
              monthly_limit_usd: Number(monthly.value),
              free_quota_daily: Number(quota.value),
            });
            toast('Guardado', 'ok');
            render(view);
          } catch (err) { toast(err.message, 'danger'); }
        } },
    ],
  });
}

async function resetPassword(user) {
  const ok = await confirmSheet(
    `Se generara una contrasena temporal para ${user.email} y se cerraran sus sesiones.`,
    { title: 'Restablecer contrasena', confirmLabel: 'Restablecer' });
  if (!ok) return;
  try {
    const data = await api.post(`/api/admin/users/${user.id}/reset-password`);
    const code = el('input', { type: 'text', value: data.temporary_password,
      readonly: true });
    sheet({
      title: 'Contrasena temporal',
      subtitle: data.message,
      body: el('div', {}, [code]),
      actions: [{ label: 'Copiar', onClick: () => {
        code.select();
        navigator.clipboard?.writeText(data.temporary_password)
          .then(() => toast('Copiada', 'ok'))
          .catch(() => toast('Copiala a mano'));
      } }, { label: 'Cerrar', kind: 'secondary' }],
    });
  } catch (err) { toast(err.message, 'danger'); }
}

async function deleteUser(view, user) {
  const confirmField = el('input', { type: 'text',
    placeholder: 'Escribe el correo para confirmar' });
  sheet({
    title: 'Eliminar usuario',
    subtitle: 'Se borraran su cuenta, sus fotos y sus imagenes. No hay vuelta atras.',
    body: el('div', {}, [
      note('danger', null, `Vas a eliminar ${user.email}.`),
      field('Confirma escribiendo el correo', confirmField),
    ]),
    actions: [
      { label: 'Cancelar', kind: 'secondary' },
      { label: 'Eliminar', kind: 'danger', onClick: async () => {
          if (confirmField.value.trim().toLowerCase() !== user.email.toLowerCase()) {
            toast('El correo no coincide', 'danger');
            return;
          }
          try {
            await api.del(`/api/admin/users/${user.id}`);
            toast('Usuario eliminado');
            render(view);
          } catch (err) { toast(err.message, 'danger'); }
        } },
    ],
  });
}

/* ------------------------------------------------------------------ stats */

async function renderStats(body) {
  const stats = await api.get('/api/admin/stats');
  clear(body);

  body.appendChild(el('div', { class: 'card' }, [
    el('div', { class: 'section__title', text: 'Usuarios' }),
    ...Object.entries(stats.users_by_status || {}).map(([k, v]) =>
      kv(STATUS_ES[k] || k, String(v))),
  ]));

  body.appendChild(el('div', { class: 'card' }, [
    el('div', { class: 'section__title', text: 'Produccion' }),
    kv('Fotos originales', String(stats.originals || 0)),
    ...Object.entries(stats.images_by_kind || {}).map(([k, v]) =>
      kv({ preview: 'Vistas previas', final: 'Finales', repair: 'Reparadas' }[k] || k,
        String(v))),
    kv('Intentos totales', String(stats.attempts || 0)),
    kv('Aceptadas', String(stats.accepted || 0)),
    kv('Intentos por foto', stats.attempts_per_photo
      ? String(stats.attempts_per_photo) : '-'),
  ]));

  if ((stats.spend_by_provider || []).length) {
    body.appendChild(el('div', { class: 'card' }, [
      el('div', { class: 'section__title', text: 'Gasto por proveedor' }),
      ...stats.spend_by_provider.map((row) =>
        kv(row.provider || 'local', `${moneyExact(row.cost)} (${row.n})`)),
    ]));
  }

  const reasons = stats.reject_reasons || [];
  if (reasons.length) {
    const max = Math.max(...reasons.map((r) => r.count));
    body.appendChild(el('div', { class: 'card' }, [
      el('div', { class: 'section__title', text: 'Motivos de descarte' }),
      ...reasons.map((r) => {
        const bar = el('div', { style: { height: '8px', borderRadius: '4px',
          background: 'var(--accent)', width: `${(r.count / max) * 100}%`,
          marginTop: '4px' } });
        return el('div', { style: { marginBottom: '10px' } }, [
          el('div', { style: { display: 'flex', justifyContent: 'space-between' } }, [
            el('span', { class: 'tiny', text: r.reason }),
            el('span', { class: 'tiny', text: String(r.count) }),
          ]),
          bar,
        ]);
      }),
    ]));
  }
}

/* -------------------------------------------------------------- providers */

async function renderProviders(body) {
  const data = await api.get('/api/admin/providers');
  clear(body);
  for (const [name, info] of Object.entries(data.providers || {})) {
    body.appendChild(el('div', { class: 'card' }, [
      el('div', { class: 'card__row' }, [
        el('strong', { text: name }),
        el('span', { class: 'chip chip--' + (info.available ? 'ok' : 'warn'),
          text: info.available ? 'Disponible' : 'No disponible' }),
      ]),
      kv('Tipo', info.kind === 'vision' ? 'Analisis' : 'Imagen'),
      kv('Necesita clave', info.needs_key ? 'Si' : 'No'),
      kv('Clave configurada', (info.key || {}).present ? 'Si' : 'No'),
      (info.key || {}).from_env ? kv('Origen', 'variable de entorno') : null,
      kv('Coste por imagen', info.cost_per_image_usd
        ? moneyExact(info.cost_per_image_usd) : 'gratis'),
      info.notes ? el('p', { class: 'tiny', text: info.notes }) : null,
    ]));
  }
}

/* ------------------------------------------------------------------ audit */

async function renderAudit(body) {
  const data = await api.get('/api/admin/audit?limit=80');
  clear(body);
  if (!data.events.length) {
    body.appendChild(empty({ icon: '≡', title: 'Sin eventos todavia' }));
    return;
  }
  const list = el('div', { class: 'card' });
  for (const event of data.events) {
    list.appendChild(el('div', { class: 'check-line' }, [
      el('span', { class: 'check-line__mark', text: '·' }),
      el('span', {}, [
        el('strong', { text: event.kind }),
        el('div', { class: 'tiny',
          text: `${dateLabel(event.created_at)}${event.actor ? ' - ' + event.actor : ''}` }),
      ]),
    ]));
  }
  body.appendChild(list);
}

/* ------------------------------------------------------------ maintenance */

async function renderMaintenance(view, body) {
  clear(body);
  body.appendChild(el('div', { class: 'card' }, [
    el('h3', { text: 'Recargar catalogo' }),
    el('p', { class: 'muted',
      text: 'Vuelve a escribir las opciones y estilos de fabrica. No toca los tuyos.' }),
    el('button', { class: 'btn btn--secondary', type: 'button',
      onClick: async () => {
        try {
          const data = await api.post('/api/admin/maintenance/reseed-catalog');
          toast(`${data.options} opciones, ${data.styles} estilos`, 'ok');
        } catch (err) { toast(err.message, 'danger'); }
      } }, 'Recargar catalogo'),
  ]));

  body.appendChild(el('div', { class: 'card' }, [
    el('h3', { text: 'Vaciar la papelera' }),
    el('p', { class: 'muted',
      text: 'Borra definitivamente lo eliminado hace mas de 30 dias.' }),
    el('button', { class: 'btn btn--danger', type: 'button',
      onClick: async () => {
        const ok = await confirmSheet('Se borraran definitivamente los archivos '
          + 'eliminados hace mas de 30 dias.',
          { title: 'Vaciar papelera', confirmLabel: 'Vaciar', danger: true });
        if (!ok) return;
        try {
          const data = await api.post('/api/admin/maintenance/purge-deleted',
            { days: 30 });
          toast(`${data.removed} elementos borrados`, 'ok');
        } catch (err) { toast(err.message, 'danger'); }
      } }, 'Vaciar papelera'),
  ]));
}

export default {
  async mount(view) {
    if (!store.isAdmin()) {
      view.appendChild(empty({ icon: '◇', title: 'No tienes permisos',
        text: 'Esta seccion es solo para administradores.' }));
      return;
    }
    await render(view);
  },
  unmount() {},
};
