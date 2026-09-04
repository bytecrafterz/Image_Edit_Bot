/* Reference photographs and the identity profile.

The coverage card is the important part of this page: when the profile cannot
measure a body yet, it says so and gives the exact instructions for taking
full-body reference photos, in the words the server returns. */

import { api } from '../api.js';
import { store } from '../store.js';
import {
  progressCard, explainCard, reportCard, runFirstRun, spinnerCard,
} from '../onboarding.js';
import {
  el, clear, note, toast, spinner, empty, lazyImg, confirmSheet, sheet,
  kv, pct, dateLabel,
} from '../ui.js';

let originals = [];
let byShot = {};
let profile = null;
let onboarding = null;
let buildTimer = null;

const SHOT_ES = { closeup: 'Primer plano', half: 'Medio cuerpo',
  full: 'Cuerpo entero', unknown: 'Sin identificar' };

function stopPolling() {
  if (buildTimer) { clearInterval(buildTimer); buildTimer = null; }
}

async function load() {
  const [list, profiles] = await Promise.all([
    api.get('/api/originals'),
    api.get('/api/profiles'),
  ]);
  originals = list.originals || [];
  byShot = list.by_shot_type || {};
  onboarding = list.onboarding || null;
  profile = (profiles.profiles || [])[0] || null;
}

async function render(view) {
  clear(view);
  view.appendChild(el('h1', { text: 'Mis fotos' }));
  view.appendChild(el('p', { class: 'view__sub',
    text: 'Las fotos con las que el robot aprende como eres.' }));

  // The count comes first: on this page it is the only thing that decides
  // whether she can generate at all, and she should not have to find it.
  if (onboarding) {
    view.appendChild(progressCard(onboarding, {
      action: profile && onboarding.puede_generar
        ? el('button', { class: 'btn', type: 'button',
            onClick: () => analyse(view) },
            (profile.first_run || {}).hecho
              ? 'Repetir el analisis de mis fotos'
              : 'Analizar mis fotos')
        : null,
    }));
    view.appendChild(explainCard(onboarding, { open: !onboarding.puede_generar }));
  }

  view.appendChild(profileCard(view));
  view.appendChild(uploadCard(view));

  const report = reportCard((profile || {}).first_run);
  if (report) view.appendChild(report);

  if (!originals.length) {
    view.appendChild(empty({
      icon: '◎', title: 'Aun no has subido fotos',
      text: `Sube al menos ${(onboarding || {}).minimo || 5} fotos tuyas: son `
        + 'con las que el robot comprueba que cada imagen sigues siendo tu.',
    }));
    return;
  }

  view.appendChild(el('div', { class: 'section' }, [
    el('div', { class: 'section__title',
      text: `${originals.length} fotos guardadas` }),
    el('div', { class: 'grid' }, originals.map((o, i) => card(view, o, i))),
  ]));
}

function profileCard(view) {
  if (!profile) {
    return el('div', { class: 'card' }, [
      el('h2', { text: 'Todavia no tienes perfil' }),
      el('p', { class: 'muted',
        text: 'El perfil guarda tus medidas para que ninguna imagen te cambie el '
          + 'cuerpo ni la cara sin que te enteres.' }),
      el('button', { class: 'btn', type: 'button',
        onClick: () => createProfile(view) }, 'Crear mi perfil'),
    ]);
  }

  const coverage = profile.coverage || {};
  const ready = Boolean(coverage.ready_for_body_check);
  const rows = [
    kv('Nombre', profile.person_name),
    kv('Fotos usadas', String(profile.n_sources || 0)),
    kv('Estado', profile.status === 'ready' ? 'Listo' : 'Incompleto'),
  ];
  for (const key of ['full', 'half', 'closeup']) {
    rows.push(kv(SHOT_ES[key], String(coverage[key] ?? byShot[key] ?? 0)));
  }

  const advice = coverage.advice || [];

  return el('div', { class: 'card' }, [
    el('h2', { text: 'Tu perfil' }),
    ...rows,
    ready
      ? note('ok', 'Control de proporciones activo',
          'Hay suficientes fotos de cuerpo para comprobar que no te cambian la figura.')
      : el('div', { class: 'note note--warn' }, [
          el('strong', { text: 'Faltan fotos de cuerpo entero' }),
          document.createTextNode(
            'Sin ellas se puede cuidar tu cara y tu piel, pero no se pueden medir '
            + 'tus proporciones, que es justo lo que fallo la otra vez. '
            + 'Haz 6 u 8 fotos asi:'),
          advice.length
            ? el('ul', { style: { margin: '10px 0 0', paddingLeft: '20px' } },
                advice.map((line) => el('li', { text: line,
                  style: { marginBottom: '4px' } })))
            : null,
        ]),
    profile.consent_problem
      ? note('warn', 'Falta el consentimiento', profile.consent_problem)
      : null,
    el('div', { class: 'btn-row', style: { marginTop: '12px' } }, [
      el('button', { class: 'btn btn--secondary', type: 'button',
        onClick: () => forgetOriginals(view) }, 'Olvidar fotos'),
      el('button', { class: 'btn', type: 'button',
        onClick: () => buildProfile(view) },
        profile.status === 'ready' ? 'Actualizar perfil' : 'Construir perfil'),
    ]),
  ]);
}

function uploadCard(view) {
  const input = el('input', { type: 'file', accept: 'image/*', multiple: true,
    hidden: true, onChange: (e) => upload(view, e.target.files) });

  const zone = el('div', {
    class: 'dropzone',
    onDragover: (e) => { e.preventDefault(); zone.classList.add('dropzone--over'); },
    onDragleave: () => zone.classList.remove('dropzone--over'),
    onDrop: (e) => {
      e.preventDefault();
      zone.classList.remove('dropzone--over');
      upload(view, e.dataTransfer.files);
    },
  }, [
    el('p', { style: { margin: '0 0 12px' },
      text: 'Arrastra fotos aqui, o pulsa el boton.' }),
    el('button', { class: 'btn', type: 'button',
      onClick: () => input.click() }, 'Anadir fotos'),
    input,
  ]);

  return el('div', { class: 'card' }, [
    el('div', { class: 'section__title', text: 'Anadir fotos' }),
    zone,
    el('p', { class: 'tiny', style: { marginTop: '10px', marginBottom: 0 },
      text: 'Sin filtros y sin retoque de belleza. Si la foto lleva filtro, el '
        + 'sistema aprende la version filtrada de ti.' }),
  ]);
}

async function upload(view, files) {
  if (!files || !files.length) return;
  const status = el('div', { class: 'card' }, spinner(`Subiendo ${files.length} foto(s)...`));
  view.insertBefore(status, view.firstChild);
  try {
    const data = await api.upload('/api/originals', Array.from(files));
    toast(`${data.added} foto(s) anadidas`, 'ok');
    if ((data.skipped || []).length) {
      toast(`${data.skipped.length} no se pudieron usar`, 'danger');
    }
    await load();
    await render(view);
  } catch (err) {
    toast(err.message, 'danger');
    status.remove();
  }
}

function card(view, original, index) {
  const quality = original.quality || {};
  return el('div', { class: 'card card--flat', style: { padding: '8px' } }, [
    el('div', { class: 'tile', style: { marginBottom: '8px' } }, [
      lazyImg(original.thumb_url, original.filename),
    ]),
    el('div', { class: 'tiny', text: SHOT_ES[original.shot_type] || 'Sin identificar' }),
    quality.score !== undefined
      ? el('div', { class: 'tiny', text: 'Calidad ' + pct(quality.score) }) : null,
    el('div', { style: { display: 'flex', gap: '6px', marginTop: '8px' } }, [
      el('button', { class: 'btn btn--secondary btn--sm', type: 'button',
        'aria-label': 'Subir', disabled: index === 0,
        onClick: () => move(view, index, -1) }, '↑'),
      el('button', { class: 'btn btn--secondary btn--sm', type: 'button',
        'aria-label': 'Bajar', disabled: index === originals.length - 1,
        onClick: () => move(view, index, 1) }, '↓'),
      el('button', { class: 'btn btn--danger btn--sm', type: 'button',
        onClick: async () => {
          const ok = await confirmSheet(`Se eliminara ${original.filename}.`,
            { title: 'Eliminar foto', confirmLabel: 'Eliminar', danger: true });
          if (!ok) return;
          try {
            await api.del(`/api/originals/${original.id}`);
            await load();
            await render(view);
          } catch (err) { toast(err.message, 'danger'); }
        } }, '×'),
    ]),
  ]);
}

async function move(view, index, delta) {
  const target = index + delta;
  if (target < 0 || target >= originals.length) return;
  const a = originals[index];
  const b = originals[target];
  try {
    await api.patch(`/api/originals/${a.id}`, { sort_order: b.sort_order });
    await api.patch(`/api/originals/${b.id}`, { sort_order: a.sort_order });
    await load();
    await render(view);
  } catch (err) { toast(err.message, 'danger'); }
}

async function createProfile(view) {
  const input = el('input', { type: 'text', value: 'Yo', placeholder: 'Tu nombre' });
  sheet({
    title: 'Crear perfil',
    subtitle: 'Un perfil guarda las medidas de una persona.',
    body: el('label', { class: 'field' }, [el('span', { text: 'Nombre' }), input]),
    actions: [
      { label: 'Cancelar', kind: 'secondary' },
      { label: 'Crear', onClick: async () => {
          try {
            const created = await api.post('/api/profiles',
              { person_name: input.value.trim() || 'Yo' });
            await api.post(`/api/profiles/${created.id}/consent`,
              { relationship: 'self' });
            toast('Perfil creado', 'ok');
            await load();
            await render(view);
          } catch (err) { toast(err.message, 'danger'); }
        } },
    ],
  });
}

async function buildProfile(view) {
  if (!originals.length) { toast('Sube fotos primero'); return; }
  const box = el('div', { class: 'card' }, spinner('Midiendo tus fotos...'));
  view.insertBefore(box, view.firstChild);
  try {
    const data = await api.post(`/api/profiles/${profile.id}/build`, {});
    if (data.async && data.run_id) {
      stopPolling();
      buildTimer = setInterval(async () => {
        try {
          const status = await api.get(`/api/generate/status/${data.run_id}`);
          if (['done', 'failed', 'cancelled'].includes(status.status)) {
            stopPolling();
            await load();
            await render(view);
            toast('Perfil actualizado', 'ok');
          }
        } catch { stopPolling(); }
      }, 2500);
      return;
    }
    box.remove();
    await load();
    await render(view);
    toast('Perfil actualizado', 'ok');
  } catch (err) {
    box.remove();
    toast(err.message, 'danger');
  }
}

/* The thorough reading, on demand.  Also runs by itself on the first estimate,
   but a person who has just uploaded her photographs wants to know NOW whether
   they are good enough, not after choosing an outfit. */
async function analyse(view) {
  if (!profile) { toast('Crea tu perfil primero'); return; }
  const box = spinnerCard('Mirando tus fotos con detalle...');
  view.insertBefore(box, view.firstChild);
  try {
    await runFirstRun(profile.id, (msg) => {
      box.replaceChildren(spinner(msg));
    });
    await load();
    await render(view);
    toast('Analisis terminado', 'ok');
  } catch (err) {
    box.remove();
    toast(err.message, 'danger');
  }
}

async function forgetOriginals(view) {
  const ok = await confirmSheet(
    'Se borraran las fotos originales de esta persona. Las medidas del perfil se '
    + 'mantienen y el sistema sigue funcionando igual.',
    { title: 'Olvidar fotos originales', confirmLabel: 'Borrar fotos', danger: true });
  if (!ok) return;
  try {
    const data = await api.post(`/api/profiles/${profile.id}/forget-originals`,
      { confirm: true });
    toast(data.message || 'Fotos borradas', 'ok');
    await load();
    await render(view);
  } catch (err) { toast(err.message, 'danger'); }
}

export default {
  async mount(view) {
    view.appendChild(spinner());
    try {
      await load();
    } catch (err) {
      clear(view);
      view.appendChild(note('danger', 'No se pudo cargar', err.message));
      return;
    }
    await render(view);
  },
  unmount() { stopPolling(); },
};
