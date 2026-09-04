/* What a new person is asked for, and what was measured on what she gave.

   One module because three screens say the same thing and they must not drift:
   the register screen promises it, Mis fotos counts it down, and Generar
   refuses on it.  Every sentence comes from the server (identity/onboarding.py)
   rather than being written here, so the rule and its explanation live in one
   place and the page only decides where they go. */

import { el, note, kv, spinner, toast } from './ui.js';
import { api } from './api.js';

/* The count, as a bar and a sentence.  Compact so it can sit at the top of a
   page the person is already using without pushing everything down. */
export function progressCard(state, opts) {
  const o = opts || {};
  const s = state || {};
  const have = Number(s.fotos || 0);
  const min = Number(s.minimo || 5);
  const want = Number(s.recomendado || 8);
  const done = Boolean(s.puede_generar);
  const width = Math.max(0, Math.min(100, (have / Math.max(1, want)) * 100));

  const bar = el('div', {
    style: {
      height: '10px', borderRadius: '6px', background: 'var(--line)',
      overflow: 'hidden', margin: '10px 0 6px',
    },
  }, el('div', {
    style: {
      height: '100%', width: width + '%',
      background: done ? 'var(--ok, #2e9e6b)' : 'var(--accent, #c47b3f)',
      transition: 'width .3s',
    },
  }));

  const parts = [
    el('h2', { text: done ? 'Tus fotos de referencia' : 'Faltan fotos tuyas' }),
    bar,
    el('p', { class: 'muted', style: { marginTop: 0 }, text: s.mensaje || '' }),
    kv('Fotos subidas', `${have}`),
    kv('Minimo para generar', `${min}`),
    kv('Recomendado', `${want}`),
  ];

  const byShot = s.por_plano || {};
  parts.push(kv('Cuerpo entero', String(byShot.full || 0)));
  parts.push(kv('Medio cuerpo', String(byShot.half || 0)));
  parts.push(kv('Primer plano', String(byShot.closeup || 0)));

  for (const line of (s.pendiente || [])) {
    parts.push(el('p', { class: 'tiny', text: '- ' + line }));
  }
  if (o.action) parts.push(o.action);

  return el('div', { class: 'card' }, parts);
}

/* Why the robot wants them and what happens to them.  Collapsed by default:
   it is long on purpose - it is a promise about photographs of her body - and
   nobody should have to scroll past it every time. */
export function explainCard(state, opts) {
  const s = state || {};
  const o = opts || {};
  const list = (title, lines) => el('div', { style: { marginTop: '10px' } }, [
    el('div', { class: 'section__title', text: title }),
    el('ul', { style: { margin: '6px 0 0', paddingLeft: '20px' } },
      (lines || []).map((line) => el('li', {
        text: line, style: { marginBottom: '6px' },
      }))),
  ]);

  const inner = el('div', {}, [
    list('Por que hacen falta varias', s.porque),
    list('Que se hace con ellas', s.que_pasa_con_tus_fotos),
    list('Como hacerlas', s.como_hacerlas),
  ]);

  if (o.open) return el('div', { class: 'card' }, inner);

  const details = el('details', {}, [
    el('summary', {
      style: { cursor: 'pointer', fontWeight: '600' },
      text: 'Por que necesita varias fotos y que hace con ellas',
    }),
    inner,
  ]);
  return el('div', { class: 'card' }, details);
}

/* The first run report: what CAN and CANNOT be checked for this person.
   Rendered from whatever the server measured, so a person with no full length
   photograph sees the missing half instead of a promise. */
export function reportCard(report) {
  const r = report || {};
  if (!r.hecho) return null;
  const rows = [];

  rows.push(el('h2', { text: 'Analisis de tus fotos' }));
  rows.push(el('p', { class: 'muted', style: { marginTop: '-4px' },
    text: r.resumen || '' }));

  const block = (title, b) => {
    if (!b) return null;
    return el('div', { style: { marginTop: '12px' } }, [
      el('div', { class: 'section__title', text: title }),
      el('p', { class: 'tiny', style: { margin: 0 }, text: b.veredicto || '' }),
    ]);
  };

  if (r.rostro) {
    rows.push(kv('Tu foto mas dificil', `${r.rostro.peor_foto} (limite ${r.rostro.limite})`));
    rows.push(kv('Margen', String(r.rostro.margen)));
  }
  rows.push(block('Rostro', r.rostro));
  rows.push(block('Cuerpo', r.cuerpo));
  rows.push(block('Manos', r.manos));
  rows.push(block('Piel', r.piel));

  const refs = r.referencias || {};
  if (refs.n) {
    rows.push(el('div', { style: { marginTop: '12px' } }, [
      el('div', { class: 'section__title', text: 'Fotos que se enviaran' }),
      el('p', { class: 'tiny', style: { margin: 0 }, text: refs.motivo || '' }),
    ]));
  }

  if ((r.se_puede_comprobar || []).length) {
    rows.push(el('div', { style: { marginTop: '12px' } }, [
      el('div', { class: 'section__title', text: 'Se puede comprobar' }),
      el('ul', { style: { margin: '6px 0 0', paddingLeft: '20px' } },
        r.se_puede_comprobar.map((t) => el('li', { text: t }))),
    ]));
  }
  for (const line of (r.avisos || [])) {
    rows.push(note('warn', 'Aviso sobre tus fotos', line));
  }
  if ((r.no_se_puede_comprobar || []).length) {
    rows.push(note('warn', 'Con estas fotos NO se puede comprobar',
      r.no_se_puede_comprobar.join(' ')));
  }
  if ((r.consejos || []).length) {
    rows.push(el('ul', { style: { margin: '6px 0 0', paddingLeft: '20px' } },
      r.consejos.map((t) => el('li', { class: 'tiny', text: t }))));
  }
  return el('div', { class: 'card' }, rows.filter(Boolean));
}

/* Ask the server for the thorough reading, waiting out the background job when
   there are enough photographs for it to become one. */
export async function runFirstRun(profileId, onProgress) {
  const data = await api.post(`/api/profiles/${profileId}/first-run`, {});
  if (!data.async) return data.informe;
  if (onProgress) onProgress(data.message || 'Analizando tus fotos...');
  for (let i = 0; i < 240; i += 1) {
    await new Promise((done) => setTimeout(done, 2000));
    let status;
    try {
      status = await api.get(`/api/generate/status/${data.run_id}`);
    } catch { break; }
    if (['done', 'failed', 'cancelled'].includes(status.status)) break;
  }
  const fetched = await api.get(`/api/profiles/${profileId}/first-run`);
  return fetched.informe;
}

export function spinnerCard(label) {
  return el('div', { class: 'card' }, spinner(label || 'Analizando tus fotos...'));
}
