/* The main screen: pick a photo, say what to change, see the price, watch the
   robot work, keep the good ones.

   The one rule that shapes this page: nothing is spent until she presses
   "Generar", and the amount is on screen before she does. */

import { api } from '../api.js';
import { refreshBalances } from '../app.js';
import { store } from '../store.js';
import { router } from '../router.js';
import { progressCard, explainCard, reportCard } from '../onboarding.js';
import {
  el, clear, frag, note, toast, sheet, spinner, progressBar, empty,
  money, moneyExact, lazyImg, confirmSheet, kv, pct, dragScroll,
} from '../ui.js';

const POLL_MS = 1500;

let state = null;
let pollTimer = null;

function reset() {
  state = {
    step: 1,
    originals: [],
    original: null,
    analysis: null,
    groups: [],
    styles: [],
    style: null,
    choices: {},
    nPreviews: store.restore('n_previews', 6),
    // '' (the old "Automatico") now means Gemini on the server too; see fal.py.
    engine: store.restore('engine', '') || 'identity_banana',
    sourceImage: null,     // a result being edited ("no, cambia el escote")
    referencia: null,      // the garment value she uploaded this time
    quality: store.restore('quality', 'preview'),
    plan: null,
    run: null,
    onboarding: null,
    selected: new Set(),
  };
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

/* ------------------------------------------------------------------ step 1 */

function stepHeader(view) {
  const labels = ['Elige la foto', 'Que quieres cambiar', 'Confirma el coste',
    'El robot trabaja', 'Elige las buenas'];
  return el('div', { class: 'step' }, [
    el('span', { class: 'step__n' + (state.step > 1 ? ' step__n--done' : ''),
      text: String(state.step) }),
    el('div', {}, [
      el('h1', { text: labels[state.step - 1], style: { margin: 0 } }),
      el('p', { class: 'muted', style: { margin: 0, fontSize: '.86rem' },
        text: `Paso ${state.step} de 5` }),
    ]),
  ]);
}

async function renderStep1(view) {
  clear(view);
  view.appendChild(stepHeader(view));

  const fileInput = el('input', { type: 'file', accept: 'image/*', multiple: true,
    hidden: true, onChange: (e) => uploadFiles(view, e.target.files) });

  // Below the minimum this screen stops being a chooser and becomes the ask.
  // The server refuses the estimate anyway; refusing here as well means she
  // never picks an outfit only to be told afterwards that she cannot generate.
  if (state.onboarding && !state.onboarding.puede_generar) {
    view.appendChild(progressCard(state.onboarding, {
      action: el('div', { class: 'btn-row' }, [
        el('button', { class: 'btn', type: 'button',
          onClick: () => fileInput.click() }, 'Anadir fotos ahora'),
        el('button', { class: 'btn btn--secondary', type: 'button',
          onClick: () => router.go('#/originals') }, 'Ir a Mis fotos'),
      ]),
    }));
    view.appendChild(fileInput);
    view.appendChild(explainCard(state.onboarding, { open: true }));
    return;
  }

  view.appendChild(el('div', { class: 'card' }, [
    el('button', { class: 'btn', type: 'button',
      onClick: () => fileInput.click() }, 'Anadir foto desde el movil'),
    fileInput,
    el('p', { class: 'tiny', style: { marginTop: '10px', marginBottom: 0 },
      text: 'Se abrira tu carrete. Puedes elegir varias a la vez.' }),
  ]));

  const strip = dragScroll(el('div', { class: 'scroller' }));
  if (!state.originals.length) {
    strip.appendChild(el('p', { class: 'muted',
      text: 'Todavia no tienes fotos guardadas.' }));
  }
  for (const original of state.originals) {
    const tile = el('button', {
      class: 'tile', type: 'button',
      style: { width: '108px', aspectRatio: '3/4', padding: 0, border: '1px solid var(--line)' },
      'aria-selected': state.original && state.original.id === original.id,
      onClick: () => chooseOriginal(view, original),
    }, [
      lazyImg(original.thumb_url, original.filename),
      el('div', { class: 'tile__meta' }, [
        el('span', { text: shotLabel(original.shot_type) }),
        recordBadge(original),
      ]),
    ]);
    strip.appendChild(tile);
  }
  view.appendChild(el('div', { class: 'section' }, [
    el('div', { class: 'section__title', text: 'O elige una que ya tengas' }),
    strip,
  ]));

  if (state.original) {
    view.appendChild(renderChosen(view));
  }
}

/* What the paid record says about this photograph, on the tile itself.
   Measured 2026-09-10: her two most-used photographs stood at 17 of 17 and
   3 of 10 on the face check, and nothing on this screen told her which was
   which - she paid for the eleventh attempt on the bad one. Three paid images
   is the same bar the estimate uses before it says anything. */
export function recordBadge(original) {
  const h = original.historial;
  const parts = [];
  if (h && h.pagadas >= 3) {
    // Red when the face is the problem, green only when most paid images
    // really came out, plain otherwise: "sale bien 2 de 8" in green was a
    // sentence the numbers did not support.
    const bad = h.cara_medidas >= 3 && h.cara_fallos / h.cara_medidas >= 0.4;
    const good = !bad && h.buenas / h.pagadas >= 0.6;
    parts.push(el('span', {
      text: bad ? `ha fallado ${h.cara_fallos} de ${h.cara_medidas}`
          : good ? `sale bien ${h.buenas} de ${h.pagadas}`
                 : `${h.buenas} buenas de ${h.pagadas}`,
      style: { display: 'block', fontSize: '.7rem', fontWeight: bad || good ? '600' : '400',
               color: bad ? 'var(--danger, #e5484d)' : good ? 'var(--ok, #46a758)' : 'inherit' } }));
  }
  return parts.length ? el('span', {}, parts) : null;
}

function shotLabel(shot) {
  return { closeup: 'Primer plano', half: 'Medio cuerpo', full: 'Cuerpo entero' }[shot]
    || 'Sin identificar';
}

function renderChosen(view) {
  const a = state.analysis || {};
  const quality = a.quality || {};
  const issues = quality.issues || [];
  return el('div', { class: 'card' }, [
    el('img', { src: state.original.url, alt: '',
      style: { borderRadius: '12px', maxHeight: '46vh', objectFit: 'contain',
        margin: '0 auto 12px' } }),
    kv('Tipo de foto', shotLabel(a.shot_type || state.original.shot_type)),
    kv('Calidad', quality.score !== undefined ? pct(quality.score) : 'sin medir'),
    a.measurable_body === false
      ? note('warn', 'No se ve el cuerpo entero',
          'En esta foto no se pueden medir tus proporciones, asi que ese control '
          + 'quedara limitado.')
      : null,
    issues.length ? note('warn', 'Aviso sobre esta foto', issues.join('. ')) : null,
    el('button', { class: 'btn', type: 'button',
      onClick: () => goStep2(view) }, 'Continuar'),
  ]);
}

async function uploadFiles(view, files) {
  if (!files || !files.length) return;
  const box = el('div', { class: 'card' }, spinner('Subiendo tus fotos...'));
  view.appendChild(box);
  try {
    const data = await api.upload('/api/originals', Array.from(files));
    box.remove();
    if (data.skipped && data.skipped.length) {
      /* The API already says WHY each file was left out ("no es una imagen",
         "archivo vacio"); showing only a count told her something went wrong
         and nothing she could do about it. */
      const first = data.skipped[0] || {};
      toast(data.skipped.length === 1
        ? `No se ha podido usar ${first.filename || 'un archivo'}: ${first.reason || 'no es una foto'}`
        : `${data.skipped.length} archivos no se han podido usar (${first.reason || 'no son fotos'})`);
    }
    await loadOriginals();
    const first = (data.originals || [])[0];
    if (first) await chooseOriginal(view, first);
    else await renderStep1(view);
  } catch (err) {
    box.remove();
    toast(err.message, 'danger');
  }
}

async function loadOriginals() {
  const data = await api.get('/api/originals');
  state.originals = data.originals || [];
  state.onboarding = data.onboarding || null;
}

async function chooseOriginal(view, original) {
  state.original = original;
  state.analysis = null;
  await renderStep1(view);
  try {
    state.analysis = await api.get(`/api/originals/${original.id}/analysis`);
  } catch { /* the reading is a nicety, not a blocker */ }
  await renderStep1(view);
}

/* ------------------------------------------------------------------ step 2 */

async function goStep2(view) {
  state.step = 2;
  clear(view);
  view.appendChild(stepHeader(view));
  view.appendChild(spinner('Mirando tu foto...'));
  try {
    const [options, styles] = await Promise.all([
      api.get(api.qs('/api/catalog/options', { original_id: state.original.id })),
      api.get(api.qs('/api/catalog/styles', {
        shot_type: (state.analysis && state.analysis.shot_type) || '' })),
    ]);
    state.groups = options.groups || [];
    state.styles = styles.styles || [];
    state.style = state.style || (state.styles[0] || null);

    /* CHOICES MADE ON ANOTHER PHOTOGRAPH.  Picking a photo does not clear what
       was already chosen - that would throw away her work every time she
       browses - so a floor length gown chosen on a full body shot survived
       into a head and shoulders one, where it is not on the menu at all.  It
       used to travel to the estimate and be dropped there in silence.  Now it
       is dropped here, where the menu is, and she is told which one and why. */
    const allowed = {};
    for (const g of state.groups) {
      allowed[g.group_key] = new Set((g.values || []).map((v) => v.value_key));
    }
    const gone = [];
    for (const [group, values] of Object.entries(state.choices)) {
      const keep = (values || []).filter((v) => (allowed[group] || new Set()).has(v));
      if (keep.length !== (values || []).length) {
        for (const v of values) if (!keep.includes(v)) gone.push(v);
      }
      if (keep.length) state.choices[group] = keep;
      else delete state.choices[group];
    }
    if (gone.length) {
      toast(`En esta foto no cabe ${gone.length === 1 ? 'una de tus elecciones' : gone.length + ' de tus elecciones'}: elige otra vez`);
    }

    renderStep2(view);
  } catch (err) {
    clear(view);
    view.appendChild(stepHeader(view));
    view.appendChild(note('danger', 'No se pudieron cargar las opciones', err.message));
  }
}

function choiceSentence(group) {
  const chosen = state.choices[group.group_key] || [];
  if (chosen.length === 0) {
    return 'No has elegido nada: el robot ira variando esto entre las fotos.';
  }
  if (chosen.length === 1) return 'Sera igual en todas las fotos.';
  return `Se combinaran estas ${chosen.length} opciones entre las fotos.`;
}

/* Gemini first and preselected: measured on her photographs on 2026-09-11 it
   kept her face at 0.71-0.80 where FLUX Kontext ranged 0.50-0.78 and she
   rejected the Kontext results on sight.  Same price. */
const ENGINES = [
  ['identity_banana', 'Gemini (recomendado)'],
  ['identity_multi', 'FLUX Kontext'],
  ['identity_gpt', 'OpenAI'],
];

/* Her words, typed or dictated, and a picture of a garment she likes.  What
   she asked for on 2026-09-11 in her own terms: "quiero poder hablar por voz",
   "subir una foto y decir quiero algo parecido a esto".  The sentence goes to
   /api/generate/interpretar, comes back as choices - existing ones where they
   fit, new values of her own where the catalogue had nothing - and the usual
   estimate follows.  Dictation is the browser's own (es-ES); iPhone Safari has
   it.  Nothing here spends on images. */
function freeTextCard(view) {
  const box = el('textarea', { rows: 3, placeholder:
    'Dime lo que quieres: "vestido rojo cruzado, en un restaurante, de pie" o "como la foto de la prenda pero en azul marino"',
    style: { width: '100%', boxSizing: 'border-box' } });
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let rec = null;
  const mic = el('button', { class: 'btn btn--secondary btn--sm', type: 'button', hidden: !SR,
    onClick: () => {
      if (rec) { rec.stop(); return; }
      rec = new SR(); rec.lang = 'es-ES'; rec.interimResults = true; rec.continuous = false;
      const before = box.value ? box.value.trim() + ' ' : '';
      rec.onresult = (ev) => { box.value = before + Array.from(ev.results).map((r) => r[0].transcript).join(' '); };
      rec.onend = () => { rec = null; mic.textContent = '\u{1F3A4} Dictar'; };
      rec.onerror = () => { rec = null; mic.textContent = '\u{1F3A4} Dictar'; toast('No se ha podido escuchar. Escribelo.'); };
      mic.textContent = '\u{1F534} Escuchando... (toca para parar)';
      try { rec.start(); } catch { rec = null; mic.textContent = '\u{1F3A4} Dictar'; }
    } }, '\u{1F3A4} Dictar');
  const refName = el('span', { class: 'tiny', text: state.referencia ? `Prenda: ${state.referencia.label_es}` : '' });
  const fileInput = el('input', { type: 'file', accept: 'image/*', hidden: true,
    onChange: async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      refName.textContent = 'Leyendo la prenda...';
      try {
        const data = await api.upload('/api/catalog/referencia', [file], { texto: box.value || '' });
        state.referencia = data.value;
        refName.textContent = `Prenda: ${data.value.label_es}`;
        toast('Prenda guardada como opcion tuya. Ahora dime que quieres hacer con ella, o calcula directamente.', 'ok');
      } catch (err) { refName.textContent = ''; toast(err.message, 'danger'); }
    } });
  const go = el('button', { class: 'btn', type: 'button',
    onClick: () => interpretAndPrice(view, box.value) }, 'Entender y calcular');
  const enginesRow = el('div', { class: 'chips', style: { marginTop: '8px' } },
    ENGINES.map(([key, label]) => el('button', {
      class: 'chip' + (state.engine === key ? ' chip--on' : ''), type: 'button',
      onClick: (ev) => {
        state.engine = key; store.persist('engine', key);
        for (const c of enginesRow.children) c.classList.toggle('chip--on', c === ev.currentTarget);
      } }, label)));
  return el('div', { class: 'section' }, [
    el('div', { class: 'section__title', text: 'Dilo con tus palabras' }),
    el('div', { class: 'card' }, [
      box, fileInput,
      el('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '8px', alignItems: 'center' } }, [
        mic,
        el('button', { class: 'btn btn--secondary btn--sm', type: 'button',
          onClick: () => fileInput.click() }, 'Foto de una prenda'),
        refName,
      ]),
      el('p', { class: 'tiny', style: { margin: '6px 0 0' },
        text: 'Dictar: pulsa, habla en espanol y lo que digas aparece arriba. Vuelve a pulsar para parar.' }),
      el('div', { class: 'tiny', style: { marginTop: '8px' }, text: 'Motor de imagen' }),
      enginesRow,
      el('div', { style: { marginTop: '10px' } }, go),
      el('p', { class: 'tiny', style: { margin: '8px 0 0' },
        text: 'O elige abajo con las opciones de siempre. Entender lo que dices no cuesta imagenes.' }),
    ]),
  ]);
}

/* Her sentence -> choices -> the same estimate as always.  Whatever the
   catalogue already had is picked; whatever it lacked becomes a value of
   hers; anything that would change who she is comes back as "rechazado"
   and is said, not done. */
async function interpretAndPrice(view, texto) {
  const clean = (texto || '').trim();
  if (!clean && !state.referencia) { toast('Escribe o dicta lo que quieres, o sube una prenda.'); return; }
  const btnNote = toast('Entendiendo lo que pides...');
  try {
    const data = await api.post('/api/generate/interpretar', {
      texto: clean,
      original_id: state.sourceImage ? null : (state.original ? state.original.id : null),
      source_image_id: state.sourceImage || null,
      referencia_value: state.referencia ? state.referencia.value_key : null,
    }, { timeout: 120000 });
    Object.assign(state.choices, data.choices || {});
    if (data.vistas > 0) { state.nPreviews = Math.min(6, Math.max(1, data.vistas)); store.persist('n_previews', state.nPreviews); }
    if (data.rechazado) toast(data.rechazado, 'danger');
    if (data.resumen) toast(data.resumen, 'ok');
    await goStep3(view);
  } catch (err) { toast(err.message, 'danger'); }
}

function renderStep2(view) {
  clear(view);
  view.appendChild(stepHeader(view));
  view.appendChild(freeTextCard(view));

  // Style carousel
  const styleRow = dragScroll(el('div', { class: 'scroller' }));
  for (const style of state.styles) {
    const card = el('button', {
      class: 'chip' + (state.style && state.style.key === style.key ? ' chip--on' : ''),
      type: 'button',
      // A fixed width keeps the carousel scrollable: without a maximum the
      // description stretches the card past the edge of a phone screen.
      style: { flexDirection: 'column', alignItems: 'flex-start',
        width: '168px', minWidth: '168px', maxWidth: '168px',
        whiteSpace: 'normal', textAlign: 'left', borderRadius: '14px',
        padding: '12px', height: 'auto' },
      onClick: () => { state.style = style; renderStep2(view); },
    }, [
      el('strong', { text: style.name_es, style: { fontSize: '.92rem' } }),
      el('span', { text: style.description || '',
        style: { fontSize: '.76rem', opacity: '.8', marginTop: '4px' } }),
    ]);
    styleRow.appendChild(card);
  }
  view.appendChild(el('div', { class: 'section' }, [
    el('div', { class: 'section__title', text: 'Estilo' }),
    styleRow,
  ]));

  // ?? and not ||: priority 0 is the most relevant group for this photograph,
  // and `0 || 99` quietly demoted it into the collapsed "more options" drawer -
  // so the one thing the system most wanted to suggest was the one thing hidden.
  const rank = (g) => (g.priority ?? 99);
  const primary = state.groups.filter((g) => rank(g) < 4);
  const rest = state.groups.filter((g) => rank(g) >= 4);

  for (const group of primary) view.appendChild(renderGroup(view, group));

  if (rest.length) {
    const body = el('div', { hidden: true });
    for (const group of rest) body.appendChild(renderGroup(view, group));
    const toggle = el('button', { class: 'btn btn--secondary', type: 'button',
      onClick: () => {
        body.hidden = !body.hidden;
        toggle.textContent = body.hidden ? 'Mas opciones' : 'Menos opciones';
      } }, 'Mas opciones');
    view.appendChild(el('div', { class: 'section' }, [toggle, body]));
  }

  // Count + quality
  const countValue = el('strong', { text: String(state.nPreviews) });
  const stepper = el('div', { class: 'card__row' }, [
    el('span', { text: 'Cuantas vistas previas' }),
    el('div', { style: { display: 'flex', gap: '8px', alignItems: 'center' } }, [
      el('button', { class: 'btn btn--secondary btn--sm', type: 'button',
        onClick: () => { state.nPreviews = Math.max(1, state.nPreviews - 1);
          countValue.textContent = String(state.nPreviews);
          store.persist('n_previews', state.nPreviews); } }, '-'),
      countValue,
      el('button', { class: 'btn btn--secondary btn--sm', type: 'button',
        onClick: () => { state.nPreviews = Math.min(12, state.nPreviews + 1);
          countValue.textContent = String(state.nPreviews);
          store.persist('n_previews', state.nPreviews); } }, '+'),
    ]),
  ]);

  const qualitySelect = el('select', {
    onChange: (e) => { state.quality = e.target.value; store.persist('quality', state.quality); },
  }, [
    el('option', { value: 'preview', selected: state.quality === 'preview' }, 'Vista previa (rapida)'),
    el('option', { value: 'standard', selected: state.quality === 'standard' }, 'Estandar'),
    el('option', { value: 'high', selected: state.quality === 'high' }, 'Alta'),
  ]);

  view.appendChild(el('div', { class: 'card' }, [
    stepper,
    el('div', { style: { marginTop: '12px' } }, [
      el('span', { class: 'muted', text: 'Calidad', style: { display: 'block', marginBottom: '6px' } }),
      qualitySelect,
    ]),
  ]));

  view.appendChild(el('div', { class: 'btn-row' }, [
    el('button', { class: 'btn btn--secondary', type: 'button',
      onClick: () => { state.step = 1; renderStep1(view); } }, 'Atras'),
    el('button', { class: 'btn', type: 'button',
      onClick: () => goStep3(view) }, 'Ver el coste'),
  ]));
}

function renderGroup(view, group) {
  const chips = el('div', { class: 'chips' });
  const sentence = el('p', { class: 'tiny', style: { marginTop: '8px', marginBottom: 0 } });

  const refresh = () => { sentence.textContent = choiceSentence(group); };

  for (const value of group.values) {
    const suggested = (group.suggested || []).includes(value.value_key);
    const chip = el('button', {
      class: 'chip' + (suggested ? ' chip--suggested' : ''),
      type: 'button',
      'aria-pressed': (state.choices[group.group_key] || []).includes(value.value_key),
      onClick: () => {
        const list = state.choices[group.group_key] || [];
        const index = list.indexOf(value.value_key);
        let next;
        if (index >= 0) next = list.filter((v) => v !== value.value_key);
        else if (group.multi === false) next = [value.value_key];
        else next = list.concat([value.value_key]);
        if (next.length) state.choices[group.group_key] = next;
        else delete state.choices[group.group_key];
        for (const other of chips.children) {
          const key = other.dataset.key;
          other.setAttribute('aria-pressed',
            String((state.choices[group.group_key] || []).includes(key)));
        }
        refresh();
      },
      dataset: { key: value.value_key },
    }, value.label_es);
    chips.appendChild(chip);
  }
  refresh();

  return el('div', { class: 'section' }, [
    el('div', { class: 'section__title', text: group.label_es }),
    group.reason ? el('p', { class: 'tiny', style: { marginTop: '-4px' },
      text: group.reason }) : null,
    chips,
    sentence,
  ]);
}

/* ------------------------------------------------------------------ step 3 */

async function goStep3(view) {
  state.step = 3;
  clear(view);
  view.appendChild(stepHeader(view));
  const first = !(state.onboarding || {}).analisis_hecho;
  view.appendChild(spinner(first
    ? 'Mirando tus fotos a fondo y calculando el coste. La primera vez tarda '
      + 'un poco: solo se hace una vez.'
    : 'Preparando y calculando el coste...'));
  try {
    // The FIRST estimate for a person runs the thorough reading of her
    // photographs inside this request - build the profile, choose the
    // reference trio, read her hands - which measured 18.5 s on five
    // photographs and grows with the gallery.  api.js aborts an ordinary call
    // at 20 s, so without a longer budget here the very first estimate any new
    // account ever asks for would fail with "la conexion tarda demasiado" on a
    // request that was working perfectly.  Later estimates read the stored
    // report and come back in under a second.
    const plan = await api.post('/api/generate/analyze', {
      original_id: state.original ? state.original.id : null,
      source_image_id: state.sourceImage || null,
      engine: state.engine || null,
      style: state.style ? state.style.key : null,
      options: state.choices,
      n_previews: state.nPreviews,
      quality: state.quality,
    }, { timeout: 300000 });
    state.plan = plan;
    renderStep3(view);
  } catch (err) {
    clear(view);
    view.appendChild(stepHeader(view));
    /* Guidance, not a failure: what lands here is the estimate refusing for a
       reason she can act on (a garment this photograph cannot wear, too few
       photographs to check her identity), and it has cost nothing. */
    view.appendChild(note('info', 'Con esta foto no se puede', err.message));
    view.appendChild(el('button', { class: 'btn btn--secondary', type: 'button',
      onClick: () => { state.step = 2; renderStep2(view); } }, 'Cambiar opciones'));
  }
}

function renderStep3(view) {
  clear(view);
  view.appendChild(stepHeader(view));

  const plan = state.plan;
  // What was measured on HER photographs, above the price and before the
  // button: the first estimate for a person runs the thorough analysis and
  // this is where its verdict is read.
  const first = reportCard(plan.analisis_inicial);
  if (first) view.appendChild(first);
  const est = plan.estimate || {};
  const summary = plan.plan_summary || {};
  const balances = store.get('balances') || {};
  const provider = est.provider || 'local';
  const balance = (balances[provider] || {}).balance;
  const after = (balance === null || balance === undefined)
    ? null : balance - (est.total_usd || 0);

  const lockedRows = Object.entries(summary.locked || {}).map(([group, value]) =>
    kv(group, String(value).replace(/_/g, ' ')));

  view.appendChild(el('div', { class: 'card' }, [
    el('h2', { text: est.total_usd > 0 ? money(est.total_usd) : 'Sin coste' }),
    el('p', { class: 'muted', style: { marginTop: '-4px' },
      text: `${summary.n_variants || 0} imagenes con ${est.provider || 'motor local'}` }),
    after !== null
      ? kv('Saldo despues', money(after))
      : kv('Motor', 'local gratuito (no gasta saldo)'),
    est.per_image_usd ? kv('Por imagen', moneyExact(est.per_image_usd)) : null,
    // The endpoint, not the internal role name: it is what fal's price list
    // calls the model, so she can check the figure above against it herself.
    // It is also the line that says which of the two paths this run takes -
    // v1/fill repaints a zone and never redraws her face, kontext does.
    est.endpoint ? kv('Modelo', est.endpoint) : null,
  ]));

  if (lockedRows.length) {
    view.appendChild(el('div', { class: 'card' }, [
      el('div', { class: 'section__title', text: 'Igual en todas' }),
      ...lockedRows,
    ]));
  }
  if ((summary.varied || []).length) {
    view.appendChild(el('div', { class: 'card' }, [
      el('div', { class: 'section__title', text: 'Va cambiando' }),
      el('p', { class: 'muted', style: { margin: 0 },
        text: summary.varied.join(', ') }),
    ]));
  }
  // WHAT IS LIKELY TO GO WRONG WITH THIS PETICION, AND THE FIX, BEFORE THE
  // BUTTON.  It goes above the generic warnings because it is the only block
  // on this screen she can act on with one tap, and below the price because
  // the price is what she came here to read.
  const riskCard = renderRiskCard(view, est.riesgo || {});
  if (riskCard) view.appendChild(riskCard);

  for (const warning of (plan.warnings || [])) {
    // Already said, in full, inside the risk card just above.
    if (riskCard && riskSaid.has(warning)) continue;
    view.appendChild(note('warn', null, warning));
  }
  for (const line of (summary.notes || [])) {
    view.appendChild(el('p', { class: 'tiny', text: line }));
  }

  // Not 'nothing has been spent': since 2026-09-05 the estimate itself can
  // charge for reading the photograph with Claude (about 0.0112 USD, once per
  // photograph), and that charge is written in the avisos just above.  A line
  // that flatly denied it would be the only false sentence on the screen.
  view.appendChild(note('info', 'Todavia no se ha generado ninguna imagen',
    'Las imagenes solo se cobran cuando pulses Generar. Si arriba pone que se '
    + 'ha pagado la lectura de tu foto, eso ya esta hecho y no se repite.'));

  view.appendChild(el('div', { class: 'btn-row' }, [
    el('button', { class: 'btn btn--secondary', type: 'button',
      onClick: () => { state.step = 2; renderStep2(view); } }, 'Cambiar opciones'),
    el('button', { class: 'btn', type: 'button',
      onClick: () => startRun(view) }, 'Generar'),
  ]));
}

/* The lines the risk card has already said in full, so the generic warning
   list underneath does not repeat them word for word. */
const riskSaid = new Set();

/* What the record says will go wrong with THIS request, and the request that
   avoids it - offered as a button, not as advice.

   The measurement this exists for: replaying every paid call this installation
   has ever made through the assessment (scripts/replay_risk.py), starting from
   an empty memory and judging each request only on the calls paid for before
   it, it warns about 36 of the 43 charges that produced no usable image - 1.52
   of the 1.81 USD wasted - and puts a warning in front of 4 of the 31 that did
   work.  A warning that fired on everything would be worth nothing, so that
   second number is the one this card is tuned against. */
function renderRiskCard(view, risk) {
  riskSaid.clear();
  const motivos = risk.motivos || [];
  const gratis = risk.gratis || {};
  const ajuste = risk.ajuste || {};
  if (!motivos.length && !gratis.posible) return null;

  const children = [];

  // FIRST AND LOUDEST: the path that cannot be rejected and cannot be charged.
  if (gratis.posible) {
    riskSaid.add(gratis.texto);
    children.push(note(gratis.ya ? 'ok' : 'info',
      gratis.ya ? 'Esto sale gratis' : 'Esto se puede hacer gratis',
      gratis.texto));
  }

  if (motivos.length) {
    children.push(el('div', { class: 'section__title',
      text: risk.titulo || 'Lo que suele salir mal con esta peticion' }));
    const list = el('ul', { style: { margin: '0 0 4px', paddingLeft: '18px' } });
    for (const motivo of motivos.slice(0, 4)) {
      riskSaid.add(motivo.texto);
      list.appendChild(el('li', { class: 'tiny',
        style: { marginBottom: '6px',
          fontWeight: motivo.nivel === 'confirmar' ? '600' : '400' },
        text: motivo.texto }));
    }
    children.push(list);
  }

  // AND THE ALTERNATIVE, APPLIED WITH ONE TAP.  It edits the very choices the
  // estimate was built from and asks for a new estimate, so what she confirms
  // is a real price for a real request and never a promise made by this file.
  if (ajuste && ajuste.titulo) {
    children.push(el('p', { class: 'tiny', style: { margin: '10px 0 6px' },
      text: ajuste.texto || '' }));
    const applies = (ajuste.quitar || []).length || Object.keys(ajuste.cambiar || {}).length;
    // The record can also name a PHOTOGRAPH: the same tap, but it swaps the
    // source and re-prices, because the photo is not one of the choices.
    const better = ajuste.foto
      ? (state.originals || []).find((o) => o.filename === ajuste.foto) : null;
    if (applies) {
      children.push(el('button', {
        class: 'btn', type: 'button',
        onClick: () => applyAdjustment(view, ajuste),
      }, ajuste.titulo));
    } else if (better) {
      children.push(el('button', {
        class: 'btn', type: 'button',
        onClick: () => useOtherPhoto(view, better),
      }, `Usar ${better.filename}`));
    } else {
      // Nothing to press: the fix is a switch in Ajustes, not one of her
      // choices, and pretending otherwise would be a button that lies.
      children.push(note('info', ajuste.titulo, ajuste.texto || ''));
    }
  }

  return el('div', { class: 'card' }, children);
}

/* Swap the source photograph for the one the record recommends and re-price.
   Nothing is spent.  The choices travel unchanged; a choice that does not
   exist for the new photograph's framing is dropped by the estimate itself,
   as it always was. */
async function useOtherPhoto(view, original) {
  const before = state.original;
  state.original = original;
  state.analysis = null;
  try {
    state.analysis = await api.get(`/api/originals/${original.id}/analysis`);
  } catch { /* the reading is a nicety, not a blocker */ }
  await goStep3(view);
  if (state.step === 3 && state.plan) {
    toast(`Ahora se usa ${original.filename}. Vuelto a calcular, no se ha gastado nada.`, 'ok');
  } else {
    state.original = before;
  }
}

/* Apply the suggested request and re-price it.  Nothing is spent: this is the
   same free /analyze the screen already ran. */
async function applyAdjustment(view, ajuste) {
  const before = JSON.parse(JSON.stringify(state.choices || {}));
  for (const group of (ajuste.quitar || [])) delete state.choices[group];
  for (const [group, value] of Object.entries(ajuste.cambiar || {})) {
    state.choices[group] = [value];
  }
  await goStep3(view);
  // goStep3 swallows its own failure and renders the reason instead of
  // throwing, so success is read off the screen it actually produced: a
  // cheerful toast over an error message would be the one false sentence here.
  if (state.step === 3 && state.plan) {
    toast('Peticion ajustada y vuelta a calcular. No se ha gastado nada.', 'ok');
  } else {
    state.choices = before;
  }
}

/* ------------------------------------------------------------------ step 4 */

async function startRun(view) {
  // A COMBINATION THAT HAS NEVER ONCE WORKED IS NOT CHARGED SILENTLY.  The
  // server refuses this run with the same sentence if the flag does not
  // arrive, so this sheet is the courtesy and the 409 is the guarantee - and
  // the sentence carries the count, "ha fallado N de N veces", because a
  // warning without its number is just a mood.
  const risk = (state.plan.estimate || {}).riesgo || {};
  let confirmed = false;
  if (risk.nivel === 'confirmar' && risk.confirmacion) {
    confirmed = await confirmSheet(risk.confirmacion, {
      title: 'Esto ya ha fallado antes',
      confirmLabel: 'Pagar igualmente', danger: true,
    });
    if (!confirmed) return;
  }
  try {
    await api.post('/api/generate/run',
      { run_id: state.plan.run_id, confirmar_riesgo: confirmed });
    state.step = 4;
    state.selected = new Set();
    renderStep4(view, null);
    stopPolling();
    pollTimer = setInterval(() => pollRun(view), POLL_MS);
    pollRun(view);
  } catch (err) {
    toast(err.message, 'danger');
  }
}

let lastSpent = -1;

async function pollRun(view) {
  try {
    const run = await api.get(`/api/generate/status/${state.plan.run_id}`);
    state.run = run;
    // Money moved: the app bar has to say so now, not at the next login.
    if (run.spent_usd !== lastSpent) { lastSpent = run.spent_usd; refreshBalances(); }
    if (['done', 'failed', 'cancelled', 'stopped_no_balance'].includes(run.status)) {
      stopPolling();
      if (run.status === 'done' && run.images.length) {
        if (state.finalizing) {
          // The finals are in the album; there is nothing left to do on this
          // screen.  Say so once and go back to the start for the next one.
          const n = run.images.length;
          toast(`Listo: ${n} ${n === 1 ? 'imagen' : 'imagenes'} en alta calidad, ya en tu album.`, 'ok');
          reset();
          await loadAndRender(view);
          return;
        }
        state.step = 5;
        renderStep5(view);
        return;
      }
      state.finalizing = false;
    }
    renderStep4(view, run);
  } catch (err) {
    stopPolling();
    toast(err.message, 'danger');
  }
}

function renderStep4(view, run) {
  clear(view);
  view.appendChild(stepHeader(view));

  const progress = progressBar(run ? run.progress : 0);
  view.appendChild(el('div', { class: 'card' }, [
    el('p', { style: { margin: '0 0 10px', fontWeight: '600' },
      text: (run && run.stage) || 'Empezando...' }),
    progress,
    run ? el('div', { style: { display: 'flex', justifyContent: 'space-between',
      marginTop: '10px' } }, [
      el('span', { class: 'tiny', text: `${run.accepted} aceptadas` }),
      el('span', { class: 'tiny', text: run.spent_usd > 0 ? moneyExact(run.spent_usd) : 'gratis' }),
    ]) : null,
  ]));

  /* WHAT SHE IS NOT BEING SHOWN, AND WHY.  "El robot ha descartado imagenes"
     over a list of failed checks reads like a verdict on her: the client asked
     for no rejection messages, and where one is unavoidable it has to say what
     happened to the picture, what it cost and what happens next.  The run is
     still going while this shows, so it is worded as work in progress. */
  if (run && run.discard_reasons && run.discard_reasons.length) {
    const done = ['done', 'failed', 'cancelled', 'stopped_no_balance']
      .includes(run.status);
    const text = run.discard_reasons
      .map((r) => `${r.count}: ${r.reason}`).join('. ')
      + (done ? '.' : '. El robot sigue intentandolo.');
    view.appendChild(note('info', 'Alguna imagen no ha salido bien', text));
  }

  /* THE PROVIDER DID NOT HAND OVER THE IMAGE.  This is not the same event as
     "the robot did not like the result" and it must not borrow its words: on
     2026-09-05 ten calls across five runs came back as a black file and were
     charged 0.46 USD, and the screen told the client "solo te ensena las
     imagenes en las que sales tu, y esta vez ninguna lo ha conseguido" - which
     blames her photographs for a picture no robot ever received.  The robot
     never saw these.  So this branch says what happened, that it was charged,
     and what is actually left to try - and it does NOT suggest changing the
     garment or the photograph, because that is exactly what was tried twice at
     0.46 USD and blocked both times. */
  const blocked = (run && run.bloqueadas) || { n: 0, usd: 0 };
  if (run && run.status === 'done' && !run.images.length && blocked.n > 0) {
    view.appendChild(note('warn', 'El proveedor no ha entregado las imagenes',
      `fal.ai ha devuelto un archivo en negro en ${blocked.n} `
      + `${blocked.n === 1 ? 'intento' : 'intentos'} y los cobra igual `
      + `(${moneyExact(blocked.usd)}). No es que las fotos salieran mal: no `
      + 'llegaron. Ni tu ni el robot han visto ninguna imagen, y tus fotos no '
      + 'se han tocado. Repetir la misma peticion cuesta lo mismo y acaba '
      + 'igual, asi que antes entra en Ajustes y comprueba que el repintado '
      + 'por zonas, las fotos de referencia y el texto de cobertura estan como '
      + 'vienen de fabrica (desactivados): esa es la configuracion con la que '
      + 'este robot si ha recibido imagenes.'));
  } else if (run && run.status === 'done' && !run.images.length) {
    /* THE RUN FINISHED AND THE ROBOT REALLY DID LOOK.  Here the images did
       arrive and the identity check refused them, so this sentence is true. */
    view.appendChild(note('info', 'Esta vez no ha salido ninguna buena',
      'El robot solo te ensena las imagenes en las que sales tu, y esta vez '
      + 'ninguna lo ha conseguido. Tus fotos no se han tocado. '
      + (run.spent_usd > 0
        ? `Lo que se ha probado ya esta pagado (${moneyExact(run.spent_usd)}) y no se gasta nada mas. `
        : '')
      + 'Prueba a cambiar solo la ropa, o elige otra foto tuya donde se te vea '
      + 'entera y con buena luz.'));
  }

  if (run && run.status === 'stopped_no_balance') {
    const help = run.balance_help || {};
    view.appendChild(note('danger', 'El robot se ha detenido',
      `${run.error || 'Sin saldo suficiente.'} Recarga unos ${money(help.recommended_topup || 5)} en `
      + `${help.provider || 'tu proveedor'} y vuelve a intentarlo.`));
    view.appendChild(el('a', { class: 'btn', href: '#/settings' }, 'Ir a Ajustes'));
  }
  if (run && run.status === 'failed') {
    /* run.error is written by services/jobs.user_message: either a sentence
       this product wrote for her, or one plain sentence.  A Python exception
       can no longer land here. */
    view.appendChild(note('info', 'El robot se ha quedado a medias',
      run.error || 'No se ha podido terminar. No se te cobra lo que no se hizo. '
      + 'Vuelve a intentarlo.'));
  }

  if (run && run.images.length) {
    view.appendChild(el('div', { class: 'section' }, [
      el('div', { class: 'section__title', text: 'Listas hasta ahora' }),
      el('div', { class: 'grid' }, run.images.map((img) => tile(img, false))),
    ]));
  }

  if (run && ['running', 'queued'].includes(run.status)) {
    view.appendChild(el('button', { class: 'btn btn--secondary', type: 'button',
      onClick: async () => {
        await api.post(`/api/generate/cancel/${state.plan.run_id}`);
        toast('Deteniendo...');
      } }, 'Detener'));
  } else if (run) {
    view.appendChild(el('button', { class: 'btn btn--secondary', type: 'button',
      onClick: () => { reset(); loadAndRender(view); } }, 'Empezar de nuevo'));
  }
}

/* ------------------------------------------------------------------ step 5 */

function tile(img, selectable) {
  const node = el('div', {
    class: 'tile',
    'aria-selected': selectable && state.selected.has(img.id),
    onClick: () => {
      if (!selectable) { openViewer(img); return; }
      if (state.selected.has(img.id)) state.selected.delete(img.id);
      else state.selected.add(img.id);
      node.setAttribute('aria-selected', String(state.selected.has(img.id)));
      const counter = document.getElementById('sel-count');
      if (counter) counter.textContent = String(state.selected.size);
    },
  }, [
    lazyImg(img.thumb_url, ''),
    el('div', { class: 'tile__meta' }, [
      el('span', { text: pct(img.score) }),
      el('span', { text: img.cost_usd > 0 ? moneyExact(img.cost_usd) : 'gratis' }),
    ]),
  ]);
  return node;
}

function renderStep5(view) {
  clear(view);
  view.appendChild(stepHeader(view));
  const run = state.run;

  view.appendChild(note('ok', 'El robot ha terminado',
    `${run.accepted} imagenes superaron la revision de ${run.attempts} intentos. `
    + `Coste: ${run.spent_usd > 0 ? moneyExact(run.spent_usd) : 'gratis'}.`));

  view.appendChild(el('p', { class: 'muted',
    text: 'Toca las que te gusten y luego generalas en alta calidad.' }));

  view.appendChild(el('div', { class: 'grid' }, run.images.map((img) => tile(img, true))));

  // "NO, CAMBIA EL ESCOTE."  The next request starts from the image she
  // taps, not from the photograph: say what to change, the robot re-prices,
  // and the identity check still runs against her profile.
  const change = el('textarea', { rows: 2, placeholder: 'Que cambiar en la imagen elegida: "cambia el escote", "mas elegante", "en azul"...',
    style: { width: '100%', boxSizing: 'border-box' } });
  view.appendChild(el('div', { class: 'card', style: { marginTop: '12px' } }, [
    el('div', { class: 'section__title', text: 'Cambiar algo en una imagen' }),
    el('p', { class: 'tiny', text: 'Toca una imagen y dime que cambiar: la siguiente se hace a partir de ella.' }),
    change,
    el('button', { class: 'btn btn--secondary', type: 'button', style: { marginTop: '8px' },
      onClick: () => {
        const first = Array.from(state.selected)[0];
        if (!first) { toast('Toca primero la imagen que quieres cambiar.'); return; }
        state.sourceImage = first;
        state.choices = {};
        interpretAndPrice(view, change.value);
      } }, 'Cambiar sobre la imagen elegida'),
  ]));

  view.appendChild(el('div', { class: 'card', style: { marginTop: '16px' } }, [
    el('div', { class: 'card__row' }, [
      el('span', { text: 'Elegidas' }),
      el('strong', { id: 'sel-count', text: String(state.selected.size) }),
    ]),
  ]));

  view.appendChild(el('div', { class: 'btn-row' }, [
    el('button', { class: 'btn btn--secondary', type: 'button',
      onClick: () => showReport() }, 'Ver ficha'),
    el('button', { class: 'btn', type: 'button',
      onClick: () => runFinal(view) }, 'Alta calidad'),
  ]));

  view.appendChild(el('button', { class: 'btn btn--ghost', type: 'button',
    style: { marginTop: '10px' },
    onClick: () => { reset(); loadAndRender(view); } }, 'Empezar otra foto'));
}

async function runFinal(view) {
  if (!state.selected.size) { toast('Elige al menos una imagen'); return; }
  try {
    const data = await api.post('/api/generate/final', {
      run_id: state.plan.run_id,
      image_ids: Array.from(state.selected),
      quality: 'high',
    });
    state.plan = { ...state.plan, run_id: data.run_id };
    state.finalizing = true;
    state.step = 4;
    renderStep4(view, null);
    stopPolling();
    pollTimer = setInterval(() => pollRun(view), POLL_MS);
  } catch (err) {
    toast(err.message, 'danger');
  }
}

/* ------------------------------------------------------------------ ficha */

async function showReport() {
  let report;
  try {
    report = await api.get(`/api/generate/report/${state.plan.run_id}`);
  } catch (err) { toast(err.message, 'danger'); return; }

  const body = frag([
    kv('Intentos', String(report.intentos ?? '-')),
    kv('Aceptadas', String(report.aceptadas ?? '-')),
    kv('No mostradas', String(report.descartadas ?? '-')),
    kv('Retocadas solo por zonas', String(report.reparadas ?? 0)),
    report.intentos_por_foto ? kv('Intentos por foto', String(report.intentos_por_foto)) : null,
    report.segundos ? kv('Tiempo', `${report.segundos} s`) : null,
    kv('Coste real', report.coste_usd > 0 ? moneyExact(report.coste_usd) : 'gratis'),
    kv('Modelo usado', (report.modelos || []).join(', ') || '-'),
  ]);

  /* WHAT THE ROBOT CHANGED WHEN SOMETHING FAILED, AND WHETHER IT WORKED.
     The client asked that a rejection be fixed by adjusting the options rather
     than by paying for the same request again; a fix she is never shown is
     indistinguishable from a robot that simply spent more of her money.  It
     goes above the defect list because it is the sentence that explains why
     there is a second charge for one photograph. */
  for (const made of (report.ajustes || [])) {
    body.appendChild(el('div', { class: 'section__title',
      text: made.funciono ? 'Se cambio una cosa y salio bien'
        : 'Se cambio una cosa y aun asi no salio',
      style: { marginTop: '18px' } }));
    body.appendChild(el('div', {
      class: 'check-line ' + (made.funciono ? 'check-line--ok' : 'check-line--bad') }, [
      el('span', { class: 'check-line__mark', text: made.funciono ? '✓' : '✕' }),
      el('span', {}, [
        el('div', { text: made.texto }),
        made.medida ? el('div', { class: 'tiny', text: `Se decidio asi porque ${made.medida}.` }) : null,
      ]),
    ]));
  }

  const defects = Object.entries(report.defectos_detectados || {});
  if (defects.length) {
    body.appendChild(el('div', { class: 'section__title', text: 'Lo que se vio mal',
      style: { marginTop: '18px' } }));
    for (const [name, count] of defects) body.appendChild(kv(name, String(count)));
  }

  /* Found on pixels that were never redrawn: they come from her own camera, so
     the ficha says so instead of counting them against the robot's work. */
  const own = Object.entries(report.ya_en_tu_foto || {});
  if (own.length) {
    body.appendChild(el('div', { class: 'section__title',
      text: 'Visto en la parte que no se ha vuelto a dibujar',
      style: { marginTop: '18px' } }));
    body.appendChild(el('p', { class: 'tiny',
      text: 'Esa parte de la imagen son los pixeles de tu propia foto, tal cual '
        + 'los grabo tu camara: el robot no los ha tocado.' }));
    for (const [name, count] of own) body.appendChild(kv(name, String(count)));
  }

  if ((report.motivos_descarte || []).length) {
    body.appendChild(el('div', { class: 'section__title', text: 'Por que no te las ensenamos',
      style: { marginTop: '18px' } }));
    for (const item of report.motivos_descarte) {
      body.appendChild(el('div', { class: 'check-line check-line--bad' }, [
        el('span', { class: 'check-line__mark', text: '✕' }),
        el('span', {}, [
          el('div', { text: item.motivo }),
          ...(item.detalle || []).map((d) => el('div', { class: 'tiny', text: d })),
        ]),
      ]));
    }
  }

  const first = (report.imagenes || [])[0];
  if (first) {
    body.appendChild(el('div', { class: 'section__title', text: 'Comprobaciones',
      style: { marginTop: '18px' } }));
    for (const check of first.comprobaciones || []) {
      body.appendChild(el('div', {
        class: 'check-line ' + (check.paso ? 'check-line--ok' : 'check-line--bad') }, [
        el('span', { class: 'check-line__mark', text: check.paso ? '✓' : '✕' }),
        el('span', {}, [
          el('strong', { text: check.nombre }),
          el('div', { class: 'tiny', text: check.detalle || '' }),
        ]),
      ]));
    }
  }

  sheet({ title: 'Ficha de la tirada',
    subtitle: 'Lo que hizo el robot, con numeros reales.',
    body, actions: [{ label: 'Cerrar', kind: 'secondary' }] });
}

function openViewer(img) {
  const node = el('div', { class: 'viewer' }, [
    el('button', { class: 'viewer__close', type: 'button', 'aria-label': 'Cerrar',
      onClick: () => node.remove() }, '×'),
    el('div', { class: 'viewer__img' }, el('img', { src: img.url, alt: '' })),
    el('div', { class: 'viewer__bar' }, [
      el('a', { class: 'btn', href: `/api/album/${img.id}/download` }, 'Descargar'),
      el('button', { class: 'btn', type: 'button',
        onClick: async () => {
          try { await api.post(`/api/favorites/${img.id}`); toast('Guardada en favoritos', 'ok'); }
          catch (err) { toast(err.message, 'danger'); }
        } }, 'Favorito'),
    ]),
  ]);
  document.body.appendChild(node);
}

/* ------------------------------------------------------------------- page */

async function loadAndRender(view) {
  clear(view);
  view.appendChild(spinner('Cargando tus fotos...'));
  try {
    await loadOriginals();
  } catch (err) {
    clear(view);
    view.appendChild(note('danger', 'No se pudieron cargar tus fotos', err.message));
    return;
  }
  await renderStep1(view);
}

/* Arriving from the album with "Cambiar": the tapped result becomes the
   source of the next request, its photograph stays the reference for the
   options menu, and the screen opens on step 2 to hear what to change. */
async function mountFromImage(view, imageId) {
  reset();
  clear(view);
  view.appendChild(spinner('Preparando la imagen...'));
  try {
    await loadOriginals();
    const img = await api.get(`/api/album?limit=200`).then((d) => (d.images || []).find((i) => i.id === imageId));
    if (!img) { toast('Esa imagen ya no esta en el album.'); await renderStep1(view); return; }
    state.sourceImage = imageId;
    state.original = state.originals.find((o) => o.id === img.original_id) || state.originals[0] || null;
    state.analysis = null;
    await goStep2(view);
    toast('La siguiente imagen se hara a partir de la que has elegido en el album.', 'ok');
  } catch (err) {
    clear(view);
    view.appendChild(note('danger', 'No se pudo preparar la imagen', err.message));
  }
}

export default {
  async mount(view, params) {
    if (params && params.desde) { await mountFromImage(view, params.desde); return; }
    if (!state) reset();
    await loadAndRender(view);
  },
  unmount() { stopPolling(); },
};
