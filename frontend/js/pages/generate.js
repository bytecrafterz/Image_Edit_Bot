/* The main screen: pick a photo, say what to change, see the price, watch the
   robot work, keep the good ones.

   The one rule that shapes this page: nothing is spent until she presses
   "Generar", and the amount is on screen before she does. */

import { api } from '../api.js';
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

function renderStep2(view) {
  clear(view);
  view.appendChild(stepHeader(view));

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
      original_id: state.original.id,
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
  for (const warning of (plan.warnings || [])) {
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

/* ------------------------------------------------------------------ step 4 */

async function startRun(view) {
  try {
    await api.post('/api/generate/run', { run_id: state.plan.run_id });
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

async function pollRun(view) {
  try {
    const run = await api.get(`/api/generate/status/${state.plan.run_id}`);
    state.run = run;
    if (['done', 'failed', 'cancelled', 'stopped_no_balance'].includes(run.status)) {
      stopPolling();
      if (run.status === 'done' && run.images.length) {
        state.step = 5;
        renderStep5(view);
        return;
      }
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

  /* THE RUN FINISHED AND NO IMAGE PASSED.  Without this she was left looking
     at "Listo: 0 imagenes" and a list of checks, with nothing to do next. */
  if (run && run.status === 'done' && !run.images.length) {
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

export default {
  async mount(view) {
    if (!state) reset();
    await loadAndRender(view);
  },
  unmount() { stopPolling(); },
};
