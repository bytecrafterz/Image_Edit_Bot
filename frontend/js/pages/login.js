/* One calm screen that toggles between entering and registering. */

import { api } from '../api.js';
import { router } from '../router.js';
import { store } from '../store.js';
import { el, clear, field, note, toast } from '../ui.js';
import { loadSession } from '../app.js';

let mode = 'login';

function render(view) {
  clear(view);

  const email = el('input', { type: 'email', name: 'email', autocomplete: 'email',
    inputmode: 'email', placeholder: 'tu@correo.com', required: true });
  const password = el('input', { type: 'password', name: 'password',
    autocomplete: mode === 'login' ? 'current-password' : 'new-password',
    placeholder: mode === 'login' ? 'Tu contrasena' : 'Al menos 8 caracteres',
    required: true });
  const name = el('input', { type: 'text', name: 'display_name',
    autocomplete: 'name', placeholder: 'Como quieres que te llamemos' });

  const error = el('div', { class: 'field__error', hidden: true });
  const submit = el('button', { class: 'btn', type: 'submit' },
    mode === 'login' ? 'Entrar' : 'Crear cuenta');

  const form = el('form', {
    onSubmit: async (event) => {
      event.preventDefault();
      error.hidden = true;
      submit.disabled = true;
      submit.textContent = 'Un momento...';
      try {
        const body = { email: email.value.trim(), password: password.value };
        if (mode === 'register') body.display_name = name.value.trim();
        const path = mode === 'login' ? '/api/auth/login' : '/api/auth/register';
        const data = await api.post(path, body);

        if (data.needs_approval) {
          clear(view);
          view.appendChild(el('div', { class: 'view' }, [
            el('h1', { text: 'Cuenta creada' }),
            note('info', 'Falta un paso',
              'Un administrador tiene que aprobar tu cuenta antes de que puedas '
              + 'entrar. Te avisara cuando este lista.'),
            el('button', { class: 'btn btn--secondary', type: 'button',
              onClick: () => { mode = 'login'; render(view); } }, 'Volver'),
          ]));
          return;
        }

        api.setToken(data.token);
        await loadSession();
        router.go('#/generate');
        toast('Hola de nuevo', 'ok');
      } catch (err) {
        error.textContent = err.message;
        error.hidden = false;
      } finally {
        submit.disabled = false;
        submit.textContent = mode === 'login' ? 'Entrar' : 'Crear cuenta';
      }
    },
  }, [
    mode === 'register' ? field('Tu nombre', name) : null,
    field('Correo electronico', email),
    field('Contrasena', password,
      mode === 'register' ? 'Minimo 8 caracteres.' : null),
    error,
    submit,
  ]);

  view.appendChild(el('div', { style: { maxWidth: '420px', margin: '0 auto',
    paddingTop: '8vh' } }, [
    el('div', { style: { textAlign: 'center', marginBottom: '26px' } }, [
      el('img', { src: '/icons/icon-192.png', alt: '', width: 64, height: 64,
        style: { margin: '0 auto 14px', borderRadius: '16px' } }),
      el('h1', { text: 'Photo Robot' }),
      el('p', { class: 'muted',
        text: 'Tus fotos, transformadas. Tu cuerpo y tu cara, intactos.' }),
    ]),
    el('div', { class: 'card' }, form),
    el('button', {
      class: 'btn btn--ghost', type: 'button',
      onClick: () => { mode = mode === 'login' ? 'register' : 'login'; render(view); },
    }, mode === 'login' ? 'No tengo cuenta todavia' : 'Ya tengo cuenta'),
  ]));
}

export default {
  async mount(view) { render(view); },
  unmount() {},
};
