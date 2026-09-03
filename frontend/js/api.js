/* Thin HTTP client.  Every call goes through here so that a lost session or a
   dead network produces one predictable, Spanish, non-technical message. */

const TOKEN_KEY = 'pr_token';
const TIMEOUT_MS = 20000;
const UPLOAD_TIMEOUT_MS = 180000;

export class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.status = status;
    this.payload = payload || {};
  }
}

function getToken() {
  try { return localStorage.getItem(TOKEN_KEY) || ''; } catch { return ''; }
}

function setToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch { /* private browsing */ }
}

function headers(extra) {
  const out = Object.assign({}, extra || {});
  const token = getToken();
  if (token) out.Authorization = 'Bearer ' + token;
  return out;
}

async function request(method, path, body, opts) {
  const options = opts || {};
  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    options.timeout || (body instanceof FormData ? UPLOAD_TIMEOUT_MS : TIMEOUT_MS)
  );

  let response;
  try {
    response = await fetch(path, {
      method,
      credentials: 'same-origin',
      signal: controller.signal,
      headers: body instanceof FormData
        ? headers()
        : headers(body === undefined ? {} : { 'Content-Type': 'application/json' }),
      body: body instanceof FormData ? body
        : (body === undefined ? undefined : JSON.stringify(body)),
    });
  } catch (err) {
    clearTimeout(timeout);
    if (err.name === 'AbortError') {
      throw new ApiError('La conexion tarda demasiado. Intentalo otra vez.', 0);
    }
    throw new ApiError('No hay conexion con el servidor.', 0);
  }
  clearTimeout(timeout);

  if (response.status === 204) return null;

  let payload = null;
  const type = response.headers.get('content-type') || '';
  if (type.includes('application/json')) {
    try { payload = await response.json(); } catch { payload = null; }
  }

  if (response.status === 401) {
    setToken('');
    if (!location.hash.startsWith('#/login')) location.hash = '#/login';
    throw new ApiError('Tu sesion ha caducado. Entra de nuevo.', 401, payload);
  }
  if (!response.ok) {
    const detail = payload && payload.detail
      ? (typeof payload.detail === 'string' ? payload.detail : 'No se pudo completar.')
      : 'No se pudo completar la operacion.';
    throw new ApiError(detail, response.status, payload);
  }
  return payload;
}

export const api = {
  get: (path, opts) => request('GET', path, undefined, opts),
  post: (path, body, opts) => request('POST', path, body === undefined ? {} : body, opts),
  put: (path, body, opts) => request('PUT', path, body, opts),
  patch: (path, body, opts) => request('PATCH', path, body, opts),
  del: (path, opts) => request('DELETE', path, undefined, opts),

  upload(path, files, extra) {
    const form = new FormData();
    for (const file of files) form.append('files', file, file.name);
    if (extra) for (const [k, v] of Object.entries(extra)) form.append(k, v);
    return request('POST', path, form);
  },

  qs(path, params) {
    const search = new URLSearchParams();
    for (const [k, v] of Object.entries(params || {})) {
      if (v !== undefined && v !== null && v !== '') search.append(k, v);
    }
    const query = search.toString();
    return query ? `${path}?${query}` : path;
  },

  getToken, setToken,
  isLoggedIn: () => Boolean(getToken()),
};
