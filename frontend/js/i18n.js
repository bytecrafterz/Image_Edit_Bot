/* Spanish is the product language; English exists as a fallback so a future
   client is not blocked. */

const ES = {
  'app.name': 'Photo Robot',
  'nav.create': 'Crear', 'nav.album': 'Album', 'nav.favorites': 'Favoritos',
  'nav.originals': 'Mis fotos', 'nav.settings': 'Ajustes', 'nav.admin': 'Admin',
  'common.cancel': 'Cancelar', 'common.save': 'Guardar', 'common.delete': 'Eliminar',
  'common.back': 'Atras', 'common.close': 'Cerrar', 'common.continue': 'Continuar',
  'common.loading': 'Cargando...', 'common.retry': 'Reintentar',
  'common.free': 'gratis', 'common.of': 'de',
  'auth.login': 'Entrar', 'auth.register': 'Crear cuenta',
  'auth.email': 'Correo electronico', 'auth.password': 'Contrasena',
  'auth.name': 'Tu nombre',
  'gen.title': 'Crear fotos', 'gen.step1': 'Elige la foto',
  'gen.step2': 'Que quieres cambiar', 'gen.step3': 'Confirma el coste',
  'gen.step4': 'El robot trabaja', 'gen.step5': 'Elige las buenas',
  'gen.generate': 'Generar', 'gen.stop': 'Detener',
  'gen.highQuality': 'Generar en alta calidad',
  'gen.report': 'Ver ficha',
  'balance.low': 'Saldo bajo', 'balance.zero': 'Sin saldo',
};

const EN = {
  'nav.create': 'Create', 'nav.album': 'Album', 'nav.favorites': 'Favourites',
  'nav.originals': 'My photos', 'nav.settings': 'Settings', 'nav.admin': 'Admin',
  'common.cancel': 'Cancel', 'common.save': 'Save', 'common.delete': 'Delete',
};

let locale = 'es';

export function setLocale(next) { locale = next === 'en' ? 'en' : 'es'; }

export function t(key, vars) {
  const table = locale === 'en' ? EN : ES;
  let text = table[key] || ES[key] || key;
  if (vars) {
    for (const [k, v] of Object.entries(vars)) {
      text = text.replace(new RegExp('\{' + k + '\}', 'g'), String(v));
    }
  }
  return text;
}
