# Cuando lleguen las fotos nuevas de Nayane

Orden de trabajo, de principio a fin. Todo lo que gasta dinero esta marcado.

## 1. Fotos suyas (identidad)

1. Subirlas a la cuenta que hace la demo (admin) en Originales, o que ella las suba a la suya.
   Sirven: cuerpo entero de frente y de tres cuartos, medio cuerpo, primer plano; sin filtros, sin recortes, luz pareja.
2. Administracion -> cuenta -> "Reconstruir perfil". Comprobar en Originales que cada foto nueva dice "medible" y que el perfil pasa de 24 a N fotos.
3. Saldo: fal.ai (imagenes) y Anthropic (lectura de fotos, 0.011 USD por foto, se cobra una vez). Limite diario en Ajustes: subirlo antes de un lote grande.

## 2. Sus cuatro vestidos (las fotos de muestra que envio)

Cada vestido debe existir como valor CON su imagen, para que el motor vea la prenda y no solo la descripcion:

- desde la app: paso 2 -> "Foto de una prenda" -> elegir la imagen -> queda como opcion propia; o
- desde el servidor, como usuario del servicio (gasta ~0.01 USD por prenda en Claude):

      sudo -u photorobot PHOTOROBOT_DATA=/opt/photorobot/data \
        /opt/photorobot/backend/.venv/bin/python /opt/photorobot/scripts/register_garment.py <user_id> /ruta/vestido_rojo.jpg "vestido rojo cruzado"

Las cuatro descripciones ya existen sin imagen como `vestido_rojo_cruzado`, `vestido_blanco_escote`, `vestido_negro_solapa`, `vestido_negro_escote`.

## 3. Lote de prueba (gasta dinero)

Con una sesion de admin guardada en un tarro de cookies (Mozilla):

    python scripts/demo_batch.py cookies.txt lote1.json --shots full,half --engine identity_banana --n 2

Por defecto: cada foto de cuerpo entero o medio cuerpo, con cada vestido, restaurante y calle alternando. Precio aproximado: 0.039 USD por imagen (Gemini). Si se registraron los vestidos con imagen, usar sus claves `mi_...` con `--dresses`.

## 4. Seleccion y entrega

    sudo -u photorobot /opt/photorobot/backend/.venv/bin/python /opt/photorobot/scripts/demo_bundle.py \
        --min 0.80 --name nayane-lote1 --db /opt/photorobot/data/photorobot.sqlite3 --out /opt/photorobot/data/demo lote1.json

Descarga desde cualquier PC, con sesion de admin: `https://fotografica.duckdns.org/api/admin/demo/nayane-lote1-<fecha>.zip`

## 5. Que comprueba el sistema desde 2026-09-12

- cara: la linea de identidad (0.45) y la de parecido (0.60); por debajo de 0.60 se descarta y se repite la semilla
- cuerpo: la silueta en cabezas contra la foto de origen, limite 8%; la ropa excusa un ensanchamiento hasta el 15% y nunca un estrechamiento; alturas discordantes ya no excusan mas de un 16%
- referencias: en una foto de cuerpo entero o medio cuerpo, una de las fotos de referencia es siempre de cuerpo entero
- finales: ampliacion 2x (fal clarity, ~0.19 USD por imagen de 1024x1536); si la cara baja mas de 0.05 se entrega la de 1x. Se apaga con `upscale_finals=false` en Ajustes
