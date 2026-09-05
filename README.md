# Photo Robot

Un sistema que **transforma tus fotografías reales**: analiza la foto, decide qué
debe conservarse, escribe el prompt, elige el modelo, genera, **revisa el
resultado con medidas**, corrige sólo la zona defectuosa y aprende de cada
corrección.

No es otro generador de imágenes. Es el control que hace el trabajo que hoy se
hace a mano, con un objetivo medible: **lo que cuesta 5 intentos manuales debe
costarle al robot 1 o 2**, con el coste de cada imagen a la vista.

---

## El principio

> El sistema **no se imagina tu cuerpo**. Parte de una fotografía real tuya y la
> transforma. Cuando cambia la ropa, el fondo o la luz, tus proporciones no se
> recalculan: vienen del píxel de tu foto.

Y encima de eso hay **medidas, no buenas intenciones**. A un generador se le
puede pedir amablemente que no te adelgace; sólo una medición demuestra que no
lo hizo.

---

## El recorrido

```
     tu foto real
          │
    1. ANALIZAR ──── pose, rostro, silueta, piel, calidad, tipo de plano
          │
    2. PLANIFICAR ── qué se fija, qué varía, cuántas imágenes, cuánto cuesta
          │
    3. DECIDIR ───── qué modelo y por qué  →  ¿hay saldo?  → si no, PARA y avisa
          │
    4. GENERAR ───── siempre a partir de la foto real, nunca desde cero
          │
    5. VERIFICAR ─── rostro · proporciones · tono de piel · anatomía · calidad
          │
    6. REPARAR ───── inpainting sólo en la zona rota, sin rehacer la foto
          │
    7. APRENDER ──── tus "me gusta" ajustan prompts y parámetros
          │
     imagen aceptada + ficha con números reales
```

---

## Tu cara no se vuelve a dibujar

Cuando lo que pides **no mueve a la persona dentro de la foto** — otra ropa,
otro color, otro fondo — el robot no regenera la imagen: dibuja una **máscara**
sobre la zona que cambia y deja fuera tu cara y tus manos. El motor repinta sólo
lo blanco de esa máscara y todo lo demás se copia **píxel a píxel de tu propia
fotografía**.

```
     tu foto 2316x3088
          │
     máscara: blanco = la ropa (22% de la foto en IMG_7871)
              negro  = tu cara, tus manos, el resto
          │
     fal-ai/flux-pro/v1/fill  ── 0.05 USD ── repinta sólo lo blanco
          │
     se pega dentro de tu foto ── y se comprueba que fuera de la máscara
                                   NO ha cambiado ni un píxel
          │
     archivo entregado: 2316x3088, el tamaño de tu cámara
```

Lo que eso significa, y está medido: tu **parecido facial no depende del
modelo** porque tu cara no pasa por él; tus **manos son las tuyas** y la ficha
lo dice con esas palabras en vez de avisarte de que no ha podido juzgarlas; tus
**tatuajes y tus pendientes** siguen ahí; y el archivo conserva la resolución de
tu foto en vez de bajar a 1024x1024.

Cuando pides **otra postura u otro encuadre** eso no se puede hacer: mover la
pose mueve tu cuerpo dentro del cuadro (medido en sus fotos: la cabeza se
desplaza 1,23 cabezas de mediana), así que hay que volver a dibujarte entera con
`kontext/multi` (0.04 USD, viajan 3 fotos tuyas) y **entonces sí se comprueba el
rostro antes de enseñarte nada**. La pantalla de coste te dice cuál de los dos
caminos vas a pagar, con el nombre del modelo, **antes** de gastar.

Verificado sin gastar: 3 regímenes x 192 peticiones enmascaradas, **0 píxeles
cambiados fuera de la máscara** en todas ellas. Sin gastar también está su
límite: de las 28 llamadas reales que esta cuenta ha hecho a `fill`, 26 fueron
reparaciones de zonas y las 2 únicas peticiones de ropa con máscara que han
salido de verdad las bloqueó el revisor de contenido de fal.

---

## Cómo se mide la identidad

Cada imagen generada se vuelve a medir con los mismos analizadores que
construyeron el perfil, y se compara con **la propia foto de origen**.

| Comprobación | Qué mide | Peso |
|---|---|---|
| `identity_face` | firma de reconocimiento facial SFace (128 valores) contra la media de tus fotos, mínimo 0.45 | 30% |
| `body_proportions` | anchos sobre la longitud del torso, el perfil de silueta a 9 alturas del torso y el perfil de figura a hasta 29 alturas medidas en cabezas | 25% |
| `skin_tone` | croma a*/b* con tolerancia adaptada a tu propia variación | 15% |
| `anatomy` | manos, dedos, extremidades, personas de más, texturas fundidas — y **declara la mano que no ha podido juzgar** en vez de darla por buena | 20% |
| `quality` | nitidez, exposición, contraste, ruido | 10% |

### La cara: reconocimiento, no parecido geométrico

`identity_face` usaba un descriptor geométrico de 64 valores que **no separaba a
nadie**: sus 24 fotografías puntuaban 0.9832 - 0.9993 y ocho fotografías de ocho
mujeres distintas 0.9577 - 0.9945, solapadas, con un mínimo exigido de 0.72 que
no podía dispararse. Desde septiembre de 2026 la firma es un **embedding SFace de
128 valores** (OpenCV Zoo, Apache-2.0, se ejecuta en local) comparado por coseno
contra la media de los embeddings de sus fotos, con el mínimo en **0.45**.

Barrido del 2026-09-04, con la comprobación que realmente decide: sus 24
fotografías 0.6908 - 0.8737 (24 de 24 aceptadas), dos fotos suyas que el perfil
no ha visto nunca 0.7624 y 0.8222 (aceptadas), ocho mujeres distintas 0.0201 -
0.1925 (8 de 8 rechazadas) y las cuatro imágenes de pago que hay en disco y no
son ella 0.2920 - 0.3968 (4 de 4 rechazadas). Entre la mejor cara ajena (0.3968)
y su peor fotografía propia (0.6908) no hay ninguna imagen: el límite, 0.45, se
pone dentro de esa franja vacía.

Los pesos no se versionan (37 MB); se descargan con
`python scripts\fetch_face_model.py`. Sin ellos la comprobación se informa como
**no realizada**, nunca como aprobada.

### Por qué "hombros contra caderas" no basta

Es la medida obvia y es inútil. Si una herramienta te estrecha **entera**, esa
proporción no cambia y la imagen pasa el control. Por eso todo se normaliza
contra la **longitud del torso** (punto medio de hombros → punto medio de
caderas), que no se mueve cuando te adelgazan:

`shoulder_w_over_torso`, `hip_w_over_torso`, `waist_w_over_torso`,
`bust_w_over_torso`, `head_h_over_torso`.

### Y la comparación que de verdad funciona

Comparar contra una media de población arrastra el ruido del encuadre, la
distancia y el giro del cuerpo. Pero el robot **siempre parte de una foto
concreta**, así que compara el resultado contra su propio origen: misma persona,
misma pose, mismo ángulo. Casi todo el ruido se cancela en la división.

Los números medidos están en **[docs/CALIBRACION.md](docs/CALIBRACION.md)** y se
reproducen con:

```
python scripts\calibrate_identity.py --paired --measurable-only
```

Resultado medido hoy (2026-09-05) sobre sus fotos reales, con el mismo comando:
**0% de falsas alarmas** (0 de 6) y una detección neta del **0% / 50% / 50%**
para un adelgazamiento del 6%, 12% y 18%, disparando siempre la comprobación
correcta — nunca rechaza por el motivo equivocado. Ese 50% es **cobertura, no
puntería**: de las seis fotos de prueba, en tres el encuadre no deja medir la
figura en cabezas, y en esas tres el robot lo dice en vez de fingir que ha
mirado. Un adelgazamiento del 6% **no se puede separar hoy** del volumen que
añade una prenda, y así está escrito en la ficha. El límite no es el método:
son las fotos de cuerpo entero que faltan (ver más abajo).

---

## Instalación en Windows

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Busca Python 3.12, crea el entorno virtual, instala todo y lo comprueba. Si no
encuentra Python, te da el enlace de descarga.

## Arrancar

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
```

Verás dos direcciones:

```
En este ordenador:   http://localhost:8080
Desde el movil:      http://192.168.1.42:8080
```

**Desde el iPhone**: abre la segunda dirección en Safari, estando en la misma
red wifi. Luego *Compartir → Añadir a pantalla de inicio* y queda como una app.

La **primera cuenta que se registre es la administradora**. Las siguientes
quedan pendientes de aprobación.

### Recarga automática

El servidor se reinicia solo al guardar cualquier archivo `.py`, así que un
cambio en el código se aplica sin tocar nada. Dos detalles que importan:

- **Un trabajo en curso se cancela** cuando el servidor se reinicia. No guardes
  código mientras se están generando imágenes: la tirada se corta y las
  imágenes ya cobradas se pierden.
- Sólo se vigila `backend/app`, y sólo archivos `.py`. La carpeta `data/`
  queda fuera a propósito: allí se escriben las fotos generadas y la base de
  datos, y vigilarla reiniciaría el servidor en bucle durante cada tirada.

Para un uso normal, sin recarga:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1 --no-reload
```

---

## Con claves y sin claves

El sistema funciona **completo desde el primer minuto sin ninguna clave**.

| | Sin claves | Con claves |
|---|---|---|
| Generación | motor local: fondo, color de ropa, luz, grado, encuadre | fal.ai: edición generativa e inpainting real |
| Lectura de la foto | visión por computador local | Claude lee la foto y redacta el prompt: **0.0112 USD, una sola vez por foto** — se guarda con la foto y la siguiente estimación de esa misma foto no cuesta nada — y se anota en tu saldo de Anthropic |
| Coste por imagen | **0 USD** | lo decide **lo que pides**, no el nivel: **0.05 USD** si sólo cambias ropa, color o escena (se repinta con máscara y tu cara no se vuelve a dibujar), **0.04 USD** si cambias la postura o el encuadre, **0.006 USD** en borrador; siempre a la vista antes de pagar |
| Recorrido completo | sí | sí |

El motor local **no es IA generativa** y no pretende serlo: es transformación
fotográfica real sobre tu foto. Sirve para probar el robot entero de punta a
punta sin gastar un céntimo, y para el uso gratuito.

Las claves se añaden en **Ajustes → Claves de IA**. Se guardan en el servidor, se
validan al guardarlas, y **nunca se devuelven al navegador**.

**Antes de poner la clave de pago, lee
[docs/PROCESAR_CON_CLAVE.md](docs/PROCESAR_CON_CLAVE.md)**: cómo se añade, qué
enseña la pantalla de coste y qué se gasta de verdad por imagen, qué nivel de
calidad elegir, qué mide el robot en cada imagen y qué hace cuando algo falla —
con los números medidos sobre las imágenes de pago que ya hay compradas. Incluye
el ensayo completo del camino de pago que **no gasta nada**:

```powershell
backend\.venv\Scripts\python.exe scripts\rehearse_paid.py
```

---

## El dinero

Tres reglas, escritas en el código y no sólo prometidas:

1. **Avisa antes de quedarse a cero**, diciendo cuánto queda y cuánto recargar.
2. **Al no llegar para la siguiente imagen, para y avisa**, en vez de seguir
   fallando imagen tras imagen.
3. **Nunca cobra en tu tarjeta.** No hay código de pago en este proyecto.
   "Registrar recarga" sólo **anota** el saldo que ya añadiste en la web del
   proveedor.

Además: límite diario, límite mensual, coste por imagen visible antes de
generar, y la métrica que importa — **intentos por foto conseguida**.

Los tres ajustes que gobiernan el gasto de una tirada — `max_retries`,
`max_repair_rounds` y `autorepair` — **llegan a la tirada desde el 2026-09-04**;
hasta entonces se guardaban y no los leía nadie. La misma lectura alimenta el
techo que se enseña antes de pagar, así que la promesa de la pantalla y lo que
la tirada puede llegar a gastar son un solo número: para 3 imágenes en alta,
1.7100 USD de serie, 0.2400 con `autorepair` apagado y 0.0800 con 0 reintentos y
0 reparaciones.

---

## Las pantallas

| Pantalla | Para qué |
|---|---|
| **Crear** | El recorrido en 5 pasos: foto → cambios → coste → trabajo → elegir |
| **Álbum** | Todo lo generado, con visor, descarga, favoritos, ficha y *Marcar final* (pasa a "Finales" por 0.00 USD lo que ya pasó todas las comprobaciones) |
| **Favoritos** | Lo que guardaste, con descarga en lote |
| **Mis fotos** | Fotos de referencia y el perfil de identidad |
| **Ajustes** | Saldo, avisos, uso, límites, comportamiento, claves, cuenta |
| **Admin** | Usuarios, aprobaciones, estadísticas, proveedores, auditoría |

La **ficha** de cada tirada muestra: intentos, tiempo, coste real, qué modelo se
usó y por qué, qué defectos se detectaron, qué se reparó con inpainting, y qué se
descartó con el motivo medido.

---

## Fotos de referencia: lo que hace falta

Para cuidar **cara, piel y marcas** basta con lo que ya hay. Para medir
**proporciones** hacen falta fotos donde se vean hombros y caderas a la vez.

De las 24 fotos actuales, **sólo una muestra los tobillos**. Por eso el sistema
dice abiertamente que el control de proporciones está limitado, en vez de
aparentar que funciona.

**6 u 8 fotos de cuerpo entero**, así:

- Teléfono apoyado a 2–3 m, a la altura del pecho.
- **Cámara trasera**, no la frontal.
- Teléfono en **vertical**, con temporizador.
- **De la cabeza a los pies**, con los pies dentro del cuadro.
- Brazos relajados a los lados, sin cruzarlos.
- Ropa ajustada, para que se vea la silueta real.
- Luz natural de ventana, sin flash.
- **Sin filtros y sin embellecedor.**
- Dos de frente, dos de perfil, dos de tres cuartos.

Cuando lleguen, vuelve a ejecutar la calibración: se espera que la detección
suba bastante, porque la limitación es de datos, no de método.

---

## Varias personas

Está pensado así desde el principio. Cada persona tiene su **perfil
independiente** con sus propias medidas, y nada se mezcla.

Y una vez medido el perfil, **las fotos originales pueden borrarse**: lo que se
guarda son los números, no las imágenes. *Mis fotos → Olvidar fotos originales*.

Cada perfil guarda además un **registro de consentimiento**: quién lo dio, la
relación (uno mismo o cliente), cuándo y para qué. Un perfil de otra persona sin
ese registro no genera nada.

---

## Lo que no hace, y por qué

**No genera imágenes íntimas ni de contenido sexual de personas reales.**

No es desconfianza hacia nadie en concreto. Es que la herramienta no distingue
entre las fotos de su dueña y las de cualquier otra persona que suba unas
cuantas imágenes, y ese tipo concreto de sistema es el que se usa para hacerle
eso a gente que no ha dado permiso. La regla vive en
[`backend/app/safety/guard.py`](backend/app/safety/guard.py) para que la cumpla
el programa y no la memoria de una persona.

Sí hace: moda, editorial, retrato, producto, escenarios, ropa, luz, poses,
cuerpo entero — con tu cuerpo real, tu tono de piel, tus proporciones y tus
tatuajes, comprobados con números en cada foto.

---

## Cambiar de proveedor

Un archivo y una línea.

1. Escribe `backend/app/providers/mi_proveedor.py` con una clase que implemente
   `ImageProvider` de [`providers/base.py`](backend/app/providers/base.py):
   `capabilities()`, `available()`, `estimate_cost()`, `generate()` y, si puede,
   `inpaint()`.
2. Regístrala en `providers/registry.py`.

El orquestador nunca nombra a un proveedor: pide "uno que sepa hacer inpainting a
esta calidad y dentro de este presupuesto". Cambiar de vendedor no toca el resto.

---

## Estructura

```
backend/app/
  analysis/     medir píxeles: pose, rostro, cuerpo, piel, silueta, calidad, defectos
  identity/     construir el perfil de una persona y verificar cada imagen
  generation/   prompt, plan, enrutado, reparación, aprendizaje, orquestador
  providers/    motor local gratuito, fal.ai, Claude, visión heurística, registro
  catalog/      opciones (105) y estilos (22)
  safety/       límite de contenido y consentimiento
  services/     almacenamiento, dinero y avisos, trabajos en segundo plano
  routers/      la API HTTP (58 endpoints)
frontend/       app web móvil, sin compilación ni dependencias
scripts/        instalación, arranque, importación, pruebas, calibración
data/           base de datos, fotos, resultados (no se versiona)
```

---

## Comprobarlo

```powershell
# Prueba completa de punta a punta, sin servicios externos, coste 0
backend\.venv\Scripts\python.exe scripts\e2e_test.py

# ¿De verdad detecta que te han adelgazado?
backend\.venv\Scripts\python.exe scripts\calibrate_identity.py --paired --measurable-only

# Importar una carpeta de fotos y medir el perfil
backend\.venv\Scripts\python.exe scripts\import_nayane.py

# Ensayo completo del camino de pago (clave falsa, red cerrada, coste 0 USD)
backend\.venv\Scripts\python.exe scripts\rehearse_paid.py

# Sus 24 fotos a tres tamaños + las caras que no son suyas (0 de 72 rechazadas)
backend\.venv\Scripts\python.exe scripts\sweep_identity.py

# Que dos cuentas no se vean nada la una a la otra
backend\.venv\Scripts\python.exe scripts\multiuser_test.py
```

---

## API

Todo bajo `/api`. Autenticación por `Authorization: Bearer <token>` o cookie
`pr_session`. Errores: `{"detail": "mensaje en español"}`.

| Método | Ruta | Para qué |
|---|---|---|
| POST | `/auth/register` · `/auth/login` · `/auth/logout` | cuenta y sesión |
| GET | `/auth/me` | usuario, saldos, avisos sin leer |
| GET/POST | `/profiles` | perfiles de identidad |
| POST | `/profiles/{id}/build` | medir a la persona |
| POST | `/profiles/{id}/forget-originals` | borrar fotos, conservar medidas |
| GET/POST | `/originals` | fotos de referencia |
| GET | `/originals/{id}/analysis` | lectura completa de una foto |
| GET | `/catalog/options` · `/catalog/styles` | menú adaptado a la foto |
| POST | `/generate/analyze` | planificar y **presupuestar** (no gasta) |
| POST | `/generate/run` | ejecutar (aquí sí se gasta) |
| GET | `/generate/status/{id}` | progreso para el móvil |
| POST | `/generate/final` | alta calidad de las elegidas |
| GET | `/generate/report/{id}` | la ficha |
| GET/DELETE | `/album` · `/favorites` | biblioteca |
| GET/PUT | `/settings` | ajustes, claves, uso, recargas, avisos |
| GET/POST | `/admin/*` | usuarios, estadísticas, proveedores, auditoría |

Documentación viva en `http://localhost:8080/api/docs`.

---

## Si algo falla

| Síntoma | Qué mirar |
|---|---|
| `Python no encontrado` | Instala 3.12 marcando *Add python.exe to PATH* |
| El puerto está ocupado | `start.ps1 --port 8123` |
| No se abre desde el móvil | Misma wifi; abre el puerto en el firewall de Windows |
| `mediapipe` no importa | Python debe ser 3.12 de 64 bits; reinstala con `setup.ps1` |
| Todas las imágenes se descartan | Ajustes → Estrictez → *Suave*, y revisa la ficha |
| No mide proporciones | Faltan fotos de cuerpo entero (ver arriba) |
| El robot se detuvo solo | Falta saldo: Ajustes → Registrar recarga |

Los registros están en `data/logs/app.log`.
