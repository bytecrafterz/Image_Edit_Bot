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

## Cómo se mide la identidad

Cada imagen generada se vuelve a medir con los mismos analizadores que
construyeron el perfil, y se compara con **la propia foto de origen**.

| Comprobación | Qué mide | Peso |
|---|---|---|
| `identity_face` | firma de reconocimiento facial SFace (128 valores) contra la media de tus fotos, mínimo 0.45 | 30% |
| `body_proportions` | anchos sobre la longitud del torso, el perfil de silueta a 9 alturas del torso y el perfil de figura a hasta 29 alturas medidas en cabezas | 25% |
| `skin_tone` | croma a*/b* con tolerancia adaptada a tu propia variación | 15% |
| `anatomy` | manos, dedos, extremidades, personas de más, texturas fundidas | 20% |
| `quality` | nitidez, exposición, contraste, ruido | 10% |

### La cara: reconocimiento, no parecido geométrico

`identity_face` usaba un descriptor geométrico de 64 valores que **no separaba a
nadie**: sus 24 fotografías puntuaban 0.9832 - 0.9993 y ocho fotografías de ocho
mujeres distintas 0.9577 - 0.9945, solapadas, con un mínimo exigido de 0.72 que
no podía dispararse. Desde septiembre de 2026 la firma es un **embedding SFace de
128 valores** (OpenCV Zoo, Apache-2.0, se ejecuta en local) comparado por coseno
contra la media de los embeddings de sus fotos, con el mínimo en **0.45**.

Sobre las mismas imágenes: sus fotos 0.6740 - 0.8718 (24 de 24 aceptadas), ocho
mujeres distintas 0.0194 - 0.1948 (8 de 8 rechazadas) y las dos imágenes de pago
que la clienta rechazó a mano 0.1829 y 0.2862 (rechazadas). Entre la mejor cara
ajena y su peor fotografía propia no hay ninguna imagen: el límite se pone dentro
de esa franja vacía.

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

Resultado actual sobre las fotos reales: **0% de falsas alarmas** y una
detección neta del **83% / 100% / 83%** para un adelgazamiento del 6%, 12% y
18%, disparando siempre la comprobación correcta. El límite hoy no es el
método: es que faltan fotos de cuerpo entero (ver más abajo).

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
| Lectura de la foto | visión por computador local | Claude lee la foto y redacta el prompt |
| Coste | **0 USD** | según modelo, siempre a la vista |
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

---

## Las pantallas

| Pantalla | Para qué |
|---|---|
| **Crear** | El recorrido en 5 pasos: foto → cambios → coste → trabajo → elegir |
| **Álbum** | Todo lo generado, con visor, descarga, favoritos y ficha |
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
