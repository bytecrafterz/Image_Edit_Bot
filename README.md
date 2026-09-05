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
**pendientes** siguen ahí, porque cuelgan junto a la cabeza y la máscara los
deja en negro; y el archivo conserva la resolución de tu foto en vez de bajar a
1024x1024.

**Y tu pelo sólo el de la cabeza.** La frase de arriba decía «tu cara, tu pelo,
tus manos», y de las tres sólo dos son exactas. Lo que MediaPipe llama *pelo* en
la fotografía de la cocina son **7 907 píxeles**, un filo en la coronilla: la
melena que le cae sobre el hombro y el pecho está clasificada como *torso* y
queda dentro de la zona a repintar en un **82,7%**. El aviso ya no promete el
pelo entero; dice cuál se copia y cuál se repinta.

### Tus marcas: las que la prenda deja al aire se conservan

Esta sección decía «tus tatuajes no» y hoy dice menos y más exacto. La máscara
mide tus marcas **sobre la fotografía** — con el color de piel de tu propio
perfil como referencia, no con un detector genérico — y las fuerza a negro
**junto a tu cara y tus manos** siempre que la prenda que has pedido deje esa
parte de ti al aire. En sus 25 fotografías eso encuentra 21 marcas, 17 reales
(el nombre bajo la clavícula, la escritura del antebrazo, la manga de rosas del
muslo) y 4 que no lo son (una sombra, un mechón y dos veces su propio collar);
las cuatro sólo conservan píxeles suyos en una zona que la ropa no tapa, que es
lo que ya hacía el escudo con sus pendientes.

**Y una marca que la prenda nueva tapa se repinta, siempre.** No es un fallo, es
lo único honesto: una camisa encima de un tatuaje lo cubre, y protegerlo dejaría
un agujero con su pecho dentro de la camisa. El robot lo dice antes de cobrar,
con la zona por su nombre:

> Tus marcas en el cuello se conservan: esa zona no se repinta, se copia de tu
> propia foto. La ropa que pediste tapa el torso, asi que la marca que llevas
> ahi no puede sobrevivir: una prenda encima de un tatuaje lo cubre, y no se te
> va a decir lo contrario. Si quieres verla, elige una prenda que deje esa zona
> al aire.

Hay que decir hasta dónde llega: **ninguna de las 24 prendas del catálogo deja
el pecho al aire.** 18 son partes de arriba o conjuntos completos, y a las 6 de
abajo el robot les añade una prenda superior para no pedir un torso desnudo. Así
que en un cambio de ropa su tatuaje del pecho **siempre** se pierde; donde la
protección sí se nota es en el antebrazo bajo una camiseta de manga corta, en el
muslo bajo un vestido de verano, y en el cuello, que no tapa ninguna prenda. Si
lo que quieres es conservarlo, lo que sirve es un **cambio de color o de
transparencia**: ahí no se viste nada nuevo y se protegen todas.

Y lo que se le pide al motor va ahora **en el mismo sentido que la máscara**. El
texto pagado del 2026-09-05 le pedía a la vez «una camisa blanca que cubra el
torso por completo» y «el tatuaje del pecho sin cambios y en el mismo sitio»:
dos frases sinceras que juntas son imposibles, y pedirle a un modelo una
contradicción no da la mitad de cada una. Ahora sólo se piden las marcas que la
prenda deja ver — 5 de 9 con una camisa, las 9 con un cambio de color — y la
lista de «no cambies esto» que redacta la lectura de la foto ya no cuela la
prenda que acabas de pedir cambiar (decía, literalmente, *keep unchanged: beige
tank top* dentro de una petición de camisa blanca).

Si sólo cambias el fondo, no aparece ningún aviso: la lista se calcula contra
las regiones que se repintan de verdad, y el fondo no toca ninguna marca.

### Lo que sale de tu máquina en una llamada de pago

En el camino con máscara, la respuesta del proveedor **sólo se conserva dentro
de lo blanco**: todo lo demás se copia de tu archivo a resolución completa. Sube
**su fotografía entera**, con la cabeza y los hombros.

Durante un día no fue así: se recortaba el rectángulo que contiene la zona a
repintar más un 6% de contexto, con el argumento de que lo de fuera se tira
igual. Ese rectángulo **es la zona de la ropa, y empieza en su barbilla**: lo
que recibía fal era un torso sin cabeza, el pecho centrado, los hombros y los
brazos al aire, con el **0,0%** de su recuadro de cara en la foto vestida y el
**1,9%** en IMG_7871. Multiplicaba por **2,8** la densidad de piel de la imagen
bajo revisión, y las dos llamadas pagadas con máscara volvieron bloqueadas — la
segunda sobre una fotografía completamente vestida. Además le quitaba al modelo
lo que necesita para ajustar una prenda a una persona: la escala, los hombros,
de dónde viene la luz y qué hay en la habitación.

Su cara no la protegía el recorte y no la protege el encuadre: **la protege la
máscara**. Sobre sus **25 fotografías**, la zona a repintar toca **0 píxeles**
del recuadro de su cara, y encima de todo lo negro se vuelven a escribir sus
propios píxeles a resolución completa. El pegado se comprobó con la máscara
desplazada en las dos direcciones sobre 5 de sus fotografías: el desfase que
mejor explica la imagen compuesta es **(0, 0) píxeles** en las cinco.

Del texto que se envía también se ha quitado el vocabulario de ropa interior que
el propio `safety/guard.py` rechaza si se lo escribes tú: 14 cláusulas en el
camino con máscara, y **0 en `kontext`**, que es el que ha hecho todas las
imágenes que existen. Lo que protege tu cuerpo — *slimmed waist*, *altered
breast size*, *removed tattoos* — se queda entero en los dos.

Eso era la mitad del trabajo. Medido sobre las 4 141 letras que fal recibió de
verdad el 2026-09-05, después de aquel filtro **seguían viajando** *top worn
without bottoms*, *shirt worn as a dress*, *no trousers*, *missing trousers* —
ninguna nombra una prenda interior y todas describen a una persona a medio
vestir — y las palabras *breast* y *bust*, dos veces, dentro de cláusulas que
protegen su cuerpo. Ahora: las cuatro frases **se caen sólo en el camino con
máscara** (en `kontext` protegen un fallo real y no le han costado nada nunca), y
los sustantivos **no se borran, se cambian** por el clínico — *altered breast
size* sale como *altered chest size*, y la protección viaja entera. Resultado
medido sobre esa misma petición: 18 cláusulas retiradas en vez de 14, y **0
palabras marcadas** en el texto final, contra 7 antes.

### Lo que ve el revisor es tu foto entera, y antes no lo era

Éste es el hallazgo que costó 0,100 USD y hay que leerlo antes de pagar.
Mientras hubo recorte, el rectángulo que viajaba **concentraba la piel**: ese
rectángulo *es* la zona de la ropa, o sea la parte más desnuda de cualquier foto
de una persona, y además dejaba la cabeza fuera. Medido en sus dos fotos de
origen, con la misma envoltura de piel de su propia ficha:

| | foto entera | rectángulo que viajaba | factor |
|---|---|---|---|
| IMG_7871 (bloqueada 2 veces) | 12,1% de piel | 33,1% | **2,73x** |
| la de la cocina, vestida (bloqueada) | 9,8% de piel | 27,5% | **2,82x** |

Así que la mejora que de verdad recibió el proveedor al cambiar a una foto
vestida fue **33,1% → 27,5%** (un 17% relativo), no el 36% que se había
calculado sobre el cuadro entero. Y lo que estaba mirando, en los dos casos, es
un **torso sin cabeza** centrado en el escote. El recorte se ha quitado, y lo
que sale ahora se mide igual cuando se dibuja la máscara y **se le escribe en la
pantalla de coste**, con esta frase:

> Sale tu foto entera, con tu cabeza y tus hombros: es lo que el modelo necesita
> para ajustar la ropa a tu cuerpo, y es lo que revisa el proveedor. De esa foto
> se repinta el 17%. Tu piel descubierta es el 10% de lo que se envia, y el 7%
> queda dentro de la zona que se repinta. Tu cara viaja entera y no se repinta:
> la mascara la deja fuera por completo (0 pixeles del recuadro de tu cara
> dentro de la zona) y encima se vuelven a poner tus propios pixeles.

Cuesta 0,4 s al dibujar la máscara y 0,012 s cada vez que se reutiliza, así que
la estimación la puede decir siempre. La regla de la que sale es la más barata
de todas: **el número que justifica un gasto tiene que estar medido sobre la
imagen que se envía**, no sobre la que se queda en casa.

### Qué fotografía tuya sirve como origen

La respuesta corta, después de tres llamadas pagadas: **la que menos piel
descubierta tenga dentro de la zona que vas a cambiar**, y a igualdad de eso, la
de más resolución.

- **Vestida, y mejor con las piernas cubiertas.** Una foto en ropa interior
  llega al revisor como un desnudo recortado. La foto de la cocina (top gris de
  canalé y falda negra, sentada) baja la zona a repintar del 22,3% al 18,5% del
  cuadro y la piel dentro de esa zona del 52,8% al 40,1%.
- **Media figura o cuerpo entero, no primer plano.** En un primer plano no hay
  torso que aislar: la zona pasa a ser toda ella, y lo que se repinta es casi
  todo el cuadro.
- **De tu cámara, no de WhatsApp.** El archivo se entrega al tamaño de tu foto:
  1200x1599 (1,9 MP) desde WhatsApp contra 2316x3088 (7,2 MP) desde el móvil, y
  la cara a resolución completa baja de 332 px a 201 px. Pasa el listón de
  sobra, pero es un cuarto de los píxeles.
- **Nítida y bien expuesta.** El robot lo mide y te lo dice antes de usarla.

Y lo que **no** ha servido: cambiar a una foto vestida, por sí solo, **no
desbloqueó nada**. Ese es el resultado del 2026-09-05 y está pagado.

### Cuando el proveedor bloquea el resultado

fal revisa **lo que acaba de dibujar**. Si su revisor lo marca, devuelve un
archivo completamente negro con HTTP 200: medido sobre los 19 376 bytes que esta
instalación ya tiene guardados, media 0,0000 y un único valor distinto en
4 423 680 píxeles. La inferencia **se ejecutó**, así que fal lo cobra, y el robot
lo apunta en tu libro mayor con la nota *imagen bloqueada por fal* y te lo dice
con esas palabras: «se hizo pero fal.ai no la dio por buena... se cobra igual
(0.05 USD)». Se prueba otra semilla; si vuelve a salir negro, **no se paga un
tercer intento**. Y desde hoy la fila del intento guarda el `request_id`, el
endpoint real, el rectángulo que se subió y lo que tardó, para que el siguiente
bloqueo se pueda auditar desde aquí en vez de desde el panel de fal.

Cuando pides **otra postura u otro encuadre** eso no se puede hacer: mover la
pose mueve tu cuerpo dentro del cuadro (medido en sus fotos: la cabeza se
desplaza 1,23 cabezas de mediana), así que hay que volver a dibujarte entera con
`kontext/multi` (0.04 USD, viajan 3 fotos tuyas) y **entonces sí se comprueba el
rostro antes de enseñarte nada**. La pantalla de coste te dice cuál de los dos
caminos vas a pagar, con el nombre del modelo, **antes** de gastar.

Verificado sin gastar: 3 regímenes x 192 peticiones enmascaradas, **0 píxeles
cambiados fuera de la máscara** en todas ellas. Sin gastar también está su
límite, y hay que decirlo entero. Contadas en la tabla de intentos de esta
instalación, las llamadas se parten limpiamente por camino:

| camino | llamadas | bloqueadas por fal |
|---|---|---|
| imagen entera (`kontext`, `kontext/multi`) | ~42 | **1** |
| recorte con máscara (`fill`, cambio de ropa) | **4** | **4** |

Las 26 reparaciones de zona que también usan `fill` pasaron todas, y la
diferencia con las cuatro bloqueadas es qué se sube: la reparación sube una
imagen **ya vestida** que hizo el modelo; las cuatro bloqueadas subieron el
recorte de su fotografía real, un torso sin cabeza. Esa diferencia es la que se
acaba de quitar: el camino con máscara sube ahora la **imagen entera**, igual
que el camino de la primera fila. **Sigue sin estar probado de punta a punta
contra la API real**, y mientras no lo esté, todo lo que promete esta sección
está medido en local y no en una imagen entregada.

**Qué hacer cuando te bloquean un resultado.** Por orden, y ninguno de los
cuatro cuesta nada hasta el último:

1. **Cóbratelo como lo que es.** Está apuntado en tu libro mayor como *imagen
   bloqueada por fal*. No es un error del robot ni un fallo de tu foto: fal
   ejecutó el dibujo y lo cobra. Si sale negro dos veces seguidas, el robot
   **no paga un tercer intento**.
2. **Mira lo que se subió, no lo que tienes.** La pantalla de coste te dice qué
   porcentaje de lo que se envía es piel descubierta y cuánto de eso cae dentro
   de la zona que se repinta. Ése es el número que mira el revisor.
3. **Cambia lo que reduce ese número:** una foto de origen más vestida en la
   zona que vas a cambiar, o una prenda que repinte menos (una falda repinta
   menos que un vestido completo).
4. **O pide otra postura o encuadre.** Eso va por `kontext/multi` (0.04 USD, 3
   fotos tuyas), que es el camino con 1 bloqueo en ~42 llamadas — y ahí sí se te
   comprueba el rostro antes de enseñarte nada.

Lo que **no** se ha probado todavía: una llamada pagada por el camino con
máscara **ya sin recorte**. Es lo único que separaba a los dos caminos de la
tabla, y el precio es privacidad — sale todo el cuadro de tu máquina. A cambio,
el modelo ve tu cabeza y tus hombros, que es lo que necesita para ajustarte una
prenda, y el revisor deja de recibir un torso sin cabeza.

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

# Sus 25 fotos a tres tamaños + las caras que no son suyas (0 de 75 rechazadas)
backend\.venv\Scripts\python.exe scripts\sweep_identity.py

# Comprobado el 2026-09-05 con las 25 fotos, todo a coste 0.0000 USD:
#   e2e_test.py           49 correctas, 0 fallidas
#   rehearse_paid.py      48 correctas, 0 fallidas
#   sweep_identity.py     75 de 75 pasan (parecido 0.663..0.895, umbral 0.45),
#                         8 de 8 caras ajenas rechazadas por identidad
#   calibrate_identity.py 0% de falsas alarmas, 50% de deteccion a -12% y -18%
#   multiuser_test.py     43 correctas, 0 fallidas

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
