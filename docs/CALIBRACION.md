# Calibración del control de identidad

Este documento recoge **medidas reales**, no opiniones, sobre si el control
automático de identidad funciona. Cubre las dos mitades del control, y cada una
se reproduce por su lado:

* **el cuerpo** (¿detecta que la han adelgazado?) con

  ```
  python scripts\calibrate_identity.py --paired --measurable-only
  ```

* **el rostro** (¿distingue su cara de la de otra persona?) puntuando con los
  módulos del robot las fotos que ya están en disco; la receta exacta está en
  «Cómo reproducirlo», más abajo.

Ninguna de las dos llama a un servicio de pago ni saca una imagen de la máquina.
Reprodúcelas cada vez que cambien los umbrales o lleguen fotos nuevas.

---

## Qué se mide y por qué

La queja concreta de la clienta fue que una herramienta anterior la **adelgazó
sin que ella lo pidiera**, y no tuvo forma de demostrarlo. El sistema afirma que
detecta eso automáticamente. Esa afirmación hay que probarla.

El guion:

1. Construye el perfil de identidad con una parte de las fotos reales.
2. Aparta otras fotos que el perfil **no** ha visto.
3. Fabrica el fallo exacto: comprime el cuerpo horizontalmente hacia su propio
   eje un porcentaje conocido (6%, 12%, 18%), **dejando la cara intacta**, que es
   justo lo que hacen los filtros de belleza y por eso un control que solo mire
   la cara no se entera.
4. Pasa el control real sobre las fotos originales y sobre las adelgazadas.

Dos números importan, y tiran en direcciones opuestas:

| número | significado | objetivo |
|---|---|---|
| **Falsas alarmas** | fotos reales que el control rechaza | lo más bajo posible |
| **Detección neta** | fotos adelgazadas rechazadas, contando solo aquellas que se aceptaron sin tocar | lo más alto posible |

La detección se mide **neta** a propósito: un control que rechaza todo obtiene
100% de detección y no vale nada. Solo cuentan las fotos que el sistema aceptó
cuando estaban intactas.

---

## El rostro: de un control ciego a uno que separa

Hasta septiembre de 2026 la comprobación de la cara **no podía funcionar**, y
conviene que quede escrito con números porque es el defecto que más caro salió.

La firma era un descriptor geométrico y fotométrico de 64 valores
(`analysis/face.py`). Medido contra el perfil almacenado:

| población | parecido con el descriptor viejo |
|---|---|
| Sus 24 fotografías reales | 0.9832 - 0.9993 |
| 8 fotografías de **8 mujeres distintas** | 0.9577 - 0.9945 |
| Las 2 imágenes de pago que ella rechazó de un vistazo | 0.9905 y 0.9960 |

Las dos poblaciones **se solapan** (la peor foto suya, 0.9832, cae por debajo de
la mejor impostora, 0.9945) y el umbral exigido era 0.72, un cuarto de escala por
debajo de todo. La puerta no es que fallara: **no podía disparar nunca**, y por
eso aprobó dos imágenes de otra mujer.

Hoy la firma es un **embedding SFace de 128 valores** (`identity/embedding.py`,
pesos del OpenCV Zoo, Apache-2.0, se descargan con
`python scripts\fetch_face_model.py`). Cada imagen se compara por coseno contra
la **media de los embeddings de sus fotos**, guardada en el perfil.

| población | parecido medido hoy | veredicto con el límite 0.45 |
|---|---|---|
| Sus 24 fotografías, cada una contra la media de **las otras 23** | 0.6429 - 0.8715 | 24 de 24 **aceptadas** |
| Sus 24 fotografías contra la media guardada en el perfil | 0.6740 - 0.8718 | 24 de 24 **aceptadas** |
| 2 fotos suyas que el perfil **no ha visto nunca** | 0.7624 y 0.8222 | aceptadas |
| 8 fotografías de 8 mujeres distintas | 0.0194 - 0.1948 | 8 de 8 **rechazadas** |
| 6 imágenes generadas cuya cara no es la suya | 0.1829 - 0.3968 | 6 de 6 **rechazadas** |
| - de ellas, las 2 de pago que ella rechazó | 0.1829 y 0.2862 | **rechazadas** |
| 32 imágenes generadas cuya cara sí es la suya | 0.5279 - 0.7925 | aceptadas |

**Separación medida: +0.2461.** Entre la mejor cara ajena (0.3968) y su peor
fotografía propia (0.6429) no hay ninguna imagen. El límite **0.45** se pone
dentro de esa franja vacía: queda 0.056 por encima de la peor impostora y 0.19
por debajo de su peor foto. No es un número redondo elegido a mano, es el centro
de un hueco que se puede medir.

Con galerías más pequeñas el hueco aguanta: repitiendo 300 sorteos con perfiles
de 6, 8, 10, 12, 16 y 20 fotos y puntuando las que quedaban fuera (21.600
puntuaciones), **ninguna** bajó de 0.5618.

### Cómo reproducirlo

`calibrate_identity.py` mide el cuerpo, no la cara. Los números de arriba salen
de puntuar los archivos que ya están en disco con los propios módulos del robot,
sin llamar a nadie ni gastar un céntimo:

```python
from app.identity import embedding as emb
from app.analysis import face as face_mod
v = emb.face_embedding(img, face_mod.detect_face(img))   # 128 valores
emb.similarity(v, profile["face"]["embedding_mean"])     # 0..1, 1 = idéntica
```

### Lo que hizo falta para que funcionara

Tres de sus fotografías están guardadas giradas (son selfies de móvil), y
MediaPipe les ajustaba la malla del revés: el recorte alineado salía invertido y
el reconocedor leía a una desconocida (0.1835, 0.1554, 0.1549, **por debajo de
cualquier impostora**). El embedding recorta la cabeza con la malla y después
elige, entre los cuatro ángulos rectos, el que YuNet reconoce con más confianza.
Eso subió esas tres a 0.7180 / 0.8435 / 0.7447 y la peor fotografía del conjunto
de 0.1549 a 0.6362.

Por el mismo motivo, cuando la malla cae sobre algo que no es una cara (foto
girada, muy pequeña o muy recortada) se pide una segunda opinión sobre el
fotograma entero, y solo gana si es **claramente** más segura
(`MESH_TRUSTED_CONF` 0.90, `RESCUE_MARGIN` 0.10, ambos medidos y comentados en
`identity/embedding.py`). Con eso, sus 24 fotos espejadas, giradas en los cuatro
ángulos rectos y en +/-15 y +/-25 grados, oscurecidas, sobreexpuestas, a
contraluz, recortadas y comprimidas a JPEG de calidad 20 **pasan las 24**.

### Lo que no está garantizado

* La puerta acepta una cara que sea hasta un **60-65% de otra persona** por peso
  de píxel: mezclando su recorte alineado con el de una desconocida, la mitad de
  las mezclas deja de pasar en torno a alfa 0.65. Es una propiedad del coseno
  contra una media, no un fallo, y ningún umbral dentro de la franja vacía lo
  cambia.
* Entradas destruidas (una cara de 30-40 px, una sobreexposición que quema más
  de la mitad del cuadro) sí se rechazan. No es un error de la puerta, pero se
  cuenta como rechazo y **se paga otra generación**.
* Sin los pesos en disco, o con un perfil construido antes de este cambio, la
  comprobación se informa como **no realizada** ("no se pudo comprobar el
  rostro") y se queda fuera de la nota. Nunca pasa en silencio.

---

## Resultado medido (24 fotos reales, septiembre 2026)

| adelgazamiento | detección neta | falsas alarmas |
|---|---|---|
| 6 %  | 83 % (5 de 6) | **0 %** |
| 12 % | 100 % (6 de 6) | **0 %** |
| 18 % | 83 % (5 de 6) | **0 %** |

Y lo más importante: **el 100% de esas detecciones las dispara la comprobación de
proporciones corporales**, no otra por casualidad. El control acierta por el
motivo correcto.

### De dónde viene ese resultado

El punto de partida era mucho peor, y conviene dejarlo escrito porque explica las
decisiones del código:

| estado | falsas alarmas | detección real |
|---|---|---|
| Primera versión | 60 % | 0 % (la comprobación de proporciones no llegaba a dispararse nunca) |
| Con la comparación contra la foto de origen | 0 % | 60 % |
| Versión actual (regla de la cabeza) | 0 % | 83 % / 100 % / 83 % |

Cinco correcciones, todas motivadas por una medición:

1. **Bandas contaminadas por la postura.** Las bandas se construían con medidas
   que el propio medidor había marcado como poco fiables (un brazo cruzando la
   línea de la cintura, un torso girado). Eso inflaba la desviación típica con
   ruido de postura, y como la tolerancia crecía con esa desviación, la banda se
   ensanchaba hasta ±36%: un adelgazamiento del 18% cabía dentro sin rozarla.
   Ahora esas medidas no alimentan la banda, y la banda tiene un tope de ±12%.

2. **Tono de piel medido mal.** Se comparaba la distancia CIE76 completa,
   incluyendo la luminosidad. La luminosidad cambia con la exposición y la hora
   del día: en las fotos propias de la clienta varía más de 15 unidades, muy por
   encima de cualquier límite razonable. El resultado era que el sistema
   rechazaba sus fotos auténticas por "cambio de tono de piel". Ahora la decisión
   se toma sobre el **color** (el par a/b), que es estable frente a la
   exposición, con un límite adaptado a la variación que muestra cada persona en
   sus propias fotos de referencia.

3. **Manos sanas marcadas como deformes.** En 6 de las 24 fotos reales el sistema
   veía una "mano deformada" con una confianza de hasta 0.95. Eran manos
   pequeñas (60-100 px): a ese tamaño, un error de dos píxeles en un punto de
   referencia cambia una proporción de dedo un 30%, y toda mano lejana parece
   deforme. Ahora la gravedad del defecto escala con el tamaño de la mano, así
   que las manos pequeñas se informan pero no tiran la imagen a la basura. Lo
   mismo con la piel suavizada: todos los móviles modernos suavizan la piel, así
   que eso se informa pero no rechaza por sí solo.

4. **Comparación contra la foto de origen, no contra una media.** Es el cambio
   que más aporta. Comparar una imagen generada contra una banda de población
   arrastra todo el ruido de encuadre, distancia y giro. Pero el robot **siempre
   parte de una foto concreta**, así que puede comparar el resultado contra su
   propio origen: misma persona, misma pose, mismo ángulo. Casi todo el ruido se
   cancela en la división. Además la silueta se mide a **nueve alturas** del
   torso en vez de dos, y se compara el perfil completo, porque un adelgazamiento
   uniforme mueve las nueve en el mismo sentido.

5. **Una regla que no depende del encuadre: su propia cabeza.** El torso deja de
   servir en cuanto la imagen se reencuadra, porque el propio torso cambia de
   tamaño en el cuadro. La figura se mide también en alturas contadas desde la
   barbilla en unidades de **la longitud de su cabeza** (`head_profile`, hasta 29
   alturas), medida sobre un recorte de tamaño fijo y promediada con su espejo
   para que no prefiera un lado. Medido: 0,4 % de variación al cambiar la
   resolución, 0,4 % al recortar al 62 % del cuadro y 2,4 % en el peor espejo,
   frente al 8 % que se lee cuando alguien la estrecha un 8 %. Es lo que permite
   juzgar un reencuadre, donde antes no había control ninguno.

---

## Limitación honesta, y es importante

De las 24 fotos actuales, **solo una muestra los tobillos**. La mayoría son de
medio cuerpo o primer plano.

Consecuencia concreta: el control de proporciones **solo puede actuar sobre las
fotos en las que se ven los hombros y las caderas a la vez**. En un primer plano
no hay torso que medir, no hay proporciones que comparar, y el sistema lo dice en
vez de inventarse un número.

Esto es exactamente lo que ya se le pidió a la clienta: **6 a 8 fotos de cuerpo
entero**, de la cabeza a los pies. No es un trámite. Es la diferencia entre un
control que protege sus proporciones y uno que solo puede opinar sobre su cara.

Cuando lleguen esas fotos, hay que volver a ejecutar este guion. Se espera que la
detección suba bastante, porque la limitación actual es de datos, no de método.

---

## Cómo leer la salida

```
Falsas alarmas (fotos reales rechazadas):   0.0%  (0 de 6)
Base para la deteccion neta: 6 fotos aceptadas sin tocar
Deteccion NETA del adelgazamiento del 12%: 100.0%  (6 de 6)   [OK]
   de las cuales por proporciones corporales: 6
```

* Si **falsas alarmas** sube, el sistema tirará fotos buenas y cada una cuesta
  dinero regenerarla. Es el número que hay que vigilar primero.
* Si **detección** baja, las bandas se han quedado anchas o hay pocas fotos
  medibles.
* Si detecta pero la última línea dice 0, está acertando por casualidad: revisa
  qué comprobación está disparando de verdad.

Umbrales relevantes, todos en un sitio:

| constante | archivo | valor | qué hace |
|---|---|---|---|
| `face_embed_min` | `identity/profile.py` | 0.45 | parecido facial mínimo: coseno del embedding SFace contra la media de sus fotos |
| `MESH_TRUSTED_CONF` | `identity/embedding.py` | 0.90 | confianza por debajo de la cual el recorte de la malla pide una segunda opinión |
| `RESCUE_MARGIN` | `identity/embedding.py` | 0.10 | cuánto más segura tiene que ser esa segunda opinión para ganar |
| `PAIRED_TOL` | `identity/verify.py` | 0.08 | cambio máximo de forma frente a la foto de origen |
| `HEAD_TOL` | `identity/verify.py` | 0.04 | lo mismo, con la regla que no depende del encuadre (en cabezas) |
| `SMOOTH_TEXTURE_LOSS_MAX` | `identity/verify.py` | 0.14 | grano de piel perdido a partir del cual se rechaza |
| `SMOOTH_TEXTURE_LOSS_MIN` | `identity/verify.py` | 0.09 | grano de piel perdido a partir del cual se informa |
| `BAND_MAX_REL` | `identity/profile.py` | 0.12 | anchura máxima de una banda de población |
| `GATE_MIN_SAMPLES` | `identity/profile.py` | 3 | fotos necesarias antes de que una medida pueda rechazar |
| `ANATOMY_SEVERITY_MAX` | `identity/verify.py` | 0.6 | gravedad a partir de la cual un defecto rechaza |
| `HAND_MIN_PX` | `analysis/anomaly.py` | 170 | tamaño de mano por debajo del cual la geometría no es fiable |
