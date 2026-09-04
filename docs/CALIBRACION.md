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

Repetido el **2026-09-04** sobre lo que hay en disco hoy, con la comprobación
que decide (`_check_identity_face`, no una métrica escrita para el informe): sus
24 fotografías 0.6908 - 0.8737, las 2 no vistas 0.7624 y 0.8222, las 8 mujeres
distintas 0.0201 - 0.1925, y las 4 imágenes de pago que quedan en disco y no son
ella 0.2920 - 0.3968. **38 de 38 correctas.** El hueco sigue donde estaba.

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

Barrido sobre las **24 fotos**, cada una sometida a cuatro cambios que **no**
tocan su cuerpo (reguardar, releer a 1024 px, reencuadrar al 62 % del alto, y
ensanchar un 6 % como hace la ropa) y tres que sí (estrecharla un 6, 12 y 18 %).
168 verificaciones, con el mismo camino que usa la aplicacion: cada resultado
comparado contra su propia foto de origen.

| cambio | verdad | rechazos | fotos que la regla pudo juzgar |
|---|---|---|---|
| reguardada | sin tocar | **0 de 24** | 7 |
| releída a 1024 px | sin tocar | **0 de 24** | 7 |
| reencuadre al 62 % | sin tocar | **0 de 24** | 4 |
| ropa que ensancha un 6 % | sin tocar | **0 de 24** | 7 |
| estrechada un 6 % | −6 % | 0 de 24 | 7 |
| estrechada un 12 % | −12 % | **7 de 7 juzgables** | 7 |
| estrechada un 18 % | −18 % | **7 de 7 juzgables** | 7 |

Dos lecturas, y las dos importan:

* **Ninguna foto suya se rechaza jamás**, ni reencuadrada ni con ropa nueva.
  Ese era el encargo: una diferencia pequeña no puede costarle una imagen
  pagada.
* **El adelgazamiento del 12 % — el que ella denunció — se caza siempre que hay
  cuerpo que medir**, y se caza también bajo un reencuadre (4 de 4).
* El del 6 % **no se caza, a propósito**: un adelgazamiento del 6 % se lee
  entre 4,5 % y 7,5 % y un 6 % de tela añadida se lee entre 4,1 % y 6,7 %. Las
  dos lecturas se solapan, así que a ese tamaño **ningún límite** distingue un
  cuerpo estrechado de un jersey. Se mide, se escribe en el veredicto en
  castellano y no se le cobra.
* De las 24 fotos, la regla solo puede juzgar 7: el resto son primeros planos
  con menos de dos cabezas de cuerpo bajo la barbilla, o medios cuerpos que
  salen del cuadro por los lados. Ahí no se inventa un número: se informa.

Y lo más importante: **el 100 % de esas detecciones las dispara la comprobación
de proporciones corporales**, no otra por casualidad. El control acierta por el
motivo correcto.

### De dónde viene ese resultado

El punto de partida era mucho peor, y conviene dejarlo escrito porque explica las
decisiones del código:

| estado | falsas alarmas | detección real |
|---|---|---|
| Primera versión | 60 % | 0 % (la comprobación de proporciones no llegaba a dispararse nunca) |
| Con la comparación contra la foto de origen | 0 % | 60 % |
| Tres reglas a la vez | 0 % contra sí misma; sobre copias reencuadradas o reguardadas, **21 %** la regla del esqueleto (8 de 38) y **6 %** la de la silueta sobre el torso (2 de 31) | 83 % / 100 % / 83 % |
| Versión actual (solo la regla de la cabeza) | **0 %** en 96 pruebas | 0 % / 100 % / 100 % de lo juzgable |

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

   **Y hasta dónde llega eso, medido el 2026-09-04 sobre sus 24 fotografías más
   todas las generadas en disco.** De las 38 manos detectadas, 12 salen cortadas
   por el borde del cuadro y las 26 restantes están **todas** por debajo de
   `HAND_MIN_PX` = 170: la mayor mide 157 px y la mediana 68. A ese tamaño no hay
   ninguna regla que separe una mano suya de una generada rota. Por geometría, su
   propia IMG_7880 mide una palma 17.42 de largo por ancho — 7.9 veces fuera del
   límite 2.20 — mientras que una mano generada que a simple vista está derretida
   mide 3.99; y por energía de borde contra el antebrazo, sus fotos dan 0.75 -
   3.00 y las generadas 0.50 - 3.00, solapadas por completo. Conclusión: **la
   comprobación de anatomía no puede juzgar una mano en estas imágenes, y desde
   hoy lo dice.** El veredicto y el resumen del álbum escriben "No se ha podido
   comprobar 2 manos: tu mano izquierda, sale del encuadre y tu mano derecha,
   demasiado pequeña para medirla (75 px, harían falta 170)". No rechaza nada
   — bajar el umbral hasta que fallen rechazaría antes sus propias fotos — pero
   deja de decir "sin anomalías" sobre una parte que nadie ha mirado.

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
   para que no prefiera un lado. Es lo que permite juzgar un reencuadre, donde
   antes no había control ninguno.

6. **Y una sola regla decide.** Las tres se contradecían sobre la misma imagen
   pagada, así que se midieron contra una verdad que se puede construir en vez
   de discutir: sus 24 fotos reguardadas, releídas más pequeñas, recortadas al
   62 % del alto, ensanchadas un 6 % (la única dirección en la que la ropa
   puede mover una silueta) y estrechadas un 6, 12 y 18 % conocidos. 112
   parejas medidas. Lo que leyó cada regla, en porcentaje de diferencia:

   | regla | sin tocar | reencuadrada | +6 % de tela | −6 % | −12 % | −18 % |
   |---|---|---|---|---|---|---|
   | esqueleto (anchos ÷ largo del torso) | 0,7–14,4 | 13,5 | 1,5–15,3 | 2,6–19,8 | 2,7–13,4 | 3,1–25,7 |
   | silueta sobre el torso | 0,0–2,6 | 14,2–16,9 | 0,4–7,5 | 2,8–7,8 | 3,1–12,5 | 3,3–19,9 |
   | **figura en cabezas** | **0,0–1,2** | **0,3–1,6** | 4,1–6,7 | 4,5–7,5 | **11,4–13,5** | **17,2–19,0** |

   Las dos primeras no miden el cuerpo. El esqueleto marcó **14,4 % en una foto
   que solo se había vuelto a guardar** y **2,7 % en otra estrechada un 12 %**:
   no existe ningún límite con el que baje del 10 % de falsas alarmas y llegue
   al 80 % de detección. La silueta sobre el torso mide el recorte (14–17 % en
   una foto que nadie tocó) y se le escapó por completo un adelgazamiento real
   del 12 % bajo ese recorte (0,4 %). La regla de la cabeza lee la verdad.

   Así que **la cabeza es la que rechaza, y las otras dos solo informan**. No se
   promedian: una regla que mide el encuadre y otra que mide el cuerpo no
   discrepan sobre su figura, discrepan sobre qué están midiendo, y la media de
   las dos es un número sobre nada. Las dos cifras degradadas siguen apareciendo
   en el veredicto, en castellano, con la frase que explica por qué no deciden.

---

## Limitación honesta, y es importante

De las 24 fotos actuales, **solo una muestra los tobillos**. La mayoría son de
medio cuerpo o primer plano.

Consecuencia concreta: el control de proporciones **solo puede actuar sobre las
fotos con al menos dos cabezas de cuerpo por debajo de la barbilla y que no se
salgan del cuadro por los lados**. Son 7 de las 24. En un primer plano no hay
figura que medir, no hay proporciones que comparar, y el sistema lo dice en vez
de inventarse un número.

Medido foto a foto: de las 17 que no se pueden juzgar, **16 tienen entre 0,9 y
2,2 cabezas de cuerpo** por debajo de la barbilla, y en 7 de esas las pocas
filas que quedan tocan además el borde lateral del cuadro — una fila que llega
al borde no da un ancho, da un mínimo, y un adelgazamiento no puede mover un
extremo que ya está fuera de la foto. La número 17 no devuelve malla facial.

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
Deteccion NETA del adelgazamiento del 12%:  50.0%  (3 de 6)   [DEBIL]
   de las cuales por proporciones corporales: 3
```

Ese 50 % es **cobertura, no puntería**: de las seis fotos de prueba, tres son
primeros planos sin cuerpo bajo la barbilla y la regla se abstiene en vez de
inventarse un número. En las tres que sí puede medir, acierta las tres. El
barrido sobre las 24 fotos de más arriba lo separa: 7 de 7 juzgables cazadas al
12 %, 0 falsas alarmas en 96 pruebas. Si esta cifra sube, será porque llegaron
fotos de cuerpo entero, no porque el control haya mejorado.

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
| `PAIRED_TOL` | `identity/verify.py` | 0.08 | reglas sobre el torso: a partir de aquí la diferencia se **informa**. Ya no rechaza nada |
| `HEAD_TOL` | `identity/verify.py` | 0.08 | **la única que rechaza**: cambio de figura medido en cabezas. 0 falsas alarmas en 25 pruebas honestas (peor lectura 6,7 %), caza el 12 % siempre (11,4 % la lectura más floja) |
| `HEAD_TOL_MAX` | `identity/verify.py` | 0.10 | tope al que puede ensancharse ese límite por ruido de medición; más allá perdonaría el 12 % que existe para cazar |
| `SMOOTH_TEXTURE_LOSS_MAX` | `identity/verify.py` | 0.14 | grano de piel perdido a partir del cual se rechaza |
| `SMOOTH_TEXTURE_LOSS_MIN` | `identity/verify.py` | 0.09 | grano de piel perdido a partir del cual se informa |
| `BAND_MAX_REL` | `identity/profile.py` | 0.12 | anchura máxima de una banda de población |
| `GATE_MIN_SAMPLES` | `identity/profile.py` | 3 | fotos necesarias antes de que una medida pueda rechazar |
| `ANATOMY_SEVERITY_MAX` | `identity/verify.py` | 0.6 | gravedad a partir de la cual un defecto rechaza |
| `HAND_MIN_PX` | `analysis/anomaly.py` | 170 | tamaño de mano por debajo del cual la geometría no es fiable; por debajo, la mano se declara **no comprobada** y se nombra en el veredicto |
| `RISK_MIN_N` / `RISK_MAX_RATE` | `generation/router.py` | 3 / 0.50 | cuándo una opción del catálogo se avisa como arriesgada, contada sobre sus propias imágenes pagadas |
| `SAFE_MIN_N` / `SAFE_MIN_RATE` | `generation/router.py` | 4 / 0.85 | cuándo se dice que una opción sí ha funcionado |
| `SEED_UNTIL` | `generation/router.py` | 1788514580.1 | hasta aquí llega la tabla medida a mano; lo pagado después se cuenta solo desde la base de datos |
