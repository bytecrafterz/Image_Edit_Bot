# Procesar con tu clave de fal.ai

Esta guia es para el momento en que dejas de probar con el motor local gratuito
y pones tu propia clave de pago. Explica que hay que tocar, cuanto cuesta de
verdad, que revisa el robot en cada imagen y que hace cuando algo no cuadra.

Todos los numeros de este documento estan medidos en esta maquina el 4 de
septiembre de 2026, con el codigo tal y como esta hoy. Ninguno es una
estimacion de despacho.

---

## 1. Antes de gastar nada: el ensayo

Hay un ensayo que recorre entero el camino de pago **sin llamar a fal.ai**: la
clave que usa es falsa, las imagenes las saca de una carpeta, la base de datos y
el album son temporales, y toda conexion de red que no sea a esta misma maquina
esta bloqueada y se cuenta.

```powershell
backend\.venv\Scripts\python.exe scripts\rehearse_paid.py
```

Coste: **0.00 USD, siempre**. Tarda un par de minutos y termina con una linea
como esta:

```
RESULTADO: 35 correctas, 0 fallidas   |   COSTE REAL: 0.0000 USD
```

Comprueba 35 cosas con numeros: que la clave se guarda por la pantalla de
Ajustes y nunca vuelve al navegador, que la tirada se enruta al motor de pago,
que se pide tu forma y no un cuadrado, que lo retenido es igual a lo liquidado,
que cada llamada al proveedor llega al libro mayor, que la puerta de calidad
juzga cada imagen y que la reparacion se cobra a su propio precio. Si algo del
camino de pago estuviera roto, sale ahi y no en tu factura.

Opciones utiles:

```powershell
backend\.venv\Scripts\python.exe scripts\rehearse_paid.py --quality preview --variants 6
backend\.venv\Scripts\python.exe scripts\rehearse_paid.py --recharge 0.30
```

La segunda ensaya el caso de quedarse sin saldo a mitad de una tirada.

---

## 2. Poner la clave

1. Entra en **Ajustes -> Claves de IA**.
2. En la fila de **fal.ai**, pega la clave y pulsa **Guardar clave**.
3. La clave se guarda cifrada en el servidor y **nunca se devuelve al
   navegador**: a partir de ahi solo veras una pista del tipo `sk-a...9f2c`.
4. Al guardarla se intenta validar contra fal.ai. Si en ese momento no hay
   internet veras *"Guardada, pero no se pudo verificar ahora"*; la clave queda
   guardada igual.

La clave de **Anthropic** es opcional y hace otra cosa: leer tus fotos y
redactar el prompt. Sin ella el sistema usa vision local y funciona igual, con
prompts algo mas simples.

Para quitarla: la misma fila, boton de borrar. El sistema vuelve solo al motor
local gratuito, no se queda a medias.

## 3. Registrar el saldo

El dinero se recarga **en la web de fal.ai**, con tu tarjeta, fuera de esta
aplicacion. Aqui solo se anota:

**Ajustes -> Saldo -> Registrar recarga**, e introduce el importe que ya
anadiste.

> Esta aplicacion no tiene ni una linea de codigo de cobro. "Registrar recarga"
> solo apunta un numero para poder avisarte antes de que se acabe.

Si el saldo anotado no llega para la siguiente imagen, el robot **para y lo
dice**, en vez de seguir intentandolo y fallando.

---

## 4. La pantalla de coste

Al pulsar *Calcular coste* (paso 3 de Crear) veras, antes de gastar nada:

| Lo que dice la pantalla | Que significa |
|---|---|
| El total grande | lo que se espera gastar: generacion + un factor 1.35 porque 1 de cada 3 imagenes se repite |
| **Por imagen** | el precio de una sola llamada de generacion, sin repeticiones |
| **Saldo despues** | tu saldo anotado menos el total esperado |
| Aviso de resolucion | los pixeles que el nivel entrega de verdad |
| Aviso de coste | el **techo**: lo maximo que puede llegar a costar esta tirada |

Ejemplo real de hoy, calidad alta, 3 imagenes, con cambio de ropa:

```
3 variante(s) x 0.0800 USD  + factor 1.35  = 0.4290 USD
Aviso: fal entrega como maximo 1024 px de lado largo en esta calidad, no 1536.
Aviso: la estimacion cuenta con que 1 de cada 3 imagenes se repita. Si todas
fallan a la primera, esta tirada puede llegar a 2.07 USD, porque cada intento
rechazado se repinta por zonas a 0.05 USD cada una.
```

Y lo que gasto de verdad esa misma tirada en el ensayo de hoy: **0.6500 USD**
(5 generaciones a 0.08 mas 5 repintados a 0.05), un 52% por encima del estimado
y muy por debajo del techo de 2.07. **El techo es la promesa; el estimado es
solo una media.**

### Precios reales por llamada

| Nivel | Modelo de fal | Por imagen | Pide | Entrega | 3 imagenes: estimado / techo |
|---|---|---|---|---|---|
| Borrador | `flux/dev` img2img | 0.00625 USD | 384x512 | 512 px | 0.1303 / 1.4062 USD |
| Vista previa | Kontext `[pro]` | 0.0400 USD | 576x768 | 1024 px | 0.2670 / 1.7100 USD |
| Estandar | Kontext `[pro]` | 0.0400 USD | 768x1024 | 1024 px | 0.2670 / 1.7100 USD |
| **Alta** | Kontext `[max]` | 0.0800 USD | 1152x1536 | 1024 px | 0.4290 / 2.0700 USD |
| Maxima | Kontext `[max]` | 0.0800 USD | 1536x2048 | 1024 px | 0.4290 / 2.0700 USD |

Ademas, cada **repintado de una zona** cuesta 0.0500 USD, y una reparacion
repinta hasta 3 zonas. Por eso un borrador de 0.006 USD puede acabar costando
0.15 USD si su unica pega esta en tres sitios: **lo caro no es generar, es
arreglar**.

Los tamanos de la columna "Pide" salen de tu propia foto: 2316x3088 es 3:4, asi
que se pide 3:4. Si eliges un encuadre concreto manda ese: cuerpo entero
1152x1536, medio cuerpo 1232x1536, cuadrado 1536x1536, vertical 9:16 864x1536.

## 5. Que nivel elegir

**Los cinco niveles entregan practicamente el mismo tamano de archivo**, unos
1024 px de lado largo: los modelos de fal no tienen mando de resolucion. Lo que
compras al subir de nivel es **fidelidad a tus rasgos**, no pixeles.

- **Vista previa (0.04 USD)** para explorar: cuando quieres ver seis ideas de
  escena, luz o encuadre y decidir cual te gusta.
- **Alta (0.08 USD)** para la imagen que te vas a quedar. Usa Kontext `[max]`,
  que es el que mejor conserva la cara y la piel.
- **Borrador (0.006 USD)** solo para probar el circuito. Es un modelo mas
  flojo, y como cada reparacion suya cuesta ocho veces lo que costo generarla,
  sale caro en cuanto falla.
- **Maxima** cuesta y entrega hoy lo mismo que Alta; existe para cuando fal
  publique un modelo mayor.

Recomendacion practica: **vista previa para elegir, y luego alta calidad solo
sobre las que has marcado** (boton *Alta calidad* en el paso de eleccion).

---

## 6. Que revisa el robot en cada imagen

Cada imagen generada se vuelve a medir **contra tu propia foto de origen**, no
contra una media de nadie. Cinco comprobaciones:

| Comprobacion | Que mide | Limite | Peso |
|---|---|---|---|
| **El rostro** | firma de reconocimiento facial, 128 valores (SFace), comparada con la media de tus fotos | parecido >= 0.45 | 30% |
| **Las proporciones** | tu figura a hasta 29 alturas desde la barbilla, **en unidades del largo de tu cabeza**, mas la silueta del torso a 9 alturas y las medidas del esqueleto | cambio de forma <= 4% con la regla de la cabeza, <= 8% con las del torso | 25% |
| **El tono de piel** | color a*/b* de la piel, sin contar la luminosidad, que cambia con la luz | banda adaptada a tu propia variacion | 15% |
| **La anatomia** | manos, dedos, extremidades, personas de mas, zonas fundidas y **piel demasiado suavizada** | gravedad <= 0.6 | 20% |
| **La calidad tecnica** | nitidez, exposicion, contraste, ruido | >= 0.45 | 10% |

Dos detalles que importan y que no son obvios:

**El grano de tu piel se compara con el de tu foto.** Se mide la banda fina de
la piel de la cara en las dos imagenes, llevadas antes al mismo tamano de cara,
y se calcula cuanto grano falta. Por debajo del 9% no se dice nada; entre el 9%
y el 14% se informa; **a partir del 14% se rechaza la imagen** por "piel
demasiado suavizada". Es la medida que detecta un embellecedor.

**La regla de la cabeza mide aunque cambies de ropa.** Como toda la ropa del
catalogo se pone encima de la piel y ninguna quita volumen, una figura que sale
mas **ancha** puede ser la chaqueta: se informa y no se rechaza. Una figura que
sale mas **estrecha** no la puede haber estrechado un abrigo, asi que ahi la
regla si rechaza. Eso es lo que protege contra un adelgazamiento que no pediste.

Sobre tus fotos reales, esa comprobacion mide hoy
(`scripts\calibrate_identity.py --paired --measurable-only`):

| adelgazamiento fabricado | lo detecta | falsas alarmas |
|---|---|---|
| 6% | 83% (5 de 6) | **0%** |
| 12% | 100% (6 de 6) | **0%** |
| 18% | 83% (5 de 6) | **0%** |

## 7. Que pasa cuando una comprobacion falla

1. **Se intenta reparar solo la zona rota.** Si el defecto es local (una mano,
   un trozo de piel, una zona borrosa), se repinta esa zona con una mascara,
   sin rehacer la foto. Cuesta 0.05 USD por zona, hasta 3 zonas.
2. **Si el repintado no mejora la medida, se revierte.** El robot mide otra vez
   la gravedad del defecto antes y despues; si no baja, deja la imagen como
   estaba. La zona se paga igual, porque fal cobra el repintado aunque el
   resultado no sirva, y por eso ese gasto aparece en el libro mayor.
3. **Si sigue sin pasar, se genera otra vez**, hasta 2 reintentos por vista.
4. **Si tampoco, se descarta con el motivo escrito en castellano** y aparece en
   la ficha de la tirada. Los motivos posibles son exactamente estos:
   *el rostro no coincide con el tuyo*, *cambiaron tus proporciones*, *cambio tu
   tono de piel*, *hay errores anatomicos*, *la calidad tecnica es baja*.

En el ensayo de hoy (`python scripts\rehearse_paid.py`, calidad alta, 3 vistas,
coste real 0.00 USD porque las imagenes salen de disco y la red esta cerrada):
7 intentos, **1 aceptada y 6 descartadas**, 6 rondas de reparacion de las que 4
mejoraron, 7.0 intentos por foto conseguida.

Las 6 descartadas lo fueron por **el rostro no coincide con el tuyo**, y eso es
nuevo: hasta septiembre esa comprobacion no podia rechazar nada (ver *La cara*,
mas abajo), asi que el mismo ensayo aceptaba imagenes de otra cara. El numero
tiene su cara amarga: en ese ensayo se llamo 16 veces al proveedor y se
liquidaron 1.01 USD frente a los 0.43 estimados, dentro del techo anunciado de
2.07 USD. Es el precio de que el control por fin funcione con este motor.

Si en cualquier punto el saldo no cubre la siguiente llamada, la tirada se
**detiene**, guarda lo que ya habia aceptado y te avisa de cuanto falta. Nunca
se queda a deber.

---

## 8. Lo que el motor de pago le hace hoy a tu piel y a tu cuerpo

Medido hoy sobre las **15 imagenes de fal que ya hay compradas** en esta base de
datos (12 a partir de IMG_7871, 3 de IMG_8825, todas de 1024x1024, todas con
cambio de ropa). Son anteriores a los arreglos de encuadre y precio de hoy, pero
sirven para lo que importa aqui: como trata el modelo tu piel y tu figura.

### La piel: es donde el motor falla

| | media | mediana | peor | fuera de +/-9% |
|---|---|---|---|---|
| Tal como lo entrega fal | +9.2% | +13.2% | +22.6% | 10 de 15 |
| Despues del retoque del robot | +8.8% | +13.9% | +20.6% | 10 de 15 |

Un valor positivo es **grano tuyo que falta**. Traducido: el modelo te alisa la
piel de la cara entre un 13% y un 23% en la mitad larga de los casos. La puerta
rechaza las que pasan del 14%: **7 de las 15** se rechazaron por eso, con el
motivo *piel demasiado suavizada*.

El modulo que intenta devolver el grano actua solo en **5 de 15** y mueve la
medida como mucho 2 puntos; en las peores se abstiene porque no consigue
emparejar tu piel con la de la foto. **No cuentes con el**: cuenta con que la
puerta rechace la imagen y se genere otra.

Un caso al reves: una de las 15 volvio con un **14% mas** de grano del que tiene
tu fotografia. Eso es textura inventada por el modelo, y hoy la puerta no la
rechaza, porque solo mira el lado que pierde grano. Es la unica pega conocida de
esta comprobacion.

### El cuerpo: aguanta

| | resultado |
|---|---|
| Cara reconocida | ver el apartado siguiente: con la firma de hoy, **6 de las 17** imagenes de pago que hay en disco NO son tu cara |
| Salieron **mas estrechas** de lo que eres, por encima del limite | **1 de 15** (-4.2%), y la puerta la rechaza |
| Salieron mas anchas por encima del limite | 7 de 15, de +4% a +19%, explicable por el abrigo o la chaqueta que se pidio: se informa, no se rechaza |
| Desviacion mediana de la figura | 4.1% |

Es decir: de las 15 imagenes de pago, **el motor te estrecho una sola vez, y esa
la caza el control**. El resto de la desviacion va en la direccion que la ropa
nueva explica.

### La cara: lo que antes no se media

Hasta septiembre de 2026 esta comprobacion estaba **ciega**, y hay que decirlo
claro porque tu pagaste dos imagenes por ese fallo. La firma antigua daba 0.9832
a 0.9993 en tus 24 fotos y 0.9577 a 0.9945 en fotos de **ocho mujeres
distintas**: los dos rangos se solapan, asi que el minimo de 0.72 no podia
rechazar a nadie. Las dos imagenes de pago que tu rechazaste de un vistazo
sacaron 0.9905 y 0.9960 y **fueron aprobadas**.

Hoy la firma es un reconocedor facial de verdad (SFace, 128 valores, funciona en
tu maquina, no sale ninguna foto de aqui). Sobre las mismas imagenes:

| | parecido de hoy |
|---|---|
| Tus 24 fotografias | 0.6740 a 0.8718 - **las 24 pasan** |
| Fotos tuyas que el perfil nunca vio | 0.7624 y 0.8222 - pasan |
| Fotos de 8 mujeres distintas | 0.0194 a 0.1948 - **las 8 se rechazan** |
| Las 17 imagenes de pago que hay en disco | 6 entre 0.1829 y 0.3968 (**se rechazan**), 11 entre 0.5279 y 0.7173 (pasan) |
| Las 2 que tu rechazaste | 0.1829 y 0.2862 - **se rechazan** |
| Las 21 del motor gratuito | 0.7233 a 0.7925 - todas pasan |

Entre la mejor cara ajena (0.3968) y tu peor fotografia (0.6429 en la prueba mas
dura) **no hay ninguna imagen**. El minimo se pone en 0.45, en mitad de ese
hueco.

**Lo que garantiza:** si la imagen lleva la cara de otra mujer, no llega a tu
album, la ficha lo dice con el numero y el motivo es *"el rostro no coincide con
el tuyo"*.

**Lo que cuesta cuando rechaza por la cara.** Una cara equivocada **no se puede
reparar por zonas**: un retoque local puede devolver grano a una mejilla, pero no
puede convertir a otra mujer en ti. Asi que:

* la imagen rechazada **ya esta pagada** (0.04 USD en vista previa, 0.08 USD en
  alta): la llamada se hizo y el proveedor cobra igual;
* el robot **genera otra entera**, a ese mismo precio, no al de reparacion
  (0.05 USD), que aqui no se usa, y como mucho **2 reintentos por vista** antes
  de descartarla con el motivo escrito;
* el saldo se comprueba **antes de cada llamada**, asi que nunca se gasta mas de
  lo que tienes, y el aviso de coste de la pantalla ya cuenta con que 1 de cada
  3 se repita (factor 1.35) y te ensena el techo del peor caso.

**Lo que NO garantiza:**

* No promete que la imagen se te parezca *mucho*, solo que la cara es la tuya y
  no la de otra persona. Un parecido de 0.53 pasa el control y puede seguir sin
  gustarte: **la ultima palabra es tuya**.
* Si el motor devuelve una cara mezclada, el control la acepta mientras quede
  algo menos de dos tercios de la otra persona. Medido con mezclas artificiales:
  la mitad dejan de pasar en torno al 65% de cara ajena.
* Si la cara sale minuscula (30-40 px) o quemada por la luz, el control no puede
  leerla y **rechaza**. Es un rechazo honesto, pero se paga otra generacion.
* Si faltan los pesos del reconocedor en la maquina, el control **no inventa un
  aprobado**: dice "no se pudo comprobar el rostro" y se queda fuera de la nota.
  Se instalan con `python scripts\fetch_face_model.py`.

### Resumen para decidir

- La cara **esta protegida y medida de verdad desde septiembre de 2026**: 6 de
  las 17 imagenes de pago que hay en disco no son tu cara y hoy se rechazan,
  incluidas las dos que tu rechazaste a mano. Las proporciones siguen medidas
  como antes.
- La piel **es el punto debil del modelo de pago**, y por eso 7 de 15 se
  descartan. Cuenta con que una parte de lo que pagues se vaya en reintentos:
  eso es exactamente el factor 1.35 y el techo que te ensena la pantalla.
- Si una tirada te descarta muchas imagenes por *piel demasiado suavizada*, no
  es que el control se haya vuelto loco: es el modelo alisandote la cara.

---

## 9. Que mirar tu en la imagen final

El robot mide, pero **hay cosas que no mide**. Antes de dar una imagen por
buena, mira tu:

1. **Los bordes de la silueta** contra el fondo: restos del fondo antiguo,
   escalones duros, un trozo de pared pegado a la mano. La puerta no mide el
   recorte.
2. **Las manos y los dedos** de cerca. Se avisan, pero una mano pequena en la
   foto no se puede juzgar por geometria y solo se informa.
3. **Los pies y el borde inferior**: si tu foto de origen corta por los
   tobillos, la generada tambien.
4. **El grano de la piel a tamano real**, no en la miniatura.
5. **Que la ropa sea la que pediste** y no una version que se ha comido tu
   silueta.

Lo que el robot **si** garantiza, con numeros en la ficha de cada imagen: que la
cara es la tuya, que nadie te ha estrechado el cuerpo sin que se vea, que el
tono de piel es el tuyo, que el grano de piel perdido esta por debajo del 14%, y
que no se gasta un centimo mas del techo que te ensena antes de empezar.

Lo que **no** garantiza: un recorte limpio contra el fondo, manos perfectas a
tamano pequeno, encuadres que tu foto de origen no contiene, ni que la imagen te
guste.
