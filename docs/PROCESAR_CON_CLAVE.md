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
RESULTADO: 48 correctas, 0 fallidas   |   COSTE REAL: 0.0000 USD
```

Comprueba 48 cosas con numeros: que la clave se guarda por la pantalla de
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

Y esta ensaya **el unico fallo que se ha cobrado de verdad** en esta
instalacion: fal ejecuta la imagen, la cobra, su propio revisor de contenido la
marca y devuelve un archivo completamente en negro.

```powershell
$env:PHOTOROBOT_FAL_REPLAY_BLOCK = "2"
```

El ensayo lo hace en su apartado 11b y comprueba lo que importa: que los
0.1000 USD **se apuntan** en el libro mayor (dos apuntes de 0.0500 con la nota
"imagen bloqueada por fal"), que la tirada termina sola sin error tecnico, que
**no se compra un tercer intento** despues de dos negros, y que lo lees en
castellano. Hasta el 2026-09-05 ese camino existia y no se habia ejecutado
nunca: la tirada real que le paso a esta cuenta anoto 0.0000 USD por dos
imagenes que fal si cobro.

**Una tirada que gasta dinero se lanza en segundo plano y con su propio
registro, nunca en primer plano.** El 2026-09-04, durante la tirada de pago de
este proyecto, el limite de tiempo de la consola mato la sesion a mitad; la
tirada sobrevivio solo porque el proceso de Python no murio con ella, y durante
unos diez minutos no habia forma de saber si se habia facturado una segunda
llamada o no. No se perdio dinero, pero el susto era evitable:

```powershell
Start-Process -NoNewWindow -FilePath backend\.venv\Scripts\python.exe `
  -ArgumentList "scripts\rehearse_paid.py" -RedirectStandardOutput tirada.log
```

Y mientras corre, la verdad esta siempre en el libro mayor, no en la consola:

```powershell
backend\.venv\Scripts\python.exe -c "import sqlite3;print(list(sqlite3.connect('data/photorobot.sqlite3').execute('select kind,amount_usd,ref from ledger order by id desc limit 5')))"
```

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
3 variante(s) x 0.0400 USD  + factor 1.35  = 0.2670 USD
Aviso: fal entrega como maximo 1024 px de lado largo en esta calidad, no 1536.
Aviso: la estimacion cuenta con que 1 de cada 3 imagenes se repita. En el peor
caso esta tirada llega a 1.71 USD: cada imagen se intenta como mucho 3 veces
(normalmente 2, porque el robot deja de pagar intentos en cuanto la misma
comprobacion falla dos veces con el mismo resultado) y solo se repintan por
zonas (a 0.05 USD cada una) los fallos que no se pueden corregir gratis.
```

Y lo que gasto de verdad esa misma tirada en el ensayo de hoy (2026-09-04):
**0.2000 USD**, 5 generaciones a 0.04 y ningun repintado, frente a 0.2670
estimados y un techo de 1.7100. **El techo es la promesa; el estimado es solo
una media.**

### Precios reales por llamada

**El precio lo decide lo que pides, no el nivel de calidad.** Desde el
2026-09-04 el robot **protege tu cara con una mascara** siempre que el cambio
que pides no mueva a la persona dentro de la foto: se repinta solo la ropa (o
la escena) y tu cara y tus manos **se copian de tu propia fotografia, pixel a
pixel**. Ese camino es `fal-ai/flux-pro/v1/fill` y cuesta **0.0500 USD** por
imagen en los cinco niveles.

Cuando pides otra **postura** u otro **encuadre** hay que volver a dibujarte
entera - cambiar la pose mueve tu cuerpo dentro del cuadro, medido en tus
fotos: la cabeza se desplaza 1,23 cabezas de mediana - y entonces se usa
`fal-ai/flux-pro/kontext/multi`, que viaja con tres fotos tuyas y cuesta
**0.0400 USD**, o `flux/dev` a 0.00625 USD en borrador. Ahi el rostro **se
comprueba** despues, en vez de conservarse.

Medido el 2026-09-04 sobre 25 combinaciones (5 peticiones x 5 niveles), con la
estimacion y el ensayo comparados fila a fila: **coinciden las 25**.

| Lo que pides | Nivel | Modelo de fal | Por imagen | Tu cara |
|---|---|---|---|---|
| solo ropa | los cinco | `fal-ai/flux-pro/v1/fill` | 0.0500 USD | **no se vuelve a dibujar** |
| ropa + color | los cinco | `fal-ai/flux-pro/v1/fill` | 0.0500 USD | **no se vuelve a dibujar** |
| solo escena | los cinco | `fal-ai/flux-pro/v1/fill` | 0.0500 USD | **no se vuelve a dibujar** |
| postura | borrador | `fal-ai/flux/dev/image-to-image` | 0.00625 USD | se vuelve a dibujar |
| postura | previa a maxima | `fal-ai/flux-pro/kontext/multi` | 0.0400 USD | se vuelve a dibujar |
| encuadre | borrador | `fal-ai/flux/dev/image-to-image` | 0.00625 USD | se vuelve a dibujar |
| encuadre | previa a maxima | `fal-ai/flux-pro/kontext/multi` | 0.0400 USD | se vuelve a dibujar |

**El tamano del archivo tambien cambia con el camino**, y esto es lo que mas se
nota: con mascara recibes **tu foto entera al tamano de tu camara** (medido:
2316x3088) porque solo se sustituye la zona repintada; sin mascara recibes lo
que da Kontext, **1024x1024**. Sin mascara tampoco viajan garantizados tus
tatuajes ni tus pendientes: se piden en el prompt, pero el modelo redibuja y
puede perderlos.

Sin referencias (perfil sin construir) Alta y Maxima sin mascara suben a
`kontext/max`, 0.0800 USD.

### Lo que cuesta *antes* de generar

Pulsar *Calcular coste* no es gratis del todo si tienes clave de Anthropic:
Claude lee la fotografia para redactar las instrucciones y eso cuesta unos
**0.0112 USD**. Desde el 2026-09-05 ese gasto **se apunta en tu saldo de
Anthropic**, se te dice en el mismo aviso de la pantalla de coste, y **se paga
una sola vez por fotografia**: la lectura se guarda junto a la foto y volver a
presupuestar esa misma foto ya no cuesta nada. Si te quedas sin saldo de
Anthropic, la lee el lector local gratuito y se te avisa de que la descripcion
sera mas pobre. Antes de esa fecha esta llamada se hacia en **cada**
presupuesto y no aparecia en ninguna parte: 33 presupuestos preparados en esta
instalacion y 0 apuntes de anthropic en el libro mayor.

El techo de 1.7100 USD es lo maximo que las reservas de la tirada llegan a
retener: 3 intentos x (3 generaciones a 0.0400 + 3 zonas a 0.0500 por imagen).
Es un peor caso que exige que las tres imagenes fallen las tres veces con tres
zonas repintables cada vez; lo medido en los caminos de fallo normales son
0.20 - 0.38 USD.

**Tu decides ese techo.** En *Ajustes*, `max_retries` y `max_repair_rounds`
(0 a 3) y el interruptor `autorepair` se aplican de verdad a la tirada desde el
2026-09-04 - antes se guardaban y no los leia nadie. Medido sobre esta misma
tirada de 3 imagenes en Alta:

| Ajuste | Estimado | Techo | Intentos por imagen |
|---|---|---|---|
| De serie (2 reintentos, 2 rondas) | 0.2670 USD | 1.7100 USD | 3 |
| `autorepair` apagado | 0.1080 USD | 0.2400 USD | 3 |
| 0 reintentos y 0 reparaciones | 0.0800 USD | 0.0800 USD | 1 |

Esa tabla se midio sobre el camino **sin mascara** (Kontext a 0.0400). La misma
tirada **con cambio de ropa**, que va por `fill` a 0.0500, se cotizo el
2026-09-05 en **0.3075 USD estimados con un techo de 1.8000 USD** de serie. La
regla no cambia: el techo baja a la vez que el estimado cuando bajas
`max_retries` y `max_repair_rounds`, y con 0 y 0 los dos son el mismo numero.

Con 0 y 0 el estimado y el techo son el mismo numero: la tirada no puede
gastar ni un centimo mas de lo que te ensena.

Ademas, cada **repintado de una zona** cuesta 0.0500 USD, y una reparacion
repinta hasta 3 zonas. Por eso un borrador de 0.006 USD puede acabar costando
0.15 USD si su unica pega esta en tres sitios: **lo caro no es generar, es
arreglar**.

Los tamanos de la columna "Pide" salen de tu propia foto: 2316x3088 es 3:4, asi
que se pide 3:4. Si eliges un encuadre concreto manda ese: cuerpo entero
1152x1536, medio cuerpo 1232x1536, cuadrado 1536x1536, vertical 9:16 864x1536.

## 5. Que nivel elegir

**Sin mascara, los cinco niveles entregan practicamente el mismo tamano de
archivo**, unos 1024 px de lado largo: los modelos de fal no tienen mando de
resolucion. Lo que compras al subir de nivel es **fidelidad a tus rasgos**, no
pixeles. **Con mascara da igual el nivel**: el archivo que recibes conserva el
tamano de tu propia foto, porque solo se sustituye la zona repintada.

- **Vista previa** para explorar: cuando quieres ver seis ideas de escena, luz
  o encuadre y decidir cual te gusta. Cuesta 0.05 USD la imagen si solo cambias
  ropa o escena, 0.04 USD si cambias la postura.
- **Alta** para la imagen que te vas a quedar. Con cambio de ropa cuesta lo
  mismo que la vista previa (0.05 USD, el mismo `fill`); sin mascara sube tus
  fotos de referencia a 2048 px en vez de 1536, o sea un 28% mas de cara en
  cada referencia, por los mismos 0.04 USD con el perfil construido (0.08 sin
  el).
- **Borrador (0.006 USD)** solo para probar el circuito, y **solo cambia de
  precio en postura y encuadre**: un cambio de ropa se repinta con mascara y
  cuesta 0.05 USD en cualquier nivel. Es un modelo mas flojo, y como cada
  reparacion suya cuesta ocho veces lo que costo generarla, sale caro en cuanto
  falla.
- **Maxima** cuesta y entrega hoy lo mismo que Alta; existe para cuando fal
  publique un modelo mayor.

Recomendacion practica: **vista previa para elegir, y luego alta calidad solo
sobre las que has marcado** (boton *Alta calidad* en el paso de eleccion).

Y si una vista previa ya te vale tal cual, **no la vuelvas a generar**:
seleccionala en el album y pulsa *Marcar final*. Pasa a la pestana "Finales"
por 0.0000 USD, porque es el mismo archivo que ya pagaste. El boton solo acepta
imagenes que hayan pasado todas las comprobaciones; el resto sigue teniendo que
pasar por *Alta calidad*, que es una llamada nueva y se cobra.

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
| 6% | 0% (0 de 6) | **0%** |
| 12% | 50% (3 de 6) | **0%** |
| 18% | 50% (3 de 6) | **0%** |

Medido el 2026-09-05 con ese mismo comando. Las tres fotos que no detecta son
aquellas en las que el encuadre no deja medir la figura en cabezas, y ahi el
robot **lo dice** en vez de dar la imagen por buena en silencio. Un
adelgazamiento del **6% no se puede separar hoy** del volumen que anade una
prenda: es la limitacion que hay que conocer antes de fiarse del control.

## 7. Que pasa cuando una comprobacion falla

1. **Se intenta reparar solo la zona rota.** Si el defecto es local (una mano,
   un trozo de piel, una zona borrosa), se repinta esa zona con una mascara,
   sin rehacer la foto. Cuesta 0.05 USD por zona, hasta 3 zonas.
2. **Si el repintado no mejora la medida, se revierte.** El robot mide otra vez
   la gravedad del defecto antes y despues; si no baja, deja la imagen como
   estaba. La zona se paga igual, porque fal cobra el repintado aunque el
   resultado no sirva, y por eso ese gasto aparece en el libro mayor.
3. **Si sigue sin pasar, se genera otra vez**, hasta 2 reintentos por vista (3
   intentos en total, o los que hayas dejado en *Ajustes*). Y se para antes si
   la misma comprobacion falla **dos veces con el mismo resultado**: eso no es
   mala suerte, es el motor dando la misma respuesta, y no se paga un tercer
   intento por ella.
4. **Si tampoco, se descarta con el motivo escrito en castellano** y aparece en
   la ficha de la tirada. La frase empieza siempre por *No te la ensenamos
   porque...* y los motivos posibles son exactamente estos: *la cara que ha
   salido no es la tuya*, *el cuerpo ha salido con otra forma*, *tu piel ha
   salido de otro tono*, *hay algo mal dibujado, por ejemplo una mano*, *la
   imagen ha salido borrosa*.
5. **Si fal devuelve un archivo en negro** - lo hace cuando su propio revisor de
   contenido marca la imagen que acaba de dibujar - el trabajo **si se cobra**,
   porque la inferencia se ejecuto, y se te dice con esas palabras. Se prueba
   otra semilla; si el segundo intento vuelve a salir en negro, **no se paga un
   tercero**: eso ya no es mala suerte, es esa peticion. Medido el 2026-09-04
   sobre las 175 peticiones reales de esta cuenta: 12 bloqueos de 175 (6,9%),
   repartidos por los tres modelos.

   Ya no es la palabra de fal: el 2026-09-05 esta instalacion se quedo con el
   archivo. Son **19 376 bytes**, 1024x1440, y al medirlo da media **0,0000**,
   desviacion **0,0000** y **un solo valor distinto en 4 423 680 pixeles**. Negro
   entero, sin degradado y sin fantasma de la prenda. En tu libro mayor esa
   llamada aparece como *imagen bloqueada por fal*, con la referencia del intento
   y los 0.0500 USD, y en la ficha como el aviso *La imagen 1 se hizo pero fal.ai
   no la dio por buena... se cobra igual*. La fila del intento guarda ademas el
   `request_id` de fal, el endpoint real, el rectangulo que se subio y lo que
   tardo, de modo que el proximo bloqueo se puede auditar desde tu maquina; el
   del 2026-09-05 hubo que reconstruirlo del panel de fal porque esa parte
   todavia no existia.

Antes de llegar al punto 1, el robot intenta **arreglarlo gratis**: la textura
de piel se devuelve desde una de tus propias fotografias, y las manos y el
rostro tienen su propia correccion local. Nada de eso llama al proveedor ni
cuesta un centimo, y ninguna de esas tres cosas abre ya una ronda de repintado
de pago.

En el ensayo del 2026-09-05 (calidad alta, 3 vistas con cambio de ropa, coste
real 0.00 USD porque las imagenes salen de disco y la red esta cerrada): la
tirada se enruta a `fal-ai/flux-pro/v1/fill`, se cotiza **3 x 0.0500 + factor
1.35 = 0.3075 USD** con un techo de 1.8000, y liquida **0.4500 USD** en 6
intentos: 2 aceptadas y 4 descartadas por *hay algo mal dibujado, por ejemplo
una mano*. Ese numero de descartes es del ensayo y no del robot: en modo ensayo
la "respuesta de fal" es una imagen ya comprada de otra escena, pegada dentro de
la zona de la ropa, y eso mete de verdad una mano rota donde el motor tenia
permiso para pintar. Con una respuesta que si corresponde a la peticion, las
mismas 156 peticiones enmascaradas dieron **0 descartes**.

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
album, la ficha lo dice con el numero y el motivo es *"No te la ensenamos porque
la cara que ha salido no es la tuya"*.

**Lo que cuesta cuando rechaza por la cara.** Una cara equivocada **no se puede
reparar por zonas**: un retoque local puede devolver grano a una mejilla, pero no
puede convertir a otra mujer en ti. Asi que:

* la imagen rechazada **ya esta pagada** (0.04 USD: un rechazo por la cara solo
  puede ocurrir en el camino sin mascara, que es el de Kontext, porque en el
  camino con mascara tu cara no se dibuja): la llamada se hizo y el proveedor
  cobra igual;
* esa imagen pagada **ya no se borra**: se conserva, se nombra en la ficha de la
  tirada y no se vuelve a gastar en ella;
* el robot **genera otra entera**, a ese mismo precio, no al de reparacion
  (0.05 USD), que aqui no se usa, y como mucho **2 reintentos por vista** antes
  de descartarla con el motivo escrito - o solo uno, si las dos primeras
  lecturas salen iguales;
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

- La cara **esta protegida y medida de verdad desde septiembre de 2026**.
  Barrido del 2026-09-04 con la comprobacion que decide: tus 24 fotografias
  0.6908 - 0.8737 (24 de 24 aceptadas), dos fotos tuyas que el perfil no ha
  visto nunca 0.7624 y 0.8222 (aceptadas), ocho mujeres distintas 0.0201 -
  0.1925 (8 de 8 rechazadas) y las 4 imagenes de pago que hay en disco y no son
  tu cara 0.2920 - 0.3968 (4 de 4 rechazadas). El limite, 0.45, cae en una
  franja donde no hay ninguna imagen. Las proporciones siguen medidas como
  antes.
- La piel **es el punto debil del modelo de pago**, y por eso 7 de 15 se
  descartan. Cuenta con que una parte de lo que pagues se vaya en reintentos:
  eso es exactamente el factor 1.35 y el techo que te ensena la pantalla.
- Si una tirada te descarta muchas imagenes por *piel demasiado suavizada*, no
  es que el control se haya vuelto loco: es el modelo alisandote la cara.

---

## 8b. Que sale de tu maquina en cada llamada de pago

Esto es lo que un tercero llega a ver de tus fotografias, y no es lo mismo en
los dos caminos.

**Camino con mascara** (`fal-ai/flux-pro/v1/fill`, cambio de ropa, fondo,
calzado o accesorios). La respuesta del proveedor **solo se conserva dentro de
lo blanco de la mascara**: `generation/protect.py` vuelve a pegar todo lo demas
desde tu archivo a resolucion completa. Sale **tu fotografia entera**, con tu
cabeza y tus hombros.

**Aqui habia un recorte y se ha quitado, porque estaba mal.** Durante un dia se
subio solo el rectangulo que contiene la zona a repintar mas un 6% de contexto,
con el argumento de que lo de fuera se tira igual. Ese rectangulo **es la zona
de la ropa, y empieza en tu barbilla**: lo que recibia el proveedor era un torso
sin cabeza, el pecho centrado, los hombros y los brazos al aire. Medido sobre
las dos fotografias de las llamadas de pago, con la misma envoltura de tu piel
que usa tu propia ficha:

| lo que revisa el proveedor | recorte | foto entera |
|---|---|---|
| WhatsApp 2026-09-04, parte del cuadro que sube | 34,85% | **100%** |
| tu piel descubierta, sobre lo que sube | 27,50% | **9,76%** |
| tu piel dentro de la zona a repintar | 20,09% | **7,00%** |
| tu recuadro de cara que viaja | 0,0% | **100%** |
| IMG_7871, parte del cuadro que sube | 36,18% | **100%** |
| tu piel descubierta, sobre lo que sube | 33,05% | **12,11%** |
| tu piel dentro de la zona a repintar | 30,14% | **10,90%** |
| tu recuadro de cara que viaja | 1,9% | **100%** |

El recorte multiplicaba por **2,8** la densidad de piel de la imagen que se
revisa, y las **dos** llamadas de pago con mascara volvieron bloqueadas - la
segunda sobre una fotografia tuya completamente vestida. Ademas le quitaba al
modelo lo que necesita para ajustar una prenda a una persona: tu escala, tus
hombros, de donde viene la luz y que hay en la habitacion.

**Tu cara no la protegia el recorte y no la protege el encuadre: la protege la
mascara.** Sobre tus **25 fotografias**, la zona a repintar toca **0 pixeles**
del recuadro de tu cara, y encima de todo lo negro de la mascara se vuelven a
escribir tus propios pixeles a resolucion completa. Eso se mide en cada llamada
y se escribe en la fila del intento (`envio_completo`, `enviado`) en vez de
afirmarse.

**Camino sin mascara** (`kontext/multi`, cuando pides otra postura u otro
encuadre): ahi viajan **tres fotografias tuyas enteras**, porque el modelo tiene
que volver a dibujarte y necesita reconocerte. La pantalla de coste te dice cual
de los dos caminos vas a pagar, con el nombre del modelo, antes de gastar.

**El texto.** Ninguno de los seis modelos de fal acepta un campo de negativo
aparte, asi que el negativo se pega dentro de la instruccion y **sale a la red**.
El bloque que evitaba una prenda puesta encima de la ropa interior nombraba lo
que no queria - *underwear*, *lingerie*, *bra*, *thong*, *naked* - y el propio
`safety/guard.py` de este producto rechaza ese texto si se lo escribes tu. Se ha
quitado y el mismo requisito se dice en positivo (*la prenda descrita es la unica
ropa de la imagen: tejido opaco, torso, caderas y piernas cubiertos*). Medido
sobre las 93 filas de intento que guardan un negativo: se retiran **14 clausulas
en el camino con mascara** y **0 en kontext**, que es el que ha hecho todas las
imagenes que tienes. Lo que protege tu cuerpo - *slimmed waist*, *narrowed
shoulders*, *altered breast size*, *removed tattoos* - se queda entero en los
dos: una clausula que nombra un **cambio en tu cuerpo** no se retira nunca,
aunque use una palabra del cuerpo.

**Y lo que el texto seguia llevando despues de aquel filtro.** Medido sobre las
4 141 letras que fal recibio de verdad el 2026-09-05: *top worn without bottoms*,
*shirt worn as a dress*, *no trousers*, *missing trousers* - ninguna nombra una
prenda interior y todas describen a alguien a medio vestir - y *breast* una vez y
*bust* dos, dentro de clausulas que protegen tu cuerpo. Ahora las cuatro frases
se retiran **solo en el camino con mascara** (en `kontext` protegen un fallo real
y nunca han costado nada) y los sustantivos **no se borran, se cambian** por el
clinico: *altered breast size* sale como *altered chest size* y la proteccion
viaja entera. Sobre esa misma peticion: **18 clausulas retiradas** en vez de 14 y
**0 palabras marcadas** en el texto final, frente a 7 antes.

### Lo que revisa el proveedor es tu foto entera, y antes no lo era

Esto costo 0,100 USD averiguarlo y es lo mas util de esta pagina. Mientras hubo
recorte, el rectangulo que viajaba **concentraba tu piel**, porque ese
rectangulo *es* la zona de la ropa - la parte mas desnuda de cualquier foto de
una persona - y ademas dejaba la cabeza fuera:

| | foto entera | rectangulo que viajaba | factor |
|---|---|---|---|
| IMG_7871 (bloqueada dos veces) | 12,1% de piel | 33,1% | **2,73x** |
| la de la cocina, vestida (bloqueada) | 9,8% de piel | 27,5% | **2,82x** |

O sea: cambiar a una fotografia vestida mejoro lo que el revisor vio de **33,1%
a 27,5%**, un 17% relativo, y no el 36% que se habia calculado sobre el cuadro
entero. Y lo que estaba mirando en los dos casos era un **torso sin cabeza**.
El recorte se ha quitado: **sube tu fotografia entera**, y eso **se mide al
dibujar la mascara y se te dice en la pantalla de coste**, antes de pagar:

> Sale tu foto entera, con tu cabeza y tus hombros: es lo que el modelo
> necesita para ajustar la ropa a tu cuerpo, y es lo que revisa el proveedor.
> De esa foto se repinta el 17%. Tu piel descubierta es el 10% de lo que se
> envia, y el 7% queda dentro de la zona que se repinta. Tu cara viaja entera y
> no se repinta: la mascara la deja fuera por completo (0 pixeles del recuadro
> de tu cara dentro de la zona) y encima se vuelven a poner tus propios
> pixeles.

### Que fotografia tuya conviene como origen

Por orden de lo que de verdad mueve el resultado:

1. **Vestida en la zona que vas a cambiar.** La foto de la cocina (top de canale
   gris y falda negra) baja la zona a repintar del 22,3% al 18,5% del cuadro y la
   piel dentro de esa zona del 52,8% al 40,1%.
2. **Media figura o cuerpo entero.** En un primer plano no hay torso que aislar:
   la zona pasa a ser toda tu, no hay recorte que hacer y se sube el cuadro
   entero.
3. **De tu camara, no de WhatsApp.** El archivo se entrega al tamano de tu foto:
   1200x1599 (1,9 MP) contra 2316x3088 (7,2 MP), y la cara a resolucion completa
   baja de 332 px a 201 px. Pasa de sobra el control de identidad (0,872 sobre un
   limite de 0,45), pero es un cuarto de los pixeles.
4. Nitida y bien expuesta. Eso el robot ya te lo mide al importarla.

Y lo que **no** basta: cambiar a una foto vestida, por si solo, **no desbloqueo
nada**. Es el resultado pagado del 2026-09-05.

**Lo que la mascara conserva y lo que no.** Conserva, pixel a pixel de tu propio
archivo: **tu cara, tus manos y tus pendientes**, y **las marcas de tu piel que la
prenda que has pedido deja al aire** - se miden sobre tu propia fotografia con el
color de piel de tu perfil, y se fuerzan a negro junto a la cara y las manos.
Del **pelo**, solo el de la cabeza: la melena que te cae sobre el hombro cuenta
como torso y se repinta (medido en la foto de la cocina: lo que el modelo llama
pelo son 7 907 pixeles, y el resto esta dentro de la zona en un 82,7%).

**Y una marca que la prenda nueva tapa se pierde, siempre.** Una camisa encima de
un tatuaje lo cubre; protegerlo dejaria un agujero con tu piel dentro de la
camisa. El robot te lo dice antes de cobrar y con la zona por su nombre: *"La
ropa que pediste tapa el torso, asi que la marca que llevas ahi no puede
sobrevivir... Si quieres verla, elige una prenda que deje esa zona al aire."*
Hasta donde llega: **ninguna de las 24 prendas del catalogo deja el pecho al
aire**, asi que en un cambio de ropa el tatuaje del pecho se pierde siempre.
Donde si se conserva es en el antebrazo bajo manga corta, en el muslo bajo un
vestido de verano, en el cuello - y en **todas** si lo que pides es un cambio de
color o de transparencia, porque ahi no se viste nada nuevo.

Y lo que se le pide al motor va ahora en el mismo sentido: ya no se le piden a la
vez «una camisa que cubra el torso por completo» y «el tatuaje del pecho sin
cambios», que es lo que decia el texto pagado del 2026-09-05.

---

## 9. Que mirar tu en la imagen final

El robot mide, pero **hay cosas que no mide**. Antes de dar una imagen por
buena, mira tu:

1. **Los bordes de la silueta** contra el fondo: restos del fondo antiguo,
   escalones duros, un trozo de pared pegado a la mano. La puerta no mide el
   recorte.
2. **Las manos y los dedos** de cerca. **Esta es la mas importante de la lista.**
   Medido el 2026-09-04 sobre tus 24 fotografias y todas las generadas en
   disco: de las 26 manos que no salen cortadas por el borde, ninguna llega a
   los 170 px que hacen falta para medir la geometria de los dedos - la mayor
   son 157 px. A ese tamano ninguna regla distingue una mano tuya de una mano
   derretida: tu propia IMG_7880 mide una palma 7.9 veces fuera de rango y una
   mano generada que se ve mal a simple vista mide 3.99. Por eso el robot ya no
   dice "sin anomalias" cuando en realidad no ha podido mirar: la ficha y el
   resumen del album dicen ahora **"Revisa 2 manos: no se ha podido
   comprobar"**, con el motivo de cada una (sale del encuadre, o mide 75 px
   cuando harian falta 170). Cuando leas esa frase, amplia la mano tu.
3. **Los pies y el borde inferior**: si tu foto de origen corta por los
   tobillos, la generada tambien.
4. **El grano de la piel a tamano real**, no en la miniatura.
5. **Que la ropa sea la que pediste** y no una version que se ha comido tu
   silueta.

Lo que el robot **si** garantiza, con numeros en la ficha de cada imagen: que la
cara es la tuya, que nadie te ha estrechado el cuerpo sin que se vea, que el
tono de piel es el tuyo, que el grano de piel perdido esta por debajo del 14%, y
que no se gasta un centimo mas del techo que te ensena antes de empezar.

Cuando la ropa que pediste puede ensancharte, la comprobacion de proporciones
solo puede vigilar que **no te estrechen**, y desde hoy el resumen lo dice con
esas palabras en vez de afirmar que tus proporciones coinciden.

Lo que **no** garantiza: un recorte limpio contra el fondo, manos perfectas a
tamano pequeno, encuadres que tu foto de origen no contiene, **una marca que la
prenda nueva tapa**, el pelo que te cae sobre la ropa, ni que la imagen te guste.

### Que hacer cuando fal bloquea un resultado

fal revisa **lo que acaba de dibujar**, no lo que le pides: si su revisor lo
marca, devuelve un archivo completamente negro con HTTP 200. El dibujo se
ejecuto, asi que **se cobra** (0,05 USD) y queda apuntado en tu libro mayor como
*imagen bloqueada por fal*. Por orden, y nada de esto cuesta hasta el ultimo
punto:

1. **No lo tomes como un fallo del robot ni de tu foto.** Se te dice con esas
   palabras en los avisos de la tirada. Si sale negro dos veces seguidas, el
   robot **no paga un tercer intento**.
2. **Mira el porcentaje de piel del rectangulo**, el de la pantalla de coste. Si
   es mas del doble que el de tu foto entera, lo que hay que cambiar es la zona,
   no la foto.
3. **Prueba una foto de origen mas vestida en esa zona, o una prenda que repinte
   menos** (una falda repinta menos que un vestido completo).
4. **O pide otra postura o encuadre**, que va por `kontext/multi` (0,04 USD, tres
   fotos tuyas): es el camino con 1 bloqueo en unas 42 llamadas, y ahi si se
   comprueba tu rostro antes de ensenarte nada.

Y lo que todavia **no esta probado**, contado por caminos: **1 bloqueo en unas
42 llamadas de imagen entera** (`kontext`), contra **4 de 4** en el camino con
mascara mientras hubo recorte - dos el 2026-09-04, una el 2026-09-05 con el
texto ya limpio, y otra con una fotografia **vestida**, que tampoco desbloqueo
nada. Las 26 reparaciones de zona que usan el mismo endpoint pasaron todas, y la
unica diferencia es que la reparacion sube una imagen **ya vestida que hizo el
modelo**, mientras que las cuatro bloqueadas subieron el recorte de tu
fotografia real: un torso sin cabeza.

Esa diferencia es la que se acaba de quitar. Desde ahora el camino con mascara
sube tu **fotografia entera**, igual que el camino que solo se ha bloqueado 1 vez
en 42. Todo lo que promete la seccion 8b esta medido en local; **contra la API
real, el camino con mascara sigue sin entregar una imagen**, y la primera
llamada que se haga asi es la que lo dira.
