# Entrega final — competencia 01 (DMEyF 2026)

"Tips Ale Bolaño" con el FE histórico de la cátedra. Predice BAJA+2 de 202108.

## Qué hay

- `TipsAleBolanoFE_final.csv` — la lista entregada: 11.250 clientes, uno por línea, sin encabezado.
- `codigo/` — todo lo necesario para rehacerla desde el dataset crudo, de principio a fin.
  - `reproducir.py`: el único comando. Lee el dataset, arma las variables, entrena, promedia y escribe la lista.
  - `prepare.py`: lee el CSV crudo, arma `clase_ternaria` y separa lo que se entrena (hasta 202106) de lo que se puntúa (202108).
  - `modelos/tips_ale_bolano_fe.py`: el modelo (`make_features` y `train_and_predict`). El encabezado explica las tres piezas.
  - `esperado.json`: huella (sha256) de la lista de la entrega y de la lista de cada semilla sola.
  - `requirements.txt`: versiones exactas, incluidas las dos dependencias de LightGBM. Importan: la selección de variables por canaritos depende de ellas.

## El modelo, en tres líneas

1. Modelo general: LightGBM de tres clases con abril, mayo y junio; las BAJA+1 pesan la mitad al entrenar. Incluye el FE histórico que publicó la cátedra (tendencia y promedio de 3 meses).
2. Especialista de deudores (clientes con `internet` >= 1), que elige sus variables con canaritos y se combina con el general.
3. Segunda etapa: un modelo chico que aprende en mayo y junio a corregir la franja dudosa y la reordena en agosto.

La lista es un ensamble de semillas: el modelo se corre 25 veces, una por semilla (`SEMILLAS`, en `reproducir.py`), se
promedian las probabilidades y quedan los 11.250 clientes de mayor probabilidad (`CORTE`).

Los hiperparámetros y dos listas fijas de variables (`PERCENTIL_CERO_FIJO` y `VARIABLES_2`) están escritos en
`modelos/tips_ale_bolano_fe.py` y no se recalculan al correr; la única selección de variables que se hace al correr es la del
especialista (canaritos).

## Cómo reproducir

El dataset crudo va en `~/competencia_01_crudo.csv`. El que se usó pesa 492.769.580 bytes y tiene 983.061 filas (más el
encabezado), 154 columnas y `foto_mes` de 202103 a 202108; su sha256 es
`c72c8f5c1149f1d972e996c2963c5e3dbb31eb38cfe41c2bb1bbaf847d7a7ca1` (`shasum -a 256 ~/competencia_01_crudo.csv`; en Linux,
`sha256sum`). La lista depende de los valores y también del orden de las filas y de las columnas del archivo: con otra
exportación del mismo dataset puede no coincidir.

```bash
cd codigo
pip install -r requirements.txt
python reproducir.py
```

Hace falta Python 3.12 o más nuevo (se hizo con 3.14.4), en un entorno virtual nuevo para que `python` y `pip` sean los de ese
entorno (`python3 -m venv ~/venv-entrega && source ~/venv-entrega/bin/activate`, antes del `pip install`). En macOS, LightGBM
necesita OpenMP: `brew install libomp`.
La lista idéntica, cliente por cliente y en el mismo orden, está comprobada en la máquina que se describe más abajo; con otro
sistema operativo u otro procesador LightGBM puede dar diferencias chicas en las probabilidades. En ese caso el sha256 puede no
coincidir aunque la lista sea casi la misma, porque compara también el orden de las líneas; `reproducir.py` dice además
cuántos clientes tiene en común con la lista entregada, sin mirar el orden.

Escribe `codigo/salida/TipsAleBolanoFE_final.csv` y dice si salió idéntica a la de referencia (y cuántas de las 25 listas por
semilla coinciden). Tarda unos 20 minutos en una Mac de 18 núcleos (75 s la primera semilla, unos 40 s las demás) y usa unos
21 GB de memoria (más en el pico, mientras arma las variables: conviene una máquina con 32 GB o más). Cada semilla se guarda apenas termina: si se corta, al relanzar sigue por la que falta. Para rehacer todo de
cero hay que borrar `codigo/salida/`: lo que ya está guardado ahí (el dataset en parquet y las semillas) no se vuelve a calcular.

`python reproducir.py --prueba` corre sólo la primera semilla y compara su lista con la de referencia.

Al entrenar imprime dos listas de variables elegidas por el especialista (24 para 202106 y 20 para 202104; la de 202104 es
del modelo que puntúa mayo y junio para la segunda etapa). Las dos son parte
del modelo: si en otra máquina la lista no coincide, lo primero a mirar es si salieron las mismas variables y si las versiones
son las de `requirements.txt`.

Las dos líneas que imprime la corrida de referencia (el orden de las variables también es parte del modelo):

```
especialista de deudores (202106): 24 de 1160 variables le ganan al ruido: ctrx_quarter, ctrx_quarter_min3, ctrx_quarter_avg3, ctrx_quarter_lag1, mprestamos_personales_delta1, mprestamos_personales_ratiomax3, mcuenta_corriente_delta2, mpayroll, ctrx_quarter_max3, mtransferencias_recibidas, mprestamos_personales_ratiolag1, Visa_mpagominimo, mpasivos_margen_ratioavg3, ctrx_quarter_lag2, ctrx_quarter_ratiomax3, Visa_msaldototal, mprestamos_personales_ratioavg3, mpasivos_margen_ratiolag1, cdescubierto_preacordado, ctrx_quarter_ratioavg3, mpayroll_lag1, ctrx_quarter_ratiolag1, cliente_edad_lag1, Visa_msaldopesos
especialista de deudores (202104): 20 de 1160 variables le ganan al ruido: mprestamos_personales, ctrx_quarter, ctrx_quarter_min3, ctrx_quarter_avg3, mprestamos_personales_ratioavg3, mprestamos_personales_min3, ctrx_quarter_lag1, ctrx_quarter_max3, mtarjeta_visa_consumo, mprestamos_personales_ratiomax3, mprestamos_personales_ratiolag1, mprestamos_personales_avg3, cprestamos_personales_ratiolag1, cprestamos_personales_ratioavg3, cprestamos_personales, Master_status, internet_lag1, Visa_msaldototal, mtarjeta_visa_consumo_min3, cproductos
```

## Con qué se hizo

Todo se hizo y se corrió en una sola máquina, una Mac:

- MacBook Pro (Mac17,6), chip Apple M5 Max de 18 núcleos (6 + 12), 128 GB de memoria, macOS 27.0.1 (arm64).
- Python 3.14.4 (CPython), pip 26.2.1.
- LightGBM 4.7.0 (rueda de PyPI para macOS arm64, con libomp 23.1.2 de Homebrew), pandas 3.0.5, NumPy 2.5.2, PyArrow 25.0.1 y
  SciPy 1.18.1 y narwhals 2.25.0 (dependencias de LightGBM).

Las huellas de `esperado.json` salieron de esa máquina. La corrida de punta a punta, desde el CSV crudo, dio la lista
idéntica a la entregada (25 de 25 semillas y la lista final).

## Autoría

Emmanuel Valdez no escribió ni tocó una sola línea de este código. Todo (la preparación de los datos, el modelo y el script que
reproduce la lista) lo hizo su agente de IA, Claude Code (Anthropic), siguiendo sus indicaciones.
