"""
Tips Ale Bolaño: modelo de la competencia 01 (DMEyF 2026). Predice BAJA+2 de 202108.

Tres piezas, todas entrenadas con la clase conocida (hasta 202106). Hay dos grupos de
clientes definidos a mano (deudores = internet >= 1; franja dudosa = puestos 8.000 a 30.000 sin marca internet); lo que se
hace con cada uno lo aprende un modelo.

Por semilla:
  1. GENERAL: LightGBM multiclase (CONTINUA / BAJA+1 / BAJA+2) con los tres últimos meses con clase, submuestreo 0,2 de los
     CONTINUA, variables crudas + lags y deltas de 1 y 2 meses + razones y ventana de 3 meses, con sus tratamientos de deriva
     (ver los comentarios de cada bloque). Las BAJA+1 pesan la mitad al entrenar (PESO_BAJA1): la clase que paga es BAJA+2.
     Suma el FE histórico de la cátedra: tendencia y promedio de 3 meses (ver fe_historico_catedra).
  2. ESPECIALISTA DE DEUDORES: binario, sólo clientes con internet >= 1 del último mes con clase. Elige sus variables con
     canaritos (columnas de números al azar): quedan las que rinden más que el mejor canarito (24 de ~1.160). Se combina con el
     general por reparto: el nivel lo pone el máximo de los dos y el orden la suma de sus logit.
  3. SEGUNDA ETAPA: un LightGBM muy chico (4 hojas, 100 rondas, 25 variables) que aprende a corregir al modelo en la franja
     dudosa. Se entrena con los clientes sin marca internet que el modelo deja entre los puestos 8.000 y 30.000 en mayo y
     junio, puntuados por un modelo que NO vio la clase de esos meses, y después reordena esa misma franja del mes a puntuar.
     No cambia los valores de probabilidad: cambia a quién le toca cada uno.
     El modelo que puntúa mayo y junio se entrena una sola vez por corrida (no por semilla) con marzo y abril, que casi no
     tienen historia: es un modelo más pobre que el de la entrega (mayo le queda a un mes y junio a dos).
     VARIABLES_2 es una lista fija (no se recalcula en la corrida): se eligió con canaritos sobre las franjas de mayo y junio.

Se usa desde reproducir.py (make_features y train_and_predict son la interfaz); no se corre solo. train tiene que traer TODOS
los meses con clase desde el primero del dataset: con clase sólo hasta 202104 la segunda etapa no actúa.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- features

# Empiezan con c pero no son conteos: edades (crecen solas: no llevan razones ni ventana; sí lags, deltas y el FE de la
# cátedra) y banderas 0/1.
NO_MAGNITUD = {
    "cliente_edad", "cliente_antiguedad",
    "cliente_vip", "cdescubierto_preacordado", "ccajas_transacciones", "cmobile_app_trx",
}


def es_magnitud(c: str) -> bool:
    """Montos (m*) y conteos (c*), del cliente o de sus tarjetas (Visa_m*, Master_c*): 128 de las 152 variables.

    Quedan afuera las fechas (F*, f*), las banderas y los códigos (t*, status, delinquency, internet, active_quarter):
    ahí una razón o un promedio no significan nada.
    """
    nombre = c.split("_", 1)[1] if c.startswith(("Visa_", "Master_")) else c
    return nombre[0] in "mc" and c not in NO_MAGNITUD


def razon(num, den):
    """num / den; con denominador 0 da NaN (nunca infinito)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den == 0, np.nan, num / den)


# Data Quality (tips de la cátedra, 5-oct): una variable rota en un mes se pasa a NA sólo en ese mes, antes de lags y
# deltas. Tabla fija. ccajas_depositos está rota en 202105 (todos ceros).
ROTAS = {"ccajas_depositos": (202105,)}

# Data Drifting. Mirando las 152 variables mes a mes, la escala de estos montos se corre de un mes a
# otro: aguinaldo (202106: mpayroll x1,56; el saldo y la caja de ahorro quedan inflados en 202106-07, con el primer cuartil
# del saldo x5) e inflación en tarjeta, comisiones y transferencias (+15 a +55% en 202106-08 contra 202103-05). El modelo
# aprende con 202104-06 y agosto se puntúa con lags de junio y julio: un corte en pesos aprendido en abril no vale lo mismo
# en agosto, y "cayó respecto de junio" en agosto es la resaca del aguinaldo, no una fuga.
# Selección en dos pasos: (1) montos (m*) con algún cuartil de 202106-08 a más de 25% del promedio de 202103-05 y al menos
# 0,5% de la ganancia del modelo (la importancia "gain" de LightGBM); (2) de esos, sólo los que con el percentil quedan MÁS
# parecidos entre el entrenamiento (202104-06) y agosto (PSI de sus columnas, ponderado por esa ganancia: 0,057 -> 0,010
# en estos 14).
# Quedaron afuera los que el percentil empeora, que son los de pocos valores distintos o con muchos empates:
# cpayroll_trx (1, 2, 3: el percentil de "1" pasa de 30 a 15 en junio), mcomisiones_mantenimiento (lista de precios),
# mcaja_ahorro_dolares, Visa_mpagominimo, Visa_mpagospesos y mttarjeta_visa_debitos_automaticos. Los préstamos no derivan.
PERCENTIL_CERO_FIJO = [
    "mcuentas_saldo", "mpayroll", "mcaja_ahorro", "mpasivos_margen", "mtarjeta_visa_consumo", "mcuenta_corriente",
    "mtransferencias_recibidas", "mrentabilidad", "mcomisiones_otras", "mcomisiones", "Visa_msaldototal",
    "Visa_msaldopesos", "mautoservicio", "mtransferencias_emitidas",
]
# Corrimiento de calendario: son "días desde el último cierre" y el grupo mayoritario (69% de los clientes) vale
# 1 / 2 / 5 / 0 / 3 / 6 en 202103..08; el segundo grupo va 7 días atrás. Se le resta la moda del mes: queda el grupo de
# cierre del cliente (0, 7, ...), igual todos los meses. (El percentil no sirve acá: da 36,18 / 36,22 / 36,27... y esos
# decimales le dejan al árbol separar los meses.)
RESTAR_MODA = ["Visa_fultimo_cierre", "Master_fultimo_cierre"]


def percentil_cero_fijo(x: pd.Series, mes: pd.Series) -> pd.Series:
    """Percentil (0 a 100) dentro del mes: usa sólo las filas de ese mes, así que no mira ni adelante ni atrás.

    Los positivos se ordenan entre sí (0, 100], los negativos entre sí [-100, 0) y el cero queda en 0: se conservan el signo
    y el "no tiene", que es lo que el ranking simple borra.
    La escala 0-100 es para que el redondeo a centavos de la ventana no le coma resolución.
    """
    pos = x.where(x > 0).groupby(mes).rank(pct=True)
    neg = (-x).where(x < 0).groupby(mes).rank(pct=True)
    return (100 * (pos.fillna(0) - neg.fillna(0))).where(x.notna())


def lags_y_deltas(df: pd.DataFrame) -> pd.DataFrame:
    order = df.sort_values(["numero_de_cliente", "foto_mes"]).index
    df = df.loc[order]
    for c, meses in ROTAS.items():
        df[c] = df[c].where(~df.foto_mes.isin(meses))
    for c in PERCENTIL_CERO_FIJO:
        df[c] = percentil_cero_fijo(df[c], df.foto_mes)
    for c in RESTAR_MODA:
        df[c] = df[c] - df.groupby("foto_mes")[c].transform(lambda v: v.mode().iloc[0])
    base = [c for c in df.columns if c not in ("numero_de_cliente", "foto_mes")]
    g = df.groupby("numero_de_cliente")[base]
    new = {}
    for k in (1, 2):  # Feature Engineering histórico de la cátedra: lags y delta lags de orden 1 y 2
        lag = g.shift(k)
        new.update({f"{c}_lag{k}": lag[c] for c in base})
        new.update({f"{c}_delta{k}": df[c] - lag[c] for c in base})
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df.sort_index()


def historia(x, l1, l2):
    """FE histórico de la cátedra para una variable, sobre la ventana que dejan 6 meses de datos: el mes actual (x),
    lag1 y lag2, con los que haya."""
    v = np.stack([x, l1, l2])
    # los montos traen centavos: se redondea para que lo que da 0 sea 0 exacto y no un resto de punto flotante
    prom = razon(np.nansum(v, axis=0).round(2), (~np.isnan(v)).sum(axis=0))
    maxi = np.fmax.reduce(v)  # fmax y fmin saltean los NaN
    return {
        # razones sin escala: un árbol no las arma solo y no se corren con la inflación
        "ratiolag1": razon(x, l1),
        "ratioavg3": razon(x, prom),
        "avg3": prom,
        "min3": np.fmin.reduce(v),
        "max3": maxi,
        "ratiomax3": razon(x, maxi),
    }


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe todas las filas de meses <= mes a puntuar (sin clase). Una fila de salida por fila de entrada."""
    magnitudes = [c for c in df.columns if es_magnitud(c)]
    df = lags_y_deltas(df)
    # Se calcula fila a fila a partir de los lags: sólo mira hacia atrás. Las columnas nuevas no llevan lag ni delta.
    # float32: son 768 columnas más sobre ~980.000 filas y al árbol le sobran 7 dígitos.
    hist = {}
    for c in magnitudes:
        for k, a in historia(*(df[n].to_numpy(float) for n in (c, f"{c}_lag1", f"{c}_lag2"))).items():
            hist[f"{c}_{k}"] = a.astype("float32")
    return fe_historico_catedra(pd.concat([df, pd.DataFrame(hist, index=df.index)], axis=1))


# ---------------------------------------------------------------- FE histórico de la cátedra: tendencia y promedio de 3 meses
#
# Es lo que agregó la cátedra a su Feature Engineering Histórico (función TendenciaYmuchomas con ventana 3, tendencia y
# promedio): la pendiente por cuadrados mínimos de los últimos tres valores y su promedio, para todas las variables con lag.
# Con los tres meses la pendiente es (x - lag2) / 2; con dos, x - lag1; con uno queda vacía (igual que en su código).
# Quedan afuera el detalle del resumen de tarjeta (Visa_*, Master_*: su historia no entra, ver historia_de_resumen) e internet
# (tiene su propio tratamiento). El promedio sólo se agrega a las variables que no lo tenían (las que no son montos ni conteos).
# Estas columnas las ve sólo el modelo general.
FE_CATEDRA = ("_tend3", "_avg3p")


def fe_historico_catedra(d: pd.DataFrame) -> pd.DataFrame:
    crudas = [c for c in d.columns if f"{c}_lag1" in d.columns and f"{c}_lag2" in d.columns
              and not c.startswith(("Visa_", "Master_")) and c != "internet"]
    nuevo = {}
    for c in crudas:
        y = d[[f"{c}_lag2", f"{c}_lag1", c]].to_numpy(float)
        x = np.array([1.0, 2.0, 3.0]); ok = ~np.isnan(y); n = ok.sum(1); y0 = np.where(ok, y, 0)
        sx, sy, sxy, sxx = (ok * x).sum(1), y0.sum(1), (y0 * x).sum(1), (ok * x * x).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (n * sxy - sx * sy) / (n * sxx - sx * sx)
            a = sy / n
        t[n < 2] = np.nan; a[n < 2] = np.nan
        nuevo[f"{c}_tend3"] = t.astype("float32")
        if f"{c}_avg3" not in d.columns:
            nuevo[f"{c}_avg3p"] = a.astype("float32")
    return pd.concat([d, pd.DataFrame(nuevo, index=d.index)], axis=1)


# ---------------------------------------------------------------- modelo

TRAIN_MONTHS_BACK = 3  # entrega: 202104-06 (todos con lag1); excluye 202103, que no tiene historia (el modelo que puntúa mayo y junio para la segunda etapa sí lo usa: no tiene otro)

PARAMS = dict(
    objective="binary",  # sin efecto: base() lo pisa con "multiclass" de 3 clases
    learning_rate=0.027,
    num_leaves=31,
    min_data_in_leaf=200,
    feature_fraction=0.5,
    max_bin=255,
    verbosity=-1,
    num_threads=16,
)
NUM_ROUNDS = 500
UNDERSAMPLING = 0.2  # fracción de CONTINUA que se conserva
POSITIVOS = ["BAJA+1", "BAJA+2"]  # las dos clases de baja
PESO_BAJA1 = 0.5  # las BAJA+1 pesan la mitad al entrenar el general: se van en un mes y no pagan; BAJA+2 es la clase que paga


HISTORIA = tuple(f"_{s}" for s in ("lag1", "delta1", "lag2", "delta2", "ratiolag1", "ratioavg3", "avg3", "min3", "max3", "ratiomax3"))


def historia_de_resumen(c: str) -> bool:
    """Historia derivada (lags, deltas, razones, ventana) del detalle del resumen de tarjeta (Visa_*, Master_*): no entra.

    Esas 44 variables son la foto del último resumen y dependen de en qué día cae el cierre (fultimo_cierre), que se
    corre con el calendario: es el bloque que más cambia de distribución entre los meses de entrenamiento y el mes a
    puntuar (KS mediano de sus derivadas 0,037 contra 0,016 del resto; las fechas llegan a 0,7-1,0). La diferencia de
    un saldo de resumen contra el mes anterior mezcla el ciclo del resumen con lo que hizo el cliente. Entran crudas
    (el modelo sigue viendo saldo, pago mínimo, mora y límite de hoy), y la historia del uso de la tarjeta queda en los
    agregados del banco (ctarjeta_*_transacciones, mtarjeta_*_consumo), que sí conservan sus derivadas.
    Junto con el tratamiento de la deriva: Visa_msaldototal y Visa_msaldopesos (a percentil dentro del mes) y los dos
    fultimo_cierre (menos la moda del mes) son de este bloque, así que entran sólo con su valor del mes ya corregido;
    de los 14 montos a percentil, los otros 12 son del banco y conservan su historia, calculada sobre el percentil.
    """
    return c.startswith(("Visa_", "Master_")) and c.endswith(HISTORIA)

# ---------------------------------------------------------------- especialista de deudores, con canaritos
#
# Por qué existe el especialista: `internet` con valor 1 a 3 marca deudores en problemas (saldo negativo, mora de tarjeta, 3 o más
# préstamos). Sus bajas son cierres que decide el banco y cambian de un mes a otro (2,3% de BAJA+2 en 202104, 1,6% en 202105 y
# 4,3% en 202106); el general promedia los tres meses y los subestima. El especialista mira SÓLO a ese grupo y SÓLO el último
# mes con clase. Por qué con canaritos: con ~6.000 filas y ~380 bajas, 1.160
# variables son demasiadas y el especialista ajusta ruido. Los canaritos miden cuánta ganancia junta una columna de números
# al azar; sólo quedan las variables reales que juntan más que el mejor de ellos.
SEGMENTO_INTERNET = 1  # "tiene / no tiene" la marca
PARAMS_ESP = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=15,  # el segmento son ~6.000 filas y ~380 bajas: árboles chicos
    min_data_in_leaf=50,
    feature_fraction=0.3,
    max_bin=255,
    verbosity=-1,
    num_threads=4,  # con tan pocas filas más hilos no aceleran
)
NUM_ROUNDS_ESP = 200
MIN_FILAS_ESP = 200  # resguardo: con menos filas o una sola clase no hay nada que aprender
CANARITOS = 50    # cuántas columnas de ruido
REPETICIONES = 5  # corridas con canaritos que se promedian
MINIMO = 5        # nunca menos variables que éstas
_ELEGIDAS = {}    # train del segmento -> lista: la selección no depende de la semilla, se calcula una vez por proceso


def le_ganan_al_ruido(tr: pd.DataFrame, cols: list, y: np.ndarray) -> list:
    """Variables de cols cuya ganancia media supera a la del mejor canarito, ordenadas de mayor a menor ganancia."""
    mes = int(tr.foto_mes.iloc[0])
    clave = (mes, len(tr), int(tr.numero_de_cliente.sum()), int(y.sum()), tuple(cols))  # el train exacto y sus columnas
    if clave not in _ELEGIDAS:
        X = tr[cols].reset_index(drop=True)
        ruido = [f"canarito_{i}" for i in range(CANARITOS)]
        ganancias = []
        for j in range(REPETICIONES):
            rng = np.random.default_rng(10_000 * CANARITOS + 100 * (mes % 100) + j)
            D = pd.concat([X, pd.DataFrame(rng.random((len(X), CANARITOS)), columns=ruido)], axis=1)
            model = lgb.train({**PARAMS_ESP, "seed": 9000 + j}, lgb.Dataset(D, label=y), num_boost_round=NUM_ROUNDS_ESP)
            ganancias.append(model.feature_importance("gain"))
        g = np.mean(ganancias, axis=0)
        real, vara = g[:len(cols)], g[len(cols):].max()
        orden = np.argsort(-real, kind="stable")
        elegidas = [cols[i] for i in orden if real[i] > vara]
        _ELEGIDAS[clave] = elegidas if len(elegidas) >= MINIMO else [cols[i] for i in orden[:MINIMO]]
        # Para diagnóstico: el conjunto y el ORDEN de estas variables son parte del modelo (feature_fraction sortea por
        # posición). Si en otra máquina la lista final no coincide, lo primero a mirar es si salen las mismas.
        print(f"  especialista de deudores ({mes}): {len(_ELEGIDAS[clave])} de {len(cols)} variables le ganan al ruido: "
              + ", ".join(_ELEGIDAS[clave]), flush=True)
    return _ELEGIDAS[clave]


def especialista(train: pd.DataFrame, score: pd.DataFrame, cols: list, seed: int):
    """Probabilidad de baja (BAJA+1 o BAJA+2) de los clientes del segmento, en la escala del general.

    Devuelve (máscara de filas de score que son del segmento, probabilidad de esas filas) o (máscara, None) si no se pudo
    entrenar. Se entrena sin submuestreo y se corrige a la escala del general, p / (p + 0,2 (1 - p)), sólo
    con las variables que le ganan al ruido.
    """
    en_seg = (score.internet >= SEGMENTO_INTERNET).to_numpy()
    ultimo = train.foto_mes.max()
    tr = train[(train.foto_mes == ultimo).to_numpy() & (train.internet >= SEGMENTO_INTERNET).to_numpy()]
    y = (tr.clase_ternaria != "CONTINUA").to_numpy().astype(int)
    if len(tr) < MIN_FILAS_ESP or y.min() == y.max() or not en_seg.any():
        return en_seg, None
    cols_e = [c for c in cols if c != "foto_mes"]  # un solo mes: foto_mes es constante
    elegidas = le_ganan_al_ruido(tr, cols_e, y)
    model = lgb.train({**PARAMS_ESP, "seed": int(seed)}, lgb.Dataset(tr[elegidas], label=y), num_boost_round=NUM_ROUNDS_ESP)
    p = model.predict(score.loc[en_seg, elegidas])
    return en_seg, p / (p + UNDERSAMPLING * (1 - p))


def base(train: pd.DataFrame, score: pd.DataFrame, seed: int) -> np.ndarray:
    """General + especialista de deudores con reparto. Devuelve P(BAJA+1) + P(BAJA+2) del general, con el reparto en el segmento."""
    months = sorted(train.foto_mes.unique())[-TRAIN_MONTHS_BACK:]
    train = train[train.foto_mes.isin(months)]
    # Columnas: todas menos la historia del resumen de tarjetas (ver historia_de_resumen); el FE de la cátedra lo ve sólo el general.
    cols = [c for c in train.columns if c not in ("clase_ternaria", "numero_de_cliente") and not historia_de_resumen(c)]
    # El especialista va antes del submuestreo (usa todas las filas del segmento) y antes de enmascarar internet:
    # igual sólo mira el último mes con clase, que el enmascarado no toca.
    en_seg, p_esp = especialista(train, score, [c for c in cols if not c.endswith(FE_CATEDRA)], seed)  # el especialista no ve el FE de la cátedra
    # modelo general, para todos los clientes: multiclase, con submuestreo de los CONTINUA y las BAJA+1 a medio peso
    keep = np.random.default_rng(seed).random(len(train)) < UNDERSAMPLING
    train = train[keep | (train.clase_ternaria != "CONTINUA").to_numpy()]
    y = train.clase_ternaria.map({"CONTINUA": 0, "BAJA+1": 1, "BAJA+2": 2})
    w = np.where(train.clase_ternaria == "BAJA+1", PESO_BAJA1, 1.0)
    # internet cambió de régimen: a igual actividad, tener internet >= 2 dos meses seguidos no subía la baja en 202104-05
    # (1-3% contra 2-6% del resto) y en 202106 la duplica (6-7% contra 2-4%). El promedio de los tres meses borra lo que
    # vale hoy, así que su efecto se aprende SÓLO del último mes con clase: en los meses anteriores de entrenamiento va
    # en 0 (el valor del 96%). No en NaN: "internet falta" marcaría el mes y el árbol aprendería un efecto de junio.
    # Va junto con la quita de la historia del resumen porque no se pisan: aquel filtro sólo mira Visa_*/Master_* y acá
    # quedan las 5 columnas de internet (cruda, 2 lags, 2 deltas). El assert es para que un cambio de nombres no deje
    # el enmascarado vacío sin avisar (el modelo entrenaría igual, con internet promediado en los tres meses).
    internet = [c for c in cols if c.startswith("internet")]
    assert len(internet) == 5, f"se esperaban 5 columnas de internet y hay {internet}"
    train.loc[(train.foto_mes < months[-1]).to_numpy(), internet] = 0
    dtrain = lgb.Dataset(train[cols], label=y, weight=w, free_raw_data=True)
    model = lgb.train({**PARAMS, "objective": "multiclass", "num_class": 3, "seed": seed}, dtrain, num_boost_round=NUM_ROUNDS)
    p = model.predict(score[cols])[:, 1:].sum(axis=1)
    if p_esp is not None:
        # Mes por mes: el perfil y el orden son de los clientes de UN mes. Al puntuar 202108 hay un solo mes (da lo mismo),
        # pero base_fuera_de_muestra puntúa mayo y junio en una sola llamada, y repartir todo junto le daría a un cliente
        # de junio un valor sacado de los de mayo.
        mes_seg, p_seg = score.foto_mes.to_numpy()[en_seg], p[en_seg]
        for m in np.unique(mes_seg):
            k = mes_seg == m
            p_seg[k] = repartir(p_seg[k], p_esp[k])
        p[en_seg] = p_seg
    return p


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def repartir(p_gen: np.ndarray, p_esp: np.ndarray) -> np.ndarray:
    """Probabilidad final de los clientes del segmento: CUÁNTO sube el grupo lo decide el máximo; A QUIÉN, los dos modelos.

    Si cada cliente del segmento se quedara con max(general, especialista), el grupo sube, que es lo que hace falta (el
    general lo subestima), pero a quién se sube lo elige el ruido del especialista: sube el cliente en el que el
    especialista, con sus ~6.000 filas, se pasó para arriba.
    Acá se separan las dos cosas:
      - el perfil: los valores de max(general, especialista) del segmento, ordenados. Son exactamente los mismos números
        que daría el máximo, así que por semilla entra a la lista la misma cantidad de deudores y con el mismo nivel;
      - el orden: esos valores se reparten de mayor a menor según el promedio de los logit de los dos modelos, con pesos
        iguales.
    """
    perfil = np.sort(np.maximum(p_gen, p_esp))[::-1]
    orden = np.argsort(-(logit(p_gen) + logit(p_esp)), kind="stable")
    out = np.empty_like(p_gen)
    out[orden] = perfil
    return out


# ---------------------------------------------------------------- segunda etapa sobre la franja dudosa

FRANJA = (8000, 30000)  # puestos de la base, sólo clientes sin marca internet
PARAMS_2 = dict(objective="binary", learning_rate=0.03, num_leaves=4, min_data_in_leaf=200, feature_fraction=0.3, lambda_l2=20.0,
                max_bin=63, verbosity=-1, num_threads=3)
RONDAS_2 = 100
MESES_DETRAS = 2             # meses de datos que tiene que tener detrás un mes para entrenar con él
SEMILLAS_FUERA = (800011,)  # semillas de la base fuera de muestra (se promedian si hay varias); se entrena una vez por proceso
VARIABLES_2 = [  # las que les ganan a los canaritos (percentil 95), en orden de ganancia
    "mcomisiones_mantenimiento_lag2", "mrentabilidad_max3", "mcuentas_saldo_delta2", "mcomisiones_mantenimiento_ratioavg3",
    "ccomisiones_mantenimiento_lag2", "ctrx_quarter_lag2", "mtarjeta_master_consumo_lag1", "ctrx_quarter_delta2", "cliente_edad_lag2",
    "chomebanking_transacciones_ratiomax3", "cproductos_ratiomax3", "ctrx_quarter_lag1", "mcuenta_corriente", "cliente_edad_lag1",
    "mtransferencias_recibidas_ratioavg3", "mtarjeta_master_consumo_ratiolag1", "ctransferencias_recibidas_ratioavg3",
    "chomebanking_transacciones_ratioavg3", "ccaja_ahorro_lag2", "cproductos_ratioavg3", "mcomisiones_mantenimiento",
    "ccomisiones_mantenimiento_ratioavg3", "mpagomiscuentas_delta2", "ctarjeta_visa_transacciones_ratioavg3",
    "ctarjeta_visa_transacciones_ratiomax3",
]
_FUERA = {}  # train -> {mes: base fuera de muestra}; se calcula una vez por proceso


def puesto(p):
    """Puesto de cada cliente (0 = el más probable), con orden estable."""
    r = np.empty(len(p), int)
    r[np.argsort(-p, kind="stable")] = np.arange(len(p))
    return r


def grupo(internet, p, franja=None):
    """Clientes de un mes en la franja de puestos de p y sin marca internet."""
    franja = franja or FRANJA
    r = puesto(p)
    return (r >= franja[0]) & (r < franja[1]) & (np.asarray(internet) == 0)


def calibracion(l, y):
    """Regresión logística y ~ c0 + c1 * l, por Newton."""
    A = np.column_stack([np.ones(len(l)), l])
    c = np.zeros(2)
    for _ in range(40):
        mu = 1 / (1 + np.exp(-A @ c))
        c += np.linalg.solve(A.T @ (A * (mu * (1 - mu))[:, None]) + 1e-8 * np.eye(2), A.T @ (y - mu))
    return c


def meses_de_entrenamiento(train):
    """Los dos últimos meses con clase que tienen MESES_DETRAS meses de datos antes (mayo y junio en la entrega)."""
    meses = sorted(int(m) for m in train.foto_mes.unique())
    return [m for i, m in enumerate(meses) if i >= MESES_DETRAS][-2:]


def base_fuera_de_muestra(train: pd.DataFrame) -> dict:
    """{mes: (clientes, probabilidad de la base para esas filas, sin haber visto su clase)} para mayo y junio.

    UN corte: la base se entrena con las filas con clase ANTERIORES al primero de los dos meses (clase <= 202104) y puntúa
    los dos; si SEMILLAS_FUERA trae más de una, se promedian. Se calcula una vez por proceso y vale para todas las semillas de la entrega.
    """
    meses = meses_de_entrenamiento(train)
    clave = (tuple(meses), len(train), int(train.numero_de_cliente.sum()))
    if clave not in _FUERA:
        out = {}
        if meses:
            antes = train[(train.foto_mes < meses[0]).to_numpy()]
            a_puntuar = train[train.foto_mes.isin(meses).to_numpy()].drop(columns=["clase_ternaria"])
            p = np.mean([np.asarray(base(antes, a_puntuar, s), dtype=np.float64) for s in SEMILLAS_FUERA], axis=0)
            mes = a_puntuar.foto_mes.to_numpy()
            ids = a_puntuar.numero_de_cliente.to_numpy()
            out = {m: (ids[mes == m], p[mes == m]) for m in meses}
        _FUERA[clave] = out
    return _FUERA[clave]


def segunda_etapa(train: pd.DataFrame, score: pd.DataFrame, seed: int, p_base: np.ndarray) -> np.ndarray:
    """Probabilidad corregida, del mismo largo que p_base (la de base() para las filas de score, con esa semilla).

    Entrena con las franjas de mayo y junio (por el orden de la base fuera de muestra), sin marca internet y sin las BAJA+1;
    clase: BAJA+2 contra CONTINUA; punto de partida c0 + c1 * logit(base), la calibración de la base dentro de la franja.
    Sin meses con qué entrenar (clase sólo hasta abril) devuelve p_base sin tocar.
    """
    p = np.asarray(p_base, dtype=np.float64).copy()
    fuera = base_fuera_de_muestra(train)
    if not fuera:
        return p
    X, y, l = [], [], []
    for m, (ids_m, pm) in fuera.items():
        t = train[(train.foto_mes == m).to_numpy()]
        assert np.array_equal(t.numero_de_cliente.to_numpy(), ids_m), "la base fuera de muestra no está alineada con las filas del mes"
        clase = t.clase_ternaria.to_numpy()
        k = grupo(t.internet.to_numpy(), pm) & (clase != "BAJA+1")
        X.append(t.loc[k, VARIABLES_2]); y.append((clase[k] == "BAJA+2").astype(float)); l.append(logit(pm[k]))
    X, y, l = pd.concat(X, ignore_index=True), np.concatenate(y), np.concatenate(l)
    if len(y) < 2 * PARAMS_2["min_data_in_leaf"] or y.min() == y.max():  # con muy pocas filas o una sola clase no hay con qué aprender
        return p
    c = calibracion(l, y)
    modelo = lgb.train({**PARAMS_2, "seed": seed}, lgb.Dataset(X, label=y, init_score=c[0] + c[1] * l), num_boost_round=RONDAS_2)
    mes, internet = score.foto_mes.to_numpy(), score.internet.to_numpy()
    for m in np.unique(mes):  # mes por mes: el puesto y el reparto son de los clientes de UN mes
        f = np.flatnonzero(mes == m)
        idx = f[grupo(internet[f], p[f])]
        if not len(idx):
            continue
        s = modelo.predict(score.iloc[idx][VARIABLES_2], raw_score=True) + c[0] + c[1] * logit(p[idx])
        s = pd.Series(s).rank().to_numpy() + pd.Series(p[idx]).rank().to_numpy()  # suma de puestos: corregido + base
        orden = np.argsort(-s, kind="stable")
        nuevo = np.empty(len(idx))
        nuevo[orden] = np.sort(p[idx])[::-1]  # los mismos valores, repartidos por el orden nuevo
        p[idx] = nuevo
    return p



# ---------------------------------------------------------------- el modelo

def train_and_predict(train: pd.DataFrame, score: pd.DataFrame, seed: int, intermedio=None) -> np.ndarray:
    return segunda_etapa(train, score, seed, base(train, score, seed))
