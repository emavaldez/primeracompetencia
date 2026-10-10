"""
prepare.py — datos de la competencia 01 de DMEyF 2026: lee el dataset crudo, arma clase_ternaria y separa lo que se entrena
de lo que se puntúa.

  build_cache()  lee el CSV crudo, arma clase_ternaria con la receta de la cátedra (controla la cantidad de BAJA+2 de cada mes
                 contra la de la cátedra) y lo guarda en parquet, para no volver a leer el CSV en cada corrida.
  _split3()      devuelve train (los meses con clase, hasta 202106), score (202108) e intermedio (202107, sin clase).
  predecir()     llama al modelo.
make_features recibe SÓLO meses <= mes a puntuar y SIN la clase, así no hay fuga.
"""
import inspect
import os
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(os.path.expanduser("~/competencia_01_crudo.csv"))
CACHE = Path(__file__).resolve().parent / "salida" / "competencia_01.parquet"

SUBMIT = {"last_label": 202106, "score": 202108}  # último mes con clase y mes que se puntúa


def _next_month(m):
    y, mm = divmod(int(m), 100)
    return (y + 1) * 100 + 1 if mm == 12 else m + 1


def build_cache():
    df = pd.read_csv(CSV, engine="pyarrow")
    pres = df[["numero_de_cliente", "foto_mes"]]
    k = set(zip(pres.numero_de_cliente.to_numpy(), pres.foto_mes.to_numpy()))
    m1 = df.foto_mes.map(_next_month).to_numpy()
    m2 = pd.Series(m1).map(_next_month).to_numpy()
    ids = df.numero_de_cliente.to_numpy()
    in1 = np.fromiter(((c, m) in k for c, m in zip(ids, m1)), bool, len(df))
    in2 = np.fromiter(((c, m) in k for c, m in zip(ids, m2)), bool, len(df))
    last = df.foto_mes.max()
    anteultimo = sorted(df.foto_mes.unique())[-2]
    clase = np.where(~in1, "BAJA+1", np.where(~in2, "BAJA+2", "CONTINUA")).astype(object)
    clase[df.foto_mes.to_numpy() >= anteultimo] = None
    clase[(df.foto_mes.to_numpy() == anteultimo) & ~in1] = "BAJA+1"
    clase[df.foto_mes.to_numpy() == last] = None
    df["clase_ternaria"] = clase
    ref = {202103: 960, 202104: 1139, 202105: 870, 202106: 1098}
    got = df[df.clase_ternaria == "BAJA+2"].foto_mes.value_counts().to_dict()
    assert all(got.get(m) == n for m, n in ref.items()), f"clase_ternaria no coincide con la cátedra: {got}"
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)


def load_data():
    if not CACHE.exists():
        build_cache()
    return pd.read_parquet(CACHE)


def _split3(df, last_label, score_month, make_features):
    visible = df[df.foto_mes <= score_month]
    clase = visible.clase_ternaria.where(visible.foto_mes <= last_label)
    feats = make_features(visible.drop(columns=["clase_ternaria"]).copy())
    assert feats.index.equals(visible.index), "make_features debe devolver una fila por fila de entrada, mismo índice y orden"
    feats = feats.reset_index(drop=True)
    clase = clase.reset_index(drop=True)
    train = feats[clase.notna().to_numpy()].copy()
    train["clase_ternaria"] = clase[clase.notna()].to_numpy()
    score = feats[(feats.foto_mes == score_month).to_numpy()].copy()
    # meses SIN clase entre el último etiquetado y el que se puntúa (202107 al predecir 202108). No traen clase; el modelo
    # de esta entrega los recibe y no los usa (202107 sólo entra como historia de las filas de 202108: lags, deltas y ventana).
    intermedio = feats[((feats.foto_mes > last_label) & (feats.foto_mes < score_month)).to_numpy()].copy()
    return train, score, intermedio


def predecir(train_and_predict, train, score, seed, intermedio):
    """Llama a train_and_predict; si acepta un 4º argumento, le pasa los meses intermedios sin clase."""
    if len(inspect.signature(train_and_predict).parameters) >= 4:
        return np.asarray(train_and_predict(train, score, seed, intermedio))
    return np.asarray(train_and_predict(train, score, seed))
