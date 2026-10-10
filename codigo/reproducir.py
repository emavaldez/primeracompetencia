"""Reproduce, desde el dataset crudo, la lista de la entrega de la competencia 01 (DMEyF 2026): "Tips Ale Bolaño" con el FE
histórico de la cátedra.

De principio a fin:
  1. Lee ~/competencia_01_crudo.csv y arma clase_ternaria (prepare.py).
  2. Arma las variables (make_features, en modelos/tips_ale_bolano_fe.py).
  3. Entrena el modelo con la clase conocida hasta 202106 y puntúa 202108, una vez por cada semilla de SEMILLAS
     (train_and_predict, en el mismo archivo).
  4. Promedia las probabilidades de las semillas y escribe los CORTE clientes de mayor probabilidad.

Escribe en salida/:
  competencia_01.parquet       el dataset con la clase (para no volver a leer el CSV)
  parcial/<semilla>.npy        probabilidades de cada semilla; se guardan apenas terminan: si se corta, al relanzar sigue
  semilla_<semilla>.csv        la lista de cada semilla sola
  TipsAleBolanoFE_final.csv    la lista de la entrega
Al terminar compara el sha256 de cada archivo con el de referencia (esperado.json).

Uso:
  python reproducir.py            # la lista de la entrega
  python reproducir.py --prueba   # prueba rápida: sólo la primera semilla

Con qué se hizo (todo en una sola máquina, una Mac):
  MacBook Pro (Mac17,6), chip Apple M5 Max de 18 núcleos (6 + 12), 128 GB de memoria, macOS 27.0.1 (arm64).
  Python 3.14.4 (CPython) | LightGBM 4.7.0 (rueda de PyPI para macOS arm64, con libomp 23.1.2 de Homebrew) | pandas 3.0.5 |
  NumPy 2.5.2 | PyArrow 25.0.1 | SciPy 1.18.1 y narwhals 2.25.0 (dependencias de LightGBM) | pip 26.2.1.
  Las versiones exactas de las librerías están en requirements.txt. Las huellas de esperado.json salieron de esta máquina.

Autoría: Emmanuel Valdez no escribió ni tocó una sola línea de este código. Todo (la preparación de los datos, el modelo y
este script) lo hizo su agente de IA, Claude Code (Anthropic), siguiendo sus indicaciones.
"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import numpy as np

import prepare

MODELO = "tips_ale_bolano_fe"
SEMILLAS = [
    800029, 800083, 800119, 800131, 800209, 800221, 800333, 800357, 800419, 800441, 800477, 800497, 800509,
    800519, 800593, 800669, 800707, 800743, 800759, 800861, 800897, 800971, 800993, 801179, 801197,
]  # una corrida del modelo por semilla; la lista sale del promedio de todas
CORTE = 11250  # clientes de la lista

prueba = "--prueba" in sys.argv
seeds = SEMILLAS[:1] if prueba else SEMILLAS
esperado = json.loads((AQUI / "esperado.json").read_text())

spec = importlib.util.spec_from_file_location(f"modelo_{MODELO}", AQUI / "modelos" / f"{MODELO}.py")
modelo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modelo)

out = AQUI / "salida"
(out / "parcial").mkdir(parents=True, exist_ok=True)
faltan = [s for s in seeds if not (out / "parcial" / f"{s}.npy").exists()]
print(f"{len(seeds)} semillas, faltan {len(faltan)}; lista de {CORTE} clientes", flush=True)
if faltan:
    df = prepare.load_data()
    tr, sc, mid = prepare._split3(df, prepare.SUBMIT["last_label"], prepare.SUBMIT["score"], modelo.make_features)
    ids = sc.numero_de_cliente.to_numpy().astype("int64")
    np.save(out / "parcial" / "ids.npy", ids)
    for s in faltan:
        t = time.time()
        p = np.asarray(prepare.predecir(modelo.train_and_predict, tr, sc, s, mid), dtype=np.float64)
        np.save(out / "parcial" / f"{s}.tmp.npy", p)
        (out / "parcial" / f"{s}.tmp.npy").rename(out / "parcial" / f"{s}.npy")
        print(f"  semilla {s}: {time.time() - t:.0f}s", flush=True)
ids = np.load(out / "parcial" / "ids.npy")
probs = np.stack([np.load(out / "parcial" / f"{s}.npy") for s in seeds])


def lista(p):
    """Los CORTE clientes de mayor probabilidad, uno por línea, sin encabezado (orden estable ante empates)."""
    return "\n".join(str(int(x)) for x in ids[np.argsort(-p, kind="stable")[:CORTE]]) + "\n"


def escribir(ruta, texto):
    ruta.write_bytes(texto.encode())  # bytes tal cual: en Windows write_text cambiaría los saltos de línea
    return hashlib.sha256(texto.encode()).hexdigest()


iguales = sum(escribir(out / f"semilla_{s}.csv", lista(p)) == esperado["semillas"].get(str(s)) for s, p in zip(seeds, probs))
print(f"listas por semilla: {len(seeds)} archivos en {out} | idénticas a las de referencia: {iguales} de {len(seeds)}")
if not prueba:
    h = escribir(out / "TipsAleBolanoFE_final.csv", lista(probs.mean(axis=0)))
    print(f"lista de la entrega: {out / 'TipsAleBolanoFE_final.csv'} (promedio de {len(seeds)} semillas, {CORTE} clientes)")
    print(f"  sha256 {h} | idéntica a la de referencia: {'sí' if h == esperado['final'] else 'NO'}")
    entregada = AQUI.parent / "TipsAleBolanoFE_final.csv"  # la lista entregada, al lado de codigo/
    if entregada.exists():  # el sha256 mira también el orden; esto cuenta los clientes en común sin mirarlo
        comun = set(entregada.read_text().split()) & set((out / "TipsAleBolanoFE_final.csv").read_text().split())
        print(f"  clientes en común con la lista entregada ({entregada.name}): {len(comun)} de {CORTE}")
