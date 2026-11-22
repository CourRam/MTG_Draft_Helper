"""
draft_utils.py  --  Utilidades compartidas del ayudante de draft (notebook + app de Streamlit).

Contiene:
  - Catalogo de cartas (metadatos de Scryfall alineados con los nombres de 17Lands).
  - construir_features(): convierte "un pack + un pool" en UNA FILA POR CARTA CANDIDATA con
    variables que describen la carta, el pool actual y el contexto del pick.
  - puntaje(): puntaje de un modelo (probabilidad o funcion de decision).
  - recomendar(): dado un pack (lista de nombres) y un pool, devuelve las k mejores cartas.

Idea central: el problema se plantea como CLASIFICACION BINARIA a nivel de candidata:
    y = 1 si la carta fue la elegida por el jugador en ese pick, 0 si no.
Luego, dentro de cada pack, se ordenan las candidatas por puntaje y se recomiendan las mejores.
"""

import difflib
import json

import numpy as np
import pandas as pd

COLORES = ["W", "U", "B", "R", "G"]
RAREZA_ORD = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}
TIPOS = ["Creature", "Instant", "Sorcery", "Enchantment", "Artifact", "Land", "Planeswalker", "Legendary"]
CMC_MAX = 7  # cmc >= 7 se agrupa en el cubo 7


def construir_catalogo(cartas_df, nombres):
    """
    cartas_df: CSV de Scryfall (columnas nombre, nombre_cara_frontal, cmc, type_line, colores,
               rareza, power, toughness, ...).
    nombres:   lista ORDENADA con los nombres de carta tal como aparecen en 17Lands (295).
    Devuelve un diccionario de arreglos alineados con 'nombres'. Las cartas que no se encuentren
    en Scryfall reciben valores neutros (incoloras, cmc 3, rareza 'common') y se reportan en 'sin_datos'.
    """
    por_nombre = {}
    for r in cartas_df.itertuples(index=False):
        por_nombre[r.nombre] = r
        por_nombre.setdefault(r.nombre_cara_frontal, r)

    n = len(nombres)
    color_mat = np.zeros((n, len(COLORES)), dtype=np.float32)
    tipo_mat = np.zeros((n, len(TIPOS)), dtype=np.float32)
    cmc = np.full(n, 3.0, dtype=np.float32)
    rar = np.zeros(n, dtype=np.float32)
    power = np.zeros(n, dtype=np.float32)
    tough = np.zeros(n, dtype=np.float32)
    sin_datos = []

    for i, nombre in enumerate(nombres):
        r = por_nombre.get(nombre) or por_nombre.get(nombre.split(" // ")[0])
        if r is None:
            sin_datos.append(nombre)
            continue
        cols = "" if pd.isna(r.colores) else str(r.colores)
        for j, c in enumerate(COLORES):
            color_mat[i, j] = 1.0 if c in cols else 0.0
        tl = "" if pd.isna(r.type_line) else str(r.type_line)
        for j, t in enumerate(TIPOS):
            tipo_mat[i, j] = 1.0 if t in tl else 0.0
        cmc[i] = float(r.cmc) if not pd.isna(r.cmc) else 3.0
        rar[i] = RAREZA_ORD.get(str(r.rareza), 0)
        p = pd.to_numeric(pd.Series([r.power]), errors="coerce").iloc[0]
        t_ = pd.to_numeric(pd.Series([r.toughness]), errors="coerce").iloc[0]
        power[i] = 0.0 if pd.isna(p) else p
        tough[i] = 0.0 if pd.isna(t_) else t_

    return {
        "nombres": list(nombres), "indice": {n_: i for i, n_ in enumerate(nombres)},
        "color_mat": color_mat, "tipo_mat": tipo_mat, "cmc": cmc, "rareza": rar,
        "power": power, "toughness": tough, "sin_datos": sin_datos,
    }


# ----------------------------------------------------------------------
# Conversion de pick -> estructuras numericas
# ----------------------------------------------------------------------
def matriz_pool(pools_dict, catalogo):
    """Lista de diccionarios {carta: copias} -> matriz (n_picks x n_cartas) de enteros."""
    n, m = len(pools_dict), len(catalogo["nombres"])
    P = np.zeros((n, m), dtype=np.int16)
    idx = catalogo["indice"]
    for i, d in enumerate(pools_dict):
        for carta, k in d.items():
            j = idx.get(carta)
            if j is not None:
                P[i, j] = k
    return P


def indices_pack(packs_listas, catalogo):
    """Lista de listas de nombres -> lista de arreglos de indices de carta."""
    idx = catalogo["indice"]
    return [np.array([idx[c] for c in lista if c in idx], dtype=np.int32) for lista in packs_listas]


# ----------------------------------------------------------------------
# Construccion de variables (una fila por carta candidata)
# ----------------------------------------------------------------------
def construir_features(pack_idx, P, pack_number, pick_number, catalogo, pick_ids=None, elegidas=None, draft_ids=None):
    """
    pack_idx:    lista (n_picks) de arreglos con los indices de las cartas disponibles.
    P:           matriz (n_picks x n_cartas) con el pool ANTES de cada pick.
    pack_number, pick_number: arreglos (n_picks).
    pick_ids:    identificador de cada pick (para agrupar candidatas); si None, 0..n-1.
    elegidas:    (opcional) indice de la carta elegida en cada pick -> crea la columna objetivo 'y'.
    draft_ids:   (opcional) identificador del draft de cada pick (para validacion cruzada por grupos).
    Todas las operaciones son vectorizadas y NO aprenden nada de los datos (no hay fuga de informacion).
    """
    n = len(pack_idx)
    lens = np.array([len(a) for a in pack_idx])
    pick_row = np.repeat(np.arange(n), lens)           # a que pick pertenece cada fila candidata
    card = np.concatenate(pack_idx) if n else np.array([], dtype=np.int32)

    C, T = catalogo["color_mat"], catalogo["tipo_mat"]
    cmc, rar = catalogo["cmc"], catalogo["rareza"]
    Pf = P.astype(np.float32)

    # ---- contexto del pool (por pick) ----
    pool_size = Pf.sum(1)
    pool_color = Pf @ C                                  # copias de cada color en el pool
    tot_color = pool_color.sum(1)
    share = pool_color / np.maximum(tot_color, 1.0)[:, None]
    orden = np.argsort(-pool_color, axis=1)[:, :2]
    top2 = np.zeros_like(pool_color, dtype=bool)
    np.put_along_axis(top2, orden, True, axis=1)
    pool_tipo = Pf @ T                                   # copias por tipo
    cmc_b = np.clip(cmc, 0, CMC_MAX).astype(int)
    onehot_cmc = np.eye(CMC_MAX + 1, dtype=np.float32)[cmc_b]
    pool_cmc_hist = Pf @ onehot_cmc
    pool_cmc_mean = (Pf @ cmc) / np.maximum(pool_size, 1.0)

    # ---- variables de la carta candidata ----
    cm = C[card]
    ncol = cm.sum(1)
    sh = share[pick_row]
    hay_color = ncol > 0
    afin_media = np.where(hay_color, (sh * cm).sum(1) / np.maximum(ncol, 1.0), 1.0)
    afin_min = np.where(hay_color, np.where(cm > 0, sh, 9.0).min(1), 1.0)
    en_top2 = np.where(hay_color, (((cm > 0) & ~top2[pick_row]).sum(1) == 0), True).astype(np.float32)
    # Si el pool aun no tiene cartas con color, ningun color esta "elegido": todo cuenta como afin
    sin_pool_color = tot_color[pick_row] == 0
    afin_media = np.where(sin_pool_color, 1.0, afin_media)
    afin_min = np.where(sin_pool_color, 1.0, afin_min)
    en_top2 = np.where(sin_pool_color, 1.0, en_top2)

    cmc_card = cmc[card]
    cb = cmc_b[card]
    datos = {
        "pick_id": (np.arange(n) if pick_ids is None else np.asarray(pick_ids))[pick_row],
        "carta": np.array(catalogo["nombres"], dtype=object)[card],
        # carta
        "cmc": cmc_card, "rareza": catalogo["rareza"][card],
        "power": catalogo["power"][card], "toughness": catalogo["toughness"][card],
        "n_colores": ncol,
    }
    for j, c in enumerate(COLORES):
        datos[f"color_{c}"] = cm[:, j]
    for j, t in enumerate(TIPOS):
        datos[f"tipo_{t}"] = T[card][:, j]
    # encaje carta-pool
    datos["afinidad_color_media"] = afin_media.astype(np.float32)
    datos["afinidad_color_min"] = afin_min.astype(np.float32)
    datos["en_top2_colores"] = en_top2
    datos["copias_en_pool"] = Pf[pick_row, card]
    datos["n_pool_mismo_cmc"] = pool_cmc_hist[pick_row, cb]
    datos["n_pool_mismo_tipo"] = (pool_tipo[pick_row] * T[card]).sum(1)
    # contexto del pool
    datos["pool_size"] = pool_size[pick_row]
    datos["pool_cmc_medio"] = pool_cmc_mean[pick_row].astype(np.float32)
    for j, c in enumerate(COLORES):
        datos[f"pool_prop_{c}"] = sh[:, j]
    datos["pool_n_con_color"] = tot_color[pick_row]
    for j, t in enumerate(TIPOS):
        datos[f"pool_n_{t}"] = pool_tipo[pick_row, j]
    # contexto del pick / pack
    datos["pack_number"] = np.asarray(pack_number)[pick_row].astype(np.float32)
    datos["pick_number"] = np.asarray(pick_number)[pick_row].astype(np.float32)
    datos["tam_pack"] = lens[pick_row].astype(np.float32)

    df = pd.DataFrame(datos)
    # posicion relativa de la carta dentro de su pack (rareza y costo)
    g = df.groupby("pick_id", sort=False)
    df["rareza_rank_pack"] = g["rareza"].rank(pct=True, method="average").astype(np.float32)
    df["cmc_rank_pack"] = g["cmc"].rank(pct=True, method="average").astype(np.float32)
    df["es_rareza_max_pack"] = (df["rareza"] == g["rareza"].transform("max")).astype(np.float32)

    if draft_ids is not None:
        df["draft_id"] = np.asarray(draft_ids)[pick_row]
    if elegidas is not None:
        df["y"] = (card == np.asarray(elegidas)[pick_row]).astype(np.int8)

    # float32 para ahorrar memoria
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


COLS_NUMERICAS = (
    ["cmc", "rareza", "power", "toughness", "n_colores"]
    + [f"color_{c}" for c in COLORES]
    + [f"tipo_{t}" for t in TIPOS]
    + ["afinidad_color_media", "afinidad_color_min", "en_top2_colores", "copias_en_pool",
       "n_pool_mismo_cmc", "n_pool_mismo_tipo", "pool_size", "pool_cmc_medio"]
    + [f"pool_prop_{c}" for c in COLORES]
    + ["pool_n_con_color"]
    + [f"pool_n_{t}" for t in TIPOS]
    + ["pack_number", "pick_number", "tam_pack", "rareza_rank_pack", "cmc_rank_pack", "es_rareza_max_pack"]
)
COLS_CATEGORICAS = ["carta"]


# ----------------------------------------------------------------------
# Puntaje y recomendacion
# ----------------------------------------------------------------------
def puntaje(modelo, X):
    """Probabilidad de la clase 1 si el modelo la tiene; si no (SVM), funcion de decision."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def resolver_nombres(lineas, nombres_validos, cutoff=0.75):
    """
    Convierte lineas de texto (una carta por linea) en nombres validos del set.
    Tolera mayusculas/minusculas, espacios y errores de tipeo (difflib).
    Devuelve (encontradas, no_reconocidas).
    """
    mapa = {n.lower(): n for n in nombres_validos}
    mapa.update({n.lower().split(" // ")[0]: n for n in nombres_validos})
    encontradas, falla = [], []
    for linea in lineas:
        t = linea.strip()
        if not t:
            continue
        clave = " ".join(t.lower().split())
        if clave in mapa:
            encontradas.append(mapa[clave])
            continue
        cerca = difflib.get_close_matches(clave, list(mapa.keys()), n=1, cutoff=cutoff)
        if cerca:
            encontradas.append(mapa[cerca[0]])
        else:
            falla.append(t)
    vistos, unicas = set(), []
    for c in encontradas:                    # sin duplicados, conservando el orden
        if c not in vistos:
            vistos.add(c)
            unicas.append(c)
    return unicas, falla


def recomendar(modelo, catalogo, pack_nombres, pool_dict, pack_number, pick_number, k=3):
    """
    modelo: Pipeline ya entrenado (preprocesamiento + clasificador).
    pack_nombres: lista de nombres validos del pack actual. pool_dict: {carta: copias} actual.
    Devuelve un DataFrame con las k mejores cartas ordenadas, con su puntaje.
    """
    pack_idx = indices_pack([pack_nombres], catalogo)
    P = matriz_pool([pool_dict], catalogo)
    X = construir_features(pack_idx, P, [pack_number], [pick_number], catalogo)
    s = puntaje(modelo, X)
    out = pd.DataFrame({"carta": X["carta"].to_numpy(), "puntaje": s})
    if out["puntaje"].min() >= 0 and out["puntaje"].max() <= 1:      # probabilidades -> normalizar
        out["prob_relativa"] = out["puntaje"] / max(out["puntaje"].sum(), 1e-12)
    else:                                                            # funcion de decision -> softmax
        e = np.exp(out["puntaje"] - out["puntaje"].max())
        out["prob_relativa"] = e / e.sum()
    return out.sort_values("puntaje", ascending=False).head(k).reset_index(drop=True)
