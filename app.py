"""
Asistente de Draft para Magic: The Gathering (set FRA - Reality Fracture).

El usuario pega la lista de cartas del pack actual (una por linea); la app recomienda las
mejores 3 cartas con 3 modelos entrenados en el notebook (SVM lineal, Regresion Logistica y
XGBoost), el usuario marca la carta que tomo, el pool se acumula y se repite hasta completar
el draft (3 sobres x 14 picks = 42 picks; 45 si los sobres traen 15 cartas).

Autor: Joseph Daniel Ramirez Torres (matrícula 374352) — Data Science, UACH.

Ejecutar localmente:   streamlit run app.py
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import draft_utils as du   # mismas funciones de variables que se usaron al entrenar

warnings.filterwarnings("ignore")
RUTA = Path(__file__).parent / "modelos"
COLORES_NOMBRE = {"W": "Blanco", "U": "Azul", "B": "Negro", "R": "Rojo", "G": "Verde"}

st.set_page_config(page_title="Asistente de Draft MTG", page_icon="🃏", layout="wide")


@st.cache_resource(show_spinner="Cargando modelos...")
def cargar():
    catalogo = joblib.load(RUTA / "catalogo.joblib")
    meta = joblib.load(RUTA / "metadatos.joblib")
    modelos = [joblib.load(RUTA / f"modelo_{i}.joblib") for i in (1, 2, 3)]
    return catalogo, meta, modelos


catalogo, meta, modelos = cargar()
NOMBRES_MOD = meta["nombres_modelos"]
CORTO = [n.split(" (")[0] for n in NOMBRES_MOD]          # "SVM lineal", "Regresión Logística", "XGBoost"
nombres_validos = list(catalogo["nombres"])
idx_carta = catalogo["indice"]



def reiniciar():
    st.session_state.pool = {}            # {carta: copias}
    st.session_state.historial = []       # lista de dicts: pick, pack, elegida
    st.session_state.recs = None          # recomendaciones del pack actual
    st.session_state.pack_actual = None   # cartas del pack actual (ya resueltas)
    st.session_state.msg_no_rec = []


if "pool" not in st.session_state:
    reiniciar()

with st.sidebar:
    st.header("⚙️ Configuración")
    cartas_por_sobre = st.radio("Cartas por sobre", [14, 15], index=0, horizontal=True,
                                help="Premier Draft de este set: 14 picks por sobre (42 en total). "
                                     "Si tu sobre trae 15 cartas, el draft termina en 45 picks.")
    k = st.slider("Cartas a recomendar", 1, 5, 3)
    st.divider()
    if st.button("🔄 Reiniciar draft", width="stretch"):
        reiniciar()
        st.rerun()
    st.divider()
    st.caption("**Desempeño de los modelos en Test** (509 drafts nunca vistos)")
    tabla = pd.DataFrame(meta["metricas_test"]).loc[NOMBRES_MOD, ["Top-1", "Top-3", "ROC-AUC"]]
    tabla.index = CORTO
    st.dataframe(tabla.style.format("{:.3f}"), width="stretch")
    st.caption("Top-k: % de picks en que la carta elegida por jugadores fuertes quedó entre las k "
               "primeras recomendaciones.")
    st.divider()
    st.caption("**Autor:** Joseph Daniel Ramirez Torres  \n**Matrícula:** 374352  \n"
               "Proyecto Final — Data Science · UACH")

TOTAL = 3 * cartas_por_sobre
n_picks = len(st.session_state.historial)
terminado = n_picks >= TOTAL
sobre = min(n_picks // cartas_por_sobre, 2)          # 0, 1, 2 (formato del entrenamiento)
pick_en_sobre = n_picks % cartas_por_sobre           # 0..13


st.title("🃏 Asistente de Draft — Reality Fracture (FRA)")
st.caption("Pega las cartas de tu pack y recibe las mejores opciones según tu pool actual. "
           "Los modelos imitan lo que eligen jugadores con win rate ≥ 58 %; "
           "es una ayuda a la decisión, no una verdad absoluta.")
st.progress(min(n_picks / TOTAL, 1.0),
            text=f"Pick {min(n_picks + 1, TOTAL)} de {TOTAL}  ·  "
                 f"Sobre {sobre + 1}, pick {pick_en_sobre + 1}" if not terminado
            else f"Draft completo: {TOTAL} picks")

col_izq, col_der = st.columns([3, 2], gap="large")


with col_izq:
    if terminado:
        st.success("🎉 ¡Draft completo! Revisa tu pool a la derecha y descárgalo para armar tu mazo.")
    else:
        st.subheader(f"1) Pega el pack (sobre {sobre + 1}, pick {pick_en_sobre + 1})")
        texto = st.text_area("Una carta por línea", height=220, key=f"pack_{n_picks}",
                             placeholder="Simulacrum Shaper\nKarn, Gilded Guardian\nYoshimaru, Beloved Companion\n...")
        if st.button("🔎 Recomendar", type="primary"):
            lineas = [l for l in texto.splitlines() if l.strip()]
            pack, no_rec = du.resolver_nombres(lineas, nombres_validos)
            st.session_state.msg_no_rec = no_rec
            if not pack:
                st.session_state.recs = None
                st.session_state.pack_actual = None
                st.error("No reconocí ninguna carta. Revisa los nombres (uno por línea).")
            else:
                st.session_state.pack_actual = pack
                st.session_state.recs = [
                    du.recomendar(m, catalogo, pack, st.session_state.pool, sobre, pick_en_sobre, k=len(pack))
                    for m in modelos
                ]

        if st.session_state.msg_no_rec:
            st.warning("No se reconocieron estas líneas (se ignoraron): "
                       + "; ".join(st.session_state.msg_no_rec))

        pack = st.session_state.pack_actual
        recs = st.session_state.recs
        if pack and recs:
            st.subheader("2) Recomendaciones")
            n_mostrar = min(k, len(pack))
            if len(pack) < n_mostrar or len(pack) <= 3:
                st.info(f"Solo hay {len(pack)} carta(s) en el pack.")
            # consenso: promedio de las probabilidades relativas de los 3 modelos
            filas = []
            for c in pack:
                fila = {"Carta": c}
                prob = []
                for nombre, r in zip(CORTO, recs):
                    p = float(r.loc[r["carta"] == c, "prob_relativa"].iloc[0])
                    puesto = int(r.index[r["carta"] == c][0]) + 1
                    fila[f"{nombre} (puesto)"] = puesto
                    prob.append(p)
                fila["Consenso (%)"] = 100 * float(np.mean(prob))
                filas.append(fila)
            df = pd.DataFrame(filas).sort_values("Consenso (%)", ascending=False).reset_index(drop=True)
            df.index = df.index + 1

            top = df.head(n_mostrar)
            cols = st.columns(n_mostrar)
            for i, (col, (_, f)) in enumerate(zip(cols, top.iterrows())):
                with col:
                    st.metric(f"#{i + 1}", f["Carta"], f"{f['Consenso (%)']:.0f} % relativo")
                    en_cat = catalogo["indice"].get(f["Carta"])
                    if en_cat is not None:
                        cmc = int(catalogo["cmc"][en_cat])
                        cols_c = [COLORES_NOMBRE[c] for j, c in enumerate(du.COLORES)
                                  if catalogo["color_mat"][en_cat][j] > 0]
                        st.caption(f"Costo {cmc} · {'/'.join(cols_c) if cols_c else 'Incoloro'}")

            with st.expander("Ver el pack completo con el puesto de cada modelo"):
                st.dataframe(df.style.format({"Consenso (%)": "{:.1f}"}), width="stretch")
                st.caption("«Consenso» promedia la probabilidad relativa (dentro del pack) de los 3 "
                           "modelos. Si los 3 coinciden en el primer puesto, la recomendación es más sólida.")

            st.subheader("3) ¿Qué carta tomaste?")
            elegida = st.selectbox("Carta tomada", df["Carta"].tolist(), label_visibility="collapsed")
            if st.button("✅ Confirmar pick"):
                st.session_state.historial.append(
                    {"pick": n_picks + 1, "sobre": sobre + 1, "pick_sobre": pick_en_sobre + 1,
                     "carta": elegida, "pack": list(pack),
                     "recomendada": df.iloc[0]["Carta"]})
                st.session_state.pool[elegida] = st.session_state.pool.get(elegida, 0) + 1
                st.session_state.recs = None
                st.session_state.pack_actual = None
                st.session_state.msg_no_rec = []
                st.rerun()

    # Deshacer
    if st.session_state.historial:
        if st.button("↩️ Deshacer último pick"):
            ultimo = st.session_state.historial.pop()
            c = ultimo["carta"]
            st.session_state.pool[c] -= 1
            if st.session_state.pool[c] <= 0:
                del st.session_state.pool[c]
            st.session_state.recs = None
            st.session_state.pack_actual = None
            st.rerun()

# ----------------------------------------------------------------------------------------
# Columna derecha: resumen del pool
# ----------------------------------------------------------------------------------------
with col_der:
    st.subheader(f"Tu pool ({n_picks} cartas)")
    if not st.session_state.pool:
        st.write("Aún no has tomado ninguna carta.")
    else:
        P = du.matriz_pool([st.session_state.pool], catalogo)[0]
        idx = np.flatnonzero(P > 0)
        info = pd.DataFrame({
            "Carta": [catalogo["nombres"][i] for i in idx],
            "Copias": [int(P[i]) for i in idx],
            "Costo": [int(catalogo["cmc"][i]) for i in idx],
            "Colores": ["".join(c for j, c in enumerate(du.COLORES) if catalogo["color_mat"][i][j] > 0) or "—"
                        for i in idx],
        }).sort_values(["Costo", "Carta"]).reset_index(drop=True)

        # Distribucion de colores (cada carta cuenta en cada uno de sus colores)
        conteo = {COLORES_NOMBRE[c]: int(sum(P[i] for i in idx if catalogo["color_mat"][i][j] > 0))
                  for j, c in enumerate(du.COLORES)}
        st.caption("Cartas por color (las multicolor cuentan en cada color)")
        st.bar_chart(pd.Series(conteo, name="Cartas"))
        st.caption("Curva de maná (costo de maná convertido)")
        curva = info.groupby("Costo")["Copias"].sum().reindex(range(0, 8), fill_value=0)
        st.bar_chart(curva)
        st.dataframe(info, width="stretch", hide_index=True, height=300)
        st.download_button("⬇️ Descargar pool (CSV)", info.to_csv(index=False).encode("utf-8"),
                           file_name="mi_pool_draft.csv", mime="text/csv")

    if st.session_state.historial:
        aciertos = sum(h["carta"] == h["recomendada"] for h in st.session_state.historial)
        st.caption(f"Coincidencias con la 1.ª recomendación del consenso: "
                   f"{aciertos}/{len(st.session_state.historial)}")
        with st.expander("Historial de picks"):
            st.dataframe(pd.DataFrame(st.session_state.historial)[
                ["pick", "sobre", "carta", "recomendada"]], hide_index=True, width="stretch")

st.divider()
st.caption("Proyecto Final de Data Science · Joseph Daniel Ramirez Torres (374352) · UACH · "
           "Datos: 17Lands y Scryfall.")
