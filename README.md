#  Asistente de Draft para Magic: The Gathering — Reality Fracture (FRA)

Proyecto final de Data Science (UACH). Aplicación web en Streamlit que recomienda **las mejores cartas
de un sobre** durante un Premier Draft, tomando en cuenta las cartas que ya llevas (tu *pool*).

 **App desplegada:** (https://mtgdraftapper-2026.streamlit.app/)
 **Notebook del proyecto:** `Proyecto_Final_Asistente_Draft_MTG.ipynb` (EDA, preprocesamiento, modelos, evaluación)

## ¿Cómo funciona?
1. Pegas los nombres de las cartas del sobre actual (una por línea). Los nombres se reconocen aunque tengan
   errores leves de escritura; las líneas no reconocidas se avisan.
2. Los 3 modelos del notebook puntúan cada carta y la app muestra las **3 mejores** (menos si quedan pocas),
   con el puesto que le dio cada modelo y un **consenso** entre ellos.
3. Marcas la carta que tomaste: el pool se actualiza (colores, curva de maná) y pasas al siguiente pick.
4. Se repite hasta completar el draft (3 sobres × 14 picks = 42; 45 con sobres de 15 cartas).
   Hay botón de **deshacer** y de **reiniciar**, y puedes descargar tu pool en CSV.

## Modelos (entrenados en el notebook; cada archivo es un `Pipeline` de preprocesamiento + modelo)
| Archivo | Modelo | Top-1 Test | Top-3 Test | ROC-AUC Test |
|---|---|---|---|---|
| `modelos/modelo_1.joblib` | SVM lineal (optimizada) | 0.569 | 0.889 | 0.878 |
| `modelos/modelo_2.joblib` | Regresión Logística (optimizada) | 0.571 | 0.890 | 0.878 |
| `modelos/modelo_3.joblib` | XGBoost (optimizado) | 0.571 | 0.897 | 0.888 |

Línea base de popularidad: Top-1 0.453, Top-3 0.805; azar: 0.175 / 0.483.
Los modelos imitan las elecciones de jugadores con win rate ≥ 0.58 (datos públicos de 17Lands + metadatos de Scryfall);
**son una ayuda a la decisión, no garantizan la mejor carta.**

## Estructura
```
app.py              # interfaz de Streamlit
draft_utils.py      # variables (features) y funciones compartidas con el notebook
modelos/            # modelo_1/2/3.joblib, catalogo.joblib, metadatos.joblib
requirements.txt
```

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
> Los modelos se guardaron con **scikit-learn 1.9.0**; usa esa versión para evitar incompatibilidades.

## Datos y referencias
- 17Lands, *Public Datasets* (draft data, FRA Premier Draft): https://www.17lands.com/public_datasets
- Scryfall API: https://scryfall.com/docs/api
- Bertram, Fürnkranz & Müller (2024), *Predicting Human Picks in Magic: The Gathering Drafts*, arXiv:2407.05879
