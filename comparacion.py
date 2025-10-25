import streamlit as st
import pandas as pd
import numpy as np  
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Regresión Simple vs Múltiple",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Comparación de Modelos de Regresión")

# Dataset
ds = pd.read_csv("iris.data", header=None, names=[
    "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
])

# Variables
y = ds["petal_length"]
X_simple = ds[["petal_width"]]
X_multiple = ds[["sepal_length", "sepal_width", "petal_width"]]

# Ajuste de modelos
modelo_simple = LinearRegression()
modelo_simple.fit(X_simple, y)
r2_simple = modelo_simple.score(X_simple, y)

modelo_multiple = LinearRegression()
modelo_multiple.fit(X_multiple, y)
r2_multiple = modelo_multiple.score(X_multiple, y)

delta_r2 = r2_multiple - r2_simple
mejora_relativa = (delta_r2 / r2_simple) * 100

# Tabla con solo números
tabla = pd.DataFrame({
    "Métrica": ["R²", "Mejora absoluta ΔR²", "Mejora relativa (%)"],
    "Regresión Simple": [r2_simple, np.nan, np.nan],
    "Regresión Múltiple": [r2_multiple, delta_r2, mejora_relativa]
})

st.subheader("Comparación de Regresión Simple vs Múltiple")
st.table(tabla)

# Interpretación
st.subheader("Interpretación")
st.write(f"""
Con la variable predictora `petal_width` solo explica el {r2_simple*100:.1f}% de la variabilidad. Agregando `sepal_length` y `sepal_width`, explicamos el {r2_multiple*100:.1f}%.  
La mejora de {delta_r2*100:.1f}% puntos es significativa.
""")
