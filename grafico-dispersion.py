import streamlit as st
import pandas as pd
import plotly.express as px

# Cargar dataset de eficiencia energética desde el Excel
ds = pd.read_excel("ENB2012_data.xlsx")

# Renombrar columnas para que sean descriptivas
ds.rename(columns={
    "X1": "Relative Compactness",
    "X2": "Surface Area",
    "X3": "Wall Area",
    "X4": "Roof Area",
    "X5": "Overall Height",
    "X6": "Orientation",
    "X7": "Glazing Area",
    "X8": "Glazing Area Distribution",
    "Y1": "Heating Load",
    "Y2": "Cooling Load"
}, inplace=True)

# Lista de variables predictoras
predictoras = [
    "Relative Compactness",
    "Surface Area",
    "Wall Area",
    "Roof Area",
    "Overall Height",
    "Orientation",
    "Glazing Area",
    "Glazing Area Distribution"
]

st.title("Gráfico de dispersión interactivo - Eficiencia Energética")

# Selección de variable predictora
col_seleccionada = st.selectbox("Seleccionar variable predictora", predictoras)

# Variable respuesta
y_variable = "Heating Load"

# Crear gráfico de dispersión con línea de regresión
fig = px.scatter(
    ds,
    x=col_seleccionada,
    y=y_variable,
    trendline="ols",
    labels={col_seleccionada: col_seleccionada, y_variable: y_variable},
    title=f"Dispersión de {y_variable} vs {col_seleccionada}"
)

st.plotly_chart(fig)
