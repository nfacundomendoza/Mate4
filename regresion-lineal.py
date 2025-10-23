import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import statsmodels.api as sm
import os

# Dataset
ds = pd.read_csv("iris.data", header=None, names=[
    "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
])

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)  

# Variable respuesta
Y = ds["petal_length"]

# Variables predictoras
predictoras = ["sepal_length", "sepal_width", "petal_width"]

st.title("Análisis de Regresión Lineal - Iris Dataset")
col_seleccionada = st.selectbox("Seleccionar variable predictora para el gráfico", predictoras)

# Gráfico de dispersión
fig = px.scatter(
    ds,
    x=col_seleccionada,
    y=Y.name,
    trendline="ols",
    labels={col_seleccionada: col_seleccionada, Y.name: Y.name},
    title=f"Dispersión de {Y.name} vs {col_seleccionada} con línea de regresión"
)

st.plotly_chart(fig)

st.title("Resultados de Regresión Lineal Simple")

resultados = []

for col in predictoras:
    X = ds[[col]]
    X_const = sm.add_constant(X)
    modelo_sm = sm.OLS(Y, X_const).fit()
    
    sigma2 = modelo_sm.mse_resid
    x_mean = np.mean(X[col])
    
    resultados.append({
        "Variable": col,
        "Intercepto (beta0)": round(modelo_sm.params['const'], 4),
        "Coeficiente (beta1)": round(modelo_sm.params[col], 4),
        "R²": round(modelo_sm.rsquared, 4),
        "sigma2": round(sigma2, 4)
    })

tabla_resultados = pd.DataFrame(resultados)

st.subheader("Tabla de Resultados Completos")
st.dataframe(tabla_resultados)

mejor_variable = tabla_resultados.loc[tabla_resultados['R²'].idxmax()]
st.subheader("Mejor Variable Predictora")
st.write(f"**Variable:** {mejor_variable['Variable']}")
st.write(f"**R²:** {mejor_variable['R²']}")
st.write(f"**sigma²:** {mejor_variable['sigma2']}")
st.write(f"**Coeficiente (β1):** {mejor_variable['Coeficiente (beta1)']}")

st.subheader("Análisis Comparativo")
st.write("**R² por variable:**")
for idx, row in tabla_resultados.iterrows():
    st.write(f"- {row['Variable']}: {row['R²']}")
