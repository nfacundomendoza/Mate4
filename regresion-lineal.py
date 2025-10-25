import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import statsmodels.api as sm
import os

st.set_page_config(
    page_title="Regresión Lineal - Iris Dataset",
    page_icon="🌸",
    layout="wide"
)

# Dataset
ds = pd.read_csv("iris.data", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

# Variables
Y = ds["petal_length"]
predictoras = ["sepal_length", "sepal_width", "petal_width"]

st.markdown("<h1 style='text-align:center;'>🌸 Análisis de Regresión Lineal - Iris Dataset</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Relación entre las variables morfológicas del iris mediante regresión lineal simple.</p>", unsafe_allow_html=True)

st.sidebar.header("Panel de Control")
col_seleccionada = st.sidebar.selectbox("Variable predictora", predictoras)

st.sidebar.write("---")
st.sidebar.subheader("Información")
st.sidebar.write(f"**Total de observaciones:** {len(ds)}")
st.sidebar.write(f"**Variable respuesta:** `{Y.name}`")
st.sidebar.write(f"**Variable predictora:** `{col_seleccionada}`")

# Grafico
fig = px.scatter(
    ds,
    x=col_seleccionada,
    y=Y.name,
    trendline="ols",
    labels={col_seleccionada: col_seleccionada, Y.name: Y.name},
    title=f"Relación entre {Y.name} y {col_seleccionada}",
    template="plotly_white"
)

fig.update_layout(
    height=600,
    showlegend=True,
    font=dict(size=12, color="#333"),
    title_font=dict(size=18, color="#FFFFFF"),
)

st.plotly_chart(fig, use_container_width=True)

# resultados
st.markdown("### Resultados")

resultados = []

for col in predictoras:
    X = ds[[col]]
    X_const = sm.add_constant(X)
    modelo_sm = sm.OLS(Y, X_const).fit()
    sigma2 = modelo_sm.mse_resid
    r = ds[col].corr(Y)
    conf = modelo_sm.conf_int() 
    B0_ci = conf.loc['const'].tolist() 
    B1_ci = conf.loc[col].tolist()     

    resultados.append({
        "Variable": col,
        "σ²": sigma2,
        "R²": modelo_sm.rsquared,
        "r": r,
        "Coeficiente (β₁)": modelo_sm.params[col],
        "Intercepto (β₀)": modelo_sm.params['const'],
        "IC(β1)": B1_ci,
        "IC(β₀)": B0_ci
    })

tabla_resultados = pd.DataFrame(resultados)

st.dataframe(tabla_resultados, use_container_width=True)

# Análisis variable predictora
mejor_variable = tabla_resultados.loc[tabla_resultados['R²'].idxmax()]
st.markdown(f"**La variable predictora con mejor ajuste es `{mejor_variable['Variable']}` con un R² de {mejor_variable['R²']}.**")

