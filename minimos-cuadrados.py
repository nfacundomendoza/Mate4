import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Configuración de la página
st.set_page_config(
    page_title="Regresión Múltiple",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Regresión Múltiple (Mínimos Cuadrados)")

# Dataset
ds = pd.read_csv("iris.data", header=None, names=[
    "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
])
st.subheader("Dataset Iris")

# Variables
predictoras = ["sepal_length", "sepal_width", "petal_width"]
Y = ds["petal_length"]
X = ds[predictoras]

st.subheader("Variables")
st.write(f"Variable respuesta: petal_length (Y)")
st.write(f"Variables predictoras: {predictoras}")

# Modelo
X_const = sm.add_constant(X)
modelo_sm = sm.OLS(Y, X_const).fit()

# Resultados
st.subheader("Resultados de la regresión múltiple")
resultados = {"Variable": ["const"] + predictoras,
              "Coeficiente (β)": list(modelo_sm.params),
              "Error estándar": list(modelo_sm.bse),
              "Valor t": list(modelo_sm.tvalues),
}
tabla_resultados = pd.DataFrame(resultados)
st.write(tabla_resultados)        

# Mostrar ecuación de regresión
st.subheader("Ecuación de regresión múltiple")
ecuacion = f"""
petal_length = {modelo_sm.params['const']} + ({modelo_sm.params['sepal_length']} × sepal_length) + ({modelo_sm.params['sepal_width']} × sepal_width) + ({modelo_sm.params['petal_width']} × petal_width)"""
st.code(ecuacion)

# Métricas del modelo
st.subheader("Métricas del modelo")
st.write(f"R²: {modelo_sm.rsquared}")
st.write(f"R² ajustado: {modelo_sm.rsquared_adj}")
st.write(f"Error estándar: {np.sqrt(modelo_sm.mse_resid)}")

