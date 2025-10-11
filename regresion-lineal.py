import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Cargar dataset Iris
ds = pd.read_csv("iris.data", header=None, names=[
    "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
])

# Configuración de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)  

# Variable respuesta
Y = ds["petal_length"]

# Variables predictoras
predictoras = ["sepal_length", "sepal_width", "petal_width"]

# PARTE 1: GRÁFICOS INTERACTIVOS
st.title("Análisis de Regresión Lineal - Iris Dataset")

# Selección de variable predictora para el gráfico
col_seleccionada = st.selectbox("Seleccionar variable predictora para el gráfico", predictoras)

# Crear gráfico de dispersión con línea de regresión
fig = px.scatter(
    ds,
    x=col_seleccionada,
    y=Y.name,
    trendline="ols",
    labels={col_seleccionada: col_seleccionada, Y.name: Y.name},
    title=f"Dispersión de {Y.name} vs {col_seleccionada} con línea de regresión"
)

st.plotly_chart(fig)

# PARTE 2: CÁLCULO DE RESULTADOS CORREGIDO
st.title("Resultados de Regresión Lineal Simple")

resultados = []

for col in predictoras:
    X = ds[[col]]
    
    # Modelo con statsmodels para IC más precisos
    X_const = sm.add_constant(X)
    modelo_sm = sm.OLS(Y, X_const).fit()
    
    # Calcular sigma^2 (varianza residual)
    sigma2 = modelo_sm.mse_resid
    
    # Para IC de media y predicción, usar un valor representativo (media de X)
    x_mean = np.mean(X[col])
    
    # Crear matriz para predicción en la media CORREGIDA
    X_mean_for_pred = pd.DataFrame({
        'const': [1],
        col: [x_mean]
    })
    
    # Obtener predicción en la media
    pred_mean = modelo_sm.get_prediction(X_mean_for_pred)
    summary_frame_mean = pred_mean.summary_frame(alpha=0.05)
    
    # Agregar resultados CORREGIDOS
    resultados.append({
        "Variable": col,
        "Intercepto (beta0)": round(modelo_sm.params['const'], 4),
        "Coeficiente (beta1)": round(modelo_sm.params[col], 4),
        "R²": round(modelo_sm.rsquared, 4),
        "sigma2": round(sigma2, 4),
        "IC beta0_inferior": round(modelo_sm.conf_int().loc['const'][0], 4),
        "IC beta0_superior": round(modelo_sm.conf_int().loc['const'][1], 4),
        "IC beta1_inferior": round(modelo_sm.conf_int().loc[col][0], 4),
        "IC beta1_superior": round(modelo_sm.conf_int().loc[col][1], 4),
        "IC media inferior": round(summary_frame_mean['mean_ci_lower'].iloc[0], 4),
        "IC media superior": round(summary_frame_mean['mean_ci_upper'].iloc[0], 4),
        "IC predicción inferior": round(summary_frame_mean['obs_ci_lower'].iloc[0], 4),
        "IC predicción superior": round(summary_frame_mean['obs_ci_upper'].iloc[0], 4)
    })

# Crear DataFrame final
tabla_resultados = pd.DataFrame(resultados)

# Guardar resultados en archivo UTF-8 para evitar errores de codificación
tabla_resultados.to_csv(r"resultados.txt", index=False, encoding="utf-8")
# Mostrar resultados en Streamlit
st.subheader("Tabla de Resultados Completos")
st.dataframe(tabla_resultados)

# Identificar la mejor variable predictora
mejor_variable = tabla_resultados.loc[tabla_resultados['R²'].idxmax()]
st.subheader("🏆 Mejor Variable Predictora")
st.write(f"**Variable:** {mejor_variable['Variable']}")
st.write(f"**R²:** {mejor_variable['R²']}")
st.write(f"**sigma²:** {mejor_variable['sigma2']}")
st.write(f"**Coeficiente (β1):** {mejor_variable['Coeficiente (beta1)']}")

# Análisis comparativo
st.subheader("📊 Análisis Comparativo")
st.write("**R² por variable:**")
for idx, row in tabla_resultados.iterrows():
    st.write(f"- {row['Variable']}: {row['R²']}")

# Guardar resultados en archivo
tabla_resultados.to_csv("resultados_iris_corregidos.csv", index=False, encoding="utf-8")
st.success("✅ Resultados guardados en 'resultados_iris_corregidos.csv'")

# Mostrar resumen estadístico
st.subheader("Resumen Estadístico del Dataset")
st.dataframe(ds.describe())