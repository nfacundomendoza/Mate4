import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm

# Cargar dataset Iris
ds = pd.read_csv("iris.data", header=None, names=[
    "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
])

# Lista de variables predictoras
predictoras = ["sepal_length", "sepal_width", "petal_width"]

st.title("Análisis de Regresión Lineal Simple")

# Sidebar
st.sidebar.header("Configuración del Análisis")

# Selección de variable predictora
col_seleccionada = st.sidebar.selectbox(
    "Seleccionar variable predictora", 
    predictoras,
    help="Elige la variable independiente para el análisis de regresión"
)

# Variable respuesta
y_variable = "petal_length"

# información del dataset
st.sidebar.subheader("Información del Dataset")
st.sidebar.write(f"**Total de observaciones:** {len(ds)}")
st.sidebar.write(f"**Variable respuesta:** {y_variable}")
st.sidebar.write(f"**Variable predictora:** {col_seleccionada}")

# Estadísticas descriptivas
stats = ds[[col_seleccionada, y_variable]].describe()
st.sidebar.write("**Estadísticas descriptivas:**")
st.sidebar.write(f"Media {col_seleccionada}: {stats[col_seleccionada]['mean']:.2f}")
st.sidebar.write(f"Media {y_variable}: {stats[y_variable]['mean']:.2f}")

# pestañas
tab1, tab2, tab3 = st.tabs(["📈 Gráfico de Dispersión", "📋 Resultados Estadísticos", "🔍 Análisis de Residuos"])

with tab1:
    # Crear gráfico de dispersión con línea de regresión
    fig = px.scatter(
        ds,
        x=col_seleccionada,
        y=y_variable,
        color="class",
        trendline="ols",
        labels={
            col_seleccionada: f"{col_seleccionada} (cm)",
            y_variable: f"{y_variable} (cm)",
            "class": "Especie"
        },
        title=f"Relación entre {y_variable} y {col_seleccionada}",
        template="plotly_white"
    )
    
    # Personalizar el gráfico
    fig.update_layout(
        height=600,
        showlegend=True,
        font=dict(size=12)
    )
    
    # Añadir anotaciones con información de correlación
    correlacion = ds[col_seleccionada].corr(ds[y_variable])
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"Correlación: {correlacion:.3f}",
        showarrow=False,
        bgcolor="black",
        font=dict(color="white"),
        bordercolor="black",
        borderwidth=1
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretación del gráfico
    st.subheader("📝 Interpretación del Gráfico")
    if correlacion > 0.7:
        st.success("✅ **Fuerte correlación positiva**: Los puntos siguen claramente una tendencia lineal ascendente.")
    elif correlacion > 0.3:
        st.info("📊 **Correlación moderada**: Existe una relación lineal perceptible pero no muy fuerte.")
    elif correlacion > -0.3:
        st.warning("⚠️ **Correlación débil**: La relación lineal es poco evidente.")
    else:
        st.error("📉 **Correlación negativa**: Los puntos muestran una tendencia descendente.")

with tab2:
    st.subheader("📊 Resultados del Modelo de Regresión")
    
    # Calcular regresión lineal
    X = sm.add_constant(ds[col_seleccionada])
    model = sm.OLS(ds[y_variable], X).fit()
    
    # Mostrar resultados en columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R²", f"{model.rsquared:.4f}")
        st.metric("Intercepto (β₀)", f"{model.params['const']:.4f}")
    
    with col2:
        st.metric("Coeficiente (β₁)", f"{model.params[col_seleccionada]:.4f}")
        st.metric("Error estándar", f"{model.bse[col_seleccionada]:.4f}")
    
    with col3:
        st.metric("Valor-p", f"{model.pvalues[col_seleccionada]:.4f}")
        st.metric("Observaciones", len(ds))
    
    # Ecuación de regresión
    st.subheader("🧮 Ecuación de Regresión")
    st.latex(f"\\hat{{y}} = {model.params['const']:.4f} + {model.params[col_seleccionada]:.4f} \\cdot x")
    
    # Interpretación de coeficientes
    st.subheader("🔍 Interpretación de Coeficientes")
    st.write(f"**Intercepto (β₀ = {model.params['const']:.4f})**: "
             f"Cuando {col_seleccionada} es 0, el valor predicho de {y_variable} es {model.params['const']:.4f} cm")
    
    st.write(f"**Pendiente (β₁ = {model.params[col_seleccionada]:.4f})**: "
             f"Por cada aumento de 1 cm en {col_seleccionada}, {y_variable} aumenta en {model.params[col_seleccionada]:.4f} cm")

with tab3:
    st.subheader("🔍 Análisis de Residuos")
    
    # Calcular residuos
    predictions = model.predict(X)
    residuals = ds[y_variable] - predictions
    
    # Gráfico de residuos vs valores ajustados
    fig_residuals = px.scatter(
        x=predictions,
        y=residuals,
        labels={'x': 'Valores Ajustados', 'y': 'Residuos'},
        title='Residuos vs Valores Ajustados',
        template="plotly_white"
    )
    
    # Añadir línea en y=0
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig_residuals.update_layout(height=400)
    st.plotly_chart(fig_residuals, use_container_width=True)
    
    # Interpretación de residuos
    st.subheader("📝 Interpretación de Residuos")
    if abs(residuals.mean()) < 0.1:
        st.success("✅ **Residuos centrados**: La media de los residuos es cercana a cero, indicando buen ajuste.")
    else:
        st.warning("⚠️ **Residuos no centrados**: Podría indicar problemas en el modelo.")
    
    # Histograma de residuos
    fig_hist = px.histogram(
        x=residuals,
        nbins=20,
        labels={'x': 'Residuos'},
        title='Distribución de Residuos',
        template="plotly_white"
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

#st.sidebar.markdown("---")

# Mostrar datos brutos
if st.sidebar.checkbox("Mostrar datos brutos"):
    st.subheader("📄 Datos del Dataset Iris")
    st.dataframe(ds, use_container_width=True)


# Información adicional
#with st.sidebar.expander("💡 Acerca del Análisis"):
#    st.write("""
#    Este análisis muestra la relación lineal entre el largo del pétalo 
#    y otras variables morfológicas de las flores Iris. 
#    Use las pestañas para explorar diferentes aspectos del modelo.
#    """)
