import streamlit as st
import pandas as pd
import plotly.express as px

ds = pd.read_csv("winequality-red.csv", sep=';')
predictoras = ["fixed acidity", "volatile acidity", "citric acid", 
               "chlorides", "free sulfur dioxide", "total sulfur dioxide",
               "density", "pH", "sulphates", "alcohol", "quality"]

st.title("Gráfico de dispersión interactivo")

col_seleccionada = st.selectbox("Seleccionar variable predictora", predictoras)
fig = px.scatter(ds, x=col_seleccionada, y="residual sugar", trendline="ols")
st.plotly_chart(fig)
