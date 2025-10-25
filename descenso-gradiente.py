import streamlit as st
import pandas as pd
import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iter):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            loss = np.mean((y_pred - y)**2)
            self.loss_history.append(loss)
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

st.set_page_config(
    page_title="Regresión Lineal Descenso Gradiente",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Regresión Múltiple (Descenso del Gradiente)")

# Dataset
ds = pd.read_csv("iris.data", header=None, names=[
    "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
])

# Variables
predictoras = ["sepal_length", "sepal_width", "petal_width"]
Y = ds["petal_length"].values
X = ds[predictoras].values

st.subheader("Variables")
st.write(f"Variable respuesta: petal_length (Y)")
st.write(f"Variables predictoras: {predictoras}")

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std

learning_rate = 0.01
n_iter = 1000
modelo_gd = LinearRegressionGD(learning_rate=learning_rate, n_iter=n_iter)
modelo_gd.fit(X_normalized, Y)
st.write(f"Última pérdida (MSE): {modelo_gd.loss_history[-1]}")

coef_desnormalizados = modelo_gd.weights / X_std
intercepto_desnormalizado = modelo_gd.bias - np.sum(coef_desnormalizados * X_mean)

st.subheader("Resultados del Modelo")
resultados = {"Variable": predictoras, 
              "Coeficiente (β)": list(coef_desnormalizados)}
st.write(pd.DataFrame(resultados))
st.write(f"Intercepto (β₀): {intercepto_desnormalizado}")
st.write("Ecuación de regresión múltiple:")
st.code(f"petal_length = {intercepto_desnormalizado} + "
        f"({coef_desnormalizados[0]} * sepal_length) + "
        f"({coef_desnormalizados[1]} * sepal_width) + "
        f"({coef_desnormalizados[2]} * petal_width)")
