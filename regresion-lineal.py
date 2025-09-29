import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargar dataset
ds = pd.read_csv("winequality-red.csv", sep=';', quotechar='"')
ds.columns = ds.columns.str.strip()

print("Desviación estándar de graduación alcohólica dentro del dataset:", ds["alcohol"].std())

# Variables
Y = ds["quality"]

predictoras = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide",
            "density", "pH", "sulphates", "alcohol"] 

resultados = []

for col in predictoras:
    X = ds[[col]]
    
    modelo = LinearRegression()
    modelo.fit(X, Y)
    
    resultados.append({
        "Variable": col,
        "Intercepto (β0)": modelo.intercept_,
        "Coeficiente (β1)": modelo.coef_[0],
        "R²": modelo.score(X, Y)
    })

tabla_resultados = pd.DataFrame(resultados)

print("\nResumen regresiones lineales simples:\n")
print(tabla_resultados)