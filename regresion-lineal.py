import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)  

ds = pd.read_csv("winequality-red.csv", sep=';', quotechar='"')
ds.columns = ds.columns.str.strip()

Y = ds["quality"]

predictoras = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
               "chlorides", "free sulfur dioxide", "total sulfur dioxide",
               "density", "pH", "sulphates", "alcohol"] 

resultados = []

for col in predictoras:
    X = ds[[col]]

    modelo = LinearRegression()
    modelo.fit(X, Y)
    
    X_const = sm.add_constant(X)
    modelo_sm = sm.OLS(Y, X_const).fit()
    
    pred = modelo_sm.get_prediction(X_const)
    summary_frame = pred.summary_frame(alpha=0.05)

    sigma2 = np.sum((Y - modelo.predict(X))**2) / (len(Y) - 2)
    resultados.append({
        "Variable": col,
        "Intercepto (β0)": round(modelo.intercept_, 4),
        "Coeficiente (β1)": round(modelo.coef_[0], 4),
        "R²": round(modelo.score(X, Y), 4),
        "σ²": round(sigma2, 4),
        "IC β0_inferior": round(modelo_sm.conf_int().loc['const'][0], 4),
        "IC β0_superior": round(modelo_sm.conf_int().loc['const'][1], 4),
        "IC β1_inferior": round(modelo_sm.conf_int().loc[col][0], 4),
        "IC β1_superior": round(modelo_sm.conf_int().loc[col][1], 4),
        "IC media inferior": round(summary_frame['mean_ci_lower'].iloc[0], 4),
        "IC media superior": round(summary_frame['mean_ci_upper'].iloc[0], 4),
        "IC predicción inferior": round(summary_frame['obs_ci_lower'].iloc[0], 4),
        "IC predicción superior": round(summary_frame['obs_ci_upper'].iloc[0], 4)
    })


tabla_resultados = pd.DataFrame(resultados)

print("\nResumen regresiones lineales simples probando todos los parámetros:\n")
print(tabla_resultados)
