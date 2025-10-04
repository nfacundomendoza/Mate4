import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)  

# Cargar dataset desde Excel
ds = pd.read_excel("ENB2012_data.xlsx")

# Renombrar columnas para mayor claridad
ds.rename(columns={
    "X1": "Relative Compactness",
    "X2": "Surface Area",
    "X3": "Wall Area",
    "X4": "Roof Area",
    "X5": "Overall Height",
    "X6": "Orientation",
    "X7": "Glazing Area",
    "X8": "Glazing Area Distribution",
    "Y1": "Heating Load"
}, inplace=True)

# Variable respuesta
Y = ds["Heating Load"]

# Variables predictoras
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

resultados = []

for col in predictoras:
    X = ds[[col]]

    # Modelo con sklearn
    modelo = LinearRegression()
    modelo.fit(X, Y)
    
    # Modelo con statsmodels para IC
    X_const = sm.add_constant(X)
    modelo_sm = sm.OLS(Y, X_const).fit()
    
    pred = modelo_sm.get_prediction(X_const)
    summary_frame = pred.summary_frame(alpha=0.05)

    sigma2 = np.sum((Y - modelo.predict(X))**2) / (len(Y) - 2)
    
    # Agregar resultados para cada predictor
    resultados.append({
        "Variable": col,
        "Intercepto (beta0)": round(modelo.intercept_, 4),
        "Coeficiente (beta1)": round(modelo.coef_[0], 4),
        "R²": round(modelo.score(X, Y), 4),
        "sigma2": round(sigma2, 4),
        "IC beta0_inferior": round(modelo_sm.conf_int().loc['const'][0], 4),
        "IC beta0_superior": round(modelo_sm.conf_int().loc['const'][1], 4),
        "IC beta1_inferior": round(modelo_sm.conf_int().loc[col][0], 4),
        "IC beta1_superior": round(modelo_sm.conf_int().loc[col][1], 4),
        "IC media inferior": round(summary_frame['mean_ci_lower'].iloc[0], 4),
        "IC media superior": round(summary_frame['mean_ci_upper'].iloc[0], 4),
        "IC predicción inferior": round(summary_frame['obs_ci_lower'].iloc[0], 4),
        "IC predicción superior": round(summary_frame['obs_ci_upper'].iloc[0], 4)
    })

# Crear DataFrame final
tabla_resultados = pd.DataFrame(resultados)

# Guardar resultados en archivo UTF-8 para evitar errores de codificación
tabla_resultados.to_csv(r"C:\Mate4\Mate4\resultados.txt", index=False, encoding="utf-8")


print("✅ Resultados guardados en 'resultados.txt'")
