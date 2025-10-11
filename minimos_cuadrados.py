import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

print("📁 Cargando dataset Iris...")
ds = pd.read_csv("iris.data", header=None, names=[
    "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
])

# Preparar datos
X = ds[["sepal_length", "sepal_width", "petal_width"]]
y = ds["petal_length"]

print("="*60)
print("🎯 REGRESIÓN MÚLTIPLE - MÍNIMOS CUADRADOS")
print("="*60)

# MÉTODO 1: sklearn (más simple)
print("\n1. USANDO SCIKIT-LEARN:")
modelo_mc = LinearRegression()
modelo_mc.fit(X, y)

print(f"ECUACIÓN:")
print(f"petal_length = {modelo_mc.intercept_:.4f} + ")
print(f"               ({modelo_mc.coef_[0]:.4f} × sepal_length) + ")
print(f"               ({modelo_mc.coef_[1]:.4f} × sepal_width) + ")
print(f"               ({modelo_mc.coef_[2]:.4f} × petal_width)")

# MÉTODO 2: statsmodels (más detallado)
print("\n2. USANDO STATSMODELS (más detalles):")
X_const = sm.add_constant(X)
modelo_sm = sm.OLS(y, X_const).fit()

print(f"ECUACIÓN:")
print(f"petal_length = {modelo_sm.params['const']:.4f} + ")
print(f"               ({modelo_sm.params['sepal_length']:.4f} × sepal_length) + ")
print(f"               ({modelo_sm.params['sepal_width']:.4f} × sepal_width) + ")
print(f"               ({modelo_sm.params['petal_width']:.4f} × petal_width)")

# MÉTRICAS DEL MODELO
print("\n📊 MÉTRICAS DEL MODELO MÚLTIPLE:")
print(f"R²: {modelo_sm.rsquared:.4f}")
print(f"R² ajustado: {modelo_sm.rsquared_adj:.4f}")
print(f"Error estándar: {np.sqrt(modelo_sm.mse_resid):.4f}")

# SIGNIFICANCIA DE VARIABLES
print("\n🔍 SIGNIFICANCIA DE COEFICIENTES:")
print("Variable       Coeficiente   p-value   Significativo")
print("-" * 50)
for var in ['const', 'sepal_length', 'sepal_width', 'petal_width']:
    coef = modelo_sm.params[var]
    pval = modelo_sm.pvalues[var]
    signif = "SÍ" if pval < 0.05 else "NO"
    print(f"{var:12} {coef:10.4f} {pval:10.4f}     {signif}")

print("="*60)