import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Cargar datos
ds = pd.read_csv("iris.data", header=None, names=[
    "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
])

print("="*70)
print("analisis comparativo de modelos de regresión")
print("="*70)

# R² de regresión simple (petal_width solo)
X_simple = ds[["petal_width"]]
y = ds["petal_length"]
modelo_simple = LinearRegression()
modelo_simple.fit(X_simple, y)
r2_simple = modelo_simple.score(X_simple, y)

# R² de regresión múltiple (con 3 variables)
X_multiple = ds[["sepal_length", "sepal_width", "petal_width"]]
modelo_multiple = LinearRegression()
modelo_multiple.fit(X_multiple, y)
r2_multiple = modelo_multiple.score(X_multiple, y)

print(f"\n Comparación de R²:")
print(f"Regresión Simple (solo petal_width):    R² = {r2_simple:.4f}")
print(f"Regresión Múltiple (3 variables):       R² = {r2_multiple:.4f}")
print(f"Mejora absoluta:                         ΔR² = {r2_multiple - r2_simple:.4f}")
print(f"Mejora relativa:                         {((r2_multiple - r2_simple)/r2_simple*100):.2f}%")

# analisis de mejora
print(f"\n¿Mejoró la estimación?")
if r2_multiple > r2_simple:
    mejora = r2_multiple - r2_simple
    if mejora > 0.02:
        print("Sí, mejoró significativamente")
        print(f"   Las variables adicionales aportan valor predictivo importante")
    elif mejora > 0.005:
        print("Mejoró ligeramente")
        print(f"   Las variables adicionales aportan algo de valor, pero limitado")
    else:
        print("Mejoró muy poco")
        print(f"   Las variables adicionales apenas mejoran el modelo")
else:
    print("No, no mejoró")
    print(f"   El modelo simple es igual o mejor")


print(f"\n Interpretación práctica:")
print(f"• petal_width solo explica el {r2_simple*100:.1f}% de la variabilidad")
print(f"• Agregando sepal_length y sepal_width, explicamos el {r2_multiple*100:.1f}%")
print(f"• La mejora de {((r2_multiple - r2_simple)*100):.1f}% puntos es muy pequeña")
print(f"• En la práctica, usar solo petal_width es casi igual de bueno y más simple")

# RECOMENDACIÓN
print(f"\n Recomendación final:")
if r2_multiple - r2_simple < 0.01:
    print("usar el modelo simple (solo petal_width)")
    print("• Más simple de interpretar")
    print("• Casi igual de preciso") 
    print("• Menos riesgo de overfitting")
else:
    print("considerar el modelo múltiple (3 variables)")
    print("• Mayor poder predictivo")
    print("• Captura relaciones más complejas")

print("="*70)
