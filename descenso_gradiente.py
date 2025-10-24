import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        
        print("Ejecutando Descenso del Gradiente...")
        for i in range(self.n_iter):
            # Predicciones
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Gradientes
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Actualizar parámetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calcular pérdida
            loss = np.mean((y_pred - y)**2)
            self.loss_history.append(loss)
            
            # Mostrar progreso cada 100 iteraciones
            if i % 200 == 0:
                print(f"Iteración {i:4d}, Pérdida (MSE): {loss:.6f}")
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Cargar datos
print("Cargando datos del dataset")
ds = pd.read_csv("iris.data", header=None, names=[
    "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
])
print(f"Dataset cargado: {len(ds)} registros")

X = ds[["sepal_length", "sepal_width", "petal_width"]].values
y = ds["petal_length"].values

print("Variables predictoras:")
print("- sepal_length (x₁)")
print("- sepal_width  (x₂)") 
print("- petal_width  (x₃)")
print(f"Variable respuesta: petal_length (Y)")

# Normalizar características
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std

print("Medias antes de normalizar:", X_mean)
print("Desviaciones estándar:", X_std)

# Entrenar modelo con Descenso del Gradiente
print("\n Entrenando modelo con Descenso del Gradiente...")
modelo_gd = LinearRegressionGD(learning_rate=0.01, n_iter=1000)
modelo_gd.fit(X_normalized, y)

# Desnormalizar coeficientes
print("\n Desnormalizando coeficientes...")
coef_desnormalizados = modelo_gd.weights / X_std
intercepto_desnormalizado = modelo_gd.bias - np.sum(coef_desnormalizados * X_mean)

# Resultados finales
print("\n" + "="*60)
print("Resultados finales del modelo de regresión múltiple:")
print("="*60)
print(f"Ecuación de regresión multiple:")
print(f"petal_length = {intercepto_desnormalizado:.4f} + ")
print(f"               ({coef_desnormalizados[0]:.4f} × sepal_length) + ")
print(f"               ({coef_desnormalizados[1]:.4f} × sepal_width) + ")
print(f"               ({coef_desnormalizados[2]:.4f} × petal_width)")

print(f"\nCoeficientes finales:")
print(f"β₀ (Intercepto): {intercepto_desnormalizado:.4f}")
print(f"β₁ (sepal_length): {coef_desnormalizados[0]:.4f}")
print(f"β₂ (sepal_width): {coef_desnormalizados[1]:.4f}")
print(f"β₃ (petal_width): {coef_desnormalizados[2]:.4f}")

print(f"\nPérdida final (MSE): {modelo_gd.loss_history[-1]:.6f}")

# Gráfico de convergencia
plt.figure(figsize=(10, 6))
plt.plot(modelo_gd.loss_history)
plt.title('Convergencia del Descenso del Gradiente')
plt.xlabel('Iteración')
plt.ylabel('Pérdida (MSE)')
plt.grid(True)
plt.show()

print("\n Descenso del Gradiente completado")