import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress # Realiza regresión lineal simple y proporciona métricas asociadas

# Datos de ejemplo
def regressionAndGraphic():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2, 4, 5, 4, 6, 7, 8, 9, 8, 10])

    slope, intercept, _, _, _ = linregress(x, y)
    print(f"Ecuación de la recta para {x} vs {y}: y = {round(slope, 4)} * x + {round(intercept, 4)}\n")
            
    # Graficar la dispersión y la recta de regresión
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Datos', alpha=0.6)  # Puntos de dispersión
    plt.plot(x, slope * x + intercept, color='orange', label=f'Recta de regresión: y = {round(slope, 4)} * x + {round(intercept, 4)}')  # Recta de regresión
    plt.xlabel(f'{x}')
    plt.ylabel(f'{y}')
    plt.title(f'Dispersión y Recta de Regresión Lineal entre (Humedad y Presion)')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.pause(600.01)
regressionAndGraphic()

"""

# Calcular la correlación (r)
r = np.corrcoef(x, y)[0, 1]

# Crear el diagrama de dispersión
plt.figure(figsize=(8, 6))  # Tamaño de la figura
plt.scatter(x, y, color='blue', label='Datos')

# Agregar título y etiquetas
plt.title(f'Diagrama de Dispersión (r = {r:.2f})')
plt.xlabel('Valores de X')
plt.ylabel('Valores de Y')

# Agregar la correlación como texto en el gráfico
plt.text(0.1, 0.9, f'r = {r:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Mostrar la leyenda
plt.legend()

# Mostrar el gráfico
plt.grid(True)  # Agregar una cuadrícula de fondo
plt.show()"""