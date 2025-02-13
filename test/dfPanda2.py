import pandas as pd
import numpy as np

# 1) Generar un DataFrame y leer los nombres de las columnas
df = pd.read_csv("datos.csv")  # Reemplaza "nombre_de_tu_archivo.csv" con el nombre real de tu archivo
nombres_columnas = df.columns.tolist()  # Obtiene los nombres de las columnas del DataFrame

# 2) Diccionario para guardar los arreglos
arreglos = {}

# 3) Estructura cíclica para leer las columnas y guardar los valores en arreglos
for nombre_columna in nombres_columnas:
    arreglos[nombre_columna] = df[nombre_columna].values

# 4) Ordenar los arreglos
for nombre_columna, arreglo in arreglos.items():
    arreglos[nombre_columna] = sorted(arreglo)  # Ordena los arreglos de forma ascendente

# 5) Crear un nuevo DataFrame con los arreglos ordenados
df_ordenado = pd.DataFrame(arreglos)

# 6) Imprimir el DataFrame ordenado por consola
print(df_ordenado)

# 7) Generar arreglos con los valores elevados al cuadrado (opcional)
arreglos_cuadrado = {}
for nombre_columna, arreglo in arreglos.items():
    arreglos_cuadrado[nombre_columna] = arreglo ** 2

# Imprimir los resultados (opcional)
for nombre_columna, arreglo in arreglos.items():
    print(f"{nombre_columna}: {arreglo}")

for nombre_columna, arreglo_cuadrado in arreglos_cuadrado.items():
    print(f"{nombre_columna} al cuadrado: {arreglo_cuadrado}")