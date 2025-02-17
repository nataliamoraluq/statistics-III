import pandas as pd
import numpy as np
from tabulate import *

# Lee el archivo CSV en un DataFrame de Pandas
df = pd.read_csv("datos.csv")

# Obtén los nombres de las columnas
nombres_columnas = df.columns

# Crea un diccionario para guardar los resultados de cada columna
resultados = {}

# Itera sobre las columnas del DataFrame
for nombre_columna in nombres_columnas:
    # Guarda los valores originales de la columna en un arreglo
    arreglo_original = df[nombre_columna].values

    # Calcula el cuadrado de los valores
    arreglo_cuadrado = arreglo_original**2

    #sumt ------------------------------------------------------------
    """
    numero = np.float64(12.3456789)

    print(f"{numero:.4f}")  # Output: 12.3457
    
    """
    """sumT = np.sum(arreglo_original)
    sumTal2 = np.sum(arreglo_cuadrado)
    print("len original array:"+str(len(arreglo_original)))
    print("Sum t POR COLUMNA : "+str(f"{sumT:.4f}"))
    print("Sum tcuadrado POR COLUMNA: "+str(f"{sumTal2:.4f}"))
    print("----------------------------------------")
    print("\n")

    sumT = 0
    sumTal2 = 0"""

    # Calcula la sumatoria de los valores
    sumatoria = np.sum(arreglo_original)
    # Calcula la sumatoria de los valores al cuadrado
    sumatoria_cuadrado = np.sum(arreglo_cuadrado)
    medianVar = np.median(arreglo_original)

    # Guarda los resultados en el diccionario
    resultados[nombre_columna] = {
        "originales": arreglo_original,
        "cuadrados": arreglo_cuadrado,
        "sumatoria": sumatoria,
        "sumatoria_cuadrado": sumatoria_cuadrado,
        "media":medianVar
    }

# Crea un nuevo diccionario para organizar los resultados por columna
resultados_por_columna = {
    "columna": [],
    "originales": [],
    "cuadrados": [],
    "sumatoria": [],
    "sumatoria_cuadrado": [],
    "media": []
}

"""global minValues
minValues = []
for i in resultados.items:
    nParc = len(resultados)
    # Obtén los nombres de las columnas
    t = df.columns
nt = df.values
minValues.append(nParc)
minValues.append(t)
minValues.append(nt)

for nparc, t, nt in minValues:
    print(f"\n ΣXt = {sumXt:.4f}") """

#--------------------------------------------------------------------------

# Itera sobre las columnas y sus resultados
for nombre_columna, datos in resultados.items():
    # Agrega los resultados a las listas correspondientes
    resultados_por_columna["columna"].append(nombre_columna)
    resultados_por_columna["originales"].append(", ".join(str(x) for x in datos["originales"]))
    resultados_por_columna["cuadrados"].append(", ".join(f"{x:.4f}" for x in datos["cuadrados"]))
    resultados_por_columna["sumatoria"].append(float(f"{datos['sumatoria']:.4f}"))
    resultados_por_columna["sumatoria_cuadrado"].append(float(f"{datos['sumatoria_cuadrado']:.4f}"))
    resultados_por_columna["media"].append(float(f"{datos['media']:.4f}"))

#DataFrame a partir del diccionario
df_resultados = pd.DataFrame(resultados_por_columna)

# DataFrame con los resultados
print(df_resultados)

#-----------------------------------------------------------
def printSumData(df_resultados):
    # Σsum de Xt
    sumXt = df_resultados["sumatoria"].sum()
    # Σsum de X a la 2 t
    sumX2 = df_resultados["sumatoria_cuadrado"].sum()
    # Σsum total de Xt elevada al cuadrado
    sumTotXtElev2 = sum(df_resultados["sumatoria"]**2)
    print(f"\n ΣXt = {sumXt:.4f}")
    print(f"\n ΣXt^2 = {sumX2:.4f}")
    print(f"\n Σ(Xt) ^ 2 = {sumTotXtElev2:.4f}")

printSumData(df_resultados)