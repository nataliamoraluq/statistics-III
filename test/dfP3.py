import pandas as pd
import numpy as np
from tabulate import *

df = pd.read_csv("datos.csv") # leer el csv con pandas y hacerlo un dataframe
nombres_columnas = df.columns # nombres de las columnas / nombre de las vars
contingencyTable = {} # diccionario para guardar los resultados de cada columna
# Itera sobre las columnas del DataFrame
for columnVar in nombres_columnas:
    origData = df[columnVar].values # Guarda los valores originales de la columna en un arreglo
    origDataSec = origData**2 # Calcula el cuadrado de los valores
    #origData -> arreglo original/ datos originales de entrada separados en arreglos por columna
    #origDataSec -> arreglo original al cuadrado

    # 
    sumT = np.sum(origData) # sumatoria de los valores --- ΣXt
    sumTsquare = np.sum(origDataSec) # sumatoria de los valores --- ΣXt^2
    medianVar = np.median(origData) # sumatoria de los valores --- Σ(Xt) ^2

    # save results en el diccionario
    contingencyTable[columnVar] = {
        "originales": origData,
        "cuadrados": origDataSec,
        "sumatoria": sumT,
        "sumatoria_cuadrado": sumTsquare,
        "media":medianVar
    }
# nuevo diccionario para organizar los resultados por columna
calcsPerColumn = {
    "columna": [],
    "originales": [],
    "cuadrados": [],
    "sumatoria": [],
    "sumatoria_cuadrado": [],
    "media": []
}
#demas medidas / pasos para la tabla
"""
#global minValues
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
    print(f"\n ΣXt = {sumXt:.4f}")"""

#--------------------------------------------------------------------------
# Itera sobre las columnas y sus resultados
for nombreCol, datos in contingencyTable.items():
    # Agrega los resultados a las listas correspondientes
    calcsPerColumn["columna"].append(nombreCol)
    calcsPerColumn["originales"].append(", ".join(str(x) for x in datos["originales"]))
    calcsPerColumn["cuadrados"].append(", ".join(f"{x:.4f}" for x in datos["cuadrados"]))
    calcsPerColumn["sumatoria"].append(float(f"{datos['sumatoria']:.4f}"))
    calcsPerColumn["sumatoria_cuadrado"].append(float(f"{datos['sumatoria_cuadrado']:.4f}"))
    calcsPerColumn["media"].append(float(f"{datos['media']:.4f}"))
#
#contingencyTable results after processing -- dfResults
dfResults = pd.DataFrame(calcsPerColumn) #DataFrame a partir del diccionario
#
print(dfResults) # imprimir DataFrame con results
#-----------------------------------------------------------
def printSumData(df_resultados):
    #df_resultados -> dfResults per parameters
    sumXt = df_resultados["sumatoria"].sum() # Σsum de Xt
    sumX2 = df_resultados["sumatoria_cuadrado"].sum() # Σsum de X a la 2 t
    sumTotXtElev2 = sum(df_resultados["sumatoria"]**2) # Σsum total de Xt elevada al cuadrado
    #results con 4 decimales
    print(f"\n ΣXt = {sumXt:.4f}")
    print(f"\n ΣXt^2 = {sumX2:.4f}")
    print(f"\n Σ(Xt) ^ 2 = {sumTotXtElev2:.4f}")
#
printSumData(dfResults)
#
#
#
#
#other comments n stuff
#----------------------------------------------------------------------------------------
#sumt ------------------------------------------------------------
"""
numero = np.float64(12.3456789)

    print(f"{numero:.4f}")  # Output: 12.3457
    
    sumT = np.sum(arreglo_original)
    sumTal2 = np.sum(arreglo_cuadrado)
    print("len original array:"+str(len(arreglo_original)))
    print("Sum t POR COLUMNA : "+str(f"{sumT:.4f}"))
    print("Sum tcuadrado POR COLUMNA: "+str(f"{sumTal2:.4f}"))
    print("----------------------------------------")
    print("\n")

    sumT = 0
    sumTal2 = 0

"""