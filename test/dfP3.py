#
import pandas as pd
import numpy as np
import math
from scipy.stats import f, studentized_range
from tabulate import *
#
origData = []
origDataSec = []
#
df = pd.read_csv("datos.csv") # leer el csv con pandas y hacerlo un dataframe
nombres_columnas = df.columns # nombres de las columnas / nombre de las vars
contingencyTable = {} # diccionario para guardar los resultados de cada columna
ni = 0
nt = 0
#sumBetN = []
#
#-------------------------------------------------
#para contar valores por columna
for i in nombres_columnas:
    ni = len(df[i].values)
#
#
for columnVar in nombres_columnas:
    # Se itera sobre las columnas del DataFrame
    origData = df[columnVar].values # Guarda los valores originales de la columna en un arreglo
    nt += len(df[columnVar].values)
    origDataSec = origData**2 # Calcula el cuadrado de los valores
    #origData -> arreglo original/ datos originales de entrada separados en arreglos por columna
    #origDataSec -> arreglo original al cuadrado
    #
    # 
    sumT = np.sum(origData) # sumatoria de los valores --- POR COLUMNA
    sumTsquare = np.sum(origDataSec) # sumatoria de los valores --- POR COLUMNA
    medianVar = np.median(origData) # valores de la media POR COLUMNA
    sumOt = np.sum(sumT**2) #sumT elevado a la 2
    print(f"\n sumOt = {sumOt:.4f}")
    #
    forBet = sumOt / ni
    print(f"\n forBet = {forBet:.4f}")
    sumBetN = forBet
    #
    # save results en el diccionario
    contingencyTable[columnVar] = {
        "originales": origData,
        "cuadrados": origDataSec,
        "sumatoria_xi": sumT,
        "sumatoria_de_xi2": sumTsquare,
        "sumatoria_xi_al_cuadrado":sumOt,
        "sum_xi_^2_/n":sumBetN,
        "media":medianVar
    }
# nuevo diccionario para organizar los resultados por columna
calcsPerColumn = {
    "columna": [],
    "originales": [],
    "cuadrados": [],
    "xi": [],
    "xi2": [],
    "xi^2": [],
    "xi^2/n":[],
    "media": []
}
#--------------------------------------------------------------------------
# Se itera sobre las columnas y sus resultados
for nombreCol, datos in contingencyTable.items():
    # Agrega los resultados a las listas correspondientes
    calcsPerColumn["columna"].append(nombreCol)
    calcsPerColumn["originales"].append(", ".join(str(x) for x in datos["originales"]))
    calcsPerColumn["cuadrados"].append(", ".join(f"{x:.4f}" for x in datos["cuadrados"]))
    calcsPerColumn["xi"].append(float(f"{datos['sumatoria_xi']:.4f}"))
    calcsPerColumn["xi2"].append(float(f"{datos['sumatoria_de_xi2']:.4f}"))
    calcsPerColumn["xi^2"].append(float(f"{datos['sumatoria_xi_al_cuadrado']:.4f}"))
    calcsPerColumn["xi^2/n"].append(float(f"{datos['sum_xi_^2_/n']:.4f}"))
    calcsPerColumn["media"].append(float(f"{datos['media']:.4f}"))
#
#contingencyTable results after processing -- dfResults
dfResults = pd.DataFrame(calcsPerColumn) #DataFrame a partir del diccionario
#
print(dfResults) # imprimir DataFrame con results
#-----------------------------------------------------------
def printSumData(df_resultados):
    #df_resultados -> dfResults per parameters
    sumXt = df_resultados["xi"].sum() # Σsum de Xt total
    sumX2 = df_resultados["xi2"].sum() # Σsum de X a la 2 t total
    sumTotXtElev2 = df_resultados["xi^2"].sum() # Σsum total de Xt elevada al cuadrado total
    tCols = len(df.columns)
    betN = df_resultados["xi^2/n"].sum()
    
    #results con 4 decimales al imprimirlos
    print(f"\n ΣXt = {sumXt:.4f}")
    print(f"\n ΣXt^2 = {sumX2:.4f}")
    print(f"\n Σ(Xt) ^ 2 = {sumTotXtElev2:.4f}")
    print(f"\n Σ(Xt) ^ 2 / Nt= {betN:.4f}")
    print(f"\n Ni (por col.) = {ni}")
    print(f"\n Nt = {nt}")
    print(f"\n t = {tCols}")
    #print(f"\n Σ(Xt) ^ 2 / Nt = {betN}") -- #revisar
    

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