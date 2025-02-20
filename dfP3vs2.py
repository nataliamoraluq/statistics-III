#
import pandas as pd
import numpy as np
import math
from scipy.stats import f, studentized_range
from tabulate import *
# ---------------------------------------------------------------------------------------------
# -- VARS, ARRAYS N GLOBAL STUFF --
origData = []
origDataSec = []
fTab = 0 #fTab
fCalc = 0 #fCalc
dhs = 0
pTukey = 0

"""
# calcsPerColumn - diccionario results. por columna // other data in case i need it or want to see it!!
"originales": [],
"cuadrados": [],


"xi^2": [],
"""
#
# diccionario results. por columna de la tabla de conting.
calcsPerColumn = {
    "columna": [],
    "xi": [],
    "xi2": [],
    "xi^2/n":[],
    "media": []
}
#diccionario: pre - table values / need to know em before doing the 2nd table (tabla de analisis de var.)
preTabVales = {
    "gl(Trat)":tCols-1,
    "gl(Err)":nt-tCols,
    "sigLevel":0.95,
    "alpha":0.05,
    "fTab":fTab
}
#diccionario: medidas/valores de la tabla de analisis de var.
tabVarValues = {
    "TreatmentMeasures":[], #medidas de tratamiento [SCTR,glT,MCTR]
    "ErrorMeasures":[], #medidas de error [SCE,glE,MCE]
    "TotalMeasures":[], #medidas totales [SCT,n-1,RV(fCalc)]
}
#


#
df = pd.read_csv("datos.csv") # csv como dataframen usando pandas
nombres_columnas = df.columns # nombres de las columnas / nombre de las vars
contingencyTable = {} # diccionario inicial para procesar los datos de la tabla iterando por columna
#sumBetN = []
#
def makeContingencyTable():
    #-------------------------------------------------
    #para contar valores por columna
    global ni
    for i in nombres_columnas:
        ni = len(df[i].values)
    #
    global nt
    nt = 0
    # MAIN FOR ---------- TABLA DE CONTINGENCIA / COLUMNAS / POR VARIABLE
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
        #print(f"\n sumOt = {sumOt:.4f}")
        #
        forBet = sumOt / ni #suma de los xi elev a 2 / ni (/n30)
        #print(f"\n forBet = {forBet:.4f}")
        sumBetN = forBet
        #
        # save results en el diccionario
        #"sumatoria_xi_al_cuadrado":sumOt,
        contingencyTable[columnVar] = {
            "originales": origData,
            "cuadrados": origDataSec,
            "sumatoria_xi": sumT,
            "sumatoria_de_xi2": sumTsquare,
            "sum_xi_^2_/n":sumBetN,
            "media":medianVar
        }

    #--------------------------------------------------------------------------
    # ________________ DATAFRAME CON LOS TOTALES POR VAR_______________________
    # Se itera sobre las columnas y sus resultados
    for nombreCol, datos in contingencyTable.items():
        # Agrega los resultados a las listas correspondientes
        calcsPerColumn["columna"].append(nombreCol)
        """calcsPerColumn["originales"].append(", ".join(str(x) for x in datos["originales"]))
        calcsPerColumn["cuadrados"].append(", ".join(f"{x:.4f}" for x in datos["cuadrados"]))"""
        calcsPerColumn["xi"].append(float(f"{datos['sumatoria_xi']:.4f}"))
        calcsPerColumn["xi2"].append(float(f"{datos['sumatoria_de_xi2']:.4f}"))
        #calcsPerColumn["xi^2"].append(float(f"{datos['sumatoria_xi_al_cuadrado']:.4f}"))
        calcsPerColumn["xi^2/n"].append(float(f"{datos['sum_xi_^2_/n']:.4f}"))
        calcsPerColumn["media"].append(float(f"{datos['media']:.4f}"))
    #--------------------------------------------------------------------------
#

#
def printSumData():
    makeContingencyTable() #se llama al metodo main/principal/ conting. table
    #para que se ejecute, se procesen y llenen los datos, luego procedemos a la impresion
    ##----------------------------------------------------------------------------------------------
    print("----------------------------------------------------------------------------------------------")
    print("----------------------------------- TABLA DE CONTINGENCIA ------------------------------------")
    print("______________________________________________________________________________________________")
    print("____________________ Datos y valores de la muestra original (CSV) ____________________ \n ")
    print(df) #first df with the nom proccesed data
    # **** contingencyTable Dataframe results after processing -- dfResults ***********
    dfResults = pd.DataFrame(calcsPerColumn) #DataFrame a partir del diccionario con totales por var
    """pd.set_option('display.width', 900)
    dfResults.head()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    """
    print("__________________________________________________________\n ") 
    print("____________________ Datos procesados ____________________") 
    print(dfResults)#second df DataFrame con results
    #-----------------------------------------------------------
    # _________________ "PASOS" / TOTALES DE LA TABLA DE CONTINGENC. ________________

    #df_resultados -> dfResults per parameters
    sumXt = dfResults["xi"].sum() # Σsum de Xt total
    sumX2 = dfResults["xi2"].sum() # Σsum de X a la 2 t total
    #sumTotXtElev2 = df_resultados["xi^2"].sum() # Σsum total de Xt elevada al cuadrado total
    global tCols
    tCols = len(df.columns)
    betN = dfResults["xi^2/n"].sum() #sum de todos los xi^2/ni POR COLUMNA ; el xielev2 / ni de c/u sumados
    #OJO! != de C porque C es el xielev2 / nT
    #results con 4 decimales al imprimirlos
    print("_______________________________________") 
    print(f" ΣXt = {sumXt:.4f}")
    print(f"\n ΣXt^2 = {sumX2:.4f}")
    #print(f"\n Σ(Xt) ^ 2 = {sumTotXtElev2:.4f}")
    print(f"\n Σ(Xt) ^ 2 / n = {betN:.4f}") 
    #\n
    print(f" \nNi (por col.) = {ni}")
    print(f" Nt = {nt}")
    print(f" t = {tCols}")
#
printSumData()    
#_________________________________________________________________________________________________
#-------------------------------------------------------------------------------------------------
# --------------------------------- CALCS TABLA ANALISIS DE VARIANZA -----------------------------
#_________________________________________________________________________________________________
def varianceAnalysisCalcs():
    print(f" x")
    #C
    #SCT
    #SCTR
    #SCE
    #
    #
    #MCTR
    #MCE
    #F - RV - RAZON DE VARIANZA
def tukeyTrial(ni,MCE,nminust,alpha):
    #ni = nj = 30 (muestra por col)
    global nj
    nj = ni
    #Valor qalpha t studentizada?
    print(f" x")
#
#
#
#
#other comments n stuff
#----------------------------------------------------------------------------------------
#sumt ------------------------------------------------------------
#
#
#for val1, val2 in zip(lista1, list2):   #--> leer dos listas a la vez
#
#
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