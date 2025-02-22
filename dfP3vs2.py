#
import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import f, studentized_range
from tabulate import *
# ---------------------------------------------------------------------------------------------
# -- VARS, ARRAYS N GLOBAL STUFF --
origData = []
origDataSec = []
mediasVars = []
"""
fTab = 0 #fTab
fCalc = 0 #fCalc
dhs = 0
pTukey = 0
"""


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
preTabVales = {}
#diccionario: medidas/valores de la tabla de analisis de var.
tabVarValues = {}
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
        global tCols
        tCols = len(df.columns)
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

        #array de medias
        mediasVars.append(float(f"{datos['media']:.4f}"))
    for i in mediasVars:
        print("media:",i)
        #("")+str()
    #--------------------------------------------------------------------------
#

def hipotesisProof(ftab, fcalc):
    #print("\n ") 
    print("----------------------------- PRUEBA DE HIPOTESIS ---------------------------------") 
    if(fcalc < ftab):
        #print("----------------------------------------------------------------------------------") 
        print(" Fcalc < Ftab")
        print(str(fcalc)," < ",str(ftab))
        print("(RR) - Se rechaza la Hipotesis Nula (Ho), aceptando la Hipotesis Alternativa (Ha)")
    elif(fcalc > ftab):
        #print("----------------------------------------------------------------------------------") 
        print(" Fcalc > Ftab")
        print(str(round(fcalc,4))," > ",str(round(ftab,4)))
        print("(RA) - Se acepta la Hipotesis Nula (Ho), rechazando asi la Hipotesis Alternativa (Ha)")
    else:
        print("Error! Revisar valores de Ftab y Fcalc")
    print("------------------------------------------------------------------------------------------\n ") 

#_________________________________________________________________________________________________
#-------------------------------------------------------------------------------------------------
# --------------------------------- CALCS TABLA ANALISIS DE VARIANZA -----------------------------
#_________________________________________________________________________________________________
def varianceAnalysisCalcs(dfResults, tCols, nt):
    #print(f" x")
    #
    """print("_____________________________________________________________________\n ") 
    print("___________ The other stuff pre - table (round with 4 dec)___________") """
    #-------------------
    
    #mediasVars = []
    """for posi in contingencyTable.items:
        mediasVars[posi].append(dfResults["media"])
        print("media nro:",str(posi)," =",str(mediasVars[posi].value))"""
    #-------------------
    sumXt = dfResults["xi"].sum()**2 #Σ(Xt)
    sumX2 = dfResults["xi2"].sum() #ΣX2t
    sumBetN = dfResults["xi^2/n"].sum() #Σ(Xt)2 / n
    #print("Σ(Xt) ^ 2: ",round(sumXt,4)) 
    #C = Σ(Xt) ^ 2 / Nt
    C = sumXt / nt
    #print("calc C= ",round(C, 4))
    #SCT = ΣX2t - C
    SCT = sumX2 - C
    #print("SCT = ",round(SCT, 4))
    #SCTR = ( Σ(Xt) ^ 2 / n ) - C
    SCTR = sumBetN - C
    #print("SCTR = ",round(SCTR,4))
    #SCE = SCT - SCTR
    SCE = SCT - SCTR
    #print("SCE = ",round(SCE,4))
    #
    #gl  - GRADOS DE LIBERTAD - GL DE TRATAMIENTO Y GR DE ERROR
    glTrat = tCols - 1
    glErr=nt-tCols

    """print("glTrat = ",glTrat)
    print("glError = ",glErr)"""
    #MCTR = SCE //glTrat
    MCTR = SCTR /glTrat
    #print("MCTR = ",round(MCTR,4))
    #MCE
    MCE = SCE /glErr
    #print("MCE = ",round(MCE,4))
    #F - RV - RAZON DE VARIANZA
    Fcalc = MCTR / MCE
    #print("RV / F calc = ",round(Fcalc,4))

    # **************************************************

    #F TAB - F TABULADO
    #Ftab = F glTrat;glErr;1-alpha
    #alpha = 0.05 ; 1-alpha (sigL) = 0.95
    #OJO AQUI! Comparar results y ver!!!!!!
    Ftab = f.ppf(1-0.05, dfn=glTrat,dfd=glErr)
    #print("Ftab = ", round(Ftab,4))
    

    #vers .1 del diccionario ; ordenando por tipo de medidas
    """tabVarValues = {
        "Fuente_de_Variacion":["Tratamientos","Error","Total"], #cabeceras
        "TreatmentMeasures":["SCTR:"+str(round(SCTR,4)), "gl(Tratamiento): "+str(round(glTrat,4)), "MCTR:"+str(round(MCTR,4))], #medidas de tratamiento [SCTR,glT,MCTR]
        "ErrorMeasures":["SCE: "+str(round(SCE,4)), "gl(Error): "+str(glErr), "MCE:"+str(round(MCE,4))], #medidas de error [SCE,glE,MCE]
        "TotalMeasures":["SCT: "+str(round(SCT,4)), "nt-1: "+str(nt-1), "(RV) - Fcalc: "+str(round(Fcalc,4))], #medidas totales [SCT,n-1,RV(fCalc)]
    }"""

    #vers.2 - guardados con el orden de la tabla original
    tabVarValues = {
        "Fuente_de_Variacion":["Tratamientos","Error","Total"],
        "SC":["SCTR: "+str(round(SCTR,4)), "SCE: "+str(round(SCE,4)), "SCT: "+str(round(SCT,4))], #SC
        "gl":[" gl(Trat): t-1= "+str(round(glTrat,4)), "gl(Error): n-t= "+str(glErr), "nt-1= "+str(nt-1)], #gl
        "MC":["MCTR: "+str(round(MCTR,4)), "MCE: "+str(round(MCE,4)), "(RV):Fcalc= "+str(round(Fcalc,4))], #MC y RV
    }
    #
    print("            ++ --- Tabla de Analisis de Varianza de una Via --- ++ ")
    print(tabulate(tabVarValues, headers=["Fuente de Variac.","SC","gl","MC"], tablefmt='psql'))
    #
    #save the data on the dict - datos pre tabla n other stuff
    preTabValues = {
        "gl(Trat)":glTrat,
        "gl(Err)":glErr,
        "sigLevel":0.95,
        "alpha":0.05,
        "fTab":Ftab,
        "fCalc": Fcalc,
        
    }

    #"DHS": tukeyTrial(ni,MCE,glTrat,glErr,alfa)
    alfa = preTabValues["alpha"]
    #print("alpha=",alfa)
    # -- PRUEBA DE HIPOTESIS --
    hipotesisProof(Ftab, Fcalc)

    # -- PRUEBA DE TUKEY - 
    tukeyTrial(ni,MCE,glTrat,glErr,alfa)

    #primeras pruebas de impresion
    #dataframe con los calcs para la tabla
    #print("----------------------------------------------------------------------------------\n ") 
    #dfTabRV = pd.DataFrame(tabVarValues)
    #print(dfTabRV)
# ------------------------ PRUEBA DE TUKEY -------------------
def tukeyTrial(ni,MCE,glTrat,glError,alpha):
    #ni = nj = 30 (muestra por col)
    global nj
    nj = ni
    nmt  = glError
    Qt = studentized_range.ppf(1 - alpha, glTrat + 1, glError)
    # Sea Qt -> alpha αQt, nmt studentizada // solo para impresion: α
    global DHS
    DHS = Qt*sqrt(MCE/nj)

    #
    print(" -------- PRUEBA DE TUKEY ------------")
    print("DHS: Diferencia Honestamente Significativa")
    print("nj:",nj)
    print("n-t:",nmt)
    print("MCE:",round(MCE,4))
    print("αQt,n-t:",round(Qt,4))
    print("DHS =",str(round(DHS,4)))

    return DHS
#
def medComparison(mediasVars):
    """
            y    x1   x2    x3 -> mediasVars[] -- med values of each var

        y   //  y-x1 y-x2  y-x3 
        x1 //   //  x1-x2  x1-x3
        x2 //   //    //   x2-x3
        x3 //   //    //    //
    """
    #print("t=",tCols)
    nameCols = [] # nombres de vars/columns
    #
    #global medArray
    medArray = [] #array - duplica del array orig de las medias
    global comparTable #tabla final - results de la dif entre las medias a generar 
    comparTable = []
    #nombres de las variables por columnas del CSV *; note; pudiera usarlos directo pero asi es mas ordenado
    for nombreCol in nombres_columnas:
        nameCols.append(("x̅ de %s"%(nombreCol)))
        #print(nameCols)
    #
    columnsLen = len(mediasVars) # len medias = tCols
    #nameCols = ["Var1","Var2","Var3","Var4"] #nombres etiquetas / vars
    medArray = mediasVars #copia del array para las etiquetas - vars
    comparTable = [] # """matriz""" / tabla de dif entre las medias de c/var
    #-------------- Estructura de la tabla ----------
    mainHead = [" x̅ "] + nameCols # mainHead - nombre de las variables
    secHeader = ["    "] + medArray #sec - header; para los valores como tal de cada media
    comparTable.append(mainHead)
    comparTable.append(secHeader)
    # estructura generada - calcs. de las diferencias para ver dependencias e independencia
    for i in range(columnsLen):
        # filas de la tabla se crean con pivot 
        pivot = [medArray[i]]  #pivot for each i in med values
        for j in range(columnsLen):
            if j < i:
                pivot.append("///")
                #non : /// or ||| lo que se vea mas estetico y bonito
            else:
                pivot.append(mediasVars[i] - mediasVars[j])  # i - j -> medYvar - medX1var
        comparTable.append(pivot)
    #
    print(" -----------------------------------------------------------------------------------------")
    print(" Para determinar dependencia e independencia de las variables presentes en la muestra: ")
    print(" 1: Comparamos los valores de las medias entre si ")
    print(" 2: Luego comparamos los valores con el valor del DHS (Prueba Tukey) ")
    print(" * Si los valores < DHS ; son variables dependientes ")
    print(" * Si los valores > DHS ; son variables independientes ")
    #print(\n)
    print(" --- Tabla de comparacion entre las medias de las variables --- ")
    print(tabulate(comparTable, headers="firstrow", tablefmt="grid"))
    print(" --- Variables independientes --- ")
    print("working on that vs1")
    print(" --- Variables dependientes --- ")
    print("working on that vs2")

    #ahora prosigue comparar de esta tabla c/resultado con el DHS 
    

    #print(tabulate(tabVarValues, headers=varsTo, tablefmt='psql')) 
    # #ARREGLAR!!!
    #
    print(" -----------------------------------------------------------------------------------------")
    #print("x")

#-------------------------------------- IMPRESION TABLA DE CONTINGENCIA ---------------------------------------
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
    """global tCols
    tCols = len(df.columns)"""
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

    #send the new df with the data to the next calcs to do
    varianceAnalysisCalcs(dfResults, tCols, nt) #tabla de analisis de varianza; prueba de hipotesis y prueba de Tukey
    medComparison(mediasVars) #tabla comparacion entre medias y det. variables dependient. e independient.
    # 
#
printSumData()    

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

#failed trial - con pivot y aux vs1
    #pivot=pivot+1
"""while pivot<tCols:
        #print("x")
        
        pos = 0
        while pos<len(mediasVars):
            if pos<len(mediasVars):
                medArray.append(("variable%s"%pos,mediasVars[pos]))
                pos=pos+1

    auxmedArray = mediasVars.copy()
    for i, j in zip(mediasVars,auxmedArray):
        #valuei = medArray[i][1]
        #valuej = auxmedArray[j][1]
        if(i==j):
            print("i:",i,"j+1:",j+1)
            #i = i+1
            #j = j+1
    
    #medVars = [] 0y, 1x1, 2x2, 3x3
    
            y    x1   x2    x3 -> mediasVars[]

        y   //  y-x1 y-x2  y-x3 
        x1 //   //  x1-x2  x1-x3
        x2 //   //    //   x2-x3
        x3 //   //    //    //

            y    x1   x2    x3
        y   //  y-x1 y-x2  y-x3 
        x1 //   //  x1-x2  x1-x3
        x2 //   //    //   x2-x3
        x3 //   //    //    //
        
        
    
    medTable = {
        "variables":["var:","Error","Total"],
        "SC":["SCTR: "+str(round(SCTR,4)), "SCE: "+str(round(SCE,4)), "SCT: "+str(round(SCT,4))], #SC
        "gl":[" gl(Trat): t-1= "+str(round(glTrat,4)), "gl(Error): n-t= "+str(glErr), "nt-1= "+str(nt-1)], #gl
        "MC":["MCTR: "+str(round(MCTR,4)), "MCE: "+str(round(MCE,4)), "(RV):Fcalc= "+str(round(Fcalc,4))], #MC y RV
    }

#prueba (fallida) - dictionary con keys iterables para la comparacion de medias:

                for nombreVar in varsTo:   
                    print("nombreVar", nombreVar)
                
                
                medDict = {
                    "variable%s"%pos:mediasVars[pos]
                }
                #var1 = "variable%s"%pos
                #med1 = mediasVars[pos]
                
                print("media:",med)
                print('pos + 1 =',pos+1)
                print("pos+ 1 in medVar = ",mediasVars[pos+1])
                
                var= "variable%s"%pos
                s0s=mediasVars[pos]

                medDict.clear()

                medDict.update(
                    {"variable%s"%pos=med1},{var=s0s}
                )"""