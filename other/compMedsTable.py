"""import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("datos.csv")  # Reemplaza "nombre_de_tu_archivo.csv" con el nombre real de tu archivo
X = df[['Humedad', 'Temperatura', 'Presion']]  # Variables independientes
y = df['OxidoNitroso']  # Variable dependiente
X = sm.add_constant(X)  # Agregar una constante para el intercepto
model = sm.OLS(y, X).fit()
print(model.summary())"""

import pandas as pd
from tabulate import *
from tabulate import tabulate
import string

from tabulate import tabulate
import string

dhsVal = 8.9

#descartado pq tabulate redondea y quita decimales


def crear_tabla_diferencias(medVars):
    #
    #
    columnsLen = len(medVars) # len medias = tCols
    nameCols = ["Var1","Var2","Var3","Var4"] #nombres etiquetas / vars
    medValues = medVars #copia del array para las etiquetas - vars
    comparTable = [] # """matriz""" / tabla de dif entre las medias de c/var
    dependants = []
    nondepent = []
    
    #-------------- Estructura de la tabla ----------
    #mainHead = [" x̅ "]# mainHead - nombre de las variables
    secHeader = medValues #sec - header; para los valores como tal de cada media
    comparTable.append(nameCols)
    comparTable.append(secHeader)
    # estructura generada - calcs. de las diferencias para ver dependencias e independencia
    for i in range(columnsLen):
        # filas de la tabla se crean con pivot 
        #pivot = [medValues[i]]  #pivot for each i in med values
        pivot = []
        for j in range(columnsLen):
            if j < i:
                pivot.append("///")
                #non : /// or ||| lo que se vea mas estetico y bonito
            else:
                pivot.append(medVars[i] - medVars[j])  # i - j -> medYvar - medX1var
                meandiff = medVars[i] - medVars[j] #valor de la diferencia

                # TRY 1 - dif comparada con el DHS
                #relBetVal - relationshipBetweenValues
                #or meandiff > -dhsVal
                relBetVal = "Independiente" if meandiff > dhsVal else "Dependiente"
                if(relBetVal=="Independiente"):
                    print("independiente val: ",meandiff)
                    nondepent.append(meandiff)
                else: 
                    print(" dependiente val: ",meandiff)
                    dependants.append(meandiff)
        comparTable.append(pivot)
        # Crear el DataFrame
    df = pd.DataFrame(comparTable, index=nameCols, columns=secHeader)
    return df

#impresion
# Ejemplo de uso
medVars = [10, 5, 82, 3]
difMedTable = crear_tabla_diferencias(medVars)
# Imprimir la tabla con tabulate
print(difMedTable)
# Imprimir el DataFrame
#print(df_resultado)