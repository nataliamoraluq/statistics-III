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

def crear_tabla_diferencias(medVars):
    """
    Crea una tabla de diferencias con etiquetas de fila y columna personalizadas.

    Args:
        array: El arreglo de valores.

    Returns:
        Una matriz de diferencias en formato compatible con tabulate.
    """
    #
    #
    columnsLen = len(medVars) # len medias = tCols
    nameCols = ["Var1","Var2","Var3","Var4"] #nombres etiquetas / vars
    medValues = medVars #copia del array para las etiquetas - vars
    comparTable = [] # """matriz""" / tabla de dif entre las medias de c/var
    
    #-------------- Estructura de la tabla ----------
    mainHead = [" x̅ "] + nameCols # mainHead - nombre de las variables
    secHeader = ["    "] + medValues #sec - header; para los valores como tal de cada media
    comparTable.append(mainHead)
    comparTable.append(secHeader)
    # estructura generada - calcs. de las diferencias para ver dependencias e independencia
    for i in range(columnsLen):
        # filas de la tabla se crean con pivot 
        pivot = [medValues[i]]  #pivot for each i in med values
        for j in range(columnsLen):
            if j < i:
                pivot.append("///")
                #non : /// or ||| lo que se vea mas estetico y bonito
            else:
                pivot.append(medVars[i] - medVars[j])  # i - j -> medYvar - medX1var
        comparTable.append(pivot)
    #
    return comparTable

#impresion
# Ejemplo de uso
medVars = [10, 5, 8, 3]
difMedTable = crear_tabla_diferencias(medVars)
# Imprimir la tabla con tabulate
print(tabulate(difMedTable, headers="firstrow", tablefmt="grid"))
# Imprimir el DataFrame
#print(df_resultado)