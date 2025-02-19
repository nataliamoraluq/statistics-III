"""from tabulate import *
import scipy.stats as stats
import pandas as pd
datos = pd.read_csv("datos.csv", delimiter=";")
tableData = []
for x in datos.columns:
    aux = [x]
    aux.append(datos[x].tolist())
    tableData.append(aux)
table = stats.ncf_gen()
table.x =  sum([sum(x[1]) for x in tableData])
table.x2 = sum([sum([y**2 for y in x[1]]) for x in tableData])
table.x2_2 = sum([sum(x[1])**2 for x in tableData])
table.n = sum([len(x[1]) for x in tableData])
table.xi2n = table.x2_2 / table.n
table.meaX = datos.mean(numeric_only=True)
print(table.xi2n)"""