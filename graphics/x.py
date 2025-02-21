"""import pandas as pd
import numpy as np

df = pd.read_csv("datos.csv") # leer el csv con pandas y hacerlo un dataframe
#nombres_columnas = df.columns 
pd.set_option('display.width', 500)
df.head()
print(df)"""

from tabulate import tabulate

table = [["Sun",696000,1989100000],["Earth",6371,5973.6], ["Moon",1737,73.5],["Mars",3390,641.85]]
print(tabulate(table))