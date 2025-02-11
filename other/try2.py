import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("datos.csv")  # Reemplaza "nombre_de_tu_archivo.csv" con el nombre real de tu archivo
X = df[['Humedad', 'Temperatura', 'Presion']]  # Variables independientes
y = df['OxidoNitroso']  # Variable dependiente
X = sm.add_constant(X)  # Agregar una constante para el intercepto
model = sm.OLS(y, X).fit()
print(model.summary())