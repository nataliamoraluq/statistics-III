import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact


df = pd.read_csv("datos.csv")
variables_cat = ['Humedad_cat', 'Temperatura_cat', 'Presion_cat']
tabla_contingencia = pd.crosstab(df[variables_cat], df['OxidoNitroso'])  # Ajusta 'OxidoNitroso' si es necesario
print(f"Tabla de contingencia:\n{tabla_contingencia}")
# 2. Categorizar variables continuas
# Define los rangos o niveles para cada variable

"""
df['Humedad_cat'] = pd.cut(df['Humedad'])
df['Temperatura_cat'] = pd.cut(df['Temperatura'])
df['Presion_cat'] = pd.cut(df['Presion'])"""
"""
#df['Humedad_cat'] = pd.cut(df['Humedad'], bins=[0, 30, 60, 100], labels=['Baja', 'Media', 'Alta'])
df['Temperatura_cat'] = pd.cut(df['Temperatura'], bins=[0, 70, 80, 100], labels=['Baja', 'Media', 'Alta'])
df['Presion_cat'] = pd.cut(df['Presion'], bins=[0, 29.2, 29.5, 30], labels=['Baja', 'Media', 'Alta'])"""

# 3. Crear tablas de contingencia y realizar prueba F de Fisher para cada variable

"""
for var in variables_cat:
    tabla_contingencia = pd.crosstab(df[var], df['OxidoNitroso'])  # Ajusta 'OxidoNitroso' si es necesario
    print(f"Tabla de contingencia para {var}:\n{tabla_contingencia}")

    # Prueba F de Fisher
    oddsratio, pvalue = fisher_exact(tabla_contingencia)
    print(f"Resultado de la prueba F de Fisher para {var}:")
    print(f"  Odds ratio: {oddsratio}")
    print(f"  Valor p: {pvalue}")

    # Interpretación del valor p
    alpha = 0.05  # Nivel de significancia (95%)
    if pvalue < alpha:
        print(f"  Existe una asociación significativa entre {var} y OxidoNitroso.")
    else:
        print(f"  No hay evidencia suficiente para concluir una asociación entre {var} y OxidoNitroso.")
    print("-" * 50)"""