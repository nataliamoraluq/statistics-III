import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import linregress
import statsmodels.api as sm

# 1. Cargar datos desde el archivo CSV
data = pd.read_csv("datos.csv")

# 2. Aplicar prueba F de Fisher para correlación
# Agrupar los datos por las categorías de la variable independiente
grupos = data.groupby('Humedad')['OxidoNitroso'].apply(list)  # Reemplaza 'Variable1' con el nombre de tu columna
# Realizar la prueba F de Fisher
f_statistic, p_value = f_oneway(*grupos)
print("Estadística F:", f_statistic)
print("Valor p:", p_value)

# 3. Regresión lineal simple (para cada variable independiente)
resultados_lineales = {}
for variable in ['Humedad', 'Temperatura', 'Presion']:
    slope, intercept, r_value, p_value, std_err = linregress(data[variable], data['OxidoNitroso'])
    resultados_lineales[variable] = {
        'pendiente': slope,
        'intercepto': intercept,
        'valor_r': r_value,
        'valor_p': p_value
    }
"""
# 4. Regresión lineal múltiple
X = data[['Variable1', 'Variable2', 'Variable3']]  # Variables independientes
y = data['VariableDependiente']  # Variable dependiente
X = sm.add_constant(X) # Agregar una constante para el intercepto
model = sm.OLS(y, X).fit()
resultados_multiple = model.summary()"""

# Imprimir resultados
print("\nResultados Prueba F de Fisher:")
print(f"Estadística F: {f_statistic}")
print(f"Valor p: {p_value}")

print("\nResultados Regresión Lineal Simple:")
for variable, resultados in resultados_lineales.items():
    print(f"\n{variable}:")
    print(f"  Pendiente: {resultados['pendiente']}")
    print(f"  Intercepto: {resultados['intercepto']}")
    print(f"  Valor r: {resultados['valor_r']}")
    print(f"  Valor p: {resultados['valor_p']}")

"""
print("\nResultados Regresión Lineal Múltiple:")
print(resultados_multiple)"""