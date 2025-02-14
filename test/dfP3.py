import pandas as pd
import numpy as np

# Lee el archivo CSV en un DataFrame de Pandas
df = pd.read_csv("datos.csv")

# Obtén los nombres de las columnas
nombres_columnas = df.columns

# Crea un diccionario para guardar los resultados de cada columna
resultados = {}

# Itera sobre las columnas del DataFrame
for nombre_columna in nombres_columnas:
    # Guarda los valores originales de la columna en un arreglo
    arreglo_original = df[nombre_columna].values

    # Calcula el cuadrado de los valores
    arreglo_cuadrado = arreglo_original**2

    #sumt ------------------------------------------------------------
    """
    numero = np.float64(12.3456789)

    print(f"{numero:.4f}")  # Output: 12.3457
    
    """
    sumT = np.sum(arreglo_original)
    sumTal2 = np.sum(arreglo_cuadrado)
    print("len original array:"+str(len(arreglo_original)))
    print("Sum t POR COLUMNA : "+str(f"{sumT:.4f}"))
    print("Sum tcuadrado POR COLUMNA: "+str(f"{sumTal2:.4f}"))
    print("----------------------------------------")
    print("\n")

    sumT = 0
    sumTal2 = 0

    # Calcula la sumatoria de los valores
    sumatoria = np.sum(arreglo_original)
    # Calcula la sumatoria de los valores al cuadrado
    sumatoria_cuadrado = np.sum(arreglo_cuadrado)

    # Guarda los resultados en el diccionario
    resultados[nombre_columna] = {
        "originales": arreglo_original,
        "cuadrados": arreglo_cuadrado,
        "sumatoria": sumatoria,
        "sumatoria_cuadrado": sumatoria_cuadrado
    }

# Crea un nuevo diccionario para organizar los resultados por columna
resultados_por_columna = {
    "columna": [],
    "originales": [],
    "cuadrados": [],
    "sumatoria": [],
    "sumatoria_cuadrado": []
}

# Itera sobre las columnas y sus resultados
for nombre_columna, datos in resultados.items():
    # Agrega los resultados a las listas correspondientes
    resultados_por_columna["columna"].append(nombre_columna)
    resultados_por_columna["originales"].append(", ".join(str(x) for x in datos["originales"]))
    resultados_por_columna["cuadrados"].append(", ".join(f"{x:.4f}" for x in datos["cuadrados"]))
    resultados_por_columna["sumatoria"].append(float(f"{datos['sumatoria']:.4f}"))
    resultados_por_columna["sumatoria_cuadrado"].append(float(f"{datos['sumatoria_cuadrado']:.4f}"))

# Crea el DataFrame a partir del diccionario
df_resultados = pd.DataFrame(resultados_por_columna)

# Calcula la suma total de todas las sumatorias por columnas
suma_total_sumatorias = df_resultados["sumatoria"].sum()

# Calcula la suma de todos los valores al cuadrado
suma_total_cuadrados = df_resultados["sumatoria_cuadrado"].sum()

# Calcula la suma total elevada al cuadrado con 4 decimales
#sumt_elevadaAl2 = float(f"{suma_total_sumatorias**2:.4f}")  ---> REVISAR!!!

# Calcula la nueva sumatoria: sumatoriaPorValAlCuadrado
sumatoriaPorValAlCuadrado = sum(df_resultados["sumatoria"]**2)

# Muestra el DataFrame con los resultados
print(df_resultados)

print(f"\nSuma de los totales por columnas: {suma_total_sumatorias:.4f}")

print(f"\nSuma de los totales al cuadrado sumX2 al cuadrado: {suma_total_cuadrados:.4f}")

#print(f"\nSumT elevada al cuadrado: {sumt_elevadaAl2}") ---> REVISAR!!!

print(f"\nSumatoria de los cuadrados de las sumatorias: {sumatoriaPorValAlCuadrado:.4f}")