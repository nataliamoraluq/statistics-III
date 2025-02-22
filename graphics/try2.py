import pandas as pd
import string

def crear_dataframe_diferencias(array):
    """
    Crea un DataFrame de pandas con diferencias y etiquetas personalizadas.

    Args:
        array: El arreglo de valores.

    Returns:
        Un DataFrame de pandas con las diferencias.
    """

    n = len(array)
    etiquetas = ["0","1","5","9"]
    etiquetas2 = ["x","y","w","z"]
    #cols = etiquetas2.reverse()
    matriz_diferencias = []
    #matriz_diferencias.append(etiquetas2)

    for i in range(n):
        pivot = []
        #pivot = [array[i]]  #pivot for each i in med values
        for j in range(n):
            if j < i:
                pivot.append("\\")
            else:
                pivot.append(array[i] - array[j])
        matriz_diferencias.append(pivot)

    # Crear el DataFrame
    print("\n")
    #print("    "," ".join(etiquetas2))
    df = pd.DataFrame(matriz_diferencias, index=etiquetas, columns=etiquetas)
    return df

# Ejemplo de uso
array = [10, 5, 8, 3]

df_resultado = crear_dataframe_diferencias(array)

# Imprimir el DataFrame
print(df_resultado)