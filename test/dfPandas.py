#Leer los datos y ordenarlos por columnas tipo dataframe
"""import pandas as pd
df = pd.read_csv('datos.csv') 
print(df)"""

import pandas as pd
import numpy as np
from IPython.display import display

# Lee el archivo CSV
df = pd.read_csv("datos.csv")

# Calcula los cuadrados de las columnas numéricas
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col + ' elev a la 2'] = df[col] ** 2
        df[col + ' media'] = np.median(df[col])

# Aplica estilos visuales al DataFrame
styled_df = df.style.format(precision=2, decimal='.') \
                   .set_properties(**{'border': '1px solid blue', 'text-align': 'center'}) \
                   

# Muestra el DataFrame resultante
display(styled_df)
with open('dataframe_estilizado.html', 'w') as f:
    f.write(styled_df.to_html())  # Use to_html() instead of render()