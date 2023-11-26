import pandas as pd

# Lee los datos de entrada desde el archivo CSV
datos_entrada = pd.read_csv('Credit_card_label.csv')

# Lee los resultados desde el archivo CSV
resultados = pd.read_csv('Credit_card.csv')

# Combina los datos de entrada y resultados basándote en alguna columna común, como un ID
datos_combinados = pd.merge(datos_entrada, resultados, how='inner', on='Ind_ID')

# Muestra los primeros registros del conjunto combinado
print(datos_combinados.head())

# Guarda el conjunto combinado en un nuevo archivo CSV
datos_combinados.to_csv('resultado_de_merge.csv', index=False)