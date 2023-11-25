import torch
import torch.nn as nn
import pandas as pd

# Define la arquitectura del modelo
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Cargar el modelo entrenado
model = SimpleNN(16)  # Asegúrate de proporcionar el mismo tamaño de entrada que se usó durante el entrenamiento
model.load_state_dict(torch.load("modelo_entrenado.pth"))
model.eval()

# Cargar los nuevos datos desde el archivo CSV
file_path_nuevos = "nuevos_datos.csv"  # Reemplaza con la ruta correcta a tu archivo CSV
df_nuevos = pd.read_csv(file_path_nuevos)

# Preprocesar los nuevos datos de la misma manera que los datos de entrenamiento
df_nuevos = pd.get_dummies(df_nuevos, columns=['GENDER', 'Car_Owner', 'Propert_Owner', 'Type_Income', 'EDUCATION', 'Marital_status', 'Housing_type', 'Type_Occupation'], drop_first=True)
boolean_columns_nuevos = df_nuevos.select_dtypes(include=['bool']).columns
df_nuevos[boolean_columns_nuevos] = df_nuevos[boolean_columns_nuevos].astype('int32')

# Eliminar columnas innecesarias (en este ejemplo, mantén solo las características)
X_nuevos = df_nuevos.drop(['Ind_ID'], axis=1)

# Convertir a tensores de PyTorch
X_nuevos_tensor = torch.tensor(X_nuevos.values, dtype=torch.float32)

# Realizar predicciones
with torch.no_grad():
    predictions = model(X_nuevos_tensor)

# Aplicar una función de activación sigmoide para obtener probabilidades
sigmoid = nn.Sigmoid()
probabilities = sigmoid(predictions)

# Convertir las probabilidades a etiquetas binarias (0 o 1) usando un umbral
threshold = 0.5
binary_predictions = (probabilities > threshold).float()

# Agregar las predicciones al DataFrame original
df_nuevos['Predictions'] = binary_predictions.numpy()

# Guardar el DataFrame con las predicciones en un nuevo archivo CSV
df_nuevos.to_csv("nuevas_predicciones.csv", index=False)
# import torch
# import torch.nn as nn
# import pandas as pd

# # Define la arquitectura del modelo (asegúrate de que sea la misma que durante el entrenamiento)
# class SimpleNN(nn.Module):
#     def __init__(self, input_size):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(64, 1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

# # Cargar el modelo entrenado desde el archivo .pth
# input_size = 16  # Reemplaza con el tamaño correcto según tus datos de entrenamiento
# model = SimpleNN(input_size)
# model.eval()

# # Crear datos de prueba ficticios
# data_test = {
#     'GENDER': ['M', 'F', 'M'],
#     'Car_Owner': ['Y', 'N', 'N'],
#     'Propert_Owner': ['Y', 'N', 'Y'],
#     'CHILDREN': [0, 1, 2],
#     'Annual_income': [60000, 75000, 50000],
#     'Type_Income': ['Salary', 'Business', 'Pension'],
#     'EDUCATION': ['High School', 'College', 'Masters'],
#     'Marital_status': ['Single', 'Married', 'Single'],
#     'Housing_type': ['Apartment', 'House', 'Condo'],
#     'Birthday_count': [-18772, 365243, -10000],
#     'Employed_days': [365, 730, 1825],
#     'Mobile_phone': [1, 0, 1],
#     'Work_Phone': [0, 1, 0],
#     'Phone': [1, 1, 0],
#     'EMAIL_ID': [1, 0, 1],
#     'Type_Occupation': [False, True, False],  # Valores booleanos que queremos predecir
#     'Family_Members': [2, 4, 1]
# }

# df_test = pd.DataFrame(data_test)

# # Realizar el mismo preprocesamiento que hiciste para el conjunto de entrenamiento
# # Convertir columnas categóricas en variables dummy
# df = pd.get_dummies(df_test, columns=['GENDER', 'Car_Owner', 'Propert_Owner', 'Type_Income', 'EDUCATION', 'Marital_status', 'Housing_type', 'Type_Occupation'], drop_first=True)  # Usamos drop_first=True para evitar la colinealidad

# # Convertir columnas booleanas a valores numéricos (True=1, False=0)
# boolean_columns = df.select_dtypes(include=['bool']).columns
# df[boolean_columns] = df[boolean_columns].astype('int32')
# # Asegúrate de que las columnas tengan tipos numéricos
# # for column in df_test.columns:
# #     df_test[column] = pd.to_numeric(df_test[column], errors='coerce')


# # Crear un tensor PyTorch para las predicciones
# X_test = torch.tensor(df_test.values, dtype=torch.float32)

# # Realizar predicciones
# with torch.no_grad():
#     model.eval()
#     outputs = model(X_test)
#     predictions = torch.sigmoid(outputs).numpy()

# # Imprimir las predicciones
# print("Predicciones:")
# print(predictions)


# import torch
# import pandas as pd
# from torch.utils.data import TensorDataset, DataLoader
# import torch.nn as nn
# import numpy as np



# # Supongamos que tienes un DataFrame original con nuevas observaciones
# data_test = {
#     'GENDER': ['M', 'F', 'M'],
#     'Car_Owner': ['Y', 'N', 'N'],
#     'Propert_Owner': ['Y', 'N', 'Y'],
#     'CHILDREN': [0, 1, 2],
#     'Annual_income': [60000, 75000, 50000],
#     'Type_Income': ['Salary', 'Business', 'Pension'],
#     'EDUCATION': ['High School', 'College', 'Masters'],
#     'Marital_status': ['Single', 'Married', 'Single'],
#     'Housing_type': ['Apartment', 'House', 'Condo'],
#     'Birthday_count': [-18772, 365243, -10000],
#     'Employed_days': [365, 730, 1825],
#     'Mobile_phone': [1, 0, 1],
#     'Work_Phone': [0, 1, 0],
#     'Phone': [1, 1, 0],
#     'EMAIL_ID': [1, 0, 1],
#     'Type_Occupation': [False, True, False],  # Valores booleanos que queremos predecir
#     'Family_Members': [2, 4, 1]
# }


# # Define a simple neural network
# class SimpleNN(nn.Module):
#     def __init__(self, input_size):
#         super(SimpleNN, self).__init__()
#         # Capa de entrada con input_size nodos y capa oculta con 64 nodos
#         self.fc1 = nn.Linear(input_size, 64)
#         # Función de activación ReLU
#         self.relu = nn.ReLU()
#         # Capa de salida con 1 nodo (sin función de activación)
#         self.fc2 = nn.Linear(64, 1)

#     def forward(self, x):
#         # Propagación hacia adelante
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

# # Cargar datos desde el archivo CSV
# file_path = "resultado_de_merge.csv"  # Reemplaza con la ruta correcta a tu archivo CSV
# df = pd.read_csv(file_path)
# df = pd.get_dummies(df, columns=['GENDER', 'Car_Owner', 'Propert_Owner', 'Type_Income', 'EDUCATION', 'Marital_status', 'Housing_type', 'Type_Occupation'], drop_first=True)  # Usamos drop_first=True para evitar la colinealidad

# # Convertir columnas booleanas a valores numéricos (True=1, False=0)
# boolean_columns = df.select_dtypes(include=['bool']).columns
# df[boolean_columns] = df[boolean_columns].astype('int32')

# # Separar etiquetas (y_train) y características (X_train)
# X_train = df.drop(['label', 'Ind_ID'], axis=1)

# # Cargar el modelo desde el archivo .pth
# input_size_test = X_train.shape[1]
# model = SimpleNN(input_size_test)  # Asegúrate de que input_size sea el mismo que el modelo original
# model.load_state_dict(torch.load("modelo_entrenado.pth"))
# model.eval()  # Cambiar al modo de evaluación

# # Supongamos que tienes nuevos datos en un DataFrame llamado df_test
# # Repite el preprocesamiento que hiciste para el conjunto de entrenamiento

# # Convertir columnas categóricas en variables dummy
# df_test = pd.get_dummies(data_test, columns=['GENDER', 'Car_Owner', 'Propert_Owner', 'Type_Income', 'EDUCATION', 'Marital_status', 'Housing_type', 'Type_Occupation'], drop_first=True)

# # Convertir columnas booleanas a valores numéricos (True=1, False=0)
# boolean_columns_test = df_test.select_dtypes(include=['bool']).columns
# df_test[boolean_columns_test] = df_test[boolean_columns_test].astype('int32')

# # Convertir a tensores de PyTorch
# X_test = torch.tensor(df_test.values, dtype=torch.float32)

# # Crear un conjunto de datos utilizando TensorDataset
# test_dataset = TensorDataset(X_test)

# # Crear un DataLoader para facilitar el manejo de los datos durante la inferencia
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # Realizar predicciones
# predictions = []
# with torch.no_grad():
#     for batch_X in test_loader:
#         outputs = model(batch_X[0])
#         predictions.append(outputs.sigmoid().numpy())

# # Concatenar las predicciones en un array numpy
# predictions = np.concatenate(predictions)

# # Puedes utilizar 'predictions' como las predicciones de tu modelo en los nuevos datos
