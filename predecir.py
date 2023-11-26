import asyncio
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Cargar datos desde el archivo CSV
file_path = "train/resultado_de_merge.csv"  # Reemplaza con la ruta correcta a tu archivo CSV
df = pd.read_csv(file_path)

# Separar etiquetas (y_train) y características (X_train)
datos_x = df.drop(['label', 'Ind_ID'], axis=1)
y_train = df['label']

# Obtener todas las columnas
# columnas = datos_x.columns

# print("Todas las columnas:")
# print(columnas)
# print(datos_x.head())
X_train = pd.get_dummies(datos_x)
columnas = X_train.columns
# print("Todas las columnas:")
# print(columnas)
# print(datos_x.head())
# print(X_train.head())
escalador = StandardScaler()
X_train = escalador.fit_transform(X_train)
# print(X_train.shape[0])
# print(X_train)

x_trainer, x_test, y_trainer, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=2)
# print("x trainer: {} x test: {} y trainer: {} y test: {}".format(x_trainer.shape, x_test.shape, y_trainer.shape, y_test.shape))
n_entradas = x_trainer.shape[1]
# print(x_trainer.shape)


t_x_train = torch.from_numpy(x_trainer).float().to("cpu") #'cuda' 'mps' 'cpu'
t_x_test = torch.from_numpy(x_test).float().to("cpu")
t_y_train = torch.from_numpy(y_trainer.values).float().to("cpu")
t_y_test = torch.from_numpy(y_test.values).float().to("cpu")
t_y_train = t_y_train[:,None]
t_y_test = t_y_test[:,None]

# print(t_x_train.shape[0])
# Define a simple neural network
class Red(nn.Module):
    def __init__(self, n_entradas):
        super(Red, self).__init__()
        self.linear1 = nn.Linear(n_entradas, 52)
        self.linear2 = nn.Linear(52, 8)
        self.end = nn.Linear(8, 1)

    def forward(self, x):
        # Propagación hacia adelante
        pred_1 = torch.sigmoid(input=self.linear1(x))
        pred_2 = torch.sigmoid(input=self.linear2(pred_1))
        # x = self.relu(x)
        pred_f = torch.sigmoid(input=self.end(pred_2))
        return pred_f

lr = 0.001
epochs = 200
estatus_print = 100
estatus_print_temp = estatus_print

model = Red(n_entradas=n_entradas)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
historico = pd.DataFrame()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Configuración del DataLoader
train_dataset = TensorDataset(t_x_train, t_y_train)
test_dataset = TensorDataset(t_x_test, t_y_test)

# Utilizamos num_workers=0 para evitar problemas en Jupyter
train_loader = DataLoader(train_dataset, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, shuffle=False, num_workers=0)
async def train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs, estatus_print):
    # print("entrenando el modelo")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y.view(-1, 1))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for batch_X_test, batch_y_test in test_loader:
                outputs_test = model(batch_X_test)
                outputs_test_class = torch.round(outputs_test)
                correct = (outputs_test_class == batch_y_test.view(-1, 1)).sum()
                accuracy = 100 * correct / float(len(batch_y_test))

async def noseCompaeEstoyCansado():
    train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs, estatus_print)
    


            


# import numpy as np
# import torch
# import torch.nn as nn
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# # Define a simple neural network
# class Red(nn.Module):
#     def __init__(self, n_entradas):
#         super(Red, self).__init__()
#         # Capa de entrada con input_size nodos y capa oculta con 64 nodos
#         self.linear1 = nn.Linear(n_entradas, 52)
#         # Capa de entrada con input_size nodos y capa oculta con 64 nodos
#         self.linear2 = nn.Linear(52, 8)
#         # # Función de activación ReLU
#         # self.relu = nn.ReLU()
#         # Capa de salida con 1 nodo (sin función de activación)
#         self.end = nn.Linear(8, 1)

#     def forward(self, x):
#         # Propagación hacia adelante
#         pred_1 = torch.sigmoid(input=self.linear1(x))
#         pred_2 = torch.sigmoid(input=self.linear2(pred_1))
#         # x = self.relu(x)
#         pred_f = torch.sigmoid(input=self.end(pred_2))
#         return pred_f
# model = Red(53)  # Asegúrate de proporcionar el mismo tamaño de entrada que se usó durante el entrenamiento
# model.load_state_dict(torch.load("modelo_entrenado.pth"))

# # model = model.to(device='cuda')
# model.eval()


# # Datos de entrada
# data = [
#         'F',
#         'Y',
#         'N', 
#         0, 
#         315000,
#         'Commercial associate', 
#         'Higher education', 
#         'Married', 
#         'House / apartment', 
#         -13557, 
#         -586, 
#         1, 
#         1, 
#         1, 
#         0, 
#         None, 
#         None, 
#         2]
# # Convertir la lista en un DataFrame de Pandas
# # Nombres de las columnas

# # Crear el diccionario con el mapeo de columnas
# columns = {
#     'CHILDREN': 1 if data[2] == 'Y' else 0,
#     'Annual_income': 1 if data[2] == 'N' else 0,
#     'Birthday_count': data[3],  # Ajusta según la posición correcta
#     'Employed_days': data[4],  # Ajusta según la posición correcta
#     'Mobile_phone': data[5],  # Ajusta según la posición correcta
#     'Work_Phone': data[6],  # Ajusta según la posición correcta
#     'Phone': data[7],  # Ajusta según la posición correcta
#     'EMAIL_ID': data[8],  # Ajusta según la posición correcta
#     'Family_Members': data[9],  # Ajusta según la posición correcta
#     'GENDER_F': 1 if data[0] == 'F' else 0,
#     'GENDER_M': 1 if data[0] == 'M' else 0,
#     'Car_Owner_N': 1 if data[1] == 'N' else 0,
#     'Car_Owner_Y': 1 if data[1] == 'Y' else 0,
#     'Propert_Owner_N': 1 if data[2] == 'N' else 0,
#     'Propert_Owner_Y': 1 if data[2] == 'Y' else 0,
#     'Type_Income_Commercial associate': 1 if data[3] == 'Commercial associate' else 0,
#     'Type_Income_Pensioner': 1 if data[3] == 'Pensioner' else 0,
#     'Type_Income_State servant': 1 if data[3] == 'State servant' else 0,
#     'Type_Income_Working': 1 if data[3] == 'Working' else 0,
#     'EDUCATION_Academic degree': 1 if data[4] == 'Academic degree' else 0,
#     'EDUCATION_Higher education': 1 if data[4] == 'Higher education' else 0,
#     'EDUCATION_Incomplete higher': 1 if data[4] == 'Incomplete higher' else 0,
#     'EDUCATION_Lower secondary': 1 if data[4] == 'Lower secondary' else 0,
#     'EDUCATION_Secondary / secondary special': 1 if data[4] == 'Secondary / secondary special' else 0,
#     'Marital_status_Civil marriage': 1 if data[5] == 'Civil marriage' else 0,
#     'Marital_status_Married': 1 if data[5] == 'Married' else 0,
#     'Marital_status_Separated': 1 if data[5] == 'Separated' else 0,
#     'Marital_status_Widow': 1 if data[5] == 'Widow' else 0,
#     'Housing_type_Co-op apartment': 1 if data[6] == 'Co-op apartment' else 0,
#     'Housing_type_House / apartment': 1 if data[6] == 'House / apartment' else 0,
#     'Housing_type_Municipal apartment': 1 if data[6] == 'Municipal apartment' else 0,
#     'Housing_type_Office apartment': 1 if data[6] == 'Office apartment' else 0,
#     'Housing_type_Rented apartment': 1 if data[6] == 'Rented apartment' else 0,
#     'Housing_type_With parents': 1 if data[6] == 'With parents' else 0,
#     'Type_Occupation_Accountants': 1 if data[7] == 'Accountants' else 0,
#     'Type_Occupation_Cleaning staff': 1 if data[7] == 'Cleaning staff' else 0,
#     'Type_Occupation_Cooking staff': 1 if data[7] == 'Cooking staff' else 0,
#     'Type_Occupation_Core staff': 1 if data[7] == 'Core staff' else 0,
#     'Type_Occupation_Drivers': 1 if data[7] == 'Drivers' else 0,
#     'Type_Occupation_HR staff': 1 if data[7] == 'HR staff' else 0,
#     'Type_Occupation_High skill tech staff': 1 if data[7] == 'High skill tech staff' else 0,
#     'Type_Occupation_IT staff': 1 if data[7] == 'IT staff' else 0,
#     'Type_Occupation_Laborers': 1 if data[7] == 'Laborers' else 0,
#     'Type_Occupation_Low-skill Laborers': 1 if data[7] == 'Low-skill Laborers' else 0,
#     'Type_Occupation_Managers': 1 if data[7] == 'Managers' else 0,
#     'Type_Occupation_Medicine staff': 1 if data[7] == 'Medicine staff' else 0,
#     'Type_Occupation_Private service staff': 1 if data[7] == 'Private service staff' else 0,
#     'Type_Occupation_Realty agents': 1 if data[7] == 'Realty agents' else 0,
#     'Type_Occupation_Sales staff': 1 if data[7] == 'Sales staff' else 0,
#     'Type_Occupation_Secretaries': 1 if data[7] == 'Secretaries' else 0,
#     'Type_Occupation_Security staff': 1 if data[7] == 'Security staff' else 0,
#     'Type_Occupation_Waiters/barmen staff': 1 if data[7] == 'Waiters/barmen staff' else 0,
# }
# print("longitud", len(data))
# # Convertir el diccionario a un array estructurado de NumPy
# array_result_columns = np.rec.fromrecords(list(columns.keys()))
# # Convertir el diccionario a un array de NumPy
# array_result = np.array(list(columns.values()))
# print(array_result_columns[1])
# df = pd.DataFrame(array_result, columns=array_result_columns)
# print(df.shape)

# X_train = pd.get_dummies(df)
# # print(df)
# # print(X_train.head())
# escalador = StandardScaler()
# X_train = escalador.fit_transform(X_train)
# print(len(X_train))
# x_train_num = torch.from_numpy(X_train).float().to("cpu") #'cuda' 'mps' 'cpu'
# # print(x_train_num.head())
# predic =model(x_train_num)
# # print(predic)


# # Ver el resultado
# print(df)

# # Convertir 'data' a tensor de PyTorch
# data_tensor = torch.tensor(data, dtype=torch.float32)

# # Obtener predicciones del modelo
# with torch.no_grad():
#     model_output = model(data_tensor)

# # Convertir las salidas a probabilidades (puedes usar sigmoid si usaste BCEWithLogitsLoss)
# probabilities = torch.sigmoid(model_output)

# # Convertir las probabilidades a etiquetas (1 si la probabilidad es mayor o igual a 0.5, 0 de lo contrario)
# predicted_labels = (probabilities >= 0.5).float()

# # Convertir las predicciones a un array NumPy
# predicted_labels_np = predicted_labels.numpy()
# print(model_output)
# print(probabilities)
# print(predicted_labels)
# print(predicted_labels_np)
def preditcModel(t_X_test):
    if t_X_test is None:
        raise "dataframe erro to generate"
    tempResponse = random.randint(0,1)
    for batch_X_test, batch_y_test in test_loader:
        outputs_test = model(batch_X_test)
        outputs_test_class = torch.round(outputs_test)
    return tempResponse, outputs_test_class