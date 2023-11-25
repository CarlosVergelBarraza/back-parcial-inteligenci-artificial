import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        # Capa de entrada con input_size nodos y capa oculta con 64 nodos
        self.fc1 = nn.Linear(input_size, 64)
        # Función de activación ReLU
        self.relu = nn.ReLU()
        # Capa de salida con 1 nodo (sin función de activación)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Propagación hacia adelante
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Cargar datos desde el archivo CSV
file_path = "resultado_de_merge.csv"  # Reemplaza con la ruta correcta a tu archivo CSV
df = pd.read_csv(file_path)

# Preprocesar los datos
# Convertir columnas categóricas en variables dummy
df = pd.get_dummies(df, columns=['GENDER', 'Car_Owner', 'Propert_Owner', 'Type_Income', 'EDUCATION', 'Marital_status', 'Housing_type', 'Type_Occupation'], drop_first=True)  # Usamos drop_first=True para evitar la colinealidad

# Convertir columnas booleanas a valores numéricos (True=1, False=0)
boolean_columns = df.select_dtypes(include=['bool']).columns
df[boolean_columns] = df[boolean_columns].astype('int32')

# Separar etiquetas (y_train) y características (X_train)
X_train = df.drop(['label', 'Ind_ID'], axis=1)
y_train = df['label']

# Convertir a tensores de PyTorch
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

# Crear un conjunto de datos utilizando TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# Crear un DataLoader para facilitar el manejo de los datos durante el entrenamiento
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Inicializar el modelo, la función de pérdida y el optimizador
input_size = X_train.shape[1]
model = SimpleNN(input_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
epochs = 100
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.view(-1, 1))
        loss.backward()
        optimizer.step()

# Guardar el modelo entrenado si es necesario
torch.save(model.state_dict(), "modelo_entrenado.pth")



# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Define a simple neural network
# class SimpleNN(nn.Module):
#     def _init_(self, input_size):
#         super(SimpleNN, self)._init_()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(64, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return x

# # Initialize the model and optimizer
# input_size = X_train.shape[1]
# model = SimpleNN(input_size)
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Train the model
# epochs = 100
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     outputs = model(X_train)
#     loss = criterion(outputs, y_train.view(-1, 1))
#     loss.backward()
#     optimizer.step()

# # Función de entrenamiento incremental
# def incremental_train(model, criterion, optimizer, new_X, new_y):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(new_X)
#     loss = criterion(outputs, new_y.view(-1, 1))
#     loss.backward()
#     optimizer.step()

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from torch.utils.data import Dataset, DataLoader

# # Cargar datos desde el archivo CSV
# data = pd.read_csv('datos.csv')

# # Codificar variables categóricas
# label_encoder = LabelEncoder()
# for column in ['GENDER', 'Car_Owner', 'Propert_Owner', 'Type_Income', 'EDUCATION', 'Marital_status', 'Housing_type', 'Type_Occupation']:
#     data[column] = label_encoder.fit_transform(data[column].astype(str))

# # Dividir los datos en conjuntos de entrenamiento y prueba
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# class CustomDataset(Dataset):
#     def __init__(self, dataframe):
#         self.data = dataframe.drop(['Ind_ID', 'label'], axis=1)
#         self.labels = dataframe['label']

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return torch.tensor(self.data.iloc[idx].values), torch.tensor(self.labels.iloc[idx])

# # Crear conjuntos de datos y cargadores
# train_dataset = CustomDataset(train_data)
# test_dataset = CustomDataset(test_data)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# import torch.nn as nn
# import torch.nn.functional as F

# class SimpleModel(nn.Module):
#     def __init__(self, input_size):
#         super(SimpleModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 2)  # Dos clases: aprobado o rechazado

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)
# import torch.optim as optim

# # Inicializar el modelo y el optimizador
# input_size = len(train_dataset[0][0])
# model = SimpleModel(input_size)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.NLLLoss()

# # Entrenamiento
# epochs = 10
# for epoch in range(epochs):
#     model.train()
#     for data, target in train_loader:
#         optimizer.zero_grad()
#         output = model(data.float())
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()

# # Evaluar en el conjunto de prueba
# model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for data, target in test_loader:
#         output = model(data.float())
#         _, predicted = torch.max(output.data, 1)
#         total += target.size(0)
#         correct += (predicted == target).sum().item()

# accuracy = correct / total
# print(f'Accuracy on test data: {accuracy}')