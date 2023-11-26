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
# print(datos_x.head())

X_train = pd.get_dummies(datos_x)
# print(X_train.head())
escalador = StandardScaler()
X_train = escalador.fit_transform(X_train)
print(X_train.shape[0])
# print(X_train)

x_trainer, x_test, y_trainer, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=2)
print("x trainer: {} x test: {} y trainer: {} y test: {}".format(x_trainer.shape, x_test.shape, y_trainer.shape, y_test.shape))
n_entradas = x_trainer.shape[1]

t_x_train = torch.from_numpy(x_trainer).float().to("cpu") #'cuda' 'mps' 'cpu'
t_x_test = torch.from_numpy(x_test).float().to("cpu")
t_y_train = torch.from_numpy(y_trainer.values).float().to("cpu")
t_y_test = torch.from_numpy(y_test.values).float().to("cpu")
t_y_train = t_y_train[:,None]
t_y_test = t_y_test[:,None]

print(t_x_train.shape[0])
# Define a simple neural network
class Red(nn.Module):
    def __init__(self, n_entradas):
        super(Red, self).__init__()
        # Capa de entrada con input_size nodos y capa oculta con 64 nodos
        self.linear1 = nn.Linear(n_entradas, 52)
        # Capa de entrada con input_size nodos y capa oculta con 64 nodos
        self.linear2 = nn.Linear(52, 8)
        # # Función de activación ReLU
        # self.relu = nn.ReLU()
        # Capa de salida con 1 nodo (sin función de activación)
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
print("Arquitectura de modelo: {}".format(model))
historico = pd.DataFrame()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Configuración del DataLoader
train_dataset = TensorDataset(t_x_train, t_y_train)
test_dataset = TensorDataset(t_x_test, t_y_test)

# Utilizamos num_workers=0 para evitar problemas en Jupyter
train_loader = DataLoader(train_dataset, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, shuffle=False, num_workers=0)
# train_dataset = TensorDataset(t_x_train, t_y_train)
# train_loader = DataLoader(train_dataset, shuffle=True)
# # Entrenamiento del modelo
# for epoch in range(epochs):
#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y.view(-1, 1))
#         loss.backward()
#         optimizer.step()

# # Guardar el modelo entrenado si es necesario
# torch.save(model.state_dict(), "modelo_entrenado.pth")
def train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs, estatus_print):
    print("entrenando el modelo")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y.view(-1, 1))
            loss.backward()
            optimizer.step()

        if epoch % estatus_print == 0:
            print(f"\nEpoch {epoch} \t Loss: {round(loss.item(), 4)}")
            estatus_print = estatus_print_temp
        else:
            estatus_print = estatus_print -1
        model.eval()
        with torch.no_grad():
            for batch_X_test, batch_y_test in test_loader:
                outputs_test = model(batch_X_test)
                outputs_test_class = torch.round(outputs_test)
                correct = (outputs_test_class == batch_y_test.view(-1, 1)).sum()
                accuracy = 100 * correct / float(len(batch_y_test))

            if epoch % estatus_print == 0:
                print("Accuracy: {}".format(accuracy.item()))
                print(epoch)
                estatus_print = estatus_print_temp
            else:
                estatus_print = estatus_print -1



# print("entrenando el modelo")
# for epoch in range(1, epochs+1):
#     y_pred = model(t_x_train)
#     # Aplicar función de umbral (redondear)
#     # y_pred_binary = torch.round(y_pred)
#     print(f"resultado de prediccion {y_pred}")
#     print(f"deberia prediccion {t_y_train}")
#     loss = loss_fn(input=y_pred, target=t_y_train)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()

#     if epoch % estatus_print == 0:
#         print(f"\nEpoch {epoch} \t Loss: {round(loss.item(), 4)}")   
#     while torch.no_grad():
#         y_pred = model(t_x_test)
#         y_pred_class = y_pred.round()
#         correct = (y_pred_class == t_y_test).sum()
#         accuracy = 100*correct / float(len(t_y_test))
        
#         if epoch % estatus_print == 0:
#             print("Accoutacy: {}".format(accuracy.item()))  
#             print(epoch)
#             estatus_print = estatus_print_temp
#         else:
#             estatus_print = estatus_print -1
#     df_tpm = pd.DataFrame(
#         data={
#             'Epoch': epoch,
#             'Loss': round(loss.item(),4),
#             'Lossaccuracy': round(accuracy.item(),4)
#         }, index=[0]
#     )
#     historico = pd.concat(objs=[historico, df_tpm], ignore_index=True, sort=False)
# print("accuracy final: {}".format(round(accuracy.item(), 4)))
# Guardar el modelo entrenado si es necesario
# Entrenamiento del modelo
train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs, estatus_print)

torch.save(model.state_dict(), "modelo_entrenado.pth")







# # Convertir a tensores de PyTorch
# X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
# # print(X_train.values[0])
# Crear un conjunto de datos utilizando TensorDataset
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# # Crear un DataLoader para facilitar el manejo de los datos durante el entrenamiento
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Inicializar el modelo, la función de pérdida y el optimizador
# input_size = X_train.shape[1]
# print(input_size)

# model = SimpleNN(input_size)
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Entrenamiento del modelo
# epochs = 100
# for epoch in range(epochs):
#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y.view(-1, 1))
#         loss.backward()
#         optimizer.step()

# # Guardar el modelo entrenado si es necesario
# torch.save(model.state_dict(), "modelo_entrenado.pth")



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