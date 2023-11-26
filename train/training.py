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
print("Todas las columnas:")
print(columnas)
print(datos_x.head())
# print(X_train.head())
escalador = StandardScaler()
X_train = escalador.fit_transform(X_train)
# print(X_train.shape[0])
# print(X_train)

x_trainer, x_test, y_trainer, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=2)
print("x trainer: {} x test: {} y trainer: {} y test: {}".format(x_trainer.shape, x_test.shape, y_trainer.shape, y_test.shape))
n_entradas = x_trainer.shape[1]
print(x_trainer.shape)


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

train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs, estatus_print)

torch.save(model.state_dict(), "modelo_entrenado.pth")
