import numpy as np
import torch
import torch.nn as nn
import pandas as pd

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
model = Red(53)  # Asegúrate de proporcionar el mismo tamaño de entrada que se usó durante el entrenamiento
model.load_state_dict(torch.load("modelo_entrenado.pth"))

# model = model.to(device='cuda')
model.eval()

# Supongamos que 'data' es tu array NumPy o lista de Python
# Tu array
data = [0.00000e+00, 1.80000e+05, -1.87720e+04, 3.65243e+05, 1.00000e+00,
              0.00000e+00, 0.00000e+00, 0.00000e+00, 2.00000e+00, 1.00000e+00,
              1.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00,
              1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00,
              0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00,
              0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
              0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
              0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
              0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00]

# Convertir 'data' a tensor de PyTorch
data_tensor = torch.tensor(data, dtype=torch.float32)

# Obtener predicciones del modelo
with torch.no_grad():
    model_output = model(data_tensor)

# Convertir las salidas a probabilidades (puedes usar sigmoid si usaste BCEWithLogitsLoss)
probabilities = torch.sigmoid(model_output)

# Convertir las probabilidades a etiquetas (1 si la probabilidad es mayor o igual a 0.5, 0 de lo contrario)
predicted_labels = (probabilities >= 0.5).float()

# Convertir las predicciones a un array NumPy
predicted_labels_np = predicted_labels.numpy()
print(model_output)
print(probabilities)
print(predicted_labels)
print(predicted_labels_np)