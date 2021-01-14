import numpy as np #Numerical Python (Estructuras de datos y matrices)
import random 
import json 

import torch #libreria para el aprendizaje automatico, utiliza los tensorflow, ejecuta el codigo de forma nativa usando la GPU
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

#cargamos nuestra bd
with open('basededatos.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# recorrer cada oración en nuestros patrones de intenciones
for intent in intents['intenciones']:
    tag = intent['etiqueta']
    # agregar a la lista de etiquetas
    tags.append(tag)
    for pattern in intent['patrones']:
        # tokenizar cada palabra de la oración
        w = tokenize(pattern)
        # agregar a nuestra lista de palabras
        all_words.extend(w)
        # agregar al par xy
        xy.append((w, tag))

# ignorar palabra
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# eliminar duplicados y ordenar
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patrones")
print(len(tags), "etiquetas:", tags)
print(len(all_words), "palabras derivadas unicas:", all_words)

# crear datos de entrenamiento
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bolsa de palabras para cada patron de sentencia
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss solo necesita etiquetas de clase, no one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hiperparámetros
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # Admite la indexación de modo que el conjunto de datos [i] se pueda utilizar para obtener la i-ésima muestra
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # podemos llamar a len (conjunto de datos) para devolver el tamaño
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Pérdida y optimizacion
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Entrena el modelo
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Pase adelantado
        outputs = model(words)
        # si y sería one-hot, debemos aplicar
        # etiquetas = antorcha.max (etiquetas, 1) [1]
        loss = criterion(outputs, labels)
        
        # Retroceder y optimizar
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'interaciones [{epoch+1}/{num_epochs}], perdidos: {loss.item():.4f}')


print(f'perdida final: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Entrenamiento completado, archivo guardado en: {FILE}')