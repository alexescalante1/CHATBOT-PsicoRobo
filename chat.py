import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

#declaramos el objeto torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#cargamos nuestra bd
with open('basededatos.json', 'r') as json_data:
    intents = json.load(json_data)

#cargamos losdatos de entrenamiento
FILE = "data.pth"
data = torch.load(FILE)

#obtenemos los paraetros
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

#crea el modelo de reconocimiento
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval() #evalua las respuestas


bot_name = "PsicoRobo"
print(f"{bot_name}: Mi nombre es PsicoRobo. Responderé a tus consultas, si desea salir, escriba adios")
while True:
    sentence = input("Tu: ") 
    if sentence == "adios":
        print(f"{bot_name}:Que tengas un buen día")
        break

    sentence = tokenize(sentence) #tokenizamos las palabras
    X = bag_of_words(sentence, all_words) #coincidencia de palabras con la bd json
    X = X.reshape(1, X.shape[0]) 
    X = torch.from_numpy(X).to(device) #Objeto de entrenamiento y posibles respuestas

    output = model(X) 
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()] #etiquetas y predicciones

    probs = torch.softmax(output, dim=1) #entrada de n dimensiones y rescalado de salidas de n dimensiones
    prob = probs[0][predicted.item()] #prediccion de la respuesta mas acertada
    if prob.item() > 0.75: #probabilidades mayores a 75%
        for intent in intents['intenciones']: #posibles intenciones
            if tag == intent["etiqueta"]:
                print(f"{bot_name}: {random.choice(intent['respuestas'])}")
    else:
        print(f"{bot_name}: No entiendo...")