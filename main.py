from perceptron import Perceptron
from database import Database,IRIS_VIRGINICA
from random import sample

# Customize Perceptron

trainRate = 0.5 # Proporção da base de dados destinada a treino
learnRate = 0.05 # Taxa de aprendizado do Perceptron
epochs = 120 # Número de repetições no treino do Perceptron

# Load database
database = Database()

# Remover classes não utilizadas
database.cleanDatabase(IRIS_VIRGINICA)

databaseTrain,databaseTest = database.separateDatabase(trainRate=trainRate)

# Create Perceptron

perceptron = Perceptron(nInputs=database.inputSize)
print("Pesos Iniciais: ",perceptron.weights)

# Train Perceptron

print("Treinando Perceptron...")
perceptron.train(databaseTrain,learnRate=learnRate,epoch=epochs)
print("Perceptron Treinado")
print("Pesos Finais: ",perceptron.weights)

# Test Perceptron
acertos = 0
erros = 0
newDatabaseTest = sample(databaseTest,k=len(databaseTest)) # Embaralhar base de teste Evitar dados todos em sequência

for test in newDatabaseTest:

    inputTest = test[0:-1]
    classReal = test[-1]
    prevision = perceptron.prever(inputTest)
    
    if(prevision == classReal):
        acertos += 1
    else:
        erros += 1

print(f"Acertos: {acertos}")
print(f"Erros: {erros}")
