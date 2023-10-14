from perceptron import Perceptron
from database import Database,IRIS_VIRGINICA
from random import sample

def assertividade(max_acertos,mean_acertos):
    return (mean_acertos)/max_acertos

# Customize Perceptron

trainRate = 0.5 # Proporção da base de dados destinada a treino
learnRate = 0.05 # Taxa de aprendizado do Perceptron
epochs = 120 # Número de repetições no treino do Perceptron

# Load database
database = Database()

# Remover classes não utilizadas
database.cleanDatabase(IRIS_VIRGINICA)
databaseTrain,databaseTest = database.separateDatabase(trainRate=trainRate)

for param in range(10,200,10):
    
    epochs = param
    acertosTotal = []
    errosTotal = []
    
    for _ in range(30):
        
        # Create Perceptron

        perceptron = Perceptron(nInputs=database.inputSize)
        perceptron.train(databaseTrain,learnRate=learnRate,epoch=epochs)

        # Test Perceptron
        acertos = 0
        erros = 0
        newDatabaseTest = sample(databaseTest,k=len(databaseTest)) # Embaralhar base de teste
        for test in newDatabaseTest:
            inputTest = test[0:-1]
            classReal = test[-1]
            prevision = perceptron.prever(inputTest)
            if(prevision == classReal):
                acertos += 1
            else:
                erros += 1
        
        acertosTotal.append(acertos)
        errosTotal.append(erros)
    
    # Registrar Resultados
    with open("./resultadosTestes/tresclasses.txt","a+") as f:
        sum = 0
        for i in acertosTotal:
            sum += i
            
        mediaAcertos = sum/len(acertosTotal)
        maxAcertos = round((1-trainRate)*database.size)
        
        assertividadeTeste = assertividade(max_acertos=maxAcertos,mean_acertos=mediaAcertos)
        
        f.write(f"{epochs} {mediaAcertos} {assertividadeTeste}\n")
        f.close()
