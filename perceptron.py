from random import seed,uniform
from database import IRIS_SETOSA,IRIS_VIRGINICA,IRIS_VERSICOLOR
seed(123)

def generateWeights(nInputs:int) -> list:
    randomWeights = []
    for _ in range(nInputs):
        randomWeights.append(uniform(-1,1))
    return randomWeights

# funcao ativação 3 parametros
# def activeFunction(value):
#     if(value < -0.3334):
#         return IRIS_SETOSA
#     elif(value >= -0.3334 and value < 0.3332):
#         return IRIS_VERSICOLOR
#     else:
#         return IRIS_VIRGINICA
    
def activeFunction(value):
    if(value > 0):
        return IRIS_VERSICOLOR
    else:
        return IRIS_SETOSA

class Perceptron:
    def __init__(self,nInputs:int) -> None:
        self.weights = generateWeights(nInputs=nInputs)
        
    def train(self,data,learnRate,epoch):
        
        for _ in range(epoch):
            for line in data:
                inputData = line[0:-1]
                classData = line[-1]
                
                # Produto Escalar
                prod = 0
                for i in range(len(self.weights)):
                    prod += float(inputData[i]) * self.weights[i]
                
                # Função ativação
                prevClass = activeFunction(prod)
                
                # Atualizar pesos
                erro = classData - prevClass
                for i in range(len(self.weights)):
                    self.weights[i] = self.weights[i] + learnRate * erro * inputData[i]
                    prod += inputData[i] * self.weights[i]
        
    def prever(self,input):
        
        # Produto Escalar
        prod = 0
        for i in range(len(self.weights)):
            prod += input[i] * self.weights[i]
        
        # Função ativação
        return activeFunction(prod)
       