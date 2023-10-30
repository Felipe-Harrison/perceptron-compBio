from random import seed,uniform
from database import IRIS_SETOSA,IRIS_VIRGINICA,IRIS_VERSICOLOR
seed(123)

# Configurações 100% -> 
# [-0.21574777107043808, -0.27576151746860694, 0.7958571366259992, -0.023853694757445573]
# [-0.40574777107043836, -0.5557615174686068, 1.2258571366259994, -0.043853694757445716]
# [-0.3457477710704384, -0.9557615174686067, 1.4458571366259996, 0.10614630524255442]
def generateWeights(nInputs:int) -> list:
    randomWeights = []
    for _ in range(nInputs):
        randomWeights.append(uniform(-1,1))
    return randomWeights

# funcao ativação 3 classes
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
    def __init__(self,nInputs:int,weights = []) -> None:
        self.weights = generateWeights(nInputs=nInputs) if len(weights) == 0 else weights
        
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
       