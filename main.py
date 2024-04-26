
import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import pickle
import os

class Perceptron:
    entrada = None
    target = None
    
    entrada_test = None
    target_test = None
    
    entradas = None
    saidas = None
    
    alpha = None
    
    neuronios = None
    
    camadaX = None
    camadaA = None
    camadaB = None
    camadaC = None
    camadaY = None
    
    A = None
    B = None
    C = None
    Y = None
    list_Y = None
    
    XA = None
    AB = None
    BC = None
    CY = None
    
    
    biasA = None
    biasB = None
    biasC = None
    biasY = None
    
    steps = None
    
    targetsTeste = []
    
    y_pred = []
    y_pred_test = []
    
    ciclo = 0
    
    erroTotal = np.Infinity
    erroTotal_test = np.Infinity
    trained = 0
    
    listaCiclo = []
    erros = []
    
    listaCiclo_test = []
    erros_test = []
    
    firstErro = 0
    
    erro_anterior = 0
     
    accuracy_train = 0
    accuracy_test = 0
    
    list_accuracy_train = []
    list_accuracy_test = []
    
    aleatorio = 0.5
    ciclos = None
    
    def get_weights_biases(self):
        return {
            'XA': self.XA,
            'AB': self.AB,
            'BC': self.BC,
            'CY': self.CY,
            'biasA': self.biasA,
            'biasB': self.biasB,
            'biasC': self.biasC,
            'biasY': self.biasY
        }
    
    def save_weights_biases(self, filename):
        
        weights_biases = self.get_weights_biases()
        
        with open(filename, 'wb') as file:
            pickle.dump(weights_biases, file)
    
    def load_weights_biases(self, filename):
        with open(filename, 'rb') as file:
            weights_biases = pickle.load(file)
            self.XA = weights_biases['XA']
            self.AB = weights_biases['AB']
            self.BC = weights_biases['BC']
            self.FY = weights_biases['CY']
            self.biasA = weights_biases['biasA']
            self.biasB = weights_biases['biasB']
            self.biasC = weights_biases['biasC']
            self.biasY = weights_biases['biasY']
            
    def relu(self,resultadosPuros):
        linhas,colunas = resultadosPuros.shape
        
        resultados = np.zeros((linhas,colunas))
        
        for i in range(linhas):
            for j in range(colunas):
                if(resultadosPuros[i][j] >= 0):
                    resultados[i,j] = resultadosPuros[i][j]
                else:
                    resultados[i,j] = 0
        
        return resultados
    
    def derivadaRelu(self,resultado):
        linhas, colunas = resultado.shape
        resultadoDerivada = np.zeros((linhas,colunas))
        
        for i in range(linhas):
            for j in range(colunas):
                if(resultado[i][j]>=0):
                    resultadoDerivada[i][j] = 1
                else:
                    resultadoDerivada[i][j] = 0
                    
        return resultadoDerivada
    
    def adaptive_learning_rate(self, erro):
        
        deltaErro = erro - self.erro_anterior
        
        if deltaErro < 0.5:
            self.alpha = self.alpha * 1.5  # Reduza a taxa de aprendizagem
        else:
            self.alpha = self.alpha * 0.5  # Aumente a taxa de aprendizagem

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def derivadaSigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def reset(self):
        print('================================')
        print('reinicando os pesos!!!')
        print('================================')
         
        for i in range(self.entradas):
            for j in range(self.neuronios):
                self.XA[i][j] = rd.uniform(-self.aleatorio,self.aleatorio)
                   
        for j in range(self.neuronios):
            self.biasA[0][j] = rd.uniform(-self.aleatorio,self.aleatorio)
        
        for j in range(self.neuronios):
            for k in range(self.neuronios):
                self.AB[j][k] = rd.uniform(-self.aleatorio,self.aleatorio)
                   
        for k in range(self.neuronios):
            self.biasB[0][k] = rd.uniform(-self.aleatorio,self.aleatorio)
            
        for k in range(self.neuronios):
            for l in range(self.neuronios):
                self.BC[k][l] = rd.uniform(-self.aleatorio,self.aleatorio)
                   
        for l in range(self.neuronios):
            self.biasC[0][l] = rd.uniform(-self.aleatorio,self.aleatorio)
            
           
        for l in range(self.neuronios):
            for m in range(self.saidas):
                self.CY[l][m] = rd.uniform(-self.aleatorio,self.aleatorio)
                   
        for m in range(self.saidas):
            self.biasY[0][m] = rd.uniform(-self.aleatorio,self.aleatorio)
               
        self.listaCiclo = []
        self.erros = []
        
        self.listaCiclo_test = []
        self.erros_test = []
        self.list_Y = []
        
        self.ciclo = 0
        
        self.erro = 1000
        
        self.trained = 0
        
        self.firstErro = 0
        
    def __init__(self, entrada,target,neuronios,alpha,ciclos,steps):
        self.entrada = entrada
        self.target = target
        self.entradas = entrada.shape[1]
        self.saidas = target.shape[1]
        self.steps = steps
        self.alpha = alpha
        self.neuronios = neuronios
        self.ciclos = ciclos
        self.XA = np.zeros((self.entradas,self.neuronios))
        self.AB = np.zeros((self.neuronios,self.neuronios))
        self.BC = np.zeros((self.neuronios,self.neuronios))
        self.CY = np.zeros((self.neuronios,self.saidas))
        self.biasA = np.zeros((1,self.neuronios))
        self.biasB = np.zeros((1,self.neuronios))
        self.biasC = np.zeros((1,self.neuronios))
        self.biasY = np.zeros((1,self.saidas))
        self.list_Y = []
        self.camadaX = np.zeros((1,self.entradas))
        self.camadaA = np.zeros((1,self.neuronios))
        self.camadaB = np.zeros((1,self.neuronios))
        self.camadaC = np.zeros((1,self.neuronios))
        self.camadaY = np.zeros((1,self.saidas))
        self.listaCiclo = []
        self.erros = []
        self.listaCiclo_test = []
        self.erros_test = []
        self.firstErro = 0
        
    def train(self):
        if(self.trained==0):
            while self.ciclo<self.ciclos and self.erro>1:
                     
                self.camadaA = np.dot(np.asarray(self.entrada), self.XA) + self.biasA
    
                self.A = np.tanh(self.camadaA)
                
                self.camadaB = np.dot(self.A, self.AB) + self.biasB
    
                self.B = np.tanh(self.camadaB)
                
                self.camadaC = np.dot(self.B, self.BC) + self.biasC
    
                self.C = np.tanh(self.camadaC)
            
                self.camadaY = np.dot(self.C, self.CY) + self.biasY
    
                self.Y = self.camadaY
                
                self.list_Y.append(self.Y)
                
                self.erro = 0.5*np.sum((np.asarray(self.target)-self.Y)**2)
                      
                print('================================')
                print('Ciclo: ' + str(self.ciclo))
                print('Alpha: ' + str(self.alpha))
                print('Treinamento LMSE: ' + str(self.erro))
                print('================================')
                
                deltinhaCY = (np.asarray(self.target)-self.Y)
            
                deltinhaBC = np.dot(deltinhaCY,self.CY.T)*(1-self.C**2)
                
                deltinhaAB = np.dot(deltinhaBC,self.BC.T)*(1-self.B**2)
                
                deltinhaXA = np.dot(deltinhaAB,self.AB.T)*(1-self.A**2)
                
                  
                deltaCY = self.alpha * np.dot(deltinhaCY.transpose(),self.C)
                deltaBiasY = self.alpha * np.sum(deltinhaCY)  
                
                deltaBC = self.alpha * np.dot(deltinhaBC.transpose(),self.B)
                deltaBiasC = self.alpha * np.sum(deltinhaBC) 
                
                deltaAB = self.alpha * np.dot(deltinhaAB.transpose(),self.A)
                deltaBiasB = self.alpha * np.sum(deltinhaAB) 
                
                deltaXA = self.alpha * np.dot(deltinhaXA.transpose(),self.entrada)
                deltaBiasA = self.alpha * np.sum(deltinhaXA)
                
                self.CY = self.CY + deltaCY.transpose()
                self.biasY = self.biasY + deltaBiasY.transpose()
                
                self.BC = self.BC + deltaBC.transpose()
                self.biasC = self.biasC + deltaBiasC.transpose()
                
                self.AB = self.AB + deltaAB.transpose()
                self.biasB = self.biasB + deltaBiasB.transpose()
                
                self.XA = self.XA + deltaXA.transpose()
                self.biasA = self.biasA + deltaBiasA.transpose()
                    
                self.list_accuracy_train.append(self.accuracy_train)
                self.list_accuracy_test.append(self.accuracy_test)
                  
                self.listaCiclo.append(self.ciclo)

                self.erros.append(self.erro)
                             
                X = self.entrada.sort_index()
                Y = pd.DataFrame(self.Y)
                
                # Combinando os dataframes de treinamento e teste
                x_ = np.asarray(X["x"]).reshape(self.steps, self.steps)
                y_ = np.asarray(X["y"]).reshape(self.steps, self.steps)
                z_nn = np.asarray(Y).reshape(self.steps, self.steps)
                
                data = {
                    'Epoch': self.listaCiclo,
                    'Error': self.erros,
                }
                
                df = pd.DataFrame(data)

                # Plotar os gráficos
                fig = plt.figure(figsize=(18, 6))

                plt.subplot(1, 3, 1)
                sns.lineplot(x='Epoch', y='Error', data=df, label='Training Error')
                
                plt.xlabel('Epoch')
                plt.ylabel('Error')
                plt.title('Training Error over Epochs')
                # plt.ylim(0, 120)
                
                ax = fig.add_subplot(1, 3, 2, projection='3d')
                ax.plot_surface(x_, y_, z_nn, cmap='viridis')
                
                # Adicionando rótulos aos eixos
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('f')
                
                
                Y_real = self.target.sort_index()
                z_real = np.asarray(Y_real).reshape(self.steps, self.steps)
                
                ax = fig.add_subplot(1, 3, 3, projection='3d')
                
                ax.plot_surface(x_, y_, z_real, cmap='viridis')
                
                # Adicionando rótulos aos eixos
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('f')
                
                # Mostrando os gráficos
                plt.tight_layout()
                plt.show()
                
                self.ciclo = self.ciclo + 1
                
            self.trained = 1
        
        else:
            print('================================')
            print('Modelo ja treinado')
            print('================================')   
          
    def test(self, entrada):
        Y_list = []
        for i in range(entrada.shape[0]):
            camadaA = np.dot(entrada[i], self.XA) + self.biasA
            
            A = np.tanh(camadaA)
    
            camadaB = np.dot(A, self.AB) + self.biasB
            
            B = np.tanh(camadaB)
            
            camadaC = np.dot(B, self.BC) + self.biasC
            
            C = np.tanh(camadaC)
    
            camadaY = np.dot(C, self.CY) + self.biasY
            
            Y_list.append(camadaY)
    
        return np.asarray(Y_list)

def criar_diretorio(path):
    if(not os.path.exists(path)):
        os.makedirs(path)

def main():
    
    # Definindo os valores de x e y
    steps = 100
    x = np.linspace(-10, 10, steps)
    y = np.linspace(-10, 10, steps)
    x, y = np.meshgrid(x, y)
    
    # f = (np.sin(x/1.5) * np.sin(y/1.5)) / (x * y)

    f = np.sin(x/2) + np.sin(y/2)

    # f = 3*(np.e**(-(x**2+y**2)/0.5))

    # Transformando os arrays em dataframes
    df_inputs = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
    df_targets = pd.DataFrame({'f': f.flatten()})
    
    listaR2Treinamento = []
    listaR2 = []
    
    for neuronios in range(5, 15, 5):
        
        pathModel = "models/"
        pathTreinamento = "treinamentos/"

        criar_diretorio(pathModel)
        criar_diretorio(pathTreinamento)

        criar_diretorio(pathModel+('/neuronios_%d'%(neuronios)))
        criar_diretorio(pathTreinamento+('/neuronios_%d'%(neuronios)))

        for alpha in range(10, 30, 10):
            criar_diretorio(pathModel+('/neuronios_%d'%(neuronios))+('/alpha_%d'%alpha))
            criar_diretorio(pathTreinamento+('/neuronios_%d'%(neuronios))+('/alpha_%d'%alpha))

            for ciclos in range(100, 1500, 100):
                criar_diretorio(pathModel+('/neuronios_%d'%(neuronios))+('/alpha_%d'%alpha)+('/ciclos_%d'%ciclos))
                criar_diretorio(pathTreinamento+('/neuronios_%d'%(neuronios))+('/alpha_%d'%alpha)+('/ciclos_%d'%ciclos))
                
                r2_treinamento_list = []

                for i in range(2):
                    model = Perceptron(df_inputs, df_targets,neuronios,alpha/10**(6),ciclos,steps)
                    model.reset()
                    model.train()
    
                    nameModel = '/model_teste_%d.h5'%(i)
                        
                    model.save_weights_biases(pathModel+('/neuronios_%d'%(neuronios))+('/alpha_%d'%alpha)+('/ciclos_%d'%ciclos)+nameModel)
                    
                    taxas2 = model.test(np.asarray(df_inputs))
                    taxas2 = taxas2.reshape((taxas2.shape[0],taxas2.shape[1]))
                    alvo2 = np.asarray(df_targets)
                    
                    
                    print(taxas2.shape)
                    print(alvo2.shape)
                    
                    r2_treinamento = round(r2_score(alvo2, taxas2), 5)
                    
                    if(r2_treinamento>1 or r2_treinamento<-1):
                        r2_treinamento = 0 
                    
                    r2_treinamento_list.append(r2_treinamento)
                    
                    listaR2Treinamento.append(r2_treinamento)
                    
            
                    listaR2.append(listaR2Treinamento) 

                    matrizR2 = np.asarray(listaR2)
                    matrizR2 = np.transpose(matrizR2)
            
                    df = pd.DataFrame(data=matrizR2, columns=['R2 treinamento'])
                    
                    nameTreinamento = '/teste_%d.csv'%(i)
                    
                    df.to_csv(pathTreinamento+('/neuronios_%d'%(neuronios))+('/alpha_%d'%alpha)+('/ciclos_%d'%ciclos)+nameTreinamento)
            
                    listaR2Treinamento.clear()
                    listaR2.clear()
                    
                # Calculando a média, desvio padrão e o maior valor de cada lista
                media_treinamento = sum(r2_treinamento_list) / len(r2_treinamento_list)
                desvio_padrao_treinamento = (sum([(x-media_treinamento)**2 for x in r2_treinamento_list]) / len(r2_treinamento_list))**0.5
                maximo_treinamento = max(r2_treinamento_list)
                           
                # Criando um DataFrame com esses valores
                df = pd.DataFrame({
                    'Metrica': ['Media', 'Desvio Padrao', 'Maximo'],
                    'Treinamento': [media_treinamento, desvio_padrao_treinamento, maximo_treinamento],
                })
                
                # Salvando o DataFrame em um arquivo CSV
                df.to_csv(pathTreinamento+('/neuronios_%d'%(neuronios))+('/alpha_%d'%alpha)+('/ciclos_%d'%ciclos)+'/Metricas.csv', index=False)  
                    

if __name__ == "__main__":
    main()


