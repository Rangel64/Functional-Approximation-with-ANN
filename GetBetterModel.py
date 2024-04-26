import pandas as pd
import numpy as np
import sys

def main():
    global dataSet_test, kfolds
    
    np.set_printoptions(threshold=sys.maxsize)

    best_r2_treinamento = -float('inf')
    best_dataframe_path = ''
    
    pathTreinamento = "treinamentos/"
    
    for neuronios in range(5, 15, 5):
        for alpha in range(10, 30, 10):
            for ciclos in range(1, 3, 1):
               
                for i in range(2):
                    nameTreinamento = '/teste_%d.csv'%(i)
                    
                    df = pd.read_csv(pathTreinamento+('/neuronios_%d'%(neuronios))+('/alpha_%d'%alpha)+('/ciclos_%d'%ciclos)+nameTreinamento)
                    
                    r2_treinamento = df['R2 treinamento'].iloc[0]
                    
                    if r2_treinamento > best_r2_treinamento:
                        
                        best_r2_treinamento = r2_treinamento
                        best_dataframe_path = pathTreinamento+('/neuronios_%d'%(neuronios))+('/alpha_%d'%alpha)+('/ciclos_%d'%ciclos)+nameTreinamento
            


    print(f'O melhor R2 treinamento encontrado foi {best_r2_treinamento} no dataframe {best_dataframe_path}')
    print('Finalizando...')


if __name__ == "__main__":
    main()

