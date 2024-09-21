import numpy as np
from ejecutarTest import guardarMatriz_file
from ejecutarTest import ejecutar
import matplotlib.pyplot as plt

def generate_matrizError(epsilon):
    autovalores = [10, 10 - epsilon, 5, 2, 1]
    D = np.diag(autovalores)
    v = np.random.rand(D.shape[0])*10
    
    v = v / np.linalg.norm(v)
    B = np.eye(len(v)) - 2 * np.outer(v, v)

    M = B.T @ D @ B
    
    return M

def convergencia(repes):
    matrices_path = '.'
    filename = "matrix.txt"
    epsilons = np.logspace(-4, 0, num=80)
    matriz_iterProm = np.zeros((5, len(epsilons)))
    matriz_errorProm = np.zeros((5, len(epsilons)))
    #matrices desvio estandar
    matriz_iterDE = np.zeros((5, len(epsilons)))
    matriz_errorDE = np.zeros((5, len(epsilons)))
    
    for i in range(epsilons.shape[0]):
        # n columnas por cantidad de repeticiones  con cada epsilon
        matrix_iter = np.zeros((5, repes))
    
        matrix_error = np.zeros((5,repes))
        print(i)
        
        for j in range(repes):
            M = generate_matrizError(epsilons[i])
            guardarMatriz_file(matrices_path, filename, M, 5)
            ejecutar(matrices_path, filename, 4000, "autovalores.txt", "autovectores.txt", "iteraciones.txt")
            
            test_iteraciones = np.loadtxt("iteraciones.txt")        #array con la cantidad de iteraciones que tardo cada caso (aca deberia hacer el promedio y guardarlo en otro para comparar desp?)
            test_values = np.loadtxt("autovalores.txt")             #array con los autovalores
            test_vectores = np.loadtxt("autovectores.txt")          #array con los autovectores
         
            """mido el error, hago todos los errores y busco el promedio"""
            errores = np.zeros(test_values.shape[0])
            for x in range(test_values.shape[0]):
                error = M @ test_vectores[:,x]  
                error = np.linalg.norm(error - test_values[x] * test_vectores[:,x])
                errores[x] = error
            
            for fila in range(5):
                matrix_iter[fila, j] = test_iteraciones[fila]
                matrix_error[fila, j] = errores[fila]   
                
        
        for f in range(5):
            matriz_iterProm[f, i] = np.mean(matrix_iter[f, :])   
            matriz_iterDE[f, i] = np.std(matrix_iter[f, :]) 
            matriz_errorProm[f, i] = np.mean(matrix_error[f, :])   
            matriz_errorDE[f, i] = np.std(matrix_error[f, :])  
       
    
    fig, axs = plt.subplots(5, 1, figsize=(10, 10))
    for fila in range(5):
        axs[fila].errorbar(epsilons, matriz_iterProm[fila, :], yerr=matriz_iterDE[fila, :], fmt='o-', color='r')
        axs[fila].set_xscale('log')
        axs[fila].set_xlabel('Epsilons')
        axs[fila].set_ylabel('Iteraciones Promedio')
        axs[fila].set_title(f'autovalor {fila+1}')
        axs[fila].grid(True)
    plt.tight_layout()
    plt.show()
    
    
    fig, axs = plt.subplots(5, 1, figsize=(10, 10))
    for fila in range(5):
        axs[fila].errorbar(epsilons, matriz_errorProm[fila, :], yerr=matriz_errorDE[fila, :], fmt='o-', color='b')
        axs[fila].set_xscale('log')
        axs[fila].set_yscale('log')
        axs[fila].set_xlabel('Epsilons')
        axs[fila].set_ylabel('Error Promedio')
        axs[fila].set_title(f'autovalor {fila+1}')
        axs[fila].grid(True)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    repes = 30
    convergencia(repes)
   



   


   
