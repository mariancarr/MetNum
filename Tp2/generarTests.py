import numpy as np
import matplotlib.pyplot as plt
from ejecutarTest import generate_matrixTest
from ejecutarTest import guardarMatriz_file
from ejecutarTest import ejecutar

def tests(valores, niters, matrix_path, test):
    res = []
    results_testValues  = []
    results_realValues = []

    M, autovalores = generate_matrixTest(valores)
    guardarMatriz_file(matrix_path, "matrix.txt", M, valores)
    for iter in niters:
        ejecutar(matrix_path,"matrix.txt", iter, "autovalores.txt", "autovectores.txt", "iteraciones.txt")
        test_iteraciones = np.loadtxt("iteraciones.txt")
        test_values = np.loadtxt("autovalores.txt")
        results_testValues.append(test_values)
        results_realValues.append(autovalores)
        res.append(abs(test_values - autovalores)) 
        
        if(test == 1):
           
            if np.allclose(test_values, autovalores, rtol=1e-3, atol=1e-8):
                        
                print("autovalores coinciden. Matriz tamaño", valores, "x", valores, "niter = ", iter)
            else:
                print("autovalores No coinciden. Matriz tamaño", valores, "x", valores, "niter = ", iter)   
            
    
    
    if(test == 2):            
        fig, axs = plt.subplots(len(results_testValues), 1, figsize=(17, 6 * len(results_testValues)))
        
        for i in range(len(results_testValues)): 
            axs[i].scatter(range(len(results_realValues[i])), results_realValues[i], label='Valores reales', color='blue', s = 20)
            axs[i].scatter(range(len(results_testValues[i])), results_testValues[i], label='Valores calculados', color='red', marker='x', s = 20)
            axs[i].set_title(f'Autovalores para niter = {niters[i]}')
            axs[i].legend()

        plt.subplots_adjust(hspace=0.3)
        plt.show()
    
    if(test == 3):
        plt.boxplot(res)
        plt.yscale("log")
        plt.xlabel("Cantidad de iteraciones")
        plt.ylabel("Error")
        plt.title("correctitud metodo")
        plt.xticks(ticks = range(len(niters)+1), labels = [" "] + niters)
            
        plt.show()
       
    
    
if __name__ == '__main__':
    matrices_path = '.'
    valoresNpclose = [10, 25, 50, 100, 300, 784]
    niters1 = [5000]
    niters2 = [10, 100, 1000, 10000]
    niters3 = [10, 100, 1000, 10000, 20000]
    #valores para chequear usando np-allclose
    for i in valoresNpclose:
        tests(i, niters1, matrices_path, 1)

    #esto es para el primer grafico
    valores = 100
    tests(valores, niters2, matrices_path, 2)
    
    #valores para el segundo grafico
    valores = 500
    tests(valores, niters3, matrices_path, 3)


