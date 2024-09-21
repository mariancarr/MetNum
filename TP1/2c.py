from eliminacionGaussianaPivot import eliminacionGaussiana_PP
import numpy as np
import matplotlib.pyplot as plt
import sys

epsilons = np.logspace(0,-6,num=30)

#GRAFICAR CON PUNTOS, SUPERPONER AMBOS GRAFICOS 
A = [[1,2,3],
     [1,2,3],
     [1,2,3]]

x = [1,1,1]

b = [6,6,6]

data32 =[]
data64 = []

for epsilon in epsilons:
    A1 = []
    valores = [
         [1,2+epsilon,3-epsilon], 
         [1-epsilon,2,3+epsilon],
         [1+epsilon,2-epsilon,3]
         ]
    for fila in valores:
        a = [np.float32(fila[i]) for i in range(len(fila))]
        A1.append(a)



    

    res32 = eliminacionGaussiana_PP(A1,b,1e-10)
    res64 = eliminacionGaussiana_PP(valores,b,1e-10)
    max_dif = -1
    i = 0
    for res_i in res32:
        if(abs(res_i-1) > max_dif):
            max_dif = abs(res_i-1)

    data32.append(max_dif)

    max_dif = -1
    i = 0
    for res_i in res64:
        if(abs(res_i-1) > max_dif):
            max_dif = abs(res_i-1)

    data64.append(max_dif)

#norms = np.array(data)    
plt.plot(epsilons,data32, 'ro', label='Error en 32 bits')
plt.xlabel('Epsilon')
plt.ylabel('Norma INF')
plt.xscale('log')
plt.yscale('log')

plt.plot(epsilons,data64, 'bo', label='Error en 64 bits')
plt.xscale('log')
plt.yscale('log')
plt.title('Error en 32 y 64 bits')
plt.legend()
plt.show()
