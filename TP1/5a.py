import numpy as np
import timeit
import matplotlib.pyplot as plt
from eliminacionGaussianaTridiagonalVectores import eliminacionGaussianaTridigonal
from LUTridiagonalVectores import factorizacionLU
from LUTridiagonalVectores import resolverSistemaLU
from eliminacionGaussianaPivot import eliminacionGaussiana_PP

# Ej 5
eps = 1e-8
tiempos_minimos_tridiagonal = [] #aca voy a ir guardando los tiempos minimos de cada iteracion de la tridiagonal
tiempos_minimos_pivoteo = [] #aca voy a ir guardando los tiempos minimos de cada iteracion de pivoteo
tamaños = [] #aca voy guardando los distintos tamaños que voy probando

#DESPUES PLOTEO CADA LISTA EN BASE AL TAMAÑO

# Generamos la matriz laplaciano de nxn
def generarMatrizLaplaciano(n):
    a = np.zeros((n,n))
    for i in range(n):
        a[i][i] = -2
        if(i > 0):
            a[i][i-1] = 1
        if(i < n-1):
            a[i][i+1] = 1
    return a
    
# Ej 5.a)

for i in range(1,50, 5):
    d_pivoteo = np.random.rand(i).tolist()
    d_tridiagonal = d_pivoteo.copy()

    laplaciano_pivoteo = generarMatrizLaplaciano(i).tolist()
    laplaciano_tridiagonal = laplaciano_pivoteo.copy()

    timesGaussTridiagonal = timeit.repeat('eliminacionGaussianaTridigonal(laplaciano_tridiagonal,d_tridiagonal)',
                                        'from __main__ import eliminacionGaussianaTridigonal,laplaciano_tridiagonal,d_tridiagonal',
                                        repeat=20,number=100)
    minimo_trid = min(timesGaussTridiagonal)

    timesGaussPivot = timeit.repeat('eliminacionGaussiana_PP(laplaciano_pivoteo,d_pivoteo,eps)',
                                    'from __main__ import eliminacionGaussiana_PP,laplaciano_pivoteo,d_pivoteo,eps',
                                    repeat=20,number=100)
    minimo_pivot = min(timesGaussPivot)
  
    tiempos_minimos_tridiagonal.append(minimo_trid)
    tiempos_minimos_pivoteo.append(minimo_pivot)
    tamaños.append(i)

plt.plot(tamaños,tiempos_minimos_pivoteo,'-ro',label='Usando Pivoteo')
plt.xlabel('tamaño')
plt.ylabel('tiempo') 
plt.xscale('log')
plt.yscale('log')

plt.plot(tamaños,tiempos_minimos_tridiagonal,'-bo',label='Usando Tridiagonal')
plt.xscale('log')
plt.yscale('log')

plt.title('Tiempos usando pivoteo vs tridiagonal')
plt.legend()
plt.show()
