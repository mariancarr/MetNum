import numpy as np
import timeit
import matplotlib.pyplot as plt
from eliminacionGaussianaTridiagonalVectores import eliminacionGaussianaTridigonal
from LUTridiagonalVectores import factorizacionLU
from LUTridiagonalVectores import resolverSistemaLU

# Ej 5
eps = 1e-8
tiempos_minimos_tridiagonal = [] 
tiempos_minimos_precomputo_trid = [] 
iteraciones = [] 

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
    
# Ej 5.b)
d = np.random.rand(40)
laplaciano_tridiagonal = generarMatrizLaplaciano(40)



laplaciano_tridiagonal = generarMatrizLaplaciano(40)
LU_times = timeit.repeat('factorizacionLU(laplaciano_tridiagonal)','from __main__ import factorizacionLU,laplaciano_tridiagonal',
                         repeat=100,number=800)

LU_min_time = min(LU_times)

LU = factorizacionLU(laplaciano_tridiagonal)
a = LU[0]
b = LU[1]
c = LU[2]


for i in range(1,1000,10):
    d_tridiagonal_precomputo = np.copy(d).tolist()
    d_tridiagonal = np.copy(d).tolist()
    timesGaussTridiagonal = timeit.repeat('eliminacionGaussianaTridigonal(laplaciano_tridiagonal,d_tridiagonal)',
                                          'from __main__ import eliminacionGaussianaTridigonal,laplaciano_tridiagonal,d_tridiagonal',
                                           repeat=50,number=800)

    minimo_trid = min(timesGaussTridiagonal)
    tiempos_minimos_tridiagonal.append(minimo_trid)
    timesGaussTridiagonal_precomputo = timeit.repeat('resolverSistemaLU(a,b,c,d_tridiagonal_precomputo)',
                                                     'from __main__ import resolverSistemaLU,a,b,c,d_tridiagonal_precomputo',
                                                     repeat=50,number=800)

    minimo_trid_precomputo = min(timesGaussTridiagonal_precomputo)
    if(i == 1):
        tiempos_minimos_precomputo_trid.append(minimo_trid_precomputo+LU_min_time)

    else:
        tiempos_minimos_precomputo_trid.append(minimo_trid_precomputo)

    iteraciones.append(i)


plt.plot(iteraciones,tiempos_minimos_precomputo_trid,'ro',label='Usando precomputo')
plt.xlabel('iteraciones')
plt.ylabel('tiempo') 
plt.xscale('log')
plt.yscale('log')

plt.plot(iteraciones,tiempos_minimos_tridiagonal,'bo',label='Sin precomputo')
plt.xscale('log')
plt.yscale('log')

plt.title('Tiempos usando tridiagonal vs LU')
plt.legend()
plt.show()
