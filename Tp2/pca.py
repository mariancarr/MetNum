import os
import numpy as np
from ejecutarTest import guardarMatriz_file, ejecutar
from knn import cargarDatos
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Ejercicio 3) c)
def pca(X, matriz, autovalores, autovectores, iteraciones):
    matrizCov = generarMatrizCovarianza(X)
#    print("matrizCov.shape ", matrizCov.shape)
    guardarMatriz_file(".", matriz, matrizCov, matrizCov.shape[0])
    if(not os.path.isfile(autovalores)):
        ejecutar(".", matriz, 10000, autovalores, autovectores, iteraciones)
    d = np.loadtxt(autovalores)
    v = np.loadtxt(autovectores) 
#    print("shape autovalores ", d.shape)
#    print("autovalores ", d)
#    print("shape autovectores ", v.shape)
#    print("autovectores ", v)

    # Valido resultados
    pca = PCA()
    Xhat = pca.fit_transform(X)
    pca_varianza = pca.explained_variance_
    if np.allclose(d, pca_varianza, rtol=1e-3, atol=1e-10): 
        print("autovalores coinciden")
    else:
        print("autovalores no coinciden")
 #   print("pca_varianza ", pca_varianza)

    return d, v


def pca_ej3c():
    datos = cargarDatos()
    return pca(datos[0], "matrizPCA.txt", "autovaloresPCA.txt", "autovectoresPCA.txt", "iteracionesPCA.txt")

# Generacion de matriz de covarianza
def generarMatrizCovarianza(a):
    a = centrado(a)
    return (a.T @ a) / (a.shape[0] - 1)

def centrado(a):
    promedio = a.mean(axis=0)
    a = a - promedio
    return a

# Generamos el gráfico que muestre la varianza acumulada explicada hasta "p" componentes principales
def varianzaAcumuladaVsP(varianza):
    plt.plot(np.cumsum(varianza) / np.sum(varianza))
    plt.ylabel("Varianza explicada acumulada")
    plt.xlabel('# componentes')
    plt.show()

def cambioBase(X, V):
    cent = centrado(X)
    cambioBase = cent @ V 
    #print("shape cambioBase ", cambioBase.shape)
    #print("Cambio base ", cambioBase)
    return cambioBase

if __name__ == '__main__':
    # Prueba Ejercicio 3) c)
    print("Ejercicio 3) c)")
    res = pca_ej3c()
    varianzaAcumuladaVsP(res[0])
