import os
import numpy as np
from ejecutarTest import guardarMatriz_file, ejecutar
from knn import cargarDatos, knn
from pca import pca, cambioBase

# Ejercicio 3) c)
def exa(X_train, y_train, X_test, y_test):
    x_1 = X_train[:1000,]
    x_2 = X_train[1000:2000,]
    x_3 = X_train[2000:3000,]
    x_4 = X_train[3000:4000,]
    x_5 = X_train[4000:5000,]

    y_1 = y_train[:1000,]
    y_2 = y_train[1000:2000,]
    y_3 = y_train[2000:3000,]
    y_4 = y_train[3000:4000,]
    y_5 = y_train[4000:5000,]

    performance = np.zeros(5)
    res1 = []
    res2 = []
    res3 = []
    res4 = []
    res5 = []
    resFinal = []

    for i in range(5):
        if(i == 0):
            x_junta = np.concatenate((x_2, x_3, x_4, x_5))
            pca1 = pca(x_junta, "matrizPCA_1.txt", "autovaloresPCA_1.txt", "autovectoresPCA_1.txt", "iteracionesPCA_1.txt")
            y_junta = np.concatenate((y_2, y_3, y_4, y_5))
            x_desa = x_1
            y_desa = y_1
        elif(i == 1):
            x_junta = np.concatenate((x_1, x_3, x_4, x_5))
            pca2 = pca(x_junta, "matrizPCA_2.txt", "autovaloresPCA_2.txt", "autovectoresPCA_2.txt", "iteracionesPCA_2.txt")
            y_junta = np.concatenate((y_1, y_3, y_4, y_5))
            x_desa = x_2
            y_desa = y_2
        elif(i == 2):
            x_junta = np.concatenate((x_1, x_2, x_4, x_5))
            pca3 = pca(x_junta, "matrizPCA_3.txt", "autovaloresPCA_3.txt", "autovectoresPCA_3.txt", "iteracionesPCA_3.txt")
            y_junta = np.concatenate((y_1, y_2, y_4, y_5))
            x_desa = x_3
            y_desa = y_3
        elif(i == 3):
            x_junta = np.concatenate((x_1, x_2, x_3, x_5))
            pca4 = pca(x_junta, "matrizPCA_4.txt", "autovaloresPCA_4.txt", "autovectoresPCA_4.txt", "iteracionesPCA_4.txt")
            y_junta = np.concatenate((y_1, y_2, y_3, y_5))
            x_desa = x_4
            y_desa = y_4
        else:
            x_junta = np.concatenate((x_1, x_2, x_3, x_4))
            pca5 = pca(x_junta, "matrizPCA_5.txt", "autovaloresPCA_5.txt", "autovectoresPCA_5.txt", "iteracionesPCA_5.txt")
            y_junta = np.concatenate((y_1, y_2, y_3, y_4))
            x_desa = x_5
            y_desa = y_5
    
        for k in range(1, 10):
            for p in [2, 3, 4, 5, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 784]: 
                if(i == 0):
                    x_hat_train = cambioBase(x_junta, pca1[1][:,:p]) 
                    x_hat_dev = cambioBase(x_desa, pca1[1][:,:p]) 
                    res1.append([p, k, knn(x_hat_train, x_hat_dev, y_junta, y_desa, k)])
                elif(i == 1):
                    x_hat_train = cambioBase(x_junta, pca2[1][:,:p]) 
                    x_hat_dev = cambioBase(x_desa, pca2[1][:,:p]) 
                    res2.append([p, k, knn(x_hat_train, x_hat_dev, y_junta, y_desa, k)])
                elif(i == 2):
                    x_hat_train = cambioBase(x_junta, pca3[1][:,:p]) 
                    x_hat_dev = cambioBase(x_desa, pca3[1][:,:p]) 
                    res3.append([p, k, knn(x_hat_train, x_hat_dev, y_junta, y_desa, k)])
                elif(i == 3):
                    x_hat_train = cambioBase(x_junta, pca4[1][:,:p]) 
                    x_hat_dev = cambioBase(x_desa, pca4[1][:,:p]) 
                    res4.append([p, k, knn(x_hat_train, x_hat_dev, y_junta, y_desa, k)])
                else:
                    x_hat_train = cambioBase(x_junta, pca5[1][:,:p]) 
                    x_hat_dev = cambioBase(x_desa, pca5[1][:,:p]) 
                    res5.append([p, k, knn(x_hat_train, x_hat_dev, y_junta, y_desa, k)])
                
    # Calculamos el promedio
    mejorPromedio = 0
    for j in range(len(res1)):
        res = (res1[j][2] + res2[j][2] + res3[j][2] + res4[j][2] + res5[j][2]) / 5
        if(mejorPromedio < res):
            mejorPromedio = res
            mejorP = res1[j][0]
            mejorK = res1[j][1]

    print("Mejor performance: ", mejorPromedio)    
    print("Mejor P: ", mejorP)
    print("Mejor K: ", mejorK)

    pcaFinal = pca(X_train, "matrizPCA.txt", "autovaloresPCA.txt", "autovectoresPCA.txt", "iteracionesPCA.txt")
    x_hat_trainFinal = cambioBase(X_train, pcaFinal[1][:,:mejorP]) 
    x_hat_test = cambioBase(X_test, pcaFinal[1][:,:mejorP]) 
    exactitud = knn(x_hat_trainFinal, x_hat_test, y_train, y_test, mejorK)
    print("Performance en datos de test: ", exactitud)
    return exactitud

def exa_ej3d():
    X_train, X_test, y_train, y_test = cargarDatos()
    exa(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    # Prueba Ejercicio 3) d)
    print("Ejercicio 3) d)")
    exa_ej3d()
  #  res1 = [[2, 1, 0.341], [3, 1, 0.509], [4, 1, 0.648], [5, 1, 0.669], [25, 1, 0.783], [50, 1, 0.801], [75, 1, 0.806], [100, 1, 0.804], [2, 2, 0.341], [3, 2, 0.523], [4, 2, 0.636], [5, 2, 0.663], [25, 2, 0.78], [50, 2, 0.792], [75, 2, 0.788], [100, 2, 0.79], [2, 3, 0.38], [3, 3, 0.55], [4, 3, 0.661], [5, 3, 0.68], [25, 3, 0.789], 
  #  [50, 3, 0.805], [75, 3, 0.808], [100, 3, 0.818], [2, 4, 0.389], [3, 4, 0.566], [4, 4, 0.664], [5, 4, 0.702], [25, 4, 0.796], [50, 4, 0.806],[75, 4, 0.799], [100, 4, 0.806], [2, 5, 0.4], [3, 5, 0.581], [4, 5, 0.682], [5, 5, 0.703],
  #  [25, 5, 0.8], [50, 5, 0.811], [75, 5, 0.809], [100, 5, 0.815]]
  #  res2 = [[2, 1, 0.335], [3, 1, 0.511], [4, 1, 0.634], [5, 1, 0.659], [25, 1, 0.809], [50, 1, 0.81], [75, 1, 0.813], [100, 1, 0.812], [2, 2, 0.36], [3, 2, 0.512], [4, 2, 0.624], [5, 2, 0.664], 
  #  [25, 2, 0.79], [50, 2, 0.803], [75, 2, 0.807], [100, 2, 0.812], [2, 3, 0.382], [3, 3, 0.532], [4, 3, 0.645], [5, 3, 0.702],
  #  [25, 3, 0.805], [50, 3, 0.82], [75, 3, 0.822], [100, 3, 0.821], [2, 4, 0.396], [3, 4, 0.554], [4, 4, 0.649], [5, 4, 0.71], [25, 4, 0.807], [50, 4, 0.817],
  #  [75, 4, 0.82], [100, 4, 0.813], [2, 5, 0.43], [3, 5, 0.562], [4, 5, 0.661], [5, 5, 0.709], [25, 5, 0.809], [50, 5, 0.825], [75, 5, 0.826], [100, 5, 0.827]]
  #  res3 = [[2, 1, 0.351], [3, 1, 0.514], [4, 1, 0.605], [5, 1, 0.666], [25, 1, 0.802], [50, 1, 0.801], [75, 1, 0.799], [100, 1, 0.804], [2, 2, 0.36], [3, 2, 0.513], 
  #  [4, 2, 0.616], [5, 2, 0.682], [25, 2, 0.802], [50, 2, 0.805], [75, 2, 0.797], [100, 2, 0.802], [2, 3, 0.378], [3, 3, 0.548], [4, 3, 0.646], [5, 3, 0.691], [25, 3, 0.799], [50, 3, 0.796], [75, 3, 0.797], [100, 3, 0.803], [2, 4, 0.396], [3, 4, 0.556], [4, 4, 0.661], [5, 4, 0.7], [25, 4, 0.807], [50, 4, 0.811]
  #  , [75, 4, 0.807], [100, 4, 0.816], [2, 5, 0.412], [3, 5, 0.556], [4, 5, 0.675], [5, 5, 0.702], [25, 5, 0.805], [50, 5, 0.801], [75, 5, 0.801], [100, 5, 0.8]
  #  ]
  #  res4 = [[2, 1, 0.351], [3, 1, 0.501], [4, 1, 0.622], [5, 1, 0.661], [25, 1, 0.806], [50, 1, 0.815], [75, 1, 0.818], [100, 1, 0.826], [2, 2, 0.358], 
  #  [3, 2, 0.497], [4, 2, 0.623], [5, 2, 0.678], [25, 2, 0.799], [50, 2, 0.812], [75, 2, 0.813], [100, 2, 0.813], [2, 3, 0.375], [3, 3, 0.551], [4, 3, 0.653], 
  #  [5, 3, 0.706], [25, 3, 0.803], [50, 3, 0.831], [75, 3, 0.829], [100, 3, 0.829], [2, 4, 0.397], [3, 4, 0.56], [4, 4, 0.671], [5, 4, 0.715], [25, 4, 0.821], 
  #  [50, 4, 0.827], [75, 4, 0.836], [100, 4, 0.837], [2, 5, 0.422], [3, 5, 0.566], [4, 5, 0.683], [5, 5, 0.707], [25, 5, 0.814], [50, 5, 0.82], [75, 5, 0.835], [100, 5, 0.828]]
  #  res5 = [[2, 1, 0.326], [3, 1, 0.505], [4, 1, 0.623], [5, 1, 0.672], [25, 1, 0.813], [50, 1, 0.812], [75, 1, 0.813], [100, 1, 0.812], [2, 2, 0.329], 
  #  [3, 2, 0.504], [4, 2, 0.624], [5, 2, 0.654], [25, 2, 0.799], [50, 2, 0.807], [75, 2, 0.807], [100, 2, 0.805], [2, 3, 0.348], [3, 3, 0.547], [4, 3, 0.665], 
  #  [5, 3, 0.699], [25, 3, 0.814], [50, 3, 0.825], [75, 3, 0.824], [100, 3, 0.825], [2, 4, 0.386], [3, 4, 0.558], [4, 4, 0.67], [5, 4, 0.699], [25, 4, 0.817], 
  #  [50, 4, 0.815], [75, 4, 0.818], [100, 4, 0.829], [2, 5, 0.381], [3, 5, 0.577], [4, 5, 0.665], [5, 5, 0.718], [25, 5, 0.823], [50, 5, 0.823], [75, 5, 0.827],
  #  [100, 5, 0.829]]

  #  mejorPromedio = 0
  #  for j in range(len(res1)):
  #      print("j ",j )
  #      res = (res1[j][2] + res2[j][2] + res3[j][2] + res4[j][2] + res5[j][2]) / 5
  #      if(mejorPromedio < res):
   #         mejorPromedio = res
    #        mejorP = res1[j][0]
     #       mejorK = res1[j][1]

   # print("Mejor performance: ", mejorPromedio)    
   # print("Mejor P: ", mejorP)
   # print("Mejor K: ", mejorK)

  #  pcaFinal = pca(X_train, "matrizPCA.txt", "autovaloresPCA.txt", "autovectoresPCA.txt", "iteracionesPCA.txt")
  #  x_hat_trainFinal = cambioBase(X_train, pcaFinal[1][:,:mejorP]) 
  #  x_hat_test = cambioBase(X_test, pcaFinal[1][:,:mejorP]) 
  #  exactitud = knn(x_hat_trainFinal, x_hat_test, y_train, y_test, mejorK)
  #  print("Performance en datos de test: ", exactitud)
    

