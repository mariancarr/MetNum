import numpy as np
import zipfile
import os
from scipy import stats

# Ejercicio 1) a)
def knn(X_newtrain, X_dev, y_newtrain, y_dev, k):
    normas_x_dev = np.linalg.norm(X_dev, axis=1) #calculo la norma de cada fila al mismo tiempo. Devuelve un array donde la pos i es la norma 2 de la fila i de X_dev
    normas_x_new_train = np.linalg.norm(X_newtrain, axis=1)

    prod_escalar = X_dev @ X_newtrain.T      #la pos ij es el prod escalar entre la img de desarrollo i y la de entrenamiento j. tam=(1000x4000)

    m = (prod_escalar.T / normas_x_dev).T    #divido a cada fila por la norma de la img de desarrollo
    m = m / normas_x_new_train.reshape(1,-1) #divido a cada columna por la norma de la img de entrenamiento

    matriz_unos = np.ones((m.shape[0],m.shape[1]))

    dists_coseno = matriz_unos - m

    ordenadas = np.argsort(dists_coseno)
    primeros_k = np.delete(ordenadas,np.s_[k:],axis=1) #elimino de la columnas k hasta al final

    valores_en_y_new_train = y_newtrain[primeros_k] #me devuelve una matriz donde la fila i son los elementos en y_newtrain indexados por los numeros de la fila i de primeros_k
    modas = stats.mode(valores_en_y_new_train,axis=1).mode #calculo la moda de cada fila. Devuelve un array de cant_imgs(X_dev) de tamaño. El elem en la pos i del array debe ser comparado con el elem i de y_dev

    return np.count_nonzero(modas == y_dev) / X_dev.shape[0]

  # Ejercicio 3) a)
def exactitud_5():
    datos = cargarDatos()
    res = knn(datos[0], datos[1], datos[2], datos[3], 5)
    print("Exactitud 5: ", res)
    return res
   
def extraerDatos():    
    zipfilename = "datos.zip"
    
    # abro y extraigo todos los archivos del zip
    z = zipfile.ZipFile(zipfilename, "r")
    try:
        z.extractall()
    except:
        print('Error')
        pass
    z.close()

def cargarDatos():
    if(not os.path.isdir('datos')):
        extraerDatos()

    X_train = np.loadtxt("datos/X_train.csv", delimiter=",")
    y_train = np.loadtxt("datos/y_train.csv", delimiter=",").astype(int)
    X_test = np.loadtxt("datos/X_test.csv", delimiter=",")
    y_test = np.loadtxt("datos/y_test.csv", delimiter=",").astype(int)

   # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test

# Ejercicio 3) b)
# Validacion Cruzada para KNN
def validacionCruzadaKNN(X_train, y_train, k):
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

    for i in range(5):
        if(i == 0):
            x_junta = np.concatenate((x_2, x_3, x_4, x_5))
            y_junta = np.concatenate((y_2, y_3, y_4, y_5))
            x_desa = x_1
            y_desa = y_1
        elif(i == 1):
            x_junta = np.concatenate((x_1, x_3, x_4, x_5))
            y_junta = np.concatenate((y_1, y_3, y_4, y_5))
            x_desa = x_2
            y_desa = y_2
        elif(i == 2):
            x_junta = np.concatenate((x_1, x_2, x_4, x_5))
            y_junta = np.concatenate((y_1, y_2, y_4, y_5))
            x_desa = x_3
            y_desa = y_3
        elif(i == 3):
            x_junta = np.concatenate((x_1, x_2, x_3, x_5))
            y_junta = np.concatenate((y_1, y_2, y_3, y_5))
            x_desa = x_4
            y_desa = y_4
        else:
            x_junta = np.concatenate((x_1, x_2, x_3, x_4))
            y_junta = np.concatenate((y_1, y_2, y_3, y_4))
            x_desa = x_5
            y_desa = y_5

        performance[i] = knn(x_junta, x_desa, y_junta, y_desa, k)

    print(performance)
    return performance.mean()

            
def validacionCruzada_KNN():
    X_train, X_test, y_train, y_test = cargarDatos()
    exactitud = 0
    for k in range(1, 10):
        res = validacionCruzadaKNN(X_train, y_train, k)
        print("K: ", k)
        print("Exactitud: ", res)
        if(res > exactitud):
            exactitud = res
            mejorK = k
    print("exactitud del mejorK: ", exactitud)
    return mejorK
   
if __name__ == '__main__':
    # Prueba Ejercicio 3) a)
    print("Ejercicio 3) a)")
    exactitud_5()
    # Prueba Ejercicio 3) b)
    print("Ejercicio 3) b)")
    mejorK = validacionCruzada_KNN()
    print("mejorK: ", mejorK) # El mejor entre 1 y 10 es 4