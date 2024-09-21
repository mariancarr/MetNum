import numpy as np

# Ej 1. a)

#Para la eliminacion Gaussiana sin pivote tengo dos casos en los que no habria solucion: 
#1-si tengo mas incognitas que ecuaciones es decir si la matriz y b son de diferentes tamaños
#2-si tengo un 0 en la diagonal pues hay deberia hacer un intercambio de filas pero para eso necesito el pivote.

##from errores import NoPuedoEncontrarSolucion 

#aca lo que busco resolver es Ax = b. Siendo A una matriz, y b un vector

epsilon = 1e-06

def eliminacionGaussiana_SP(matriz, b):
    a = copiar(matriz)
    tamA = len(a)
    tamB = len(b)
    if(not esMatrizCuadrada(a)):
        raise Exception("No hay solución: La matriz no es cuadrada")
    elif (tamA != tamB):
        raise Exception("No hay solución: El tamaño de la matriz y el vector son incompatibles")
    else:
        #Agrego b a la matriz A asi puedo empezar a hacer las operaciones aritmeticas correspondientes
        i = 0
        for i in range(i, tamA):
            a[i].append(b[i])

        i = 0
        for i in range(tamA - 1):
            j = i + 1
            #aca chequeo si hay un 0 en la diagonal.Si lo hay, entonces debe haber todo 0 en la columna debajo de este, pues si no, no hay solucion.
            for j in range(j, tamA):
                if (np.allclose(a[i][i], 0, rtol=epsilon)):
                    k = i +  1

                    while (k < tamA and np.allclose(a[k][i], 0, rtol=epsilon)):
                        k += 1
                    #Aca si en la diagonal tengo un 0 y debajo de este todo es 0 no pasa nada pero si hay un numero diferente de 0 en la columna (al ser sin pivoteo) no tengo solucion
                    if (k != tamA):
                        raise Exception("No hay solución")
                    else:
                        break
                else:
                    c = a[j][i]/a[i][i]
                    k = i
                    while (k < tamA + 1): #aca es tamA + 1 porque tengo que cambiar el valor de b tmb
                        a[j][k] = a[j][k] - c* a[i][k]
                        k += 1

    
    #tengo q ver la ultima fila xq si me queda todo 0 entonces tengo infinitas soluciones. Pero si tengo todo 0 menos el b, no hay solucion
    if (np.allclose(a[tamA - 1][tamA - 1], 0, rtol=epsilon) and not np.allclose(a[tamA - 1][tamA], 0, rtol=epsilon)):
        raise Exception("No hay solucion")
        
    elif (np.allclose(a[tamA - 1][tamA - 1], 0, rtol=epsilon) and np.allclose(a[tamA - 1][tamA], 0, rtol=epsilon)):
        raise Exception("Hay infinitas soluciones")
    else:
        res = []
        x = a[tamA - 1][tamA] / a[tamA - 1][tamA - 1]
        res.append(x)

        fila = tamA - 2
        while (fila >= 0):
            x = a[fila][tamA]
            i = 0
            while i < len(res):
                x = x - a[fila][tamA - (i + 1)]* res[i]
                i += 1
            x = x / a[fila][fila]
            res.append(x)
            fila -= 1
        resFinal = list(reversed(res))

        return resFinal

def copiar(m):
    a = []
    for i in range(len(m)):
        a.append(m[i].copy())
    return a

def esMatrizCuadrada(a):
    tamA = len(a)
    for i in range(tamA):
        if(len(a[i]) != tamA):
            return False
    return True
 


