from eliminacionGaussiana import copiar, esMatrizCuadrada
import numpy as np

# Ej 2. a)

epsilon = 1e-06

def eliminacionGaussiana_PP(matriz, b,eps):
    a = copiar(matriz)
    tamA = len(a)
    tamB = len(b)
    flag_error = 0
    if(not esMatrizCuadrada(a)):
        raise Exception("No hay solución: La matriz no es cuadrada")
    elif (tamA != tamB):
        raise Exception("No hay solución: El tamaño de la matriz y el vector son incompatibles")  
    else:
        i = 0
        for i in range(i, tamA):
            a[i].append(b[i])

        i = 0
        for i in range(tamA - 1):
            #Hago pivoteo buscando el valor mas grande en cada iteracion
            valor_pivot = abs(a[i][i])
            pivot = i
            aux = i
 
            #si encuentro un |valor| mayor al q tengo guardado actualmente como mayor, lo cambio.
            for aux in range(i, tamA):
                if (abs(a[aux][i]) > abs(valor_pivot)):
                    pivot = aux
                    valor_pivot = abs(a[aux][i])

            #hago el cambio de filas, aca tengo q asegurarme de cambiar tmb el b 
            actual = a[i]
            nuevo = a[pivot]
            a[i] = nuevo
            a[pivot] = actual
                
            j = i + 1
            for j in range(j, tamA):
                if(abs(a[i][i]) <= eps):
                    flag_error += 1
                c = a[j][i]/ a[i][i]
                k = i
                while (k < tamA + 1): #aca es tamA + 1 porque tengo que cambiar el valor de b tmb
                    a[j][k] = a[j][k] - c* a[i][k]
                    k += 1
    
    if (np.allclose(a[tamA - 1][tamA - 1], 0, rtol=epsilon) and not np.allclose(a[tamA - 1][tamA], 0, rtol=epsilon)):
        raise Exception("No hay solucion")
    elif (np.allclose(a[tamA - 1][tamA - 1], 0, rtol=epsilon) and np.allclose(a[tamA - 1][tamA], 0, rtol=epsilon)):
        raise Exception("Hay infinitas soluciones")
        
    else:
        res = []
        if(abs(a[tamA - 1][tamA - 1]) <= eps):
            flag_error += 1
        x = a[tamA - 1][tamA] / a[tamA - 1][tamA - 1]
        res.append(x)
    
    #si tengo 0 en una diagonal hay infinitas soluciones creo
    fila = tamA - 2
    while (fila >= 0):
        if (np.allclose(a[fila][fila], 0, rtol=epsilon)):
            raise Exception("Hay infinitas soluciones")
        else:
            x = a[fila][tamA]
            i = 0
            while i < len(res):
                x = x - a[fila][tamA - (i + 1)]* res[i]
                i += 1
            if(abs(a[fila][fila]) <= eps):
                flag_error += 1
            x = x / a[fila][fila]
            res.append(x)
            fila -= 1

    resFinal = list(reversed(res))
    if(flag_error != 0):
        print("ADVERTENCIA: Puede haber errores por dividir por numeros cercanos a 0")
    #print("RESULTADO : ", resFinal)
    return resFinal


    
