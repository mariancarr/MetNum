from eliminacionGaussiana import esMatrizCuadrada
import numpy as np

#Ejercicio 3.b)

epsilon = 1e-06

def eliminacionGaussiana_tridiagonal_vectores(a, b, c, d):
    tamA = len(a)
    tamB = len(b)
    tamC = len(c)
    tamD = len(d)

    if(tamA == tamB and tamB == tamC and tamC == tamD):
        i = 0

        #tengo q guardar los c[i + 1] de cuando hago pivoteo, los voy a guardar en un vector aparte
        c_c = []

        for i in range(tamA - 1):
            if(b[i] == 0):
                if(a[i + 1] != 0):

                    #si hago pivoteo en la columna que estoy cambiando ya queda todo en 0 por lo tanto solo tengo q intercambiar las filas
                    temp = b[i]
                    b[i] = a[i + 1]
                    a[i + 1] = temp

                    temp2 = b[i + 1]
                    b[i + 1] = c[i]
                    c[i] = temp2

                    c_c.append(c[i + 1])
                    c[i + 1] = 0

                    temp3 = d[i]
                    d[i] = d[i + 1]
                    d[i + 1] = temp3

            else:
                cociente = a[i + 1]/ b[i]
                a[i + 1] = 0
                b[i + 1] = b[i + 1] - (cociente * c[i])
                d[i + 1] = d[i + 1] - (cociente * d[i])
                #si no hago pivoteo arriba del c[i+1] hay un 0 entonces no hace falta hacer nada

                c_c.append(0)

        res = []

        if (b[tamB - 1] == 0 and d[tamD - 1] != 0):
            raise Exception("No hay solucion")
            
        elif (b[tamA - 1] == 0 and d[tamA - 1] == 0):
            raise Exception("Hay infinitas soluciones")

        else:  
            x = d[tamD - 1]/ b[tamB - 1]
            res.append(x)


            fila = tamD - 2
            while (fila >= 0):

                #como no se de ante mano que filas intercambie uso c[fila] y c_c[fila] porque uno de los dos va a ser 0.
                x = d[fila] - (c[fila]* res[tamD - fila - 2] + c_c[fila]* res[tamD - 3 -fila])
                x = x / b[fila]
                res.append(x)
                fila -= 1
            
            resFinal = list(reversed(res))
        #   print("RESULTADO : ", resFinal)
            return resFinal


def eliminacionGaussianaTridigonal(m, d):
    if(not esMatrizCuadrada(m)):
        raise Exception("No hay solución: La matriz no es cuadrada")
    elif (len(m) != len(d)):
        raise Exception("No hay solución: El tamaño de la matriz y el vector son incompatibles")  
    else:
        v = convertirAVectores(m)
        
        if(len(v) == 3):
            return eliminacionGaussiana_tridiagonal_vectores(v[0], v[1], v[2], d)
        else: 
            raise Exception("No se pudieron generar los vectores de la matriz triagonal")

def convertirAVectores(m):
    f = len(m)
    a = list()
    b = list()
    c = list()
    for i in range(f):
        if i-1 >= 0:
            a.append(m[i][i-1]) 
        else:
             a.append(0)
        b.append(m[i][i])
        if i+1 < f:
            c.append(m[i][i+1])
        else:
            c.append(0)    
    
    return [a, b, c]
