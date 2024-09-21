from eliminacionGaussiana import esMatrizCuadrada
from eliminacionGaussianaTridiagonalVectores import convertirAVectores
import numpy as np

# Ejercicio 3.c)

epsilon = 1e-06

# Obtenemos la matriz LU que se va a guardar en los mismos vectores de entrada
# Donde la matriz L va a estar conformada por los vectores (a, unos, ceros) 
# mientras que U se conformara con los vectores (ceros, b, c) donde 
# ceros es un vector de todos ceros, unos es un vector de todos unos; 
# y a, b son modificados por el algoritmo

def lu_tridiagonal_vectores(a, b, c):
    tamA = len(a)
    tamB = len(b)
    tamC = len(c)

    if(tamA == tamB and tamB == tamC):  
        for k in range(1, tamA):
            if(b[k-1] == 0):
                raise Exception("No existe factorizacion LU")    
            a[k] = a[k] / b[k-1]
            b[k] = b[k] - a[k] * c[k-1]
    else:
        raise Exception("El tamaño de los vectores son incompatibles")

def factorizacionLU(A):
    if(not esMatrizCuadrada(A)):
        raise Exception("No hay solución: La matriz no es cuadrada")
    else:
        v = convertirAVectores(A)
        if(len(v) == 3):
            lu_tridiagonal_vectores(v[0], v[1], v[2])
            return v
        else: 
            raise Exception("No se pudieron generar los vectores de la matriz triagonal")

def resolverSistemaLU(a, b, c, d):
    obtenerYDeMatrizL(a, d)
    obtenerXDeMatrizU(b, c, d)
    return d

    

# Resuelvo Ly = d
def obtenerYDeMatrizL(a, d):
    tamA = len(a)
    tamD = len(d)
    if(tamA == tamD and tamA > 0):
            for i in range(1, tamA):
                d[i] = d[i] - a[i] * d[i-1]

# Resuelvo Ux = y
def obtenerXDeMatrizU(b, c, y):
    tamB = len(b)
    tamC = len(c)
    tamY = len(y)
    if(tamB == tamC and tamC == tamY):
        if(tamB == 0):
            raise Exception("Los vectores no puden ser vacios") 
        
        if(b[tamB-1] == 0 and y[tamY-1] != 0):
            raise Exception("No hay solucion")   
        elif (b[tamB-1] == 0 and y[tamY-1] == 0):

            raise Exception("Hay infinitas soluciones")

        y[tamY-1] = y[tamY-1]/b[tamB-1]
        for i in range(tamB-2, -1, -1):
            if(b[i] == 0 and y[i] != 0):
                raise Exception("No hay solucion")   
            elif (b[i] == 0 and y[i] == 0):
                raise Exception("Hay infinitas soluciones")
            y[i] = (y[i] - c[i] * y[i+1])/ b[i]

def resolverSistemaConLU(m, d):
    if (len(m) != len(d)):
        raise Exception("No hay solución: El tamaño de la matriz y el vector son incompatibles")  
    else:
        v = factorizacionLU(m)
        d = resolverSistemaLU(v[0], v[1], v[2], d)
        return d
