
import numpy as np
import matplotlib.pyplot as plt
from LUTridiagonalVectores import factorizacionLU 
from LUTridiagonalVectores import resolverSistemaLU

n = 101
diag_principal = np.diag(np.ones(n)) 
diag_inferior = np.diag(np.ones(n-1), -1)
diag_superior = np.diag(np.ones(n-1), 1)

matriz_lapciana = diag_principal + diag_inferior + diag_superior
matriz_lapciana[np.diag_indices(n)] = -2

preComputo = factorizacionLU(matriz_lapciana)

a = preComputo[0]
b = preComputo[1]
c = preComputo[2]
n=101
dA = np.zeros(n)
dA[n//2 + 1] = 4 / n
resA = resolverSistemaLU(a, b, c, dA)

di = 4/ n**2
dB = np.full(n, di)
resB = resolverSistemaLU(a, b, c, dB)


#el vector i me queda [0,1,2,3,...,100] no se si tengo que hacerlo desde 0 o desde 1 pero se cambia facil
#desp uso la multiplicacion de numpy * que es de elemento a elemnto asi va variando el i en el vector dC luego
i = np.arange(n)
dC = (-1+ 2 * i/(n-1)) * 12 / (n**2)
resC = resolverSistemaLU(a, b, c, dC)

x = np.arange(101)
plt.plot(x, resC, color="limegreen", label='(c)')
plt.plot(x, resB, color="darkorange", label='(b)')
plt.plot(x, resA, color="steelblue", label='(a)')
plt.legend(loc="lower left")
plt.xlabel('X')
plt.yticks(np.arange(-1, 0.5, 0.25))
plt.ylabel('u')
plt.show()