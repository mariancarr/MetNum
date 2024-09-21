import numpy as np
import matplotlib.pyplot as plt
from eliminacionGaussianaTridiagonalVectores import eliminacionGaussianaTridigonal

def difusionImplicita(alpha):
    n = 101
    n2 = n//2
    r = 10
    m = 1000

    a = np.full(n2 - r + 1, 0)
    b = np.full(n2 + r -(n2 - r + 1), 1)
    c = np.full(n2 - r + 1, 0)
    u = np.concatenate((a, b, c))
    diag_principal = np.diag(np.full(n, 2*alpha + 1)) 
    diag_inferior = np.diag(np.full(n-1, -alpha), -1)
    diag_superior = np.diag(np.full(n-1, -alpha),1)
    matriz_implicita =diag_principal + diag_inferior + diag_superior

    implicita_res = []
    i = 0
    while(i < m):
        implicita_res.append(u)
        u = eliminacionGaussianaTridigonal(matriz_implicita, u)
        i += 1

    implicita_res = np.transpose(implicita_res)
    return implicita_res

alphas = [0.25, 0.5, 1, 4]
fig, axs = plt.subplots(2, 2, sharex = True,sharey = True)
for i, alpha in enumerate(alphas):
    row = i // 2
    col = i % 2
    implicita_res = difusionImplicita(alpha)
    img = axs[row, col].imshow(implicita_res, cmap='viridis',aspect = 8)
    axs[row, col].set_title(f'Alpha = {alpha}')  


for ax in axs[:, 0]:
    ax.set_yticklabels(np.arange(120, -1, -20))
    
fig.text(0.5, 0.02, 'k', ha='center')
fig.text(0.04, 0.5, 'X', va='center', rotation='vertical')
fig.subplots_adjust(right = 0.85)
cbar_ax = fig.add_axes([0.87, 0.10, 0.05, 0.80])
cbar = axs[row, col].figure.colorbar(img, cax=cbar_ax)
cbar_ax.set_ylabel("u")


plt.show()

