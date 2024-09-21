import numpy as np
import matplotlib.pyplot as plt
from eliminacionGaussianaTridiagonalVectores import eliminacionGaussianaTridigonal

n = 101
n2 = n//2
r = 10
m = 1000

def difusionExplicita(alpha):
    a = np.full(n2 - r + 1, 0)
    b = np.full(n2 + r -(n2 - r + 1), 1)
    c = np.full(n2 - r + 1, 0)
    u = np.concatenate((a, b, c))
    diag_prin = np.diag(np.full(n, -2*alpha + 1)) 
    diag_inf = np.diag(np.full(n-1, alpha), -1)
    diag_sup = np.diag(np.full(n-1, alpha), 1)
    matriz_explicita = diag_prin + diag_inf + diag_sup


    explicita_res = []
    j = 0
    while(j < m):
        explicita_res.append(u)
        u = np.matmul(matriz_explicita, u)
        j += 1

    explicita_res = np.transpose(explicita_res)
    return explicita_res

alphas = [0.25, 0.5, 0.501, 0.502]
fig, axs = plt.subplots(2, 2, sharex = True,sharey = True)
for i, alpha in enumerate(alphas):
    row = i // 2
    col = i % 2
    explicita_res = difusionExplicita(alpha)
    img = axs[row, col].imshow(explicita_res, cmap='viridis',aspect = 8)
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

