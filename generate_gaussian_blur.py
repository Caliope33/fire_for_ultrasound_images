import numpy as np
import scipy.io as sio
import os

def save_gaussian_kernel(sigma=1.6, size=25, folder='kernels/gaussian'):
    os.makedirs(folder, exist_ok=True)
    
    # Création de la grille
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    kernel = kernel / np.sum(kernel) # Normalisation

    path = os.path.join(folder, f'blur_gauss.mat')
    sio.savemat(path, {'kernel': kernel})
    print(f"Noyau Gaussien (sigma={sigma}) sauvegardé dans : {path}")
save_gaussian_kernel(sigma=1.6)
