import numpy as np
import scipy.io as sio
import os
from scipy.ndimage import rotate

def save_motion_kernel_batch(folder='kernels'):
    os.makedirs(folder, exist_ok=True)
    
    lengths = [11, 13, 15, 17, 19, 15, 13, 11] 
    angles = np.linspace(0, 157.5, 8)         
    
    for i, (l, a) in enumerate(zip(lengths, angles)):
        # Créer le noyau
        kernel = np.zeros((l, l))
        kernel[int((l - 1) / 2), :] = 1.0
        
        # Rotation
        kernel = rotate(kernel, a, reshape=False)
        kernel = kernel / kernel.sum() 
        
        file_id = i + 1
        path = os.path.join(folder, f'blur_{file_id}.mat')
        sio.savemat(path, {'blur': kernel})
        
    print(f"8 noyaux générés dans le dossier '{folder}'")

save_motion_kernel_batch()
