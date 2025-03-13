import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cupyx_ndimage
import time
from structure_tensor.multiprocessing import parallel_structure_tensor_analysis

import torch

#from data.dataset import Dataset as D

if __name__ == "__main__":

    img = np.zeros((100, 100, 100))

    if True:
        filt = cupyx_ndimage.gaussian_filter(cp.array(img), sigma=1.0, order=1)
        print("cupyx.scipy.ndimage works!")

    if True:
        #S, val, vec = parallel_structure_tensor_analysis(img, 1.0, 2.0, devices=['cuda:0'], block_size=128)
        S, val, vec = parallel_structure_tensor_analysis(np.zeros((100, 100, 100)), 1.0, 2.0, devices=['cuda:0'],
                                                         block_size=128)
        #fanis = fractional_anisotropy(val, mask, return_vector=True)
        print("parallel_structure_tensor_analysis works!")

    print("test")