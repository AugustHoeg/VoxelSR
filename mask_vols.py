# %% Packages
import os, glob
import numpy as np
#import qim3d
from scipy import ndimage
#import math
#import matplotlib.pyplot as plt

# %% File paths
data_path = "/work3/soeba/FEMurSR/lund_data/CAD*"
hr_files_paths = sorted(glob.glob(os.path.join(data_path, "HR/", "CAD*.npy")))
ms_files_paths = sorted(glob.glob(os.path.join(data_path, "MS/", "CAD*.npy")))

# %% Process volume(s)
for i in range(len(hr_files_paths)):
    print("Loading HR file: ", hr_files_paths[i])
    hr_vol = np.load(hr_files_paths[i])
    print("HR shape: ", hr_vol.shape)
    print("Loading MS file: ", ms_files_paths[i])
    ms_vol = np.load(ms_files_paths[i])
    print(f"4xMS shape: ({ms_vol.shape[0]*4}, {ms_vol.shape[1]*4}, {ms_vol.shape[2]*4})")

    print("Reshaping mask to HR size...")
    ms_vol_hr = ndimage.zoom(ms_vol, 4, order=0)
    
    print("Masking volume...")
    masked_hr = np.zeros(ms_vol_hr.shape)

    x_min = min(ms_vol_hr.shape[0],hr_vol.shape[0])
    y_min = min(ms_vol_hr.shape[1],hr_vol.shape[1])
    z_min = min(ms_vol_hr.shape[2],hr_vol.shape[2])
    masked_hr[:x_min,:y_min,:z_min] = hr_vol[:x_min,:y_min,:z_min] * ms_vol_hr[:x_min,:y_min,:z_min]
    print("Masked HR volume shape: ", masked_hr.shape)

    # plt.imshow(masked_hr[:,:,masked_hr.shape[2]//2],cmap='gray')
    # plt.show()

    print("Saving masked volume as npy at: ", hr_files_paths[i][:-4] + "_masked.npy")
    np.save(hr_files_paths[i][:-4] + "_masked.npy", masked_hr)
    # print("Saving masked volume as tif at: ", hr_files_paths[i][:-4] + "_masked.tif")
    # qim3d.io.save(hr_files_paths[i][-4] + "_masked.tif", masked_hr, replace=True)
    print("Done.")
# %%
