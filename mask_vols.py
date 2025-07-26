# %% Packages
import os, glob
import numpy as np
#import qim3d
from scipy import ndimage
from skimage.morphology import binary_dilation
#import math
import matplotlib.pyplot as plt

# %% File paths
data_path = "/work3/soeba/FEMurSR/lund_data/CAD*"
hr_files_paths = sorted(glob.glob(os.path.join(data_path, "HR/", "CAD[0-9][0-9][0-9].npy")))
ms_files_paths = sorted(glob.glob(os.path.join(data_path, "MS/", "CAD[0-9][0-9][0-9].npy")))

# %% Process volume(s)
for i in range(1, len(hr_files_paths)):
    #i=0
    print("Loading HR file: ", hr_files_paths[i])
    hr_vol = np.load(hr_files_paths[i])
    print("HR datatype: ", hr_vol.dtype)
    print("HR shape: ", hr_vol.shape)
    print("Loading MS file: ", ms_files_paths[i])
    ms_vol = np.load(ms_files_paths[i])
    print("MS datatype: ", ms_vol.dtype)
    print(f"4xMS shape: ({ms_vol.shape[0]*4}, {ms_vol.shape[1]*4}, {ms_vol.shape[2]*4})")

    # %%
    print("Reshaping mask to HR size...")
    ms_vol_hr = ndimage.zoom(ms_vol, 4, order=0)

    # %%
    print("Performing 5 iterations of binary dilation on mask...")
    for _ in range(5):
        ms_vol_hr = binary_dilation(ms_vol_hr)

    # %%
    print("Masking volume...")
    masked_hr = np.zeros(ms_vol_hr.shape)

    # %%
    print("Finding indices to match shapes...")
    x_min = min(ms_vol_hr.shape[0],hr_vol.shape[0])
    y_min = min(ms_vol_hr.shape[1],hr_vol.shape[1])
    z_min = min(ms_vol_hr.shape[2],hr_vol.shape[2])
    masked_hr[:x_min,:y_min,:z_min] = hr_vol[:x_min,:y_min,:z_min] * ms_vol_hr[:x_min,:y_min,:z_min]
    print("Masked HR volume shape: ", masked_hr.shape)
    print("Masked HR datatype: ", masked_hr.dtype)

    # %%
    # plt.imshow(masked_hr[:,:,masked_hr.shape[2]//2],cmap='gray')
    # plt.show()
    # plt.imshow(masked_hr[:,masked_hr.shape[1]//2,:],cmap='gray')
    # plt.show()
    # plt.imshow(masked_hr[masked_hr.shape[0]//2],cmap='gray')
    # plt.show()

    # %% 
    print("Converting to uint16...")
    masked_hr = masked_hr.astype(np.uint16)

    # %%
    # print("Saving masked volume as npy at: ", hr_files_paths[i][:-4] + "_masked.npy")
    # np.save(hr_files_paths[i][:-4] + "_masked.npy", masked_hr)
    ## OVERWRITE CAD_HR file
    print("Saving masked volume as npy at: ", hr_files_paths[i][:-4] + ".npy")
    np.save(hr_files_paths[i][:-4] + ".npy", masked_hr)
    # print("Saving masked volume as tif at: ", hr_files_paths[i][:-4] + "_masked.tif")
    # qim3d.io.save(hr_files_paths[i][-4] + "_masked.tif", masked_hr, replace=True)
    print("Done.")
# %%
