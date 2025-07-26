# %% Packages
import os, glob
import numpy as np
import skimage.filters as skf
#import matplotlib.pyplot as plt
import random

# %% File paths
data_path = "/work3/soeba/FEMurSR/lund_data/CAD*"
hr_files_paths = sorted(glob.glob(os.path.join(data_path, "HR/", "CAD[0-9][0-9][0-9].npy")))

# %% Process volume(s)
for i in range(1, len(hr_files_paths)):
    #i=0
    print("Loading HR file: ", hr_files_paths[i])
    hr_vol = np.load(hr_files_paths[i])
    print("HR datatype: ", hr_vol.dtype)
    print("HR shape: ", hr_vol.shape)

    # %% Get random subset of voxels and find threshold
    print("Sampling random subset of voxels...")
    subset_inds = random.sample(range(len(hr_vol[hr_vol>0])), 1000)
    subset_vol = hr_vol[hr_vol>0][subset_inds]
    print("Calculate threshold...")
    thresh = skf.threshold_mean(subset_vol)        #_isodata, _li, _mean, _minimum, _triangle, _yen, _otsu

    # %% Threshold volume
    print("Binarize volume with threshold...")
    vol_bin = hr_vol > thresh

    # %%
    # plt.imshow(vol_bin[:,:,vol_bin.shape[2]//2],cmap='gray')
    # plt.show()
    # plt.imshow(vol_bin[:,vol_bin.shape[1]//2,:],cmap='gray')
    # plt.show()
    # plt.imshow(vol_bin[vol_bin.shape[0]//2],cmap='gray')
    # plt.show()

    # %%
    print("Converting to uint16...")
    vol_bin = vol_bin.astype(np.uint16)

    print("Saving binarized volume as npy at: ", hr_files_paths[i][:-4] + "_bin.npy")
    np.save(hr_files_paths[i][:-4] + "_bin.npy", vol_bin)
    print("Done.")
# %%
