# %% Packages
import os, glob
import numpy as np
#import qim3d
from scipy import ndimage
#import math
#import matplotlib.pyplot as plt

# %% File paths
data_path = "/work3/soeba/FEMurSR/lund_data/CAD*"
lr_files_paths = sorted(glob.glob(os.path.join(data_path, "LR/", "CAD*.npy")))
ms_files_paths = sorted(glob.glob(os.path.join(data_path, "MS/", "CAD*.npy")))

# %% Process volume(s)
for i in range(len(lr_files_paths)):
    print("Loading LR file: ", lr_files_paths[i])
    lr_vol = np.load(lr_files_paths[i])
    print("LR shape: ", lr_vol.shape)
    print("Loading MS file: ", ms_files_paths[i])
    ms_vol = np.load(ms_files_paths[i])
    print(f"MS shape: ({ms_vol.shape[0]}, {ms_vol.shape[1]}, {ms_vol.shape[2]})")
    
    print("Masking volume...")
    masked_lr = lr_vol * ms_vol

    print("Saving masked volume as npy at: ", lr_files_paths[i][:-4] + "_masked.npy")
    np.save(lr_files_paths[i][:-4] + "_masked.npy", masked_lr)

    print("Reshaping mask to HR size...")
    ms_vol_hr = ndimage.zoom(ms_vol, 4, order=0)

    print("Saving HR-sized mask as npy at: ", ms_files_paths[i][:-4] + "_HR.npy")
    np.save(ms_files_paths[i][:-4] + "_HR.npy", masked_lr)

    print("Done.")
# %%
