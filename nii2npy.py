# %% Packages
import os, glob
import numpy as np
import nibabel as nib

# %% Load .nii-files and save as .npy
data_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/LUND_data/CAD*"
lr_files_paths = sorted(glob.glob(os.path.join(data_path, "LR/", "CAD*.nii")))
#print(lr_files_paths)
for lr_file in lr_files_paths:
    print(lr_file)
    lr_vol = nib.load(lr_file)
    lr_vol = lr_vol.get_fdata()
    np.save(lr_file[:-3] + "npy", lr_vol)
    print(lr_file[:-3] + "npy")
# %%
