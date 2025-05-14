# %% Packages
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# %% Load data
vol_no = "CAD052"
vol_hr = np.load('/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/LUND_data/' + vol_no + '/HR/' + vol_no + '.npy', mmap_mode='r')
#vol_lr = nib.load('/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/LUND_data/' + vol_no + '/LR/' + vol_no + '.nii')
vol_ms = np.load('/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/LUND_data/' + vol_no + '/MS/' + vol_no + '.npy', mmap_mode='r')

# %% Print shapes
print(f'HR vol shape: {vol_hr.shape}')
#print(f'LR vol shape: {vol_ms.shape}')
print(f'4x LR vol shape: {(vol_ms.shape[0] * 4, vol_ms.shape[1] * 4, vol_ms.shape[2] * 4)}')

# %% Extract nib
vol_lr = vol_lr.get_fdata()

# %% Show slices
fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(vol_hr[vol_hr.shape[2] // 2], cmap='gray')
ax[1].imshow(vol_lr[vol_lr.shape[2] // 2], cmap='gray')
ax[2].imshow(vol_ms[vol_ms.shape[2] // 2], cmap='gray')
plt.show()
# %%
