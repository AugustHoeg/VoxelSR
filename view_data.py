# %% Packages
import numpy as np
import matplotlib.pyplot as plt

# %% Load data
#vol_list = ["045", "046", "047", "049", "050", "051", "052", "053", "054", "055", "057", "060"]
vol_list = ["045", "046", "047", "049", "050"]
for vol in vol_list:
    vol_no = "CAD" + vol
    vol_hr = np.load('/work3/soeba/FEMurSR/lund_data/' + vol_no + '/HR/' + vol_no + '_masked.npy', mmap_mode='r') #, mmap_mode='r'
    vol_lr = np.load('/work3/soeba/FEMurSR/lund_data/' + vol_no + '/LR/' + vol_no + '.npy', mmap_mode='r')
    #vol_ms = np.load('/work3/soeba/FEMurSR/lund_data/' + vol_no + '/MS/' + vol_no + '.npy', mmap_mode='r')

    # Print shapes
    print(f'HR vol shape: {vol_hr.shape}')
    print(f'LR vol shape: {vol_lr.shape}')
    #print(f'MS vol shape: {vol_ms.shape}')
    print(f'4x LR vol shape: {(vol_lr.shape[0] * 4, vol_lr.shape[1] * 4, vol_lr.shape[2] * 4)}')
    #print(f'4x MS vol shape: {(vol_ms.shape[0] * 4, vol_ms.shape[1] * 4, vol_ms.shape[2] * 4)}\n\n')

# %% Extract nib
vol_lr = vol_lr.get_fdata()

# %% Print F- or C-contiguous
print("HR is C-contiguous: ", vol_hr.data.contiguous)
print("LR is C-contiguous: ", vol_lr.data.contiguous)
#print("MS is C-contiguous: ", vol_ms.data.contiguous)

# %% Show slices
fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(vol_hr[vol_hr.shape[2] // 2], cmap='gray')
ax[1].imshow(vol_lr[vol_lr.shape[2] // 2], cmap='gray')
#ax[2].imshow(vol_ms[vol_ms.shape[2] // 2], cmap='gray')
plt.show()
# %%
