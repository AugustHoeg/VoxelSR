# %% Packages
import numpy as np
import matplotlib.pyplot as plt

# %% Choose cube
cube_no = '010'

# %% Load cube
cube = np.load(f'/work3/soeba/Superresolution/LAM_3d/saved_image_cubes_bone/Synthetic_2022_QIM_52_Bone_4x/HR/cube_{cube_no}.npy')
cube = cube[0]
z = cube.shape[2]

# %% Show slices
nrows = 5
ncols = 5
fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))
fig.suptitle(f'Cube {cube_no}')

for i, ax in enumerate(axs.flatten()):
    slice_idx = i * z//(nrows*ncols) + 1
    ax.imshow(cube[:,:,slice_idx],cmap='gray')
    ax.set_title(f'Slice {slice_idx}')
    ax.axis('off')

plt.tight_layout()
plt.show()
# %%
