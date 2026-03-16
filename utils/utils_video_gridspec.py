import os
import zarr
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -----------------------------
# PARAMETERS
# -----------------------------

fps = 20
slice_subsample = 4

size_hr = 256
size_lr = 64

offset_y = -200
offset_x = -200

cell_size = 256  # display size

# -----------------------------
# FONT STYLE
# -----------------------------

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = True

# -----------------------------
# DATA
# -----------------------------

base_path = "/work3/s173944/Python/venv_srgan/3D_datasets/datasets/VoDaSuRe/ome/train/"
#base_path = "../../3D_datasets/datasets/VoDaSuRe/ome/train/"

sample_paths = [
    #"Femur_15_80kV_ome.zarr",
    #"Femur_15_80kV_ome.zarr",
    #"Femur_15_80kV_ome.zarr",
    #"Femur_15_80kV_ome.zarr",
    #"Femur_15_80kV_ome.zarr",
    #"Femur_15_80kV_ome.zarr",
    #"Femur_15_80kV_ome.zarr",
    "Bamboo_A_bin1x1_ome_1.zarr",
    "Cardboard_A_bin1x1_ome_1.zarr",
    "Elm_A_bin1x1_ome_1.zarr",
    "Larch_B_bin1x1_ome_1.zarr",
    "MDF_A_bin1x1_ome_1.zarr",
    "Oak_A_bin1x1_ome_1.zarr",
    "Cypress_A_bin1x1_ome_1.zarr"
]

sample_paths = [os.path.join(base_path, f) for f in sample_paths]

col_titles = [os.path.basename(p).split("_")[0] for p in sample_paths]
row_titles = ["High resolution", "Downsampled", "Registered"]

# -----------------------------
# LOAD DATA
# -----------------------------

hr_vols = [zarr.open(p, mode="r")["HR/0"] for p in sample_paths]
lr_vols = [zarr.open(p, mode="r")["HR/2"] for p in sample_paths]
reg_vols = [zarr.open(p, mode="r")["REG/0"] for p in sample_paths]

datasets = [hr_vols, lr_vols, reg_vols]

rows = len(datasets)
cols = len(sample_paths)

num_slices_lr = min(v.shape[0] for v in lr_vols)

# -----------------------------
# SLICE HELPER
# -----------------------------

def extract_slice(vol, z, size, offset_y=0, offset_x=0):

    H, W = vol.shape[1:]

    cy = H // 2 + offset_y
    cx = W // 2 + offset_x

    y0 = max(cy - size // 2, 0)
    y1 = min(cy + size // 2, H)

    x0 = max(cx - size // 2, 0)
    x1 = min(cx + size // 2, W)

    return vol[z, y0:y1, x0:x1]

# -----------------------------
# CREATE MATPLOTLIB LAYOUT
# -----------------------------

fig = plt.figure(figsize=(cols * 2.0, rows * 2.0))

fig.patch.set_facecolor("#f5f5f5")

fig.subplots_adjust(
    left=0.04,
    right=0.96,
    bottom=0.05,
    top=0.95,
    wspace=0.02,
    hspace=0.02
)

gs = GridSpec(rows, cols, figure=fig)

axes = []
images = []

for r in range(rows):

    row_axes = []
    row_imgs = []

    for c in range(cols):

        ax = fig.add_subplot(gs[r, c])

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

        img = ax.imshow(
            np.zeros((cell_size, cell_size)),
            cmap="gray",
            vmin=0,
            vmax=255
        )

        if r == 0:
            ax.set_title(col_titles[c], fontsize=12, y=1.01)

        if r == 0 and c == 0:
            ax.set_ylabel(row_titles[r], fontsize=12, labelpad=10)
        elif r == 1 and c == 0:
            ax.set_ylabel(row_titles[r], fontsize=12, labelpad=10)
        elif r == 2 and c == 0:
            ax.set_ylabel(row_titles[r], fontsize=12, labelpad=10)

        row_axes.append(ax)
        row_imgs.append(img)

    axes.append(row_axes)
    images.append(row_imgs)

# -----------------------------
# VIDEO SETUP
# -----------------------------

fig.canvas.draw()

width, height = fig.canvas.get_width_height()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    "vodasure_flythrough.mp4",
    fourcc,
    fps,
    (width, height)
)

# -----------------------------
# GENERATE VIDEO
# -----------------------------

for z_hr in tqdm(range(0, num_slices_lr, slice_subsample)):

    for r, dataset in enumerate(datasets):

        for c, vol in enumerate(dataset):

            if r > 0:
                z_lr = z_hr // 4
                img = extract_slice(
                    vol,
                    z_lr,
                    size_lr,
                    offset_y // 4,
                    offset_x // 4
                )
            else:
                img = extract_slice(
                    vol,
                    z_hr,
                    size_hr,
                    offset_y,
                    offset_x
                )

            img = (img >> 8).astype(np.uint8)

            img = cv2.resize(
                img,
                (cell_size, cell_size),
                interpolation=cv2.INTER_AREA
            )

            images[r][c].set_data(img)

    fig.canvas.draw()

    frame = np.frombuffer(
        fig.canvas.tostring_rgb(),
        dtype=np.uint8
    )

    frame = frame.reshape(height, width, 3)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    writer.write(frame)

writer.release()

print("Video saved.")