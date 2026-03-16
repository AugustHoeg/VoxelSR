import os
import zarr
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# PARAMETERS
# -----------------------------

fps = 20
slice_step = 1  # legacy, now replaced by slice_subsample

cell_h = 256
cell_w = 256

size_hr = 256
size_lr = 64

# New parameters
offset_y = 200  # in-plane offset for y (vertical)
offset_x = 200  # in-plane offset for x (horizontal)
slice_subsample = 2  # use every Nth slice

# -----------------------------
# DATA
# -----------------------------

base_path = "/work3/s173944/Python/venv_srgan/3D_datasets/datasets/VoDaSuRe/ome/train/"

sample_paths = [
    "Bamboo_A_bin1x1_ome_1.zarr",
    "Cardboard_A_bin1x1_ome_1.zarr",
    "Elm_A_bin1x1_ome_1.zarr",
    "Larch_B_bin1x1_ome_1.zarr",
    "MDF_A_bin1x1_ome_1.zarr",
    "Oak_A_bin1x1_ome_1.zarr",
    # "Femur_15_80kV_ome.zarr",
    # "Vertebrae_A_80kV_ome.zarr",
]

sample_paths = [os.path.join(base_path, f) for f in sample_paths]

hr_vols = [zarr.open(p, mode="r")["HR/0"] for p in sample_paths]
lr_vols = [zarr.open(p, mode="r")["HR/2"] for p in sample_paths]
reg_vols = [zarr.open(p, mode="r")["REG/0"] for p in sample_paths]

datasets = [hr_vols, lr_vols, reg_vols]

rows = len(datasets)
cols = len(sample_paths)

# Determine slice count
num_slices_lr = min(v.shape[0] for v in lr_vols)

# -----------------------------
# VIDEO SIZE
# -----------------------------

frame_h = rows * cell_h
frame_w = cols * cell_w

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    "vodasure_flythrough.mp4",
    fourcc,
    fps,
    (frame_w, frame_h),
    isColor=False,
)

# -----------------------------
# FIT IMAGE INTO CELL
# -----------------------------

def fit_to_cell(img):
    h, w = img.shape
    scale = min(cell_w / w, cell_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((cell_h, cell_w), dtype=img.dtype)
    y0 = (cell_h - new_h) // 2
    x0 = (cell_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = img
    return canvas

# -----------------------------
# SLICE EXTRACTION HELPER
# -----------------------------

def extract_slice(vol, z, size, offset_y=0, offset_x=0):
    H, W = vol.shape[1:]
    center_y = H // 2 + offset_y
    center_x = W // 2 + offset_x
    y0 = max(center_y - size // 2, 0)
    y1 = min(center_y + size // 2, H)
    x0 = max(center_x - size // 2, 0)
    x1 = min(center_x + size // 2, W)
    img = vol[z, y0:y1, x0:x1]
    return img

# -----------------------------
# GENERATE VIDEO
# -----------------------------

for z_hr in tqdm(range(0, num_slices_lr, slice_subsample)):

    frame = np.zeros((frame_h, frame_w), dtype=np.uint8)

    for r, dataset in enumerate(datasets):
        for c, vol in enumerate(dataset):
            if r > 0:
                z_lr = z_hr // 4
                img = extract_slice(vol, z_lr, size_lr, offset_y, offset_x)
            else:
                img = extract_slice(vol, z_hr, size_hr, offset_y, offset_x)

            img = (img >> 8).astype(np.uint8)  # Convert from uint16 to uint8
            img = fit_to_cell(img)
            y0 = r * cell_h
            x0 = c * cell_w
            frame[y0:y0+cell_h, x0:x0+cell_w] = img

    writer.write(frame)

writer.release()
print("Video saved.")