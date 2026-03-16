import zarr
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# PARAMETERS
# -----------------------------

fps = 20
n_cols = 4
slice_step = 1

# Paths to datasets
hr_paths = [...]
lr_paths = [...]
reg_paths = [...]

# -----------------------------
# LOAD ZARR DATASETS
# -----------------------------

hr_vols = [zarr.open(p, mode="r")["0"] for p in hr_paths]
lr_vols = [zarr.open(p, mode="r")["0"] for p in lr_paths]
reg_vols = [zarr.open(p, mode="r")["0"] for p in reg_paths]

# Determine slice count
num_slices = min(v.shape[0] for v in hr_vols)

# -----------------------------
# NORMALIZATION FUNCTION
# -----------------------------

def normalize(img):

    img = img.astype(np.float32)

    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)

    return (img * 255).astype(np.uint8)

# -----------------------------
# DETERMINE FRAME SIZE
# -----------------------------

test_img = normalize(hr_vols[0][0])
h, w = test_img.shape

frame_h = h * 3
frame_w = w * n_cols

# -----------------------------
# VIDEO WRITER
# -----------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    "vodasure_flythrough.mp4",
    fourcc,
    fps,
    (frame_w, frame_h),
    isColor=False,
)

# -----------------------------
# GENERATE VIDEO
# -----------------------------

for z in tqdm(range(0, num_slices, slice_step)):

    rows = []

    for dataset in [hr_vols, lr_vols, reg_vols]:

        imgs = []

        for vol in dataset[:n_cols]:

            img = normalize(vol[z, :, :])
            imgs.append(img)

        row = np.concatenate(imgs, axis=1)
        rows.append(row)

    frame = np.concatenate(rows, axis=0)

    writer.write(frame)

writer.release()

print("Video saved.")