import os
import zarr
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# PARAMETERS
# -----------------------------

fps = 20

cell_h = 192
cell_w = 192

size_hr = 256
size_lr = 64

slice_subsample = 4

# initial crop offsets
offset_y = -200
offset_x = -200

# pan parameters
pan_amp_x = 0
pan_amp_y = 0
pan_period = 800  # slices per cycle

# layout margins
top_margin = 60
left_margin = 80

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
    "Cypress_A_bin1x1_ome_1.zarr"
]

sample_paths = [os.path.join(base_path, f) for f in sample_paths]

col_titles = [os.path.basename(p).split("_")[0] for p in sample_paths]
row_titles = ["HR", "LR", "Registered"]

# -----------------------------
# LOAD DATASETS
# -----------------------------

hr_vols = [zarr.open(p, mode="r")["HR/0"] for p in sample_paths]
lr_vols = [zarr.open(p, mode="r")["HR/2"] for p in sample_paths]
reg_vols = [zarr.open(p, mode="r")["REG/0"] for p in sample_paths]

datasets = [hr_vols, lr_vols, reg_vols]

rows = len(datasets)
cols = len(sample_paths)

num_slices_lr = min(v.shape[0] for v in lr_vols)

# -----------------------------
# VIDEO SIZE
# -----------------------------

frame_h = rows * cell_h + top_margin
frame_w = cols * cell_w + left_margin

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    "vodasure_flythrough.mp4",
    fourcc,
    fps,
    (frame_w, frame_h),
    isColor=False,
)

# -----------------------------
# IMAGE HELPERS
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


def extract_slice(vol, z, size, offset_y, offset_x):

    H, W = vol.shape[1:]

    center_y = H // 2 + offset_y
    center_x = W // 2 + offset_x

    y0 = max(center_y - size // 2, 0)
    y1 = min(center_y + size // 2, H)

    x0 = max(center_x - size // 2, 0)
    x1 = min(center_x + size // 2, W)

    return vol[z, y0:y1, x0:x1]


def draw_vertical_text(frame, text, x, y):

    text_img = np.zeros((200, 50), dtype=np.uint8)

    cv2.putText(
        text_img,
        text,
        (5, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255,
        2,
        cv2.LINE_AA
    )

    text_img = cv2.rotate(text_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w = text_img.shape

    frame[y:y+h, x:x+w] = np.maximum(frame[y:y+h, x:x+w], text_img)


# -----------------------------
# GENERATE VIDEO
# -----------------------------

frame_idx = 0

for z_hr in tqdm(range(0, num_slices_lr, slice_subsample)):

    frame = np.zeros((frame_h, frame_w), dtype=np.uint8)

    # slow pan motion
    t = frame_idx

    pan_x = int(pan_amp_x * np.sin(2 * np.pi * t / pan_period))
    pan_y = int(pan_amp_y * np.cos(2 * np.pi * t / pan_period))

    current_offset_x = offset_x + pan_x
    current_offset_y = offset_y + pan_y

    # draw slices
    for r, dataset in enumerate(datasets):

        for c, vol in enumerate(dataset):

            if r > 0:
                z_lr = z_hr // 4
                img = extract_slice(
                    vol,
                    z_lr,
                    size_lr,
                    current_offset_y // 4,
                    current_offset_x // 4
                )
            else:
                img = extract_slice(
                    vol,
                    z_hr,
                    size_hr,
                    current_offset_y,
                    current_offset_x
                )

            img = (img >> 8).astype(np.uint8)

            img = fit_to_cell(img)

            y0 = top_margin + r * cell_h
            x0 = left_margin + c * cell_w

            frame[y0:y0+cell_h, x0:x0+cell_w] = img

    # column titles
    for c, title in enumerate(col_titles):

        x = left_margin + c * cell_w + cell_w // 2 - 40

        cv2.putText(
            frame,
            title,
            (x, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            255,
            1,
            cv2.LINE_AA
        )

    # row titles
    for r, title in enumerate(row_titles):

        y = top_margin + r * cell_h + cell_h // 2 - 40

        draw_vertical_text(frame, title, 10, y)

    writer.write(frame)

    frame_idx += 1

writer.release()

print("Video saved.")