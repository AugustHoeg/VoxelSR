import os
from pathlib import Path

import numpy as np
import qim3d as qim

print(os.getcwd())

#path = "Larch_A_bin1x1/Larch_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos2_recon.txm" # "Elm_A_bin1x1", "Cypress_A_bin1x1", "Bamboo_A_bin1x1"]
#path = "Larch_A_bin1x1/Larch_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_recon.txm"

path_list = []
base_path = "/dtu/3d-imaging-center/projects/2025_DANFIX_163_VoDaSuRe/raw_data_extern/"

# Elm
path_list.append("Elm_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon_000.txm")
path_list.append("Elm_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_pos1_recon_000.txm")
path_list.append("Elm_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_pos1_recon_001.txm")

# Cypress
path_list.append("Cypres_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon_000.txm")
path_list.append("Cypres_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_recon.txm")

# Bamboo
path_list.append("Bamboo_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon_000.txm")
path_list.append("Bamboo_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_pos1_recon_000.txm")
path_list.append("Bamboo_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_pos1_recon_001.txm")

# Oak
path_list.append("Oak_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon_000.txm")
path_list.append("Oak_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_pos1_recon_000.txm")
path_list.append("Oak_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_pos1_recon_001.txm")

# Larch
#path_list.append("Elm_A_bin1x1/Elm_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_pos1_Stitch_Export.tiff")
#path_list.append("Elm_A_bin1x1/Elm_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_Stitch_Export.tiff")

for file_name in path_list:
    out_ext = ".tiff"
    out_name = Path(file_name).stem + out_ext
    file_path = base_path + file_name

    vol = qim.io.load(file_path, virtual_stack=False, progress_bar=True)

    qim.io.save(out_name, vol, replace=True, progress_bar=True)