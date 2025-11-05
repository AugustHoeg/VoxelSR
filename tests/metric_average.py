import os
import glob
import re
import numpy as np

up_factor = 2
base_path = f"../logs/VoDaSuRe_OME/wandb/"
experiment_id = "ID009501"

search_path = os.path.join(base_path, f"*{up_factor}x_VoDaSuRe_OME_{experiment_id}/files/performance_statistics/*.txt")
print("Search path:", search_path)

experiments = glob.glob(search_path)
experiments.sort(key=os.path.getmtime, reverse=True)  # latest first
print("Found experiments:", experiments)

# Compile regex patterns for extracting metric lists
psnr_pattern = re.compile(r"PSNR SAMPLE LIST:\s*\[([^\]]+)\]")
ssim_pattern = re.compile(r"SSIM SAMPLE LIST:\s*\[([^\]]+)\]")
nrmse_pattern = re.compile(r"NRMSE SAMPLE LIST:\s*\[([^\]]+)\]")

results = []

exp_file = experiments[0]
with open(exp_file, "r") as f:
    content = f.read()

psnr_match = psnr_pattern.search(content)
ssim_match = ssim_pattern.search(content)
nrmse_match = nrmse_pattern.search(content)

if psnr_match and ssim_match and nrmse_match:
    psnr_values = np.fromstring(psnr_match.group(1), sep=' ')
    ssim_values = np.fromstring(ssim_match.group(1), sep=' ')
    nrmse_values = np.fromstring(nrmse_match.group(1), sep=' ')

    psnr_values = np.delete(psnr_values, [-2]) # remove next to last element (invalid scan)
    ssim_values = np.delete(ssim_values, [-2])  # remove next to last element (invalid scan)
    nrmse_values = np.delete(nrmse_values, [-2])  # remove next to last element (invalid scan)

    #print(psnr_values)
    #print(ssim_values)
    #print(nrmse_values)

    print("Experiment id:", experiment_id)
    print("Experiment file:", experiments)
    print(f"PSNR MEAN: {np.mean(psnr_values):.2f}")
    print(f"SSIM MEAN: {np.mean(ssim_values):.4f}")
    print(f"NRMSE MEAN: {np.mean(nrmse_values):.4f}")
