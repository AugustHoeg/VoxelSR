import os
import argparse
from huggingface_hub import HfApi, login

# Add HR Token as a command-line argument
parser = argparse.ArgumentParser(description="Upload files to Hugging Face Hub")
parser.add_argument("--HF_TOKEN", type=str, help="Hugging Face API token")
args = parser.parse_args()

# Option 1: login using token (recommended for HPC)
HF_TOKEN = args.HF_TOKEN

if HF_TOKEN is None:
    raise ValueError("Please set HF_TOKEN")

login(token=HF_TOKEN)

api = HfApi()

dataset_path = "/dtu/3d-imaging-center/projects/2025_DANFIX_163_VoDaSuRe/analysis/VoDaSuRe_dataset_p4/"

folder_path = os.path.join(dataset_path, "VoDaSuRe")  # <-- the large folder
api.upload_large_folder(
    repo_id="AugustHoeg/VoDaSuRe",
    repo_type="dataset",
    folder_path=folder_path,
)

print("Large upload initiated")
