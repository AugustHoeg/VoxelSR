import os
import glob
from huggingface_hub import HfApi

api = HfApi()

dataset_path = "/dtu/3d-imaging-center/projects/2025_DANFIX_163_VoDaSuRe/analysis/VoDaSuRe_dataset_p4/VoDaSuRe/"

api.upload_file(
    path_or_fileobj=os.path.join(dataset_path, "extract_files.sh"),
    path_in_repo="extract_files.sh",
    repo_id="AugustHoeg/VoDaSuRe",
    repo_type="dataset",
)