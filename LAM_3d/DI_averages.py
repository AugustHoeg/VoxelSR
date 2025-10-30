import os
import re
import glob
from collections import defaultdict


def extract_diffusion_scores(file_path):
    """
    Extracts 'Diffusion index' and 'Diffusion index (MEAN)' scores for each model
    from a given text file.

    Returns:
        dict: {
            model_name: {
                "id": str,
                "diffusion": float,
                "mean": float
            }
        }
    """
    pattern = re.compile(
        r"Diffusion index(?: \(MEAN\))? for ([^,]+), ([^:]+): ([\d\.eE+-]+)"
    )

    results = {}

    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                model_name = match.group(1).strip()
                full_id = match.group(2).strip()
                score = float(match.group(3))

                if model_name not in results:
                    results[model_name] = {"id": full_id, "diffusion": None, "mean": None}

                if "(MEAN)" in line:
                    results[model_name]["mean"] = score
                elif "(MEAN, no pad)" in line:
                    results[model_name]["mean_no_pad"] = score
                elif "(no pad)" in line:
                    results[model_name]["no_pad"] = score
                else:
                    results[model_name]["diffusion"] = score

    return results


if __name__ == "__main__":

    scores = extract_diffusion_scores("Results/HCP_1200_cube_027_win48_h40-w40-d40_new/LAM_DI.txt")
    for model in scores:
        print(f"\"{model}\": {scores[model]['mean']},")

    print("Current working directory is:", os.getcwd())

    #file_paths = glob.glob("Results/Synthetic*/LAM_DI.txt")
    file_paths = glob.glob("Results/HCP_1200*/LAM_DI.txt")
    print("file paths", file_paths)

    # Collect values across all files
    aggregated = defaultdict(lambda: {"diffusion": [], "mean": [], "ids": set()})

    for file_path in file_paths:
        scores = extract_diffusion_scores(file_path)

        for model, vals in scores.items():
            if vals["diffusion"] is not None:
                aggregated[model]["diffusion"].append(vals["diffusion"])
            if vals["mean"] is not None:
                aggregated[model]["mean"].append(vals["mean"])
            aggregated[model]["ids"].add(vals["id"])  # keep track of IDs seen

    # Compute averages
    for model, vals in aggregated.items():
        avg_diffusion = sum(vals["diffusion"]) / len(vals["diffusion"]) if vals["diffusion"] else None
        avg_mean = sum(vals["mean"]) / len(vals["mean"]) if vals["mean"] else None
        avg_no_pad = sum(vals["no_pad"]) / len(vals["no_pad"]) if vals["no_pad"] else None
        avg_mean_no_pad = sum(vals["mean_no_pad"]) / len(vals["mean_no_pad"]) if vals["mean_no_pad"] else None
        ids = ", ".join(sorted(vals["ids"]))  # optional: show all IDs

        print(f"{model} ({ids}):")
        print(f"  Average Diffusion index      = {avg_diffusion}")
        print(f"  Average Diffusion index no pad = {avg_no_pad}")
        print(f"  Average Diffusion index MEAN = {avg_mean}")
        print(f"  Average Diffusion index MEAN no pad = {avg_mean_no_pad}")
        print()
