import os
import re
import glob
from collections import defaultdict


def extract_diffusion_scores(file_path):
    """
    Extracts the four diffusion scores for each model from a given text file:
      - Diffusion index
      - Diffusion index (MEAN)
      - Diffusion index (no pad)
      - Diffusion index (MEAN, no pad)

    Returns:
        dict: {
            model_name: {
                "id": str,
                "diffusion": float,
                "mean": float,
                "no_pad": float,
                "mean_no_pad": float
            }
        }
    """

    # Match all four variants:
    #   Diffusion index for MODEL, ID: VALUE
    #   Diffusion index (MEAN) for MODEL, ID: VALUE
    #   Diffusion index (no pad) for MODEL, ID: VALUE
    #   Diffusion index (MEAN, no pad) for MODEL, ID: VALUE
    pattern = re.compile(
        r"Diffusion index(?: \((MEAN)?(?:, )?(no pad)?\))? for ([^,]+), ([^:]+): ([\d\.eE+-]+)"
    )

    results = {}

    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue

            mean_flag = match.group(1)
            nopad_flag = match.group(2)
            model_name = match.group(3).strip()
            full_id = match.group(4).strip()
            score = float(match.group(5))

            if model_name not in results:
                results[model_name] = {
                    "id": full_id,
                    "diffusion": None,
                    "mean": None,
                    "no_pad": None,
                    "mean_no_pad": None,
                }

            # Determine which variant we are dealing with
            if mean_flag and nopad_flag:
                results[model_name]["mean_no_pad"] = score
            elif mean_flag:
                results[model_name]["mean"] = score
            elif nopad_flag:
                results[model_name]["no_pad"] = score
            else:
                results[model_name]["diffusion"] = score

    return results


if __name__ == "__main__":

    print("Current working directory is:", os.getcwd())

    dataset = "LIDC-IDRI"
    print(f"Running DI averages for: {dataset}")

    file_paths = glob.glob(f"Results/{dataset}*/LAM_DI.txt")
    #print("file paths", file_paths)

    # Collect values across all files
    aggregated = defaultdict(lambda: {"diffusion": [], "mean": [], "no_pad": [], "mean_no_pad": [], "ids": set()})

    for file_path in file_paths:
        scores = extract_diffusion_scores(file_path)

        for model, vals in scores.items():
            if vals["diffusion"] is not None:
                aggregated[model]["diffusion"].append(vals["diffusion"])
            if vals["mean"] is not None:
                aggregated[model]["mean"].append(vals["mean"])
            if vals["no_pad"] is not None:
                aggregated[model]["no_pad"].append(vals["no_pad"])
            if vals["mean_no_pad"] is not None:
                aggregated[model]["mean_no_pad"].append(vals["mean_no_pad"])
            aggregated[model]["ids"].add(vals["id"])

    # Compute averages
    for model, vals in aggregated.items():
        avg_diffusion = sum(vals["diffusion"]) / len(vals["diffusion"]) if vals["diffusion"] else None
        avg_mean = sum(vals["mean"]) / len(vals["mean"]) if vals["mean"] else None
        avg_no_pad = sum(vals["no_pad"]) / len(vals["no_pad"]) if vals["no_pad"] else None
        avg_mean_no_pad = sum(vals["mean_no_pad"]) / len(vals["mean_no_pad"]) if vals["mean_no_pad"] else None
        ids = ", ".join(sorted(vals["ids"]))

        print(f"{model} ({ids}):")
        print(f"  Average Diffusion index              = {avg_diffusion}")
        print(f"  Average Diffusion index (no pad)     = {avg_no_pad}")
        print(f"  Average Diffusion index (MEAN)       = {avg_mean}")
        print(f"  Average Diffusion index (MEAN, no pad) = {avg_mean_no_pad}")
        print()
