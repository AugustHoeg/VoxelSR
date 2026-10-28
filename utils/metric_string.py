
import re

text = """
DATASET VoDaSuRe2 FID SCORE: 58.14445672406538

DATASET SAMPLE AVERAGES
METRIC AVERAGE PSNR: 21.366318
METRIC AVERAGE SSIM: 0.588679
METRIC AVERAGE LPIPS: 0.272316
METRIC AVERAGE MANIQA: 0.436148
METRIC AVERAGE CLIPIQA: 0.621038
METRIC AVERAGE MUSIQ: 54.055416
METRIC AVERAGE DISTS: 0.230851
METRIC AVERAGE NIQE: 7.353216
METRIC AVERAGE FRC: 0.43081
"""

# Desired order and decimal places
metrics = [
    ("PSNR", 2),
    ("SSIM", 4),
    ("LPIPS", 4),
    ("FRC", 4),
    ("MANIQA", 4),
    ("NIQE", 4),
    ("CLIPIQA", 4),
    ("MUSIQ", 2),
    ("DISTS", 4),
    ("FID", 2),
]

values = {}

# Extract metric averages
for name, value in re.findall(
    r"METRIC AVERAGE (\w+):\s*([\d.]+)", text
):
    values[name] = float(value)

# Extract FID score
match = re.search(r"FID SCORE:\s*([\d.]+)", text)
if match:
    values["FID"] = float(match.group(1))

# Format values in the requested order
formatted = []
for name, decimals in metrics:
    if name not in values:
        raise ValueError(f"Missing metric: {name}")
    formatted.append(f"{values[name]:0{2 + (decimals > 0)}.{decimals}f}")

# Print LaTeX row
print("& " + " & ".join(formatted))
