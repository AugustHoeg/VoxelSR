
import re

text = """
DATASET VoDaSuRe2 FID SCORE: 16.153431351902327

DATASET SAMPLE AVERAGES
METRIC AVERAGE PSNR: 27.600782
METRIC AVERAGE SSIM: 0.788069
METRIC AVERAGE LPIPS: 0.184895
METRIC AVERAGE MANIQA: 0.369041
METRIC AVERAGE CLIPIQA: 0.622812
METRIC AVERAGE MUSIQ: 38.5331
METRIC AVERAGE DISTS: 0.15096
METRIC AVERAGE NIQE: 9.045596
METRIC AVERAGE FRC: 0.371249
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
