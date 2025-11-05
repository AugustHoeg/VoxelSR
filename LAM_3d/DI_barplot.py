import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"

models = [
    "HAT", "RCAN", "EDDSR", "MFER", "mDCSRN", "SuperFormer", "RRDBNet3D", "MTVNet"
]

lits =        [3.540, 13.751, 5.871, 8.330, 16.779, 7.355, 25.233, 26.209]
ctspine1k =   [3.633, 13.728, 5.972, 8.139, 16.753, 7.245, 24.270, 25.310]
lidc_idri =   [3.532, 13.271, 5.802, 8.187, 17.241, 6.396, 24.768, 25.355]
vodasure =    [15.692, 16.974, 6.190, 12.921, 20.524, 18.074, 25.352, 32.711]

# ($\times4$)
datasets = {
    r"LiTS": lits,
    r"CTSpine1K": ctspine1k,
    r"LIDC-IDRI": lidc_idri,
    r"VoDaSuRe (Downsampled)": vodasure,
}

# ---- Helper: underline text ----
def underline_text(s):
    return ''.join([c + '\u0332' for c in s])

# ---- Plot setup ----
x = np.arange(len(models)) * 1.6  # wider spacing between model groups
width = 0.12
gap = 0.25
colors =  ["#edada3", "#ce6a6c", "#94c4c1", "#42899b"] #["#9BC4C3", "#DDBAAA", "#9E9795", "#ACB39F"] # ["lightcoral", "darkseagreen", "cornflowerblue", "goldenrod"]

fig, ax = plt.subplots(figsize=(12, 8))

bars_dict = {}
num_datasets = len(datasets)
offset_center = (num_datasets - 1) / 2

# --- Draw bars ---
for i, (name, values) in enumerate(datasets.items()):
    offset = (i - offset_center) * (width + gap)
    bars = ax.bar(
        x + offset, values, width,
        label=name,
        color=colors[i],
        edgecolor="black", linewidth=1.5
    )
    bars_dict[name] = bars

# ---- Find best (largest) and second-best per model (across datasets) ----
# Shape into 2D array: rows=models, cols=datasets
values_array = np.array(list(datasets.values())).T  # shape (num_models, num_datasets)

# ---- Annotate ----
for model_idx, model in enumerate(models):
    vals = values_array[model_idx]
    if np.all(vals == 0):
        continue  # skip empty rows

    # Indices of largest and 2nd largest values
    sorted_idx = np.argsort(vals)
    max_idx = sorted_idx[-1]
    second_idx = sorted_idx[-2] if len(sorted_idx) > 1 else None

    for j, (name, bars) in enumerate(bars_dict.items()):
        bar = bars[model_idx]
        height = bar.get_height()
        if height == 0:
            continue

        # Formatting based on highlight
        if j == max_idx:
            text = f"$\\mathbf{{{height:.2f}}}$"  # bold = largest
        elif j == second_idx:
            text = f"{height:.2f}" #underline_text(f"{height:.2f}")  # underline = 2nd largest
        else:
            text = f"{height:.2f}"

        if height < 10:
            xytext = (-10, 20)
        else:
            xytext = (-14, 24)

        ax.annotate(
            text,
            xy=(bar.get_x() + bar.get_width() * 0.65, height),  # moved right
            xytext=xytext,
            textcoords="offset points",
            ha="center", va="bottom",
            rotation=-60, rotation_mode="anchor",
            fontsize=14,
        )

        # Small vertical tick line above each bar
        ax.plot(
            [bar.get_x() + bar.get_width() / 2, bar.get_x() + bar.get_width() / 2],
            [height, height + 0.4],
            color="black", linewidth=1.5
        )

# ---- Labels, grid, legend ----
ax.set_ylabel("DI average", fontsize=22)
ax.set_title("Mean Diffusion Index averages across datasets", fontsize=25, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right", fontsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.set_ylim(0, max(max(v) for v in datasets.values()) * 1.2)

ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
ax.set_axisbelow(True)
ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=16)

plt.tight_layout()
plt.savefig("DI_barplot_highlighted_12_8.pdf", dpi=600, bbox_inches="tight")
plt.show()
