import matplotlib.pyplot as plt
import numpy as np

# Use Times/serif font
plt.rcParams["font.family"] = "serif"

# Models
models = [
    "HAT", "RCAN", "EDDSR", "ArSSR",
    "MFER", "mDCSRN", "SuperFormer", "RRDBNet3D", "MTVNet"
]

# Example DI values for each dataset (replace with your own)
lits =        [8.1, 7.4, 5.9, 6.8, 7.6, 8.3, 9.2, 10.8, 11.3]
ctspine1k =   [9.8, 8.9, 6.0, 7.1, 7.3, 7.8, 12.7, 10.1, 13.0]
lidc_idri =   [10.5, 9.7, 6.2, 7.4, 8.0, 8.7, 13.5, 11.6, 14.1]
vodasure =    [11.7, 10.6, 6.1, 7.8, 11.5, 12.4, 13.7, 25.0, 27.3]

datasets = {
    "LITS": lits,
    "CTSpine1K": ctspine1k,
    "LIDC-IDRI": lidc_idri,
    "VoDaSuRe": vodasure,
}

# ---- Helper functions ----

def get_best_indices(values):
    """Return indices of best (min) and second-best (next smallest) values."""
    sorted_idx = np.argsort(values)
    return sorted_idx[0], sorted_idx[1]

def underline_text(s):
    """Return underlined text using Unicode combining underline."""
    return ''.join([c + '\u0332' for c in s])

def annotate_bars(ax, bars, best_idx, second_idx):
    """Annotate bars, highlighting best (bold) and second-best (underline)."""
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i == best_idx:
            text = f"$\\mathbf{{{height:.2f}}}$"
        elif i == second_idx:
            text = underline_text(f"{height:.2f}")
        else:
            text = f"{height:.2f}"
        ax.annotate(
            text,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=10
        )

# ---- Plot setup ----

x = np.arange(len(models))
width = 0.18  # narrower bars for 4 datasets
gap = 0.05

colors = ["lightcoral", "darkseagreen", "cornflowerblue", "goldenrod"]

fig, ax = plt.subplots(figsize=(13, 7))

bars_list = []
for i, (name, values) in enumerate(datasets.items()):
    offset = (i - 1.5) * (width + gap)  # center around model positions
    bars = ax.bar(
        x + offset, values, width,
        label=name,
        color=colors[i],
        edgecolor="black", linewidth=1.0
    )
    bars_list.append((bars, name, values))

# ---- Annotate best / second-best per dataset ----
for bars, name, values in bars_list:
    best, second = get_best_indices(values)
    annotate_bars(ax, bars, best, second)

# ---- Labels, grid, legend ----
ax.set_ylabel("DI average", fontsize=22)
ax.set_title("Mean Diffusion Index averages across datasets", fontsize=25, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right", fontsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.set_ylim(0, max(max(v) for v in datasets.values()) * 1.15)

ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
ax.set_axisbelow(True)

ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=16)

plt.tight_layout()
plt.savefig("DI_barplot_4datasets.pdf", dpi=600, bbox_inches="tight")
plt.show()
