import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image



def plot_histogram(hist, bin_edges, title="Histogram", color="darkgray", savefig=False, log_scale=False):
    # Normalize histogram to probabilities (optional, looks cleaner)
    hist = hist.astype(float) / hist.sum()

    # Plot
    plt.figure(figsize=(16, 10))
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge", color=color, alpha=0.7)
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel("Intensity", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    if log_scale:
        plt.yscale('log')
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if savefig:
        plt.savefig(f"figures/{title}.pdf", dpi=300, bbox_inches='tight')
    else:
        plt.show()

def single_histogram(image, data_min=0.0, data_max=1.0, num_bins=256, label="Image", color="darkgray", ax=None):

    values = np.ravel(image)

    # Compute histograms
    hist, bin_edges = np.histogram(values, bins=num_bins, range=(data_min, data_max))

    # Normalize to probability
    hist = hist.astype(float) / hist.sum()

    # Plot
    if ax is None:
        plt.figure(figsize=(16, 10))
        plt.step(bin_edges[:-1], hist, where="mid", color=color, label=label, linewidth=2)
        plt.fill_between(bin_edges[:-1], hist, step="mid", alpha=0.3, color=color)

        plt.legend(fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
    else:
        ax.step(bin_edges[:-1], hist, where="mid", color=color, label=label, linewidth=2)
        ax.fill_between(bin_edges[:-1], hist, step="mid", alpha=0.3, color=color)

        ax.grid(axis="y", linestyle="--", alpha=0.6)

    return hist, bin_edges

def compare_histograms(image1, image2,
                       data_min=0.0, data_max=1.0, num_bins=256,
                       labels=("Image 1", "Image 2"),
                       colors=("steelblue", "darkorange"),
                       ax=None):
    """
    Compute and plot the histograms of two images/volumes for comparison.

    Parameters
    ----------
    image1, image2 : np.ndarray
        Input 2D or 3D images (any dtype, will be flattened).
    data_min, data_max : float
        Range for histogram computation.
    num_bins : int
        Number of bins for the histograms.
    labels : tuple
        Labels for the two histograms.
    colors : tuple
        Colors for the two histograms.
    title : str
        Title for the plot.
    """
    # Flatten
    values1 = np.ravel(image1)
    values2 = np.ravel(image2)

    # Compute histograms
    hist1, bin_edges = np.histogram(values1, bins=num_bins, range=(data_min, data_max))
    hist2, _ = np.histogram(values2, bins=num_bins, range=(data_min, data_max))

    # Normalize to probability
    hist1 = hist1.astype(float) / hist1.sum()
    hist2 = hist2.astype(float) / hist2.sum()

    # Plot
    if ax is None:
        plt.figure(figsize=(16, 10))
        plt.step(bin_edges[:-1], hist1, where="mid", color=colors[0], label=labels[0], linewidth=1)
        plt.step(bin_edges[:-1], hist2, where="mid", color=colors[1], label=labels[1], linewidth=1)
        plt.fill_between(bin_edges[:-1], hist1, step="mid", alpha=0.3, color=colors[0])
        plt.fill_between(bin_edges[:-1], hist2, step="mid", alpha=0.2, color=colors[1])

        plt.legend(fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
    else:
        ax.step(bin_edges[:-1], hist1, where="mid", color=colors[0], label=labels[0], linewidth=1)
        ax.step(bin_edges[:-1], hist2, where="mid", color=colors[1], label=labels[1], linewidth=1)
        ax.fill_between(bin_edges[:-1], hist1, step="mid", alpha=0.3, color=colors[0])
        ax.fill_between(bin_edges[:-1], hist2, step="mid", alpha=0.2, color=colors[1])

        ax.grid(axis="y", linestyle="--", alpha=0.6)

    return (hist1, hist2), bin_edges


if __name__ == "__main__":

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["text.usetex"] = True

    base_path = "../../downloaded_data/VoDaSuRe/supplementary/matching/"
    source_vals = np.load(os.path.join(base_path, "source_vals_100.npy"))
    reference_vals = np.load(os.path.join(base_path, "reference_vals_100.npy"))
    matched_vals = np.load(os.path.join(base_path, "matched_vals_100.npy"))

    HR = np.array(Image.open(os.path.join(base_path, "HR_100_matched.png")))
    LR = np.array(Image.open(os.path.join(base_path, "LR_100_matched.png")))
    REG_matched = np.array(Image.open(os.path.join(base_path, "REG_100_matched.png")))
    REG_unmatched = np.array(Image.open(os.path.join(base_path, "REG_100_unmatched.png")))

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)

    # Main grid: narrow left column, wide right column
    gs_main = gridspec.GridSpec(
        1, 2,
        figure=fig,
        width_ratios=[1, 3],  # narrow left, wide right
        wspace=0.02
    )

    # ----------------------------------------------------------------------
    # LEFT COLUMN: 3 small images stacked vertically
    # ----------------------------------------------------------------------
    gs_left = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=gs_main[0, 0],
        hspace=0.05
    )

    ax_left_small1 = fig.add_subplot(gs_left[0])
    ax_left_small1.imshow(HR, cmap='gray', vmin=0, vmax=65535)  # plot HR image on ax_left_small1
    ax_left_small1.text(0.5, -0.02, "Downscaled HR slice", ha='center', va='top', fontsize=22, transform=ax_left_small1.transAxes)
    ax_left_small1.set_title("VoDaSuRe - Bamboo", fontsize=24)

    ax_left_small2 = fig.add_subplot(gs_left[1])
    ax_left_small2.imshow(REG_unmatched, cmap='gray', vmin=0, vmax=65535)  # plot REG unmatched image on ax_left_small2
    ax_left_small2.text(0.5, -0.02, "Unmatched LR slice", ha='center', va='top', fontsize=22, transform=ax_left_small2.transAxes)

    ax_left_small3 = fig.add_subplot(gs_left[2])
    ax_left_small3.imshow(REG_matched, cmap='gray', vmin=0, vmax=65535)  # plot REG matched image on ax_left_small3
    ax_left_small3.text(0.5, -0.02, "Matched LR slice", ha='center', va='top', fontsize=22, transform=ax_left_small3.transAxes)

    # ----------------------------------------------------------------------
    # RIGHT COLUMN: 2 big images stacked vertically
    # Want big image height = 1.5 * small image height
    #
    # Total small height = 3 × 1 = 3
    # Total big height = 2 × 1.5 = 3  → perfectly matches
    # ----------------------------------------------------------------------
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=gs_main[0, 1],
        hspace=0.05,
        height_ratios=[1.5, 1.5]   # enforce size ratio relative to left column
    )

    # plot histogram of reference on ax_right_big1
    ax_right_big1 = fig.add_subplot(gs_right[0])
    #single_histogram(reference_vals, 0, 65535, num_bins=100, label="reference", color="darkorange", ax=ax_right_big1)
    compare_histograms(reference_vals, source_vals, 1, 65535, num_bins=512, labels=("HR slice", "Unmatched slice"), colors=("steelblue", "seagreen"), ax=ax_right_big1)
    ax_right_big1.legend(fontsize=21, loc="upper left")
    ax_right_big1.set_title("Slice intensity histogram", fontsize=24)
    ax_right_big1.tick_params(axis='both', which='major', labelsize=20)
    ax_right_big1.set_ylabel("Probability", fontsize=20)
    #ax_right_big1.yaxis.set_label_position("right")
    ax_right_big1.yaxis.tick_right()
    #ax_right_big1.set_xlabel("Intensity", fontsize=16)

    # plot histogram of source and matched on ax_right_big2
    ax_right_big2 = fig.add_subplot(gs_right[1])
    compare_histograms(reference_vals, matched_vals, 1, 65535, num_bins=512, labels=("HR slice", "Matched slice"), colors=("steelblue", "darkorange"), ax=ax_right_big2)
    ax_right_big2.legend(fontsize=21, loc="upper left")
    ax_right_big2.tick_params(axis='both', which='major', labelsize=20)
    ax_right_big2.set_ylabel("Probability", fontsize=20)
    #ax_right_big2.yaxis.set_label_position("right")
    ax_right_big2.yaxis.tick_right()
    ax_right_big2.set_xlabel("Intensity", fontsize=20)

    # ----------------------------------------------------------------------
    # Example: remove axes (optional)
    # ----------------------------------------------------------------------
    for ax in [
        ax_left_small1, ax_left_small2, ax_left_small3,
    ]:
        ax.set_xticks([])
        ax.set_yticks([])

    #plt.tight_layout()
    save_path = f"../figures/supplementary_matching_{100}.pdf"
    fig.savefig(save_path, format="pdf")
    plt.show()
