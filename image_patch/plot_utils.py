import numpy as np
import random
import torch
import os
import pickle
from PIL import Image, ImageDraw

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS_FOLDER = "plots/"
os.makedirs(PLOTS_FOLDER, exist_ok=True)


def plot_loss_curve(checkpoint_path, since=10):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    loss_history = checkpoint.get("loss_history")
    if loss_history is None:
        raise KeyError(
            "loss_history missing from checkpoint. Ensure the model was saved with util.save_model."
        )

    plt.plot(loss_history[since:], "-", label=f"loss from the {since}-th iternation")
    plt.legend()
    plt.savefig(PLOTS_FOLDER + "loss.png")
    plt.close()


def plot_outputs_hist(outputs, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    temp = outputs.flatten().cpu().detach().numpy()
    ax.hist(temp, bins=100, density=True, log=True, alpha=0.7)

    ax.set_xlabel("Output value", fontsize=16)
    ax.set_ylabel("Probability Density", fontsize=16)
    # ax.set_title('Output Values Histogram', fontsize=20)

    # Increase axis tick size
    ax.tick_params(axis="both", which="major", labelsize=14)

    return ax


def plot_outputs_dist(outputs, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    dist = outputs.sum(dim=0).int().numpy().flatten().astype(float)
    dist /= outputs.shape[0]

    # Specify bar width slightly less than 1.0
    bar_width = 1
    ax.bar(range(len(dist)), dist, width=bar_width, alpha=0.7)

    # plot a horizontal line for the mean of the data
    mean = np.mean(dist)
    mean_line = ax.axhline(mean, color="r", linestyle="dashed", linewidth=1)

    ax.set_xlabel("Output Node Index", fontsize=16)
    ax.set_ylabel("Activation Probability", fontsize=16)
    # ax.set_title('Activation Distribution', fontsize=20)

    # add a grid
    ax.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.7)

    # Increase axis tick size
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Expand y-axis limits
    ax.set_ylim([0, 1.1 * ax.get_ylim()[1]])

    # Add a legend
    ax.legend(
        [mean_line], ["Mean: {:.3f}".format(mean)], fontsize=14, loc="upper right"
    )

    # Set x-axis limits
    ax.set_xlim([-2, len(dist) + 1])

    return ax


def plot_samples_vs_activenodes(outputs):
    ur, counts = torch.unique(torch.round(outputs), dim=0, return_counts=True)
    activation_count = np.array(ur.sum(axis=1))
    sample_vs_active_neurons = {}
    for sc, ac in zip(counts, activation_count):
        sample_vs_active_neurons.setdefault(ac, []).append(sc.item())

    sample_vs_an = {}
    for k, v in sample_vs_active_neurons.items():
        sample_vs_an[k] = sum(v)

    # Convert the dictionary keys to a list of integers (or floats if necessary)
    x_values = np.array(list(sample_vs_an.keys()))
    values = np.array(list(sample_vs_an.values())).astype(float)
    values /= values.sum()

    # Create a new figure and axis object
    fig, ax = plt.subplots()

    ax.bar(x=x_values, height=values)
    ax.set_xlabel("number of active nodes", fontsize=14)
    ax.set_ylabel("proportion of samples", fontsize=14)
    ax.set_yscale("log")

    plt.savefig(PLOTS_FOLDER + "num_samples_vs_num_activenodes.png")
    plt.close(fig)  # Close the figure to free up memory


def plot_patch(ax, this_patch):
    this_patch = this_patch.transpose((1, 2, 0))  # color
    if this_patch.shape[2] == 3:
        ax.imshow(this_patch)
    else:
        this_patch = this_patch[..., 0]  # gray
        ax.imshow(this_patch, cmap="gray", vmin=0, vmax=1)


def save_patch_mosaic(
    patches,  # tensor or array, shape (N, C, H, W), values in [0..1]
    picked,  # list of length R (the “query” indices)
    indices,  # np.ndarray shape (R, C) of neighbor indices
    out_path,
    zoom: int = 20,
    pad: int = 2,
    pad_color: int = 255,
):
    # → uint8 numpy
    if torch.is_tensor(patches):
        patches = patches.cpu().numpy()
    patches = (patches * 255).astype(np.uint8)

    N, ch, H, W = patches.shape
    R, C = indices.shape

    # mosaic dims including padding between tiles
    mosaic_h = R * H + (R - 1) * pad
    mosaic_w = C * W + (C - 1) * pad

    # init background
    if ch == 3:
        mosaic = np.full((mosaic_h, mosaic_w, 3), fill_value=pad_color, dtype=np.uint8)
    else:
        mosaic = np.full((mosaic_h, mosaic_w), fill_value=pad_color, dtype=np.uint8)

    # paste each patch
    for i in range(R):
        for j in range(C):
            idx = indices[i, j]
            tile = patches[idx].transpose(1, 2, 0)  # (H, W, C) or (H, W, 1)
            if ch != 3:
                tile = tile[..., 0]
            y0 = i * (H + pad)
            x0 = j * (W + pad)
            mosaic[y0 : y0 + H, x0 : x0 + W] = tile

    # upsample
    img = Image.fromarray(mosaic)
    if zoom != 1:
        img = img.resize((mosaic_w * zoom, mosaic_h * zoom), resample=Image.NEAREST)

    # draw a 1px rectangle around each patch
    draw = ImageDraw.Draw(img)
    for i in range(R):
        for j in range(C):
            x0 = j * (W + pad) * zoom
            y0 = i * (H + pad) * zoom
            x1 = x0 + W * zoom - 1
            y1 = y0 + H * zoom - 1
            draw.rectangle([x0, y0, x1, y1], outline=0)

    img.save(out_path)


def plot_similar_patches(
    seed, img_patches, outputs, binary=True, zoom: int = 6, pad: int = 2
):
    # 1) binarize if you want
    if binary:
        outputs = outputs.round()

    # 2) tensor → numpy once
    if isinstance(outputs, torch.Tensor):
        outputs_np = outputs.cpu().numpy()
    else:
        outputs_np = outputs
    N, D = outputs_np.shape

    # 3) exactly your old picking
    patch_ind = list(range(N))
    random.seed(seed)
    random.shuffle(patch_ind)
    picked = patch_ind[:16]

    # 4) compute all 16×N L1 distances in one go
    queries = outputs_np[picked]  # (16, D)
    all_dists = np.abs(queries[:, None, :] - outputs_np[None, :, :]).sum(
        axis=2
    )  # (16, N)

    # 5) full argsort to preserve the exact same ranking you had
    ncols = 20
    top20 = np.argsort(all_dists, axis=1)[:, :ncols]  # (16, 20)
    top20[:, 0] = picked  # make sure the query is first

    # 6) one single mosaic‐save
    out_path = PLOTS_FOLDER + f"similar_patches_seed{seed}.png"
    save_patch_mosaic(
        img_patches, picked, top20, out_path, zoom=zoom, pad=pad, pad_color=255
    )


#   visualize_binary_vectors(outputs[picked].numpy(), filename=f"vis_act_nodes_seed{seed}.png")


def get_unique_pattern_and_counts(outputs, return_top_n=None):
    out_rounded = outputs.round()
    ur, counts = torch.unique(out_rounded, dim=0, return_counts=True)
    num_unique_act_pattern = ur.shape[0]
    print(f"Unique number of acitivation patterns: {num_unique_act_pattern}")

    if return_top_n:
        sorted_counts, _ = counts.sort(descending=True)
        # Filter out long tail to speed up later steps
        ur = ur[torch.where((counts > sorted_counts[return_top_n]))]
    return ur, counts, num_unique_act_pattern


def gen_sample_and_pattern_inds(outputs, ur, device="cuda:0"):
    out_rounded = outputs.round()
    sample_inds_for_pattern_ind = {}
    ur = ur.type(torch.bool)
    out_rounded = out_rounded.type(torch.bool)
    ur = ur.to(device)
    out_rounded = out_rounded.to(device)
    for ur_ind, ur_row in enumerate(ur):
        sample_inds_for_pattern_ind[ur_ind] = torch.where(
            torch.all(out_rounded == ur_row, axis=1)
        )[0].tolist()

    ur = ur.to("cpu")
    out_rounded = out_rounded.to("cpu")

    pattern_inds_with_n_active_nodes = {}
    for ur_ind, ur_row in enumerate(ur):
        s = int(ur_row.sum().item())
        pattern_inds_with_n_active_nodes.setdefault(s, []).append(ur_ind)
    return sample_inds_for_pattern_ind, pattern_inds_with_n_active_nodes


def plot_patches_with_n_active_nodes(
    pattern_inds_with_n_active_nodes,
    sample_inds_for_pattern_ind,
    img_patches,
    num_active_nodes,
    num_patterns,
    n_examples,
):
    all_pattern_inds = pattern_inds_with_n_active_nodes[num_active_nodes]
    pattern_inds = random.sample(
        all_pattern_inds, min(num_patterns, len(all_pattern_inds))
    )

    nrows = len(pattern_inds)
    ncols = n_examples
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols / 3.0, nrows / 3.0)
    )
    if nrows == 1:
        axes = axes[np.newaxis, :]

    ind = 0
    for i in range(nrows):
        all_patch_ind_in_this_partition = sample_inds_for_pattern_ind[pattern_inds[i]]
        this_ncols = min(len(all_patch_ind_in_this_partition), ncols)
        # print(ind)
        n_random_patch_ind_in_this_partition = random.sample(
            all_patch_ind_in_this_partition, this_ncols
        )
        for j in range(ncols):
            ax = axes[i, j]
            # ax.set_axis_off()
            # ax.axis('off')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # ax.set_xticks([])
            # ax.set_yticks([])
            # print(i, j, ind)
            if j < this_ncols:
                this_patch = img_patches[n_random_patch_ind_in_this_partition[j]]
                plot_patch(ax, this_patch)
        ind += 1
    fig.savefig(
        PLOTS_FOLDER
        + f"example_patches_for_{num_patterns}patterns_with_{num_active_nodes}activenodes.png"
    )
    plt.close()


def plot_receptive_fields(img_patches, outputs):
    nrows = 8
    ncols = 8
    out_rounded = outputs.round()
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(8, 8)
    )
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            activated_patches = img_patches[out_rounded[:, i * ncols + j] == 1]
            this_patch = activated_patches.mean(axis=0)
            this_patch = this_patch.transpose((1, 2, 0))
            this_patch = (this_patch - this_patch.min()) / (
                this_patch.max() - this_patch.min()
            )
            ax.imshow(this_patch)
    fig.savefig(PLOTS_FOLDER + f"receptive_fields.png")
    plt.close()


def plot_pca_feature_map(outputs, shape, out_path=None):
    """Visualize feature maps using PCA reduction to RGB."""
    from sklearn.decomposition import PCA

    # Flatten the feature maps
    features = outputs.reshape(outputs.shape[1], -1).T  # shape: (H*W, C)

    # Initialize PCA to reduce to 3 components
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(features)  # reduced shape will be (H*W, 3)

    # Reshape back to spatial dimensions and scale to 0-255
    reduced_image = reduced.reshape(shape[0], shape[1], 3)
    reduced_image -= reduced_image.min()
    reduced_image /= reduced_image.max()
    reduced_image *= 255
    reduced_image = reduced_image.astype(np.uint8)

    # Create and save the image
    img_pca = Image.fromarray(reduced_image, "RGB")
    if out_path:
        img_pca.save(out_path)
    return img_pca


def visualize_binary_vectors(vectors, filename=None):
    scale_factor = 0.25  # You can adjust this value based on your preference

    fig_width = len(vectors[0]) * scale_factor
    fig_height = len(vectors) * scale_factor
    assert max(fig_height, fig_width) < 30  # Just in case

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(vectors, cmap="gray_r", aspect="auto")

    # Set the ticks
    ax.set_xticks(np.arange(len(vectors[0])))
    ax.set_yticks(np.arange(len(vectors)))

    # Set the tick labels
    ax.set_xticklabels(np.arange(len(vectors[0])))
    ax.set_yticklabels(np.arange(len(vectors)))

    # Add gridlines
    ax.set_xticks(np.arange(-0.5, len(vectors[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(vectors), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)

    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.title("Binary Vectors Visualization")
    if not filename:
        filename = "visualize_activenodes.png"
    plt.savefig(PLOTS_FOLDER + filename)
    plt.close()
