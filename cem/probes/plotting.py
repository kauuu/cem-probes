"""
Visualisation of inter-concept leakage matrices, with concepts sorted and
boxed by concept group.

All matrices follow the convention of `cem.probes.linear`: entry (i, j) is
about TARGET concept i (rows) being predicted from SOURCE embedding j
(columns).
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def group_blocks(concept_group_map, n_concepts):
    """
    Orders concepts so that every concept group is contiguous.

    :param Dict[str, List[int]] concept_group_map: Map from group names to
        the indices of the concepts in that group. If None, every concept is
        its own group.
    :param int n_concepts: Total number of concepts.

    :returns Tuple[np.ndarray, List[Tuple[str, int, int]]]: The concept order
        and, for every group, its (name, start, size) within that order.
        Concepts missing from the map are appended as singleton groups.
    """
    if concept_group_map is None:
        concept_group_map = {str(i): [i] for i in range(n_concepts)}
    order = []
    blocks = []
    for name, idxs in concept_group_map.items():
        idxs = [i for i in idxs if i not in order]
        if not idxs:
            continue
        blocks.append((str(name), len(order), len(idxs)))
        order.extend(idxs)
    for i in range(n_concepts):
        if i not in order:
            blocks.append((str(i), len(order), 1))
            order.append(i)
    return np.array(order), blocks


def group_membership(concept_group_map, n_concepts):
    """
    :returns np.ndarray: A k x k boolean matrix whose (i, j) entry is True iff
        concepts i and j belong to the same group.
    """
    _, blocks = group_blocks(concept_group_map, n_concepts)
    order, _ = group_blocks(concept_group_map, n_concepts)
    group_of = np.zeros(n_concepts, dtype=int)
    for group_idx, (_, start, size) in enumerate(blocks):
        group_of[order[start:start + size]] = group_idx
    return group_of[:, None] == group_of[None, :]


def group_matrix(matrix, concept_group_map, reduce=np.nanmean):
    """
    Aggregates a k x k concept matrix into a G x G group matrix, ignoring the
    diagonal (i == j) entries.

    :param np.ndarray matrix: k x k matrix.
    :param Dict[str, List[int]] concept_group_map: Concept group map.
    :param Fun[np.ndarray, float] reduce: Reduction applied to every block.

    :returns Tuple[np.ndarray, List[str]]: The G x G matrix and group names.
    """
    n_concepts = matrix.shape[0]
    order, blocks = group_blocks(concept_group_map, n_concepts)
    masked = matrix.astype(float).copy()
    np.fill_diagonal(masked, np.nan)
    masked = masked[order][:, order]
    result = np.full((len(blocks), len(blocks)), np.nan)
    for a, (_, start_a, size_a) in enumerate(blocks):
        for b, (_, start_b, size_b) in enumerate(blocks):
            block = masked[start_a:start_a + size_a, start_b:start_b + size_b]
            if np.any(~np.isnan(block)):
                result[a, b] = reduce(block)
    return result, [name for name, _, _ in blocks]


def within_between_summary(matrix, concept_group_map):
    """
    Splits the off-diagonal entries of a k x k matrix into within-group and
    between-group pairs.

    :returns Dict[str, np.ndarray]: "within" and "between" arrays of values.
    """
    n_concepts = matrix.shape[0]
    same = group_membership(concept_group_map, n_concepts)
    off_diag = ~np.eye(n_concepts, dtype=bool)
    valid = ~np.isnan(matrix)
    return {
        "within": matrix[same & off_diag & valid],
        "between": matrix[~same & off_diag & valid],
    }


def plot_concept_matrix(
    matrix,
    concept_group_map=None,
    ax=None,
    title=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    mask_diagonal=True,
    colorbar_label=None,
    min_group_size_to_label=1,
):
    """
    Plots a k x k concept matrix with concepts sorted by group and a box drawn
    around every (within-group) diagonal block.

    :param np.ndarray matrix: k x k matrix (rows: target concept i, columns:
        source embedding j).
    :param Dict[str, List[int]] concept_group_map: Concept group map.
    :param matplotlib.axes.Axes ax: Axes to plot in.
    :param str title: Title of the plot.
    :param str cmap: Colormap.
    :param float vmin: Lower limit of the colour scale.
    :param float vmax: Upper limit of the colour scale.
    :param bool mask_diagonal: Whether to hide the i == j entries.
    :param str colorbar_label: Label of the colour bar.
    :param int min_group_size_to_label: Groups smaller than this get no tick
        label (avoids clutter when there are many singleton groups).

    :returns matplotlib.axes.Axes: The axes plotted in.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    n_concepts = matrix.shape[0]
    order, blocks = group_blocks(concept_group_map, n_concepts)
    values = matrix.astype(float).copy()
    if mask_diagonal:
        np.fill_diagonal(values, np.nan)
    values = values[order][:, order]

    cmap = matplotlib.colormaps[cmap].copy()
    cmap.set_bad("lightgrey")
    image = ax.imshow(
        np.ma.masked_invalid(values),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    for _, start, size in blocks:
        if size < 2:
            continue
        ax.add_patch(matplotlib.patches.Rectangle(
            (start - 0.5, start - 0.5),
            size,
            size,
            fill=False,
            edgecolor="red",
            linewidth=1.2,
        ))

    labelled = [b for b in blocks if b[2] >= min_group_size_to_label]
    ticks = [start + (size - 1) / 2 for _, start, size in labelled]
    names = [name for name, _, _ in labelled]
    ax.set_xticks(ticks)
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticks(ticks)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("source embedding j")
    ax.set_ylabel("target concept i")
    if title:
        ax.set_title(title)
    return ax


def plot_leakage_figure(results, input_name, concept_group_map=None, path=None):
    """
    Plots probe AUC (M), oracle AUC (O) and leakage (M - O) side by side for a
    given embedding space.

    :param Dict[str, Dict[str, np.ndarray]] results: Output of
        `cem.probes.linear.leakage_matrices`.
    :param str input_name: Embedding space to plot (e.g., "pos_neg").
    :param Dict[str, List[int]] concept_group_map: Concept group map.
    :param str path: If given, the figure is saved here.

    :returns matplotlib.figure.Figure: The figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    plot_concept_matrix(
        results[input_name]["test_auc"],
        concept_group_map,
        ax=axes[0],
        title=f"Probe AUC M ({input_name})",
        vmin=0.5,
        vmax=1,
        colorbar_label="test AUC",
    )
    plot_concept_matrix(
        results["oracle"]["test_auc"],
        concept_group_map,
        ax=axes[1],
        title="Oracle AUC O (ground-truth c_j)",
        vmin=0.5,
        vmax=1,
        colorbar_label="test AUC",
    )
    leakage = results[input_name]["leakage"]
    bound = np.nanmax(np.abs(leakage[~np.eye(leakage.shape[0], dtype=bool)]))
    plot_concept_matrix(
        leakage,
        concept_group_map,
        ax=axes[2],
        title=f"Leakage M - O ({input_name})",
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
        colorbar_label="AUC difference",
    )
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_group_summary(results, input_name, concept_group_map=None, path=None):
    """
    Plots the group-aggregated leakage matrix (mean M - O over every block)
    and a box plot of within-group vs between-group leakage.

    :returns matplotlib.figure.Figure: The figure.
    """
    leakage = results[input_name]["leakage"]
    grouped, names = group_matrix(leakage, concept_group_map)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    bound = np.nanmax(np.abs(grouped))
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("lightgrey")
    image = axes[0].imshow(
        np.ma.masked_invalid(grouped),
        cmap=cmap,
        vmin=-bound,
        vmax=bound,
        interpolation="nearest",
    )
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04).set_label(
        "mean AUC difference"
    )
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=90, fontsize=6)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=6)
    axes[0].set_xlabel("source group")
    axes[0].set_ylabel("target group")
    axes[0].set_title(f"Group-mean leakage M - O ({input_name})")

    summary = within_between_summary(leakage, concept_group_map)
    axes[1].boxplot(
        [summary["within"], summary["between"]],
        labels=[
            f"within group\n(n={len(summary['within'])})",
            f"between groups\n(n={len(summary['between'])})",
        ],
    )
    axes[1].axhline(0, color="grey", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("leakage M - O (test AUC)")
    axes[1].set_title("Within vs between groups")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig
