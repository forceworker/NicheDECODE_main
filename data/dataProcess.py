import pandas as pd
import numpy as np
from tqdm import tqdm
import scanpy as sc
import warnings
import matplotlib.pyplot as plt
import random
import anndata as ad
import os
import pickle
from sklearn.neighbors import NearestNeighbors
import networkx as nx

warnings.filterwarnings("ignore")


class data_process(object):
    def __init__(self, type_list, tissue_name, rand_n, rand_cell_num, label_key='Niche', latent_key=None,
                 spatial_key='spatial'):
        self.type_list = type_list
        self.tissue_name = tissue_name
        self.label_key = label_key
        self.latent_key = latent_key
        self.spatial_key = spatial_key
        self.rand_n = rand_n
        self.rand_cell_num = rand_cell_num

    def generate_pseudo_bulk(self, data, angle_deg, strip_width, overlap_ratio, min_cells=10, min_cell_types=2):
        data_x = pd.DataFrame(data.X)
        data_x = data_x.fillna(0)
        data_x[data_x < 0] = 0

        data_y = pd.DataFrame(data.obs[self.label_key])
        data_y.reset_index(inplace=True, drop=True)

        if self.latent_key is not None:
            data_x_latent = pd.DataFrame(data.obsm[self.latent_key])

        coords = data.obsm[self.spatial_key]
        coords_df = pd.DataFrame(coords, columns=['x', 'y'])
        coords_df['cell_idx'] = coords_df.index
        theta = np.deg2rad(angle_deg)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        rotated_coords = coords_df[['x', 'y']].values @ rotation_matrix.T
        coords_df['x_rot'] = rotated_coords[:, 0]
        coords_df['y_rot'] = rotated_coords[:, 1]

        x_min, x_max = coords_df['x_rot'].min(), coords_df['x_rot'].max()
        y_min, y_max = coords_df['y_rot'].min(), coords_df['y_rot'].max()

        strip_height = y_max - y_min

        x_sim = []
        y = []

        if self.latent_key is not None:
            latent_sim = []

        stride_x = strip_width * (1 - overlap_ratio)
        x_start = x_min
        strip_positions = []

        while x_start + strip_width <= x_max:
            strip_positions.append((x_start, y_min, x_start + strip_width, y_min + strip_height))
            x_start += stride_x

        valid_strips = 0
        for i, (x_start, y_start, x_end, y_end) in enumerate(strip_positions):
            cells_in_strip = coords_df[
                (coords_df['x_rot'] >= x_start) & (coords_df['x_rot'] <= x_end) &
                (coords_df['y_rot'] >= y_start) & (coords_df['y_rot'] <= y_end)
                ]
            if len(cells_in_strip) >= min_cells:
                cell_indices = cells_in_strip['cell_idx'].tolist()
                strip_expression = data_x.iloc[cell_indices]
                strip_celltypes = data_y.iloc[cell_indices]
                celltype_counts = strip_celltypes[self.label_key].value_counts()
                total_cells = len(cell_indices)
                fracs_complete = [0] * len(self.type_list)
                celltype_diversity = 0
                for j, ct in enumerate(self.type_list):
                    if ct in celltype_counts:
                        fracs_complete[j] = celltype_counts[ct] / total_cells
                        if fracs_complete[j] > 0:
                            celltype_diversity += 1
                s = sum(fracs_complete)
                if celltype_diversity >= min_cell_types and s > 0:
                    fracs_complete = [f / s for f in fracs_complete]
                    bulk_expression = strip_expression.sum(axis=0)
                    x_sim.append(bulk_expression)
                    y.append(fracs_complete)

                    if self.latent_key is not None:
                        strip_latent = data_x_latent.iloc[cell_indices]
                        bulk_latent = strip_latent.sum(axis=0)
                        latent_sim.append(bulk_latent)

                    valid_strips += 1

        print(f"Success rate: {len(x_sim) / len(strip_positions) * 100:.1f}%")

        if self.latent_key is not None:
            return x_sim, latent_sim, y
        else:
            return x_sim, y

    def build_pseudo_bulk_no_noise(self, data):
        data_x = pd.DataFrame(data.X)
        data_x = data_x.fillna(0)
        data_x[data_x < 0] = 0
        data_y = pd.DataFrame(data.obs[self.label_key])
        data_y.reset_index(inplace=True, drop=True)

        x_sim = []
        y = []
        inx = 0
        total_num = self.rand_n

        with tqdm(total=total_num) as pbar:
            while len(x_sim) < total_num:
                result = self.mix_cells(data_x, data_y, cell_type_list=self.type_list)
                if result is None:
                    continue
                sample, label = result
                x_sim.append(sample)
                y.append(label)
                inx += 1
                pbar.update(1)
                if inx >= total_num:
                    break
        return x_sim, y

    def mix_cells(self, x, y, cell_type_list):
        fracs = self.mixup_fraction(len(cell_type_list))
        samp_fracs = np.multiply(fracs, self.rand_cell_num)
        samp_fracs = list(map(round, samp_fracs))
        fracs = np.divide(samp_fracs, sum(samp_fracs))
        # Make complete fracions

        fracs_complete = [0] * len(cell_type_list)

        for i, act in enumerate(cell_type_list):
            idx = cell_type_list.index(act)
            fracs_complete[idx] = fracs[i]

        artificial_samples = []
        for i, ct in enumerate(cell_type_list):
            cells_sub = x.loc[y[self.label_key] == ct]
            if cells_sub.shape[0] > 0 and samp_fracs[i] <= len(cells_sub):
                cells_fraction = np.random.randint(0, cells_sub.shape[0], samp_fracs[i])
                cells_sub = cells_sub.iloc[cells_fraction, :]
                artificial_samples.append(cells_sub)
            else:
                return None

        df_samp = pd.concat(artificial_samples, axis=0)
        df_samp = df_samp.sum(axis=0)

        return df_samp, fracs_complete

    def mixup_fraction(self, cell_num):
        fracs = np.random.rand(cell_num)
        fracs_sum = np.sum(fracs)
        fracs = np.divide(fracs, fracs_sum)
        return fracs

    def normalize(self, series_list, method='max'):

        normalized_series_list = []

        for series in series_list:
            if method == 'max':

                max_value = series.max()
                if max_value == 0:
                    normalized_series = series.copy()
                else:
                    normalized_series = series / max_value

            elif method == 'minmax':
                min_value = series.min()
                max_value = series.max()

                if max_value == min_value:

                    normalized_series = series.copy() * 0 + 0.5
                else:
                    normalized_series = (series - min_value) / (max_value - min_value)

            elif method == 'negative_one_to_one':

                min_value = series.min()
                max_value = series.max()

                if max_value == min_value:

                    normalized_series = series.copy() * 0
                else:

                    normalized_series = 2 * (series - min_value) / (max_value - min_value) - 1

            else:
                raise ValueError(f"不支持的标准化方法: {method}。请使用 'max', 'minmax' 或 'negative_one_to_one'")

            normalized_series_list.append(normalized_series)

        return normalized_series_list


def nn_separation_analysis(adata, label_col1, label_col2, n_neighbors=15):
    """Separation analysis based on nearest neighbors (corrected version)"""

    # Ensure neighbor graph is computed
    if 'neighbors' not in adata.uns:
        print("Calculating neighbor graph...")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)

    def calculate_within_cluster_fraction(adata, label_col, n_neighbors):
        # Get connectivity matrix of neighbor graph
        connectivities = adata.obsp['connectivities']

        same_type_ratios = []
        for i in range(adata.n_obs):
            cell_label = adata.obs[label_col].iloc[i]

            # Get neighbor indices for cell i
            # Get column indices of non-zero elements from connectivity matrix (i.e., neighbors)
            row = connectivities[i]
            neighbors = row.indices

            # If number of neighbors exceeds n_neighbors, take top n_neighbors
            if len(neighbors) > n_neighbors:
                # Sort by connection strength
                data = row.data
                sorted_indices = np.argsort(data)[::-1]  # Descending order
                neighbors = neighbors[sorted_indices[:n_neighbors]]

            # Exclude self (if present)
            neighbors = neighbors[neighbors != i]

            if len(neighbors) > 0:
                neighbor_labels = adata.obs[label_col].iloc[neighbors]
                same_type_ratio = (neighbor_labels == cell_label).mean()
                same_type_ratios.append(same_type_ratio)

        return np.mean(same_type_ratios) if same_type_ratios else 0

    frac1 = calculate_within_cluster_fraction(adata, label_col1, n_neighbors)
    frac2 = calculate_within_cluster_fraction(adata, label_col2, n_neighbors)

    return frac1, frac2


import matplotlib.pyplot as plt
from pathlib import Path


def plot_spatial_by_key(
        adata,
        color_key,
        save_path=None,
        spatial_key='spatial',
        figsize=(10, 8),
        point_size=15,
        alpha=0.7,
        cmap_name='tab20',
        dpi=300,
        show_legend=True,
        legend_title=None,
        title=None,
        xlabel='Spatial X',
        ylabel='Spatial Y',
        grid=False,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        show=True,

        show_ticks=False,
        axis_off=False,

        group_colors=None,
        return_colors=True,
):
    # Get spatial coordinates
    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial coordinates not found in adata.obsm['{spatial_key}']")

    spatial_coords = adata.obsm[spatial_key]

    # Get grouping information
    if color_key not in adata.obs:
        raise KeyError(f"Column '{color_key}' not found in adata.obs")

    groups = adata.obs[color_key]

    # Ensure groups are categorical
    if not hasattr(groups, 'cat'):
        groups = groups.astype('category')

    unique_groups = groups.cat.categories

    if group_colors is None:

        n_groups = len(unique_groups)
        cmap = plt.cm.get_cmap(cmap_name, n_groups)
        group_colors = {group: cmap(i) for i, group in enumerate(unique_groups)}
    else:

        missing = [g for g in unique_groups if g not in group_colors]
        if len(missing) > 0:
            cmap = plt.cm.get_cmap(cmap_name, len(group_colors) + len(missing))

            start = len(group_colors)
            for i, g in enumerate(missing):
                group_colors[g] = cmap(start + i)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each group
    for group in unique_groups:
        group_mask = (groups == group)
        group_coords = spatial_coords[group_mask]

        if len(group_coords) == 0:
            continue

        ax.scatter(
            group_coords[:, 0],
            group_coords[:, 1],
            s=point_size,
            c=[group_colors[group]],
            label=str(group),
            alpha=alpha,
            edgecolors='none'
        )

    # Legend
    if show_legend:
        legend_title = legend_title if legend_title is not None else color_key
        ax.legend(title=legend_title, bbox_to_anchor=bbox_to_anchor, loc=loc)

    # Title
    if title is None:
        title = f'Spatial Projection by {color_key}'
    ax.set_title(title)

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Grid
    if grid:
        ax.grid(True, linestyle='--', alpha=0.3)
    else:
        ax.grid(False)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False)

    if axis_off:
        ax.set_axis_off()

    plt.tight_layout()

    # Save
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        else:
            if not save_path.suffix:
                save_path = save_path.with_suffix('.png')
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()

    if return_colors:
        return fig, group_colors
    return fig


def build_knn_graph(embeddings, gene_names, k=10):
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    G = nx.Graph()

    for i, gene in enumerate(gene_names):
        G.add_node(gene, embedding=embeddings[i])

    for i in range(len(indices)):
        for j in range(1, k + 1):
            neighbor_idx = indices[i][j]
            similarity = 1 - distances[i][j]
            G.add_edge(gene_names[i], gene_names[neighbor_idx],
                       weight=similarity, distance=distances[i][j])

    return G


def plot_knn_graph_basic(
        graph,
        figsize=(12, 10),

        auto_cluster=True,
        cluster_method="greedy_modularity",
        weight_key="weight",
        min_cluster_size=8,
        max_clusters=None,
        cluster_seed=42,

        clusters=None,  # dict or array/list aligned with graph.nodes()
        default_cluster=-1,
        default_color="lightgray",
        cmap_name="tab20",

        save_path=None,
        dpi=300,
        show=True,

        # ===== layout =====
        layout_seed=42,
):
    nodes = list(graph.nodes())
    n_nodes = len(nodes)

    if clusters is None and auto_cluster:

        use_weight = None
        if weight_key is not None:

            for _, _, d in graph.edges(data=True):
                if isinstance(d, dict) and weight_key in d:
                    use_weight = weight_key
                    break

        if cluster_method == "greedy_modularity":
            from networkx.algorithms.community import greedy_modularity_communities
            comms = list(greedy_modularity_communities(graph, weight=use_weight))
        elif cluster_method == "lpa":
            from networkx.algorithms.community import asyn_lpa_communities

            comms = list(asyn_lpa_communities(graph, weight=use_weight, seed=cluster_seed))
        else:
            raise ValueError("cluster_method must be 'greedy_modularity' or 'lpa'.")

        comms = sorted(comms, key=len, reverse=True)

        cluster_map = {n: default_cluster for n in nodes}
        kept = 0
        for cid, community in enumerate(comms):
            if len(community) < min_cluster_size:
                continue
            if (max_clusters is not None) and (kept >= max_clusters):
                continue
            for n in community:
                cluster_map[n] = kept
            kept += 1

        clusters = cluster_map

    cluster_map = None
    if clusters is not None:
        if isinstance(clusters, dict):
            cluster_map = clusters
        else:
            arr = np.asarray(clusters)
            if arr.shape[0] != n_nodes:
                raise ValueError(
                    f"clusters length ({arr.shape[0]}) must match number of nodes ({n_nodes}) "
                    f"when clusters is not a dict."
                )
            cluster_map = {n: arr[i] for i, n in enumerate(nodes)}

    if cluster_map is None:
        node_colors = ["skyblue"] * n_nodes
        n_clusters = 0
    else:
        valid_cids = []
        for n in nodes:
            cid = cluster_map.get(n, default_cluster)
            if cid is None or cid == default_cluster:
                continue
            valid_cids.append(cid)

        uniq = sorted(set(valid_cids))
        n_clusters = len(uniq)
        cmap = plt.get_cmap(cmap_name, max(n_clusters, 1))
        color_of = {cid: cmap(i) for i, cid in enumerate(uniq)}

        node_colors = []
        for n in nodes:
            cid = cluster_map.get(n, default_cluster)
            if cid is None or cid == default_cluster:
                node_colors.append(default_color)
            else:
                node_colors.append(color_of.get(cid, default_color))

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(
        graph,
        k=1 / np.sqrt(max(n_nodes, 1)),
        iterations=50,
        seed=layout_seed
    )

    nx.draw_networkx_edges(graph, pos, alpha=0.18, edge_color="gray")
    nx.draw_networkx_nodes(
        graph, pos,
        node_size=20,
        node_color=node_colors,
        alpha=0.9,
        linewidths=0
    )

    title = f"KNN Graph (Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()})"
    if cluster_map is not None:
        title += f" | colored clusters: {n_clusters} (min_size={min_cluster_size})"
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


import numpy as np


def filter_rare_cell_types_logspace(
        adata,
        cell_type_col: str = "niche",
        sigma_cut: float = 0.1,
        eps: float = 1e-8,
        min_thr: int = 1,
        verbose: bool = True,
):
    """
    Filter out rare cell types based on a log-space threshold computed from cell-type counts.

    Threshold logic:
        counts = value_counts(adata.obs[cell_type_col])
        logv = log(counts + eps)
        thr_log = mean(logv) - sigma_cut * std(logv)
        thr_count = floor(exp(thr_log))
        thr_count = max(thr_count, min_thr)

    Parameters
    ----------
    adata : AnnData-like
        Object with `adata.obs` containing `cell_type_col`.
    cell_type_col : str
        Column name in `adata.obs` that stores cell type labels.
    sigma_cut : float
        Controls how strict the threshold is in log-space.
    eps : float
        Small constant to avoid log(0).
    min_thr : int
        Minimum threshold on the count (default: 1).
    verbose : bool
        If True, print summary logs in English.

    Returns
    -------
    adata_filt : same type as `adata`
        Filtered object (a copy).
    info : dict
        Diagnostic information (thresholds, removed types, removal stats, raw counts).
    """
    if cell_type_col not in adata.obs:
        raise KeyError(f"'{cell_type_col}' not found in adata.obs columns.")

    counts = adata.obs[cell_type_col].value_counts(dropna=True)
    vals = counts.values.astype(float)

    logv = np.log(vals + eps)
    mu_log = float(logv.mean())
    sigma_log = float(logv.std(ddof=1)) if len(logv) > 1 else 0.0

    thr_log = mu_log - sigma_cut * sigma_log
    thr_count = int(np.floor(np.exp(thr_log)))
    thr_count = max(thr_count, int(min_thr))

    to_remove = counts[counts < thr_count].index

    total_samples = int(adata.shape[0])
    samples_removed = int(adata[adata.obs[cell_type_col].isin(to_remove)].shape[0])
    removal_pct = (samples_removed / total_samples) * 100 if total_samples else 0.0

    if verbose:
        print(f"log-space stats: mean={mu_log:.3f}, std={sigma_log:.3f}")
        print(f"threshold (count): {thr_count}")
        print(f"cell types to remove ({len(to_remove)}): {list(to_remove)}")
        print(f"removing {removal_pct:.2f}% of samples ({samples_removed}/{total_samples})")

    adata_filt = adata[~adata.obs[cell_type_col].isin(to_remove)].copy()

    info = {
        "cell_type_col": cell_type_col,
        "sigma_cut": sigma_cut,
        "eps": eps,
        "mu_log": mu_log,
        "sigma_log": sigma_log,
        "thr_log": float(thr_log),
        "thr_count": thr_count,
        "to_remove": list(to_remove),
        "n_cell_types_total": int(counts.shape[0]),
        "n_cell_types_removed": int(len(to_remove)),
        "total_samples": total_samples,
        "samples_removed": samples_removed,
        "removal_percentage": float(removal_pct),
        "counts": counts,  # pandas Series
    }
    return adata_filt, info


def spatial_sliding_window_stats(
        adata,
        window_width: float,
        step: float = None,
        overlap_rate: float = 0.0
):
    coords = adata.obsm["spatial"]
    x = coords[:, 0]
    y = coords[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    step = window_width * (1 - overlap_rate)

    windows = []
    x_start = x_min
    while x_start + window_width <= x_max:
        x_end = x_start + window_width
        windows.append((x_start, x_end))
        x_start += step

    cells_per_window = []
    for x_start, x_end in windows:
        mask = (x >= x_start) & (x < x_end) & (y >= y_min) & (y <= y_max)
        count = np.sum(mask)
        cells_per_window.append(count)

    mean_cells = np.mean(cells_per_window)
    n_windows = len(windows)

    result = {
        "n_windows": n_windows,
        "mean_cells_per_window": mean_cells,
        "cells_per_window": cells_per_window,
        "window_ranges": windows,
    }

    return result
