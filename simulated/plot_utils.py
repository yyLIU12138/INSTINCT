import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def plot_result_simulated(cas_list, result, sp_cmap, model_name, num_clusters, save_root, sp_embedding,
                          frame_color=None, legend=True, save=True, show=False):

    fig, axs = plt.subplots(2, 3, figsize=(11, 5))
    fig.suptitle(f'{model_name} Results Clustering', fontsize=16)
    for i in range(len(cas_list)):
        real_colors = list(cas_list[i].obs['real_spot_clusters'].astype('str').map(sp_cmap))
        axs[0, i].scatter(cas_list[i].obsm['spatial'][:, 0], cas_list[i].obsm['spatial'][:, 1], linewidth=0, s=30,
                          marker=".", color=real_colors)
        axs[0, i].set_title(f'Slice {i} (Ground Truth)', size=12)
        axs[0, i].invert_yaxis()
        axs[0, i].axis('off')

        cluster_colors = list(cas_list[i].obs['my_clusters'].astype('str').map(sp_cmap))
        axs[1, i].scatter(cas_list[i].obsm['spatial'][:, 0], cas_list[i].obsm['spatial'][:, 1],  linewidth=0, s=30,
                          marker=".", color=cluster_colors)
        axs[1, i].set_title(f'Slice {i} (Cluster Results)', size=12)
        axs[1, i].invert_yaxis()
        axs[1, i].axis('off')

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=sp_cmap[f'{i}'], label=f"{i}")
        for i in range(num_clusters)
    ]
    axs[0, 2].legend(handles=legend_handles,
                     fontsize=8, title='Spot-types', title_fontsize=10, bbox_to_anchor=(1, 1))
    axs[1, 2].legend(handles=legend_handles,
                     fontsize=8, title='Clusters', title_fontsize=10, bbox_to_anchor=(1, 1))
    save_path = save_root + f'/{model_name}_clustering.pdf'
    if save:
        plt.savefig(save_path)

    # umap
    n_spots = result.shape[0]
    size = 10000 / n_spots

    order = np.arange(n_spots)

    color_list = [[0.2298057, 0.29871797, 0.75368315],
                  [0.70567316, 0.01555616, 0.15023281],
                  [0.2298057, 0.70567316, 0.15023281]]
    slice_cmap = {f'{i}': color_list[i] for i in range(3)}
    colors = list(result.obs['slice_index'].astype('str').map(slice_cmap))

    plt.figure(figsize=(5, 5))
    if frame_color:
        plt.rc('axes', edgecolor=frame_color, linewidth=2)
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    if legend:
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=slice_cmap[f'{i}'], label=f"{i}")
            for i in range(3)
        ]
        plt.legend(handles=legend_handles,
                   fontsize=8, title='Slices', title_fontsize=10, bbox_to_anchor=(1, 1), loc=1)

    save_path = save_root + f"/{model_name}_slices_umap.pdf"
    if save:
        plt.savefig(save_path)

    colors = list(result.obs['real_spot_clusters'].astype('str').map(sp_cmap))

    plt.figure(figsize=(5, 5))
    if frame_color:
        plt.rc('axes', edgecolor=frame_color, linewidth=2)
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    if legend:
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=sp_cmap[f'{i}'], label=f"{i}")
            for i in range(num_clusters)
        ]
        plt.legend(handles=legend_handles,
                   fontsize=8, title='Spot-types', title_fontsize=10, bbox_to_anchor=(1, 1), loc=1)

    save_path = save_root + f"/{model_name}_real_clusters_umap.pdf"
    if save:
        plt.savefig(save_path)

    colors = list(result.obs['my_clusters'].astype('str').map(sp_cmap))

    plt.figure(figsize=(5, 5))
    if frame_color:
        plt.rc('axes', edgecolor=frame_color, linewidth=2)
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    if legend:
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=sp_cmap[f'{i}'], label=f"{i}")
            for i in range(num_clusters)
        ]
        plt.legend(handles=legend_handles,
                   fontsize=8, title='Clusters', title_fontsize=10, bbox_to_anchor=(1, 1), loc=1)

    save_path = save_root + f"/{model_name}_clusters_umap.pdf"
    if save:
        plt.savefig(save_path)
    if show:
        plt.show()
