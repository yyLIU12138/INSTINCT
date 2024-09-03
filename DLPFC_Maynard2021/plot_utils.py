import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def plot_DLPFC(rna_list, adata_concat, ground_truth_key, matched_clusters_key, model, group_idx, cluster_to_color_map,
               matched_to_color_map, cluster_orders, slice_name_list, cls_list, sp_embedding,
               save_root=None, frame_color=None, file_format='pdf', save=False, plot=False):

    samples = ['A', 'B', 'C']

    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    fig.suptitle(f'{model} Clustering Results (Sample {samples[group_idx]})', fontsize=16)
    for i in range(len(rna_list)):
        real_colors = list(rna_list[i].obs[ground_truth_key].astype('str').map(cluster_to_color_map))
        axs[0, i].scatter(rna_list[i].obsm['spatial'][:, 0], rna_list[i].obsm['spatial'][:, 1], linewidth=0.5, s=30,
                          marker=".", color=real_colors, alpha=0.9)
        axs[0, i].set_title(f'{slice_name_list[i]} (Ground Truth)', size=12)
        axs[0, i].invert_yaxis()
        axs[0, i].axis('off')

        cluster_colors = list(rna_list[i].obs[matched_clusters_key].map(matched_to_color_map))
        axs[1, i].scatter(rna_list[i].obsm['spatial'][:, 0], rna_list[i].obsm['spatial'][:, 1], linewidth=0.5, s=30,
                          marker=".", color=cluster_colors, alpha=0.9)
        axs[1, i].set_title(f'{slice_name_list[i]} (Cluster Results)', size=12)
        axs[1, i].invert_yaxis()
        axs[1, i].axis('off')

    legend_handles_1 = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=cluster_to_color_map[cluster],
               label=cluster) for cluster in cls_list
    ]
    axs[0, 3].legend(
        handles=legend_handles_1,
        fontsize=8, title='Spot-types', title_fontsize=10, bbox_to_anchor=(1, 1.15))
    legend_handles_2 = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=matched_to_color_map[order],
               label=f'{i}') for i, order in enumerate(cluster_orders)
    ]
    axs[1, 3].legend(
        handles=legend_handles_2,
        fontsize=8, title='Clusters', title_fontsize=10, bbox_to_anchor=(1, 1.1))
    plt.gcf().subplots_adjust(left=0.05, top=None, bottom=None, right=0.85)
    if save:
        save_path = save_root + f'{model}/{model}_group{group_idx}_clustering_results.{file_format}'
        plt.savefig(save_path, dpi=500)

    n_spots = adata_concat.shape[0]
    size = 10000 / n_spots
    order = np.arange(n_spots)
    colors_for_slices = [[0.2298057, 0.29871797, 0.75368315],
                         [0.70567316, 0.01555616, 0.15023281],
                         [0.2298057, 0.70567316, 0.15023281],
                         [0.5830223, 0.59200322, 0.12993134]]
    slice_cmap = {slice_name_list[i]: colors_for_slices[i] for i in range(len(slice_name_list))}
    colors = list(adata_concat.obs['slice_name'].astype('str').map(slice_cmap))
    plt.figure(figsize=(5, 5))
    if frame_color:
        plt.rc('axes', edgecolor=frame_color, linewidth=2)
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    plt.title(f'Slices ({model}/Sample {samples[group_idx]})', fontsize=14)
    if save:
        save_path = save_root + f"{model}/{model}_group{group_idx}_slices_umap.{file_format}"
        plt.savefig(save_path)

    colors = list(adata_concat.obs[ground_truth_key].astype('str').map(cluster_to_color_map))
    plt.figure(figsize=(5, 5))
    if frame_color:
        plt.rc('axes', edgecolor=frame_color, linewidth=2)
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    plt.title(f'Annotated Spot-types ({model}/Sample {samples[group_idx]})', fontsize=14)
    if save:
        save_path = save_root + f"{model}/{model}_group{group_idx}_annotated_clusters_umap.{file_format}"
        plt.savefig(save_path)

    colors = list(adata_concat.obs[matched_clusters_key].map(matched_to_color_map))
    plt.figure(figsize=(5, 5))
    if frame_color:
        plt.rc('axes', edgecolor=frame_color, linewidth=2)
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    plt.title(f'Identified Clusters ({model}/Sample {samples[group_idx]})', fontsize=14)
    if save:
        save_path = save_root + f"{model}/{model}_group{group_idx}_identified_clusters_umap.{file_format}"
        plt.savefig(save_path)

    if plot:
        plt.show()
