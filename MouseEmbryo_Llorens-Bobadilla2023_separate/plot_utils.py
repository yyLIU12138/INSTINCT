import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def plot_mouseembryo_3(cas_list, adata_concat, ground_truth_key, matched_clusters_key, model,
                       cluster_to_color_map, matched_to_color_map, cluster_orders,
                       slice_name_list, sp_embedding,
                       save_root=None, frame_color=None, save=False, plot=False):

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    fig.suptitle(f'{model} Clustering Results', fontsize=16)
    for i in range(len(cas_list)):
        if slice_name_list[i] == 'E12_5-S1' or slice_name_list[i] == 'E12_5-S2':
            size = 20
        else:
            size = 15
        if slice_name_list[i] == 'E15_5-S1':
            axs[i].invert_xaxis()
            axs[i].invert_yaxis()
        cluster_colors = list(cas_list[i].obs[matched_clusters_key].map(matched_to_color_map))
        axs[i].scatter(cas_list[i].obsm['spatial'][:, 1], cas_list[i].obsm['spatial'][:, 0], linewidth=0.5, s=size,
                       marker=".", color=cluster_colors, alpha=0.9)
        axs[i].set_title(f'{slice_name_list[i]}', size=12)
        axs[i].axis('off')

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=matched_to_color_map[order],
               label=f'{i}') for i, order in enumerate(cluster_orders)
    ]
    axs[2].legend(
        handles=legend_handles,
        fontsize=8, title='Clusters', title_fontsize=10, bbox_to_anchor=(1, 1))
    plt.gcf().subplots_adjust(left=0.05, top=0.8, bottom=0.0, right=0.85)
    if save:
        save_path = save_root + f'{model}/{model}_clustering_results.pdf'
        plt.savefig(save_path)

    n_spots = adata_concat.shape[0]
    size = 10000 / n_spots
    order = np.arange(n_spots)
    colors_for_slices = ['deeppink', 'darkgoldenrod', 'c']
    slice_cmap = {slice_name_list[i]: colors_for_slices[i] for i in range(len(slice_name_list))}
    colors = list(adata_concat.obs['slice_name'].astype('str').map(slice_cmap))
    plt.figure(figsize=(5, 5))
    if frame_color:
        plt.rc('axes', edgecolor=frame_color, linewidth=2)
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=slice_cmap[slice_name_list[i]],
               label=slice_name_list[i])
        for i in range(len(slice_name_list))
    ]
    plt.legend(handles=legend_handles, fontsize=8, title='Slices', title_fontsize=10,
               loc='upper left')
    plt.title(f'Slices ({model})', fontsize=16)
    if save:
        save_path = save_root + f"{model}/{model}_slices_umap.pdf"
        plt.savefig(save_path)

    colors = list(adata_concat.obs[ground_truth_key].astype('str').map(cluster_to_color_map))
    plt.figure(figsize=(5, 5))
    if frame_color:
        plt.rc('axes', edgecolor=frame_color, linewidth=2)
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    plt.title(f'Annotated Spot-types ({model})', fontsize=16)
    if save:
        save_path = save_root + f"{model}/{model}_annotated_clusters_umap.pdf"
        plt.savefig(save_path)

    colors = list(adata_concat.obs[matched_clusters_key].map(matched_to_color_map))
    plt.figure(figsize=(5, 5))
    if frame_color:
        plt.rc('axes', edgecolor=frame_color, linewidth=2)
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    plt.title(f'Identified Clusters ({model})', fontsize=16)
    if save:
        save_path = save_root + f"{model}/{model}_identified_clusters_umap.pdf"
        plt.savefig(save_path)

    if plot:
        plt.show()


def plot_mouseembryo_6(cas_list, adata_concat, ground_truth_key, matched_clusters_key, model,
                       cluster_to_color_map, matched_to_color_map, cluster_orders,
                       slice_name_list, sp_embedding,
                       save_root=None, frame_color=None, save=False, plot=False):

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle(f'{model} Clustering Results', fontsize=16)
    for i in range(len(cas_list)):
        if slice_name_list[i] == 'E12_5-S1' or slice_name_list[i] == 'E12_5-S2':
            size = 20
        else:
            size = 15
        if slice_name_list[i] == 'E15_5-S1':
            axs[int(i % 2), int(i / 2)].invert_xaxis()
            axs[int(i % 2), int(i / 2)].invert_yaxis()
        cluster_colors = list(cas_list[i].obs[matched_clusters_key].map(matched_to_color_map))
        axs[int(i % 2), int(i / 2)].scatter(cas_list[i].obsm['spatial'][:, 1], cas_list[i].obsm['spatial'][:, 0],
                                            linewidth=0.5, s=size, marker=".", color=cluster_colors, alpha=0.9)
        axs[int(i % 2), int(i / 2)].set_title(f'{slice_name_list[i]} (Cluster Results)', size=12)
        axs[int(i % 2), int(i / 2)].axis('off')

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=matched_to_color_map[order],
               label=f'{i}') for i, order in enumerate(cluster_orders)
    ]
    axs[0, 2].legend(
        handles=legend_handles,
        fontsize=8, title='Clusters', title_fontsize=10, bbox_to_anchor=(1, 1))
    plt.gcf().subplots_adjust(left=0.05, top=None, bottom=None, right=0.85)
    if save:
        save_path = save_root + f'{model}/{model}_clustering_results.pdf'
        plt.savefig(save_path)

    n_spots = adata_concat.shape[0]
    size = 10000 / n_spots
    order = np.arange(n_spots)
    colors_for_slices = ['deeppink', 'hotpink', 'darkgoldenrod', 'goldenrod', 'c', 'cyan']
    slice_cmap = {slice_name_list[i]: colors_for_slices[i] for i in range(len(slice_name_list))}
    colors = list(adata_concat.obs['slice_name'].astype('str').map(slice_cmap))
    plt.figure(figsize=(5, 5))
    if frame_color:
        plt.rc('axes', edgecolor=frame_color, linewidth=2)
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=slice_cmap[slice_name_list[i]],
               label=slice_name_list[i])
        for i in range(len(slice_name_list))
    ]
    plt.legend(handles=legend_handles, fontsize=8, title='Slices', title_fontsize=10,
               loc='upper left')
    plt.title(f'Slices ({model})', fontsize=16)
    if save:
        save_path = save_root + f"{model}/{model}_slices_umap.pdf"
        plt.savefig(save_path)

    colors = list(adata_concat.obs[ground_truth_key].astype('str').map(cluster_to_color_map))
    plt.figure(figsize=(5, 5))
    if frame_color:
        plt.rc('axes', edgecolor=frame_color, linewidth=2)
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    plt.title(f'Annotated Spot-types ({model})', fontsize=16)
    if save:
        save_path = save_root + f"{model}/{model}_annotated_clusters_umap.pdf"
        plt.savefig(save_path)

    colors = list(adata_concat.obs[matched_clusters_key].map(matched_to_color_map))
    plt.figure(figsize=(5, 5))
    if frame_color:
        plt.rc('axes', edgecolor=frame_color, linewidth=2)
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    plt.title(f'Identified Clusters ({model})', fontsize=16)
    if save:
        save_path = save_root + f"{model}/{model}_identified_clusters_umap.pdf"
        plt.savefig(save_path)

    if plot:
        plt.show()

