import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def plot_mousebrain(cas_list, adata_concat, ground_truth_key, matched_clusters_key, model,
                    cluster_to_color_map, matched_to_color_map, cluster_orders, slice_name_list, cls_list,
                    sp_embedding, save_root=None, frame_color=None, save=False, plot=False):

    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    fig.suptitle(f'{model} Clustering Results', fontsize=16)
    for i in range(len(cas_list)):
        real_colors = list(cas_list[i].obs[ground_truth_key].astype('str').map(cluster_to_color_map))
        axs[0, i].scatter(cas_list[i].obsm['spatial'][:, 0], cas_list[i].obsm['spatial'][:, 1], linewidth=0.5, s=30,
                          marker=".", color=real_colors, alpha=0.9)
        axs[0, i].set_title(f'{slice_name_list[i]} (Ground Truth)', size=12)
        axs[0, i].invert_yaxis()
        axs[0, i].axis('off')

        cluster_colors = list(cas_list[i].obs[matched_clusters_key].map(matched_to_color_map))
        axs[1, i].scatter(cas_list[i].obsm['spatial'][:, 0], cas_list[i].obsm['spatial'][:, 1], linewidth=0.5, s=30,
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
        save_path = save_root + f'{model}/{model}_clustering_results.pdf'
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
    # legend_handles = [
    #     Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=slice_cmap[slice_name_list[i]], label=slice_name_list[i])
    #     for i in range(len(slice_name_list))
    # ]
    # plt.legend(handles=legend_handles, fontsize=8, title='Slices', title_fontsize=10,
    #            loc='upper left')
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


def plot_mousebrain_verti(cas_list, adata_concat, ground_truth_key, annotation_key, cluster_to_color_map,
                          slice_name_list, cls_list, sp_embedding, mode,
                          save_root=None, save=False, plot=False):

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'{mode} Annotation Results', fontsize=14)

    real_colors = list(cas_list[0].obs[ground_truth_key].astype('str').map(cluster_to_color_map))
    axs[0].scatter(cas_list[0].obsm['spatial'][:, 0], cas_list[0].obsm['spatial'][:, 1], linewidth=0.5, s=50,
                   marker=".", color=real_colors, alpha=0.9)
    axs[0].set_title(f'{slice_name_list[0]} (Ture Labels)', size=12)
    axs[0].invert_yaxis()
    axs[0].axis('off')

    anno_colors = list(cas_list[1].obs[annotation_key].astype('str').map(cluster_to_color_map))
    axs[1].scatter(cas_list[1].obsm['spatial'][:, 0], cas_list[1].obsm['spatial'][:, 1], linewidth=0.5, s=50,
                   marker=".", color=anno_colors, alpha=0.9)
    axs[1].set_title(f'{slice_name_list[1]} (Annotation)', size=12)
    if mode == 'E13_5':
        axs[1].invert_xaxis()
    else:
        axs[1].invert_yaxis()
    axs[1].axis('off')

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=cluster_to_color_map[cluster], label=cluster)
        for cluster in cls_list
    ]
    axs[1].legend(
        handles=legend_handles,
        fontsize=8, title='Spot-types', title_fontsize=10, bbox_to_anchor=(1, 1))
    plt.gcf().subplots_adjust(left=0.05, top=0.8, bottom=0.1, right=0.75)
    if save:
        save_path = save_root + f'annotation_results.pdf'
        plt.savefig(save_path)

    spots_count = [0]
    n = 0
    for sample in cas_list:
        num = sample.shape[0]
        n += num
        spots_count.append(n)

    n_spots = adata_concat.shape[0]
    size = 10000 / n_spots
    # order = np.arange(n_spots)
    colors_for_slices = [[0.70567316, 0.01555616, 0.15023281],
                         [0.2298057, 0.70567316, 0.15023281]]
    slice_cmap = {slice_name_list[i]: colors_for_slices[i] for i in range(len(slice_name_list))}
    colors = list(adata_concat.obs['slice_name'].astype('str').map(slice_cmap))
    plt.figure(figsize=(5, 5))
    plt.rc('axes', linewidth=1)
    plt.scatter(sp_embedding[spots_count[1]:spots_count[2], 0], sp_embedding[spots_count[1]:spots_count[2], 1],
                s=size, c=colors[spots_count[1]:spots_count[2]])
    plt.scatter(sp_embedding[spots_count[0]:spots_count[1], 0], sp_embedding[spots_count[0]:spots_count[1], 1],
                s=size, c=colors[spots_count[0]:spots_count[1]])
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=slice_cmap[slice_name_list[i]],
               label=slice_name_list[i])
        for i in range(len(slice_name_list))
    ]
    plt.legend(handles=legend_handles, fontsize=8, title='Slices', title_fontsize=10,
               loc='upper left')
    plt.title(f'{mode} Slices', fontsize=14)
    if save:
        save_path = save_root + f"slices_umap.pdf"
        plt.savefig(save_path)

    colors = list(cas_list[0].obs[ground_truth_key].astype('str').map(cluster_to_color_map))
    plt.figure(figsize=(5, 5))
    plt.rc('axes', linewidth=1)
    plt.scatter(sp_embedding[spots_count[1]:spots_count[2], 0], sp_embedding[spots_count[1]:spots_count[2], 1],
                s=size, c='gray')
    plt.scatter(sp_embedding[spots_count[0]:spots_count[1], 0], sp_embedding[spots_count[0]:spots_count[1], 1],
                s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    plt.title(f'{mode} True Labels', fontsize=14)
    if save:
        save_path = save_root + f"true_labels_umap.pdf"
        plt.savefig(save_path)

    colors = list(cas_list[1].obs[annotation_key].astype('str').map(cluster_to_color_map))
    plt.figure(figsize=(5, 5))
    plt.rc('axes', linewidth=1)
    plt.scatter(sp_embedding[spots_count[1]:spots_count[2], 0], sp_embedding[spots_count[1]:spots_count[2], 1],
                s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    plt.title(f'{mode} Annotation', fontsize=14)
    if save:
        save_path = save_root + f"annotation_umap.pdf"
        plt.savefig(save_path)

    if plot:
        plt.show()
