import numpy as np
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from umap.umap_ import UMAP
from sklearn.mixture import GaussianMixture

from plot_utils import plot_mousebrain, plot_mousebrain_single

from codes.evaluation_utils import match_cluster_labels

from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
save_dir = '../../results/MouseBrain_Jiang2023/single/'
save = False

cls_list_1 = ['Primary_brain_1', "Primary_brain_2", 'Midbrain', 'Diencephalon_and_hindbrain',
              'Cartilage_2',
              'Mesenchyme', 'Muscle', 'DPallm']
cls_list_2 = ['Primary_brain_1', 'Midbrain', 'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
              'Cartilage_1', 'Cartilage_2', 'Cartilage_3', 'Cartilage_4',
              'Mesenchyme', 'Muscle', 'DPallm', 'DPallv']
cls_list_3 = ['Primary_brain_1', 'Midbrain', 'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
              'Subpallium_2', 'Cartilage_2', 'Cartilage_3',
              'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv']
cls_list_4 = ['Midbrain', 'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
              "Subpallium_1", 'Subpallium_2', 'Cartilage_1', 'Cartilage_2', 'Cartilage_3', 'Cartilage_4',
              'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv']
cls_list_all = [cls_list_1, cls_list_2, cls_list_3, cls_list_4]

cls_list = ['Primary_brain_1', 'Primary_brain_2', 'Midbrain', 'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
            'Subpallium_1', 'Subpallium_2', 'Cartilage_1', 'Cartilage_2', 'Cartilage_3', 'Cartilage_4',
            'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv']

colors_for_clusters = ['red', 'tomato', 'chocolate', 'orange', 'goldenrod',
                       'b', 'royalblue', 'g', 'limegreen', 'lime', 'springgreen',
                       'deepskyblue', 'pink', 'fuchsia', 'yellowgreen', 'olivedrab']

order_for_clusters = [11, 12, 9, 7, 0,
                      13, 14, 1, 2, 3, 4,
                      8, 10, 15, 5, 6]
order_1 = [6, 7, 4, 2, 0, 3, 5, 1]
order_2 = [11, 9, 7, 0, 1, 2, 3, 4, 8, 10, 5, 6]
order_3 = [9, 7, 5, 0, 10, 1, 2, 6, 8, 11, 3, 4]
order_4 = [9, 7, 0, 11, 12, 1, 2, 3, 4, 8, 10, 13, 5, 6]
order_list_all = [order_1, order_2, order_3, order_4]

cluster_to_color_map = {cluster: color for cluster, color in zip(cls_list, colors_for_clusters)}
order_to_cluster_map = {order: cluster for order, cluster in zip(order_for_clusters, cls_list)}

reducer = UMAP(n_neighbors=30, n_components=2, metric="correlation", n_epochs=None, learning_rate=1.0,
               min_dist=0.3, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=1, repulsion_strength=1,
               negative_sample_rate=5, a=None, b=None, random_state=1234, metric_kwds=None,
               angular_rp_forest=False, verbose=False)

models = ['INSTINCT', 'SCALE', 'STAGATE']
colors_for_labels = ['darkviolet', 'darkgoldenrod', 'steelblue']

slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

cas_list = [ad.read_h5ad(f"../../results/MouseBrain_Jiang2023/filtered_merged_{name}_atac.h5ad") for name in slice_name_list]
spots_count = [0]
n = 0
for sample in cas_list:
    num = sample.shape[0]
    n += num
    spots_count.append(n)

# single
for idx, slice_name in enumerate(slice_name_list):

    for j, model_name in enumerate(models):

        if model_name != 'INSTINCT':
            adata = ad.read_h5ad(data_dir + slice_name + '_atac.h5ad')
            if 'insertion' in adata.obsm:
                del adata.obsm['insertion']
            if model_name == 'SCALE':
                sc.pp.filter_cells(adata, min_genes=100)
                sc.pp.filter_genes(adata, min_cells=3)
        else:
            adata = cas_list[idx]

        print(f'{model_name}')

        if model_name != 'INSTINCT':
            embed = pd.read_csv(save_dir + f'{model_name}/{model_name}_embed_2_{slice_name}.csv', header=None).values
            adata.obsm['latent'] = embed

            gm = GaussianMixture(n_components=len(cls_list_all[idx]), covariance_type='tied', random_state=1234)
            y = gm.fit_predict(adata.obsm['latent'], y=None)
            adata.obs["gm_clusters"] = pd.Series(y, index=adata.obs.index, dtype='category')
        else:
            embed = pd.read_csv(save_dir + f'{model_name}/{model_name}_embed_2.csv', header=None).values
            adata.obsm['latent'] = embed[spots_count[idx]: spots_count[idx + 1], :]

            gm = GaussianMixture(n_components=len(cls_list), covariance_type='tied', random_state=1234)
            y = gm.fit_predict(embed, y=None)
            adata.obs["gm_clusters"] = pd.Series(y[spots_count[idx]: spots_count[idx + 1]], index=adata.obs.index,
                                                 dtype='category')

        adata.obs['matched_clusters'] = pd.Series(match_cluster_labels(adata.obs['Annotation_for_Combined'],
                                                                       adata.obs["gm_clusters"]),
                                                  index=adata.obs.index, dtype='category')

        my_clusters = np.sort(list(set(adata.obs['matched_clusters'])))
        order_to_cluster_map_single = {order: cluster for order, cluster in zip(order_list_all[idx], cls_list_all[idx])}
        matched_colors = [cluster_to_color_map[order_to_cluster_map_single[order]] for order in my_clusters]
        matched_to_color_map = {matched: color for matched, color in zip(my_clusters, matched_colors)}

        sp_embedding = reducer.fit_transform(adata.obsm['latent'])

        plot_mousebrain_single(adata, slice_name, 'Annotation_for_Combined', 'matched_clusters', cluster_to_color_map,
                               matched_to_color_map, cls_list_all[idx], sp_embedding, model_name,
                               save_root=save_dir+f'{model_name}/', frame_color=colors_for_labels[j],
                               legend=False, save=save, plot=True)


models = ['SCALE', 'STAGATE']
colors_for_labels = ['darkgoldenrod', 'steelblue']

# plot clustering results
for j, model in enumerate(models):

    # load raw data
    cas_list = []
    for sample in slice_name_list:
        sample_data = ad.read_h5ad(data_dir + sample + '_atac.h5ad')

        if 'insertion' in sample_data.obsm:
            del sample_data.obsm['insertion']

        if model == 'SCALE':
            sc.pp.filter_cells(sample_data, min_genes=100)
            sc.pp.filter_genes(sample_data, min_cells=3)

        cas_list.append(sample_data)
    adata_concat = ad.concat(cas_list, label='slice_name', keys=slice_name_list)

    spots_count = [0]
    n = 0
    for sample in cas_list:
        num = sample.shape[0]
        n += num
        spots_count.append(n)

    embed = [pd.read_csv(save_dir + f'{model}/{model}_embed_2_{name}.csv', header=None).values
             for name in slice_name_list]
    embed = np.vstack(embed)
    adata_concat.obsm['latent'] = embed

    gm = GaussianMixture(n_components=len(cls_list), covariance_type='tied', random_state=1234)
    y = gm.fit_predict(adata_concat.obsm['latent'], y=None)
    adata_concat.obs["gm_clusters"] = pd.Series(y, index=adata_concat.obs.index, dtype='category')
    adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(
        adata_concat.obs['Annotation_for_Combined'], adata_concat.obs["gm_clusters"]),
        index=adata_concat.obs.index, dtype='category')
    my_clusters = np.sort(list(set(adata_concat.obs['matched_clusters'])))
    matched_colors = [cluster_to_color_map[order_to_cluster_map[order]] for order in my_clusters]
    matched_to_color_map = {matched: color for matched, color in zip(my_clusters, matched_colors)}

    for i in range(len(cas_list)):
        cas_list[i].obs['matched_clusters'] = adata_concat.obs['matched_clusters'][spots_count[i]:spots_count[i+1]]

    sp_embedding = reducer.fit_transform(adata_concat.obsm['latent'])

    plot_mousebrain(cas_list, adata_concat, 'Annotation_for_Combined', 'matched_clusters', model, cluster_to_color_map,
                    matched_to_color_map, my_clusters, slice_name_list, cls_list, sp_embedding, save_root=save_dir,
                    frame_color=colors_for_labels[j], save=save, plot=True)






