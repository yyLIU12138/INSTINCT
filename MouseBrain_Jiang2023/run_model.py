import os
import csv
import torch
import numpy as np
import pandas as pd
import anndata as ad

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from ..INSTINCT import *
from ..evaluation_utils import cluster_metrics, bio_conservation_metrics, batch_correction_metrics, knn_cross_validation, match_cluster_labels

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

save = False
# plot = True

# mouse brain
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']
slice_index_list = list(range(len(slice_name_list)))

save_dir = '../../results/MouseBrain_Jiang2023/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# load raw data
cas_dict = {}
for sample in slice_name_list:
    sample_data = ad.read_h5ad(data_dir + sample + '_atac.h5ad')

    if 'insertion' in sample_data.obsm:
        del sample_data.obsm['insertion']

    cas_dict[sample] = sample_data
cas_list = [cas_dict[sample] for sample in slice_name_list]

# merge peaks
cas_list = peak_sets_alignment(cas_list)

# save the merged data
for idx, adata in enumerate(cas_list):
    adata.write_h5ad(f'{data_dir}merged_{slice_name_list[idx]}_atac.h5ad')

# load the merged data
cas_list = [ad.read_h5ad(data_dir + 'merged_' + sample + '_atac.h5ad') for sample in slice_name_list]
for j in range(len(cas_list)):
    cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]

# concatenation
adata_concat = ad.concat(cas_list, label="slice_name", keys=slice_name_list)
# adata_concat.obs_names_make_unique()
print(adata_concat.shape)

# preprocess CAS data
print('Start preprocessing')
preprocess_CAS(cas_list, adata_concat, use_fragment_count=True, min_cells_rate=0.03)
print(adata_concat.shape)
print('Done!')

adata_concat.write_h5ad(save_dir + f"preprocessed_concat_atac.h5ad")
for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_merged_{slice_name_list[i]}_atac.h5ad")

cas_list = [ad.read_h5ad(save_dir + f"filtered_merged_{sample}_atac.h5ad") for sample in slice_name_list]
origin_concat = ad.concat(cas_list, label="slice_name", keys=slice_name_list)
adata_concat = ad.read_h5ad(save_dir + f"preprocessed_concat_atac.h5ad")

print(f'Applying PCA to reduce the feature dimension to 100 ...')
pca = PCA(n_components=100, random_state=1234)
input_matrix = pca.fit_transform(adata_concat.X.toarray())
np.save(save_dir + 'input_matrix_atac.npy', input_matrix)
print('Done !')

input_matrix = np.load(save_dir + 'input_matrix_atac.npy')
adata_concat.obsm['X_pca'] = input_matrix

# calculate the spatial graph
create_neighbor_graph(cas_list, adata_concat)

spots_count = [0]
n = 0
for sample in cas_list:
    num = sample.shape[0]
    n += num
    spots_count.append(n)

INSTINCT_model = INSTINCT_Model(cas_list, adata_concat, device=device)

INSTINCT_model.train(report_loss=True, report_interval=100)

INSTINCT_model.eval(cas_list)

result = ad.concat(cas_list, label="slice_name", keys=slice_name_list)

if save:
    with open(save_dir + 'INSTINCT_embed.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result.obsm['INSTINCT_latent'])

    with open(save_dir + 'INSTINCT_noise_embed.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result.obsm['INSTINCT_latent_noise'])

gm = GaussianMixture(n_components=16, covariance_type='tied', random_state=1234)
y = gm.fit_predict(result.obsm['INSTINCT_latent'], y=None)
result.obs["gm_clusters"] = pd.Series(y, index=result.obs.index, dtype='category')
result.obs['matched_clusters'] = pd.Series(match_cluster_labels(result.obs['Annotation_for_Combined'],
                                                                result.obs["gm_clusters"]),
                                           index=result.obs.index, dtype='category')

ari, ami, nmi, fmi, comp, homo = cluster_metrics(result.obs['Annotation_for_Combined'],
                                                 result.obs['matched_clusters'].tolist())
map, c_asw, i_asw, i_f1 = bio_conservation_metrics(result, use_rep='INSTINCT_latent',
                                                   label_key='Annotation_for_Combined', batch_key='slice_name')
b_asw, b_pcr, kbet, g_conn = batch_correction_metrics(result, origin_concat, use_rep='INSTINCT_latent',
                                                      label_key='Annotation_for_Combined',
                                                      batch_key='slice_name')
accu, kappa, mf1, wf1 = knn_cross_validation(result.obsm['INSTINCT_latent'], result.obs['Annotation_for_Combined'],
                                             k=20, batch_idx=result.obs['slice_name'])


# cls_list = ['Primary_brain_1', 'Primary_brain_2', 'Midbrain', 'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
#             'Subpallium_1', 'Subpallium_2', 'Cartilage_1', 'Cartilage_2', 'Cartilage_3', 'Cartilage_4',
#             'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv']
#
# colors_for_clusters = ['red', 'tomato', 'chocolate', 'orange', 'goldenrod',
#                        'b', 'royalblue', 'g', 'limegreen', 'lime', 'springgreen',
#                        'deepskyblue', 'pink', 'fuchsia', 'yellowgreen', 'olivedrab']
#
# order_for_clusters = [11, 12, 9, 7, 0, 13, 14, 1, 2, 3, 4, 8, 10, 15, 5, 6]
#
# cluster_to_color_map = {cluster: color for cluster, color in zip(cls_list, colors_for_clusters)}
# order_to_cluster_map = {order: cluster for order, cluster in zip(order_for_clusters, cls_list)}
#
# from umap.umap_ import UMAP
# reducer = UMAP(n_neighbors=30, n_components=2, metric="correlation", n_epochs=None, learning_rate=1.0,
#                min_dist=0.3, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=1, repulsion_strength=1,
#                negative_sample_rate=5, a=None, b=None, random_state=1234, metric_kwds=None,
#                angular_rp_forest=False, verbose=False)
#
# gm = GaussianMixture(n_components=len(cls_list), covariance_type='tied', random_state=1234)
# y = gm.fit_predict(result.obsm['INSTINCT_latent'], y=None)
# result.obs["gm_clusters"] = pd.Series(y, index=result.obs.index, dtype='category')
# result.obs['matched_clusters'] = pd.Series(match_cluster_labels(
#     result.obs['Annotation_for_Combined'], result.obs["gm_clusters"]),
#     index=result.obs.index, dtype='category')
# my_clusters = np.sort(list(set(result.obs['matched_clusters'])))
# matched_colors = [cluster_to_color_map[order_to_cluster_map[order]] for order in my_clusters]
# matched_to_color_map = {matched: color for matched, color in zip(my_clusters, matched_colors)}
#
# for i in range(len(cas_list)):
#     cas_list[i].obs['matched_clusters'] = result.obs['matched_clusters'][spots_count[i]:spots_count[i+1]]
#
# sp_embedding = reducer.fit_transform(result.obsm['INSTINCT_latent'])
#
# from .plot_utils import plot_mousebrain
# plot_mousebrain(cas_list, result, 'Annotation_for_Combined', 'matched_clusters', 'INSTINCT', cluster_to_color_map,
#                 matched_to_color_map, my_clusters, slice_name_list, cls_list, sp_embedding, save_root=save_dir,
#                 frame_color='darkviolet', save=save, plot=plot)

