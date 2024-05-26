import numpy as np
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
from umap.umap_ import UMAP
from sklearn.mixture import GaussianMixture

from plot_utils import plot_mousebrain

from codes.evaluation_utils import match_cluster_labels

from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
save_dir = '../../results/MouseBrain_Jiang2023/'
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

cls_list = ['Primary_brain_1', 'Primary_brain_2', 'Midbrain', 'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
            'Subpallium_1', 'Subpallium_2', 'Cartilage_1', 'Cartilage_2', 'Cartilage_3', 'Cartilage_4',
            'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv']

colors_for_clusters = ['red', 'tomato', 'chocolate', 'orange', 'goldenrod',
                       'b', 'royalblue', 'g', 'limegreen', 'lime', 'springgreen',
                       'deepskyblue', 'pink', 'fuchsia', 'yellowgreen', 'olivedrab']

order_for_clusters = [11, 12, 9, 7, 0, 13, 14, 1, 2, 3, 4, 8, 10, 15, 5, 6]

cluster_to_color_map = {cluster: color for cluster, color in zip(cls_list, colors_for_clusters)}
order_to_cluster_map = {order: cluster for order, cluster in zip(order_for_clusters, cls_list)}

reducer = UMAP(n_neighbors=30, n_components=2, metric="correlation", n_epochs=None, learning_rate=1.0,
               min_dist=0.3, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=1, repulsion_strength=1,
               negative_sample_rate=5, a=None, b=None, random_state=1234, metric_kwds=None,
               angular_rp_forest=False, verbose=False)

models = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']
colors_for_labels = ['darkviolet', 'chocolate', 'sandybrown', 'peachpuff', 'darkslategray', 'c', 'cyan']

slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

cas_list = [ad.read_h5ad(save_dir + f"filtered_merged_{sample}_atac.h5ad") for sample in slice_name_list]
adata_concat = ad.read_h5ad(save_dir + f"preprocessed_concat_atac.h5ad")
raw_pca = np.load(save_dir + 'input_matrix_atac.npy')
sp_embedding = reducer.fit_transform(raw_pca)

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
plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                labelleft=False, labelbottom=False, grid_alpha=0)
legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=slice_cmap[slice_name_list[i]], label=slice_name_list[i])
        for i in range(len(slice_name_list))
    ]
plt.legend(handles=legend_handles, fontsize=8, title='Slices', title_fontsize=10,
           loc='upper left')
plt.title('Slices (Raw)', fontsize=16)
if save:
    save_path = save_dir + f"/raw_slices_umap.pdf"
    plt.savefig(save_path)

colors = list(adata_concat.obs['Annotation_for_Combined'].astype('str').map(cluster_to_color_map))
plt.figure(figsize=(5, 5))
plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                labelleft=False, labelbottom=False, grid_alpha=0)
plt.title('Annotated Spot-types (Raw)', fontsize=16)
if save:
    save_path = save_dir + f"/raw_annotated_clusters_umap.pdf"
    plt.savefig(save_path)
# plt.show()

cas_list = [ad.read_h5ad(save_dir + f"filtered_merged_{sample}_atac.h5ad") for sample in slice_name_list]
adata_concat = ad.concat(cas_list, label='slice_name', keys=slice_name_list)

spots_count = [0]
n = 0
for sample in cas_list:
    num = sample.shape[0]
    n += num
    spots_count.append(n)

# plot clustering results
for j, model in enumerate(models):

    embed = pd.read_csv(save_dir + f'comparison/{model}/{model}_embed_2.csv', header=None).values
    adata_concat.obsm['latent'] = embed

    gm = GaussianMixture(n_components=len(cls_list), covariance_type='tied', random_state=1234)
    y = gm.fit_predict(adata_concat.obsm['latent'], y=None)
    adata_concat.obs["gm_clusters"] = pd.Series(y, index=adata_concat.obs.index, dtype='category')
    adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(
        adata_concat.obs['Annotation_for_Combined'], adata_concat.obs["gm_clusters"]),
        index=adata_concat.obs.index, dtype='category')
    # adata_concat.obs['matched_clusters'] = list(adata_concat.obs['matched_clusters'].map(order_to_cluster_map))
    my_clusters = np.sort(list(set(adata_concat.obs['matched_clusters'])))
    matched_colors = [cluster_to_color_map[order_to_cluster_map[order]] for order in my_clusters]
    matched_to_color_map = {matched: color for matched, color in zip(my_clusters, matched_colors)}

    for i in range(len(cas_list)):
        cas_list[i].obs['matched_clusters'] = adata_concat.obs['matched_clusters'][spots_count[i]:spots_count[i+1]]

    sp_embedding = reducer.fit_transform(adata_concat.obsm['latent'])

    plot_mousebrain(cas_list, adata_concat, 'Annotation_for_Combined', 'matched_clusters', model, cluster_to_color_map,
                    matched_to_color_map, my_clusters, slice_name_list, cls_list, sp_embedding, save_root=save_dir+'comparison/',
                    frame_color=colors_for_labels[j], save=save, plot=True)






