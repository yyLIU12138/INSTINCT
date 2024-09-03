import numpy as np
import anndata as ad
import pandas as pd
from umap.umap_ import UMAP
from sklearn.mixture import GaussianMixture

from plot_utils import plot_mousebrain

from ..evaluation_utils import match_cluster_labels

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

save_dir = '../../results/MouseBrain_Jiang2023/'
save = False

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

slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

cas_list = [ad.read_h5ad(save_dir + f"filtered_merged_{sample}_atac.h5ad") for sample in slice_name_list]
adata_concat = ad.concat(cas_list, label='slice_name', keys=slice_name_list)

save_dir = '../../results/model_validity/MouseBrain_Jiang2023/sensitivity/'

spots_count = [0]
n = 0
for sample in cas_list:
    num = sample.shape[0]
    n += num
    spots_count.append(n)

name = 'rad_coef=2.5'
embed = pd.read_csv(save_dir + f'radius/{name}_embed_0.csv', header=None).values
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

plot_mousebrain(cas_list, adata_concat, 'Annotation_for_Combined', 'matched_clusters', name, cluster_to_color_map,
                matched_to_color_map, my_clusters, slice_name_list, cls_list, sp_embedding, save_root=save_dir,
                frame_color='yellowgreen', save=save, plot=True)






