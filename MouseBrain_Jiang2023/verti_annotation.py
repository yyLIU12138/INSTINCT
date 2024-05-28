import pandas as pd
import anndata as ad

from umap.umap_ import UMAP
from plot_utils import plot_mousebrain_verti
from ..evaluation_utils import knn_label_translation

from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

save = False
model = 'INSTINCT'
mode_index = 3
mode_list = ['E11_0', 'E13_5', 'E15_5', 'E18_5']
mode = mode_list[mode_index]

# mouse brain
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
save_dir = f'../../results/MouseBrain_Jiang2023/vertical/{mode}/'
slice_name_list = [f'{mode}-S1', f'{mode}-S2']

cas_list = [ad.read_h5ad(save_dir + f'filtered_merged_{sample}_atac.h5ad') for sample in slice_name_list]
cas_list[1].obs['Annotation_for_Combined'] = 'Unidentified'
adata_concat = ad.concat(cas_list, label="slice_name", keys=slice_name_list)

spots_count = [0]
n = 0
for sample in cas_list:
    num = sample.shape[0]
    n += num
    spots_count.append(n)

# from sklearn.decomposition import PCA
# from codes.INSTINCT.utils import preprocess_CAS
# preprocess_CAS(cas_list, adata_concat, use_fragment_count=True, min_cells_rate=0.03)
# # pca = PCA(n_components=100, random_state=1234)
# # input_matrix = pca.fit_transform(adata_concat.X.toarray())
# pca = PCA(n_components=30, random_state=1234)
# adata_concat.obsm['latent'] = pca.fit_transform(adata_concat.X.toarray())

adata_concat.obsm['latent'] = pd.read_csv(save_dir + f'{model}/{mode}_INSTINCT_embed.csv', header=None).values
for j in range(len(cas_list)):
    cas_list[j].obsm['latent'] = adata_concat.obsm['latent'][spots_count[j]:spots_count[j + 1]].copy()

cas_list[1].obs['predicted_labels'] = knn_label_translation(cas_list[0].obsm['latent'].copy(),
                                                            cas_list[0].obs['Annotation_for_Combined'].copy(),
                                                            cas_list[1].obsm['latent'].copy(), k=20)

if save:
    cas_list[1].write(save_dir + f'{model}/annotated_{slice_name_list[1]}_atac.h5ad')

reducer = UMAP(n_neighbors=30, n_components=2, metric="correlation", n_epochs=None, learning_rate=1.0,
               min_dist=0.3, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=1, repulsion_strength=1,
               negative_sample_rate=5, a=None, b=None, random_state=1234, metric_kwds=None,
               angular_rp_forest=False, verbose=False)

sp_embedding = reducer.fit_transform(adata_concat.obsm['latent'])

cls_list_all = ['Primary_brain_1', 'Primary_brain_2', 'Midbrain', 'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
                'Subpallium_1', 'Subpallium_2', 'Cartilage_1', 'Cartilage_2', 'Cartilage_3', 'Cartilage_4',
                'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv']

colors_for_all = ['red', 'tomato', 'chocolate', 'orange', 'goldenrod',
                  'b', 'royalblue', 'g', 'limegreen', 'lime', 'springgreen',
                  'deepskyblue', 'pink', 'fuchsia', 'yellowgreen', 'olivedrab']

cls_list = list(set(list(cas_list[0].obs['Annotation_for_Combined'])))
cls_list_reordered = [cls for cls in cls_list_all if cls in cls_list]
colors_for_clusters = [colors_for_all[i] for i in range(len(colors_for_all)) if cls_list_all[i] in cls_list]

cluster_to_color_map = {cluster: color for cluster, color in zip(cls_list_reordered, colors_for_clusters)}
print(cluster_to_color_map)

plot_mousebrain_verti(cas_list, adata_concat, 'Annotation_for_Combined', 'predicted_labels', cluster_to_color_map,
                      slice_name_list, cls_list_reordered, sp_embedding, mode,
                      save_root=save_dir+f'{model}/', save=save, plot=True)



