import numpy as np
import pandas as pd
import anndata as ad
import seaborn as sns

from sklearn.mixture import GaussianMixture
from umap.umap_ import UMAP

from ..evaluation_utils import match_cluster_labels
from plot_utils import plot_result_simulated

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1234)

save = False

scenario = 1

slice_name_list = [
    'Tech_0_0_Bio_0_0.5',
    'Tech_0_0.1_Bio_0_0',
    'Tech_0_0.1_Bio_0_0.5',
             ]

slice_index_list = list(range(len(slice_name_list)))

name_concat = slice_name_list[0]
for mode in slice_name_list[1:]:
    name_concat = name_concat + '_' + mode

save_dir = f'../../results/simulated/scenario_{scenario}/T_' + name_concat + '/'

slice_index_list = [str(i) for i in range(len(slice_name_list))]
num_iters = 8
num_clusters = 5

models = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']
color_list = ['darkviolet', 'chocolate', 'sandybrown', 'peachpuff', 'darkslategray', 'c', 'cyan']

cas_list = [ad.read_h5ad(save_dir + f"filtered_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
adata_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)

sp_cmap = {f'{i}': sns.color_palette()[i] for i in range(num_clusters)}

spots_count = [0]
n = 0
for sample in cas_list:
    num = sample.shape[0]
    n += num
    spots_count.append(n)

# plot umap
reducer = UMAP(n_neighbors=30, n_components=2, metric="correlation", n_epochs=None, learning_rate=1.0,
               min_dist=0.3, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=1, repulsion_strength=1,
               negative_sample_rate=5, a=None, b=None, random_state=1234, metric_kwds=None,
               angular_rp_forest=False, verbose=False)

for j, model_name in enumerate(models):

    embed = pd.read_csv(save_dir + f'comparison/{model_name}/{model_name}_embed_0.csv', header=None).values
    adata_concat.obsm['latent'] = embed

    gm = GaussianMixture(n_components=num_clusters, covariance_type='tied', random_state=1234)
    y = gm.fit_predict(adata_concat.obsm['latent'], y=None)
    adata_concat.obs["GM"] = y
    adata_concat.obs['my_clusters'] = pd.Series(match_cluster_labels(adata_concat.obs['real_spot_clusters'],
                                                                     adata_concat.obs["GM"]),
                                                index=adata_concat.obs.index, dtype='category')
    for i in range(len(cas_list)):
        cas_list[i].obs['my_clusters'] = adata_concat.obs['my_clusters'][spots_count[i]: spots_count[i + 1]]

    sp_embedding = reducer.fit_transform(adata_concat.obsm['latent'])

    plot_result_simulated(cas_list, adata_concat, sp_cmap, model_name, num_clusters, save_dir+f'comparison/{model_name}/',
                          sp_embedding, frame_color=color_list[j], legend=False, save=save, show=False)

