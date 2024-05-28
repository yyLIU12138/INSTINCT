import os
import csv
import torch
import numpy as np
import pandas as pd
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from umap.umap_ import UMAP

from INSTINCT import *
from ..evaluation_utils import match_cluster_labels, cluster_metrics
from plot_utils import plot_result_simulated

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

reducer = UMAP(n_neighbors=30, n_components=2, metric="correlation", n_epochs=None, learning_rate=1.0,
               min_dist=0.3, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=1, repulsion_strength=1,
               negative_sample_rate=5, a=None, b=None, random_state=1234, metric_kwds=None,
               angular_rp_forest=False, verbose=False)

num_clusters = 5
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

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cas_list = [ad.read_h5ad(f"../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
for j in range(len(cas_list)):
    print(cas_list[j].shape)
    cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]
adata_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)
preprocess_CAS(cas_list, adata_concat)

adata_concat.write_h5ad(save_dir + f"preprocessed_concat.h5ad")
for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_spot_level_slice_{slice_name_list[i]}.h5ad")

cas_list = [ad.read_h5ad(save_dir + f"filtered_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
adata_concat = ad.read_h5ad(save_dir + f"preprocessed_concat.h5ad")
print(adata_concat.shape)

print(f'Applying PCA to reduce the feature dimension to 100 ...')

pca = PCA(n_components=100, random_state=1234)
input_matrix = pca.fit_transform(adata_concat.X.toarray())
np.save(save_dir + f'input_matrix.npy', input_matrix)

print('Done !')

input_matrix = np.load(save_dir + f'input_matrix.npy')
adata_concat.obsm['X_pca'] = input_matrix

# calculate the spatial graph
create_neighbor_graph(cas_list, adata_concat)

sp_cmap = {f'{i}': sns.color_palette()[i] for i in range(num_clusters)}

INSTINCT_model = INSTINCT_Model(cas_list,
                                adata_concat,
                                input_mat_key='X_pca',  # the key of the input matrix in adata_concat.obsm
                                input_dim=100,  # the input dimension
                                hidden_dims_G=[50],  # hidden dimensions of the encoder and the decoder
                                latent_dim=30,  # the dimension of latent space
                                hidden_dims_D=[50],  # hidden dimensions of the discriminator
                                lambda_adv=1,  # hyperparameter for the adversarial loss
                                lambda_cls=10,  # hyperparameter for the classification loss
                                lambda_la=20,  # hyperparameter for the latent loss
                                lambda_rec=10,  # hyperparameter for the reconstruction loss
                                seed=1234,  # random seed
                                learn_rates=[1e-3, 5e-4],  # learning rate
                                training_steps=[500, 500],  # training_steps
                                early_stop=False,  # use the latent loss to control the number of training steps
                                min_steps=500,  # the least number of steps when training the whole model
                                use_cos=True,  # use cosine similarity to find the nearest neighbors
                                margin=10,  # the margin of latent loss
                                alpha=1,  # the hyperparameter for triplet loss
                                k=50,  # the amount of neighbors to find
                                device=device)

INSTINCT_model.train(report_loss=True, report_interval=100)

INSTINCT_model.eval(cas_list)

result = ad.concat(cas_list, label="slice_index", keys=slice_index_list)

if save:
    with open(save_dir + '/INSTINCT_embed.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result.obsm['INSTINCT_latent'])

    with open(save_dir + '/INSTINCT_noise_embed.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result.obsm['INSTINCT_latent_noise'])

spots_count = [0]
n = 0
for sample in cas_list:
    num = sample.shape[0]
    n += num
    spots_count.append(n)

gm = GaussianMixture(n_components=num_clusters, covariance_type='tied', random_state=1234)
y = gm.fit_predict(result.obsm['INSTINCT_latent'], y=None)
result.obs["gm_clusters"] = pd.Series(y, index=result.obs.index, dtype='category')
result.obs['my_clusters'] = pd.Series(match_cluster_labels(result.obs['real_spot_clusters'],
                                                           result.obs["gm_clusters"]),
                                      index=result.obs.index, dtype='category')

ari, ami, nmi, fmi, comp, homo = cluster_metrics(result.obs['real_spot_clusters'],
                                                 result.obs['my_clusters'].tolist())

for i in range(len(cas_list)):
    cas_list[i].obs['my_clusters'] = result.obs['my_clusters'][spots_count[i]: spots_count[i + 1]]

sp_embedding = reducer.fit_transform(result.obsm['INSTINCT_latent'])

plot_result_simulated(cas_list, result, sp_cmap, 'INSTINCT', num_clusters, save_dir,
                      sp_embedding, frame_color='darkviolet', legend=False, save=save, show=True)

# noise umap
sp_embedding = reducer.fit_transform(result.obsm['INSTINCT_noise_latent'])

n_spots = result.shape[0]
size = 10000 / n_spots

order = np.arange(n_spots)

color_list = [[0.2298057, 0.29871797, 0.75368315],
              [0.70567316, 0.01555616, 0.15023281],
              [0.2298057, 0.70567316, 0.15023281]]
slice_cmap = {f'{i}': color_list[i] for i in range(3)}
colors = list(result.obs['slice_index'].astype('str').map(slice_cmap))

plt.figure(figsize=(5, 5))
plt.rc('axes', edgecolor='lightslategrey', linewidth=2)
plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
plt.title('Noise', fontsize=16)
plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                labelleft=False, labelbottom=False, grid_alpha=0)
plt.legend(handles=[mpatches.Patch(color=slice_cmap[f'{i}'], label=f"{i}") for i in range(len(slice_name_list))],
           fontsize=8, title='Slices', title_fontsize=10, bbox_to_anchor=(1, 1), loc=1)

if save:
    save_path = save_dir + f"/INSTINCT_noise_umap.png"
    plt.savefig(save_path)
plt.show()

