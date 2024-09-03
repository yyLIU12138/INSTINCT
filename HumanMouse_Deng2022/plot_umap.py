import csv
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

from umap.umap_ import UMAP

from matplotlib.lines import Line2D
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

save = False
file_format = 'png'

data_dir = '../../data/spCASdata/HumanMouse_Deng2022/preprocessed/'
save_dir = '../../results/HumanMouse_Deng2022/'
slice_name_list = ['GSM5238385_ME11_50um', 'GSM5238386_ME13_50um', 'GSM5238387_ME13_50um_2']
label_list = ['GSM5238385', 'GSM5238386', 'GSM5238387']
slice_used = [0, 1, 2]
slice_name_list = [slice_name_list[i] for i in slice_used]
label_list = [label_list[i] for i in slice_used]
slice_index_list = list(range(len(slice_name_list)))

save_dir = f'../../results/HumanMouse_Deng2022/{slice_used}/'

method = 'leiden'

cas_list = [ad.read_h5ad(save_dir + f"filtered_merged_{sample}.h5ad") for sample in slice_name_list]
result = ad.concat(cas_list, label="slice_name", keys=label_list)

spots_count = [0]
n = 0
for sample in cas_list:
    num = sample.shape[0]
    n += num
    spots_count.append(n)

result.obsm['INSTINCT_latent'] = pd.read_csv(save_dir + f'INSTINCT_embed.csv', header=None).values

sc.pp.neighbors(result, use_rep='INSTINCT_latent', random_state=1234)
# sc.tl.louvain(result, random_state=1234)
sc.tl.leiden(result, resolution=1, random_state=1234)
for i in range(len(cas_list)):
    # cas_list[i].obs['louvain'] = result.obs['louvain'][spots_count[i]:spots_count[i + 1]].copy()
    cas_list[i].obs['leiden'] = result.obs['leiden'][spots_count[i]:spots_count[i + 1]].copy()
    if save:
        cas_list[i].write(save_dir + f'clustered_{slice_name_list[i]}.h5ad')

reducer = UMAP(n_neighbors=30, n_components=2, metric="correlation", n_epochs=None, learning_rate=1.0,
               min_dist=0.3, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=1, repulsion_strength=1,
               negative_sample_rate=5, a=None, b=None, random_state=1234, metric_kwds=None,
               angular_rp_forest=False, verbose=True)

# raw
raw_pca = np.load(save_dir + f'input_matrix.npy')
sp_embedding = reducer.fit_transform(raw_pca)
if save:
    with open(save_dir + f'sp_embeddings_raw.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(sp_embedding)
n_spots = result.shape[0]
size = 10000 / n_spots
order = np.arange(n_spots)[::-1]
colors_for_slices = [[0.2298057, 0.29871797, 0.75368315],
                     [0.70567316, 0.01555616, 0.15023281],
                     [0.2298057, 0.70567316, 0.15023281],]
slice_cmap = {label_list[i]: colors_for_slices[i] for i in range(len(label_list))}
colors = list(result.obs['slice_name'].astype('str').map(slice_cmap))[::-1]
plt.figure(figsize=(5, 5))
plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                labelleft=False, labelbottom=False, grid_alpha=0)
legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=slice_cmap[label_list[i]], label=label_list[i])
        for i in range(len(label_list))
    ]
plt.legend(handles=legend_handles, fontsize=8, title='Slices', title_fontsize=10,
           loc='upper left')
plt.title(f'Raw', fontsize=16)
if save:
    save_path = save_dir + f"raw_slices_umap.{file_format}"
    plt.savefig(save_path)

# integrated
sp_embedding = reducer.fit_transform(result.obsm['INSTINCT_latent'])
if save:
    with open(save_dir + f'sp_embeddings_integrated.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(sp_embedding)
n_spots = result.shape[0]
size = 10000 / n_spots
order = np.arange(n_spots)[::-1]
colors_for_slices = [[0.2298057, 0.29871797, 0.75368315],
                     [0.70567316, 0.01555616, 0.15023281],
                     [0.2298057, 0.70567316, 0.15023281],]
slice_cmap = {label_list[i]: colors_for_slices[i] for i in range(len(label_list))}
colors = list(result.obs['slice_name'].astype('str').map(slice_cmap))[::-1]
plt.figure(figsize=(5, 5))
plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                labelleft=False, labelbottom=False, grid_alpha=0)
legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=slice_cmap[label_list[i]], label=label_list[i])
        for i in range(len(label_list))
    ]
plt.legend(handles=legend_handles, fontsize=8, title='Slices', title_fontsize=10,
           loc='lower left')
plt.title(f'Integrated', fontsize=16)
if save:
    save_path = save_dir + f"integrated_slices_umap.{file_format}"
    plt.savefig(save_path)


unique_labels = result.obs[method].unique()
print(len(unique_labels))
if method == 'louvain':
    color_palette = ['gold', 'dodgerblue', 'orange', 'deepskyblue',
                     'g', 'limegreen', 'gainsboro', 'y',
                     'darkorange', 'darkgray', 'saddlebrown', 'chocolate']
    if len(unique_labels) > len(color_palette):
        color_palette = sns.color_palette("tab20", n_colors=len(unique_labels))
elif method == 'leiden':
    color_palette = ['orange', 'dodgerblue', 'wheat', 'deepskyblue', 'g',
                     'gold', 'crimson', 'limegreen', 'yellowgreen', 'lightcoral',
                     'fuchsia', 'sienna', 'lightgray', 'violet', 'hotpink',]
    if len(unique_labels) > len(color_palette):
        color_palette = sns.color_palette("tab20", n_colors=len(unique_labels))
color_list = [color_palette[i] for i in range(len(unique_labels))]
color_dict = {f'{i}': color_palette[i] for i in range(len(unique_labels))}
legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8)
                   for label, color in zip(list(range(len(unique_labels))), color_list)]
colors = list(result.obs[method].astype('str').map(color_dict))[::-1]
plt.figure(figsize=(5, 5))
plt.rc('axes', linewidth=1)
plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                labelleft=False, labelbottom=False, grid_alpha=0)
plt.title(f'Identified Clusters', fontsize=16)
plt.legend(handles=legend_elements, fontsize=8, title='Clusters', title_fontsize=10, bbox_to_anchor=(1, 1))
plt.gcf().subplots_adjust(left=0.05, top=None, bottom=None, right=0.85)
if save:
    save_path = save_dir + f"{method}_identified_clusters_umap.{file_format}"
    plt.savefig(save_path)

if len(cas_list) == 2:
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
elif len(cas_list) == 3:
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle(f'Clustering Results', fontsize=16)
for i in range(len(cas_list)):
    cluster_colors = list(cas_list[i].obs[method].astype('str').map(color_dict))
    axs[i].scatter(cas_list[i].obsm['spatial'][:, 0], cas_list[i].obsm['spatial'][:, 1], linewidth=1, s=40,
                   marker=".", color=cluster_colors, alpha=0.9)
    axs[i].invert_yaxis()
    axs[i].set_title(f'{label_list[i]}', size=12)
    axs[i].axis('off')
legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8)
                   for label, color in zip(list(range(len(unique_labels))), color_list)]
axs[len(cas_list)-1].legend(handles=legend_elements,
              fontsize=8, title='Clusters', title_fontsize=10, bbox_to_anchor=(1, 1))
plt.gcf().subplots_adjust(left=0.05, top=0.8, bottom=0.05, right=0.90)
if save:
    save_path = save_dir + f'{method}_clustering_results.{file_format}'
    plt.savefig(save_path)


# load the merged data
cas_list = [ad.read_h5ad(data_dir + f'{slice_used}/merged_{sample}.h5ad') for sample in slice_name_list]
for j in range(len(cas_list)):
    cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]
    if 'in_tissue' in cas_list[j].obs.keys():
        cas_list[j] = cas_list[j][cas_list[j].obs['in_tissue'] == 1, :]
adata_concat = ad.concat(cas_list, label="slice_name", keys=label_list)
adata_concat = adata_concat[result.obs_names, :]
adata_concat.obs['nFrags'] = adata_concat.X.sum(axis=1)
adata_concat.obs[method] = result.obs[method]
mean_nFrags_per_cluster = adata_concat.obs.groupby(method)['nFrags'].mean()
for cluster, mean_nFrags in mean_nFrags_per_cluster.items():
    print(f"cluster: {cluster}, mean_nFrags: {round(mean_nFrags)}")
# plt.show()


# add gene score matrix to .h5ad data
from scipy.io import mmread
from scipy.sparse import coo_matrix, csr_matrix

slice_index_list = list(range(len(slice_name_list)))

for i in range(len(slice_name_list)):

    adata = ad.read_h5ad(save_dir + 'clustered_' + slice_name_list[i] + '.h5ad')
    obs_names = adata.obs_names
    obs_names = [slice_name_list[i] + '#' + name.split('_', 1)[0] for name in obs_names]
    adata.obs_names = obs_names

    gene_score_matrix = mmread(save_dir + f'{label_list[i]}/gene_score_matrix.mtx')

    gene_names = pd.read_csv(save_dir + f'{label_list[i]}/gene_names.txt', header=None).squeeze().tolist()
    spot_names = pd.read_csv(save_dir + f'{label_list[i]}/spot_names.txt', header=None).squeeze().tolist()

    rna_sample = ad.AnnData(X=gene_score_matrix.T)
    rna_sample.obs_names = spot_names
    rna_sample.var_names = gene_names

    if isinstance(rna_sample.X, coo_matrix):
        rna_sample.X = csr_matrix(rna_sample.X)

    rna_sample = rna_sample[adata.obs_names, :]
    rna_sample.obs = adata.obs
    rna_sample.obsm = adata.obsm
    print(rna_sample.shape)

    if save:
        rna_sample.write(save_dir + f'{label_list[i]}/clustered_{slice_name_list[i]}_rna.h5ad')

