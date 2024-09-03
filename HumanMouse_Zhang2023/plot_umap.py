import csv
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

CAS_samples = ["GSM6206884_HumanBrain_50um", "GSM6801813_ME13_50um", "GSM6204624_ME13_100barcodes_25um",
               "GSM6204623_MouseBrain_20um", "GSM6758284_MouseBrain_20um_repATAC", "GSM6758285_MouseBrain_20um_100barcodes_ATAC",
               "GSM6204621_MouseBrain_20um_H3K27ac", "GSM6704977_MouseBrain_20um_rep_H3K27ac", "GSM6704978_MouseBrain_20um_100barcodes_H3K27me3",
               "GSM6704979_MouseBrain_20um_100barcodes_H3K27ac", "GSM6704980_MouseBrain_20um_100barcodes_H3K4me3"]
RNA_samples = ["GSM6206885_HumanBrain_50um", "GSM6799937_ME13_50um", "GSM6204637_ME13_100barcodes_25um",
               "GSM6204636_MouseBrain_20um", "GSM6753041_MouseBrain_20um_repATAC", "GSM6753043_MouseBrain_20um_100barcodes_ATAC",
               "GSM6204635_MouseBrain_20um_H3K27ac", "GSM6753042_MouseBrain_20um_repH3K27ac", "GSM6753044_MouseBrain_20um_100barcodes_H3K27me3",
               "GSM6753045_MouseBrain_20um_100barcodes_H3K27ac", "GSM6753046_MouseBrain_20um_100barcodes_H3K4me3"]

save = False
slice_name_list = ["GSM6204623_MouseBrain_20um", "GSM6758284_MouseBrain_20um_repATAC", "GSM6758285_MouseBrain_20um_100barcodes_ATAC"]
label_list = ["GSM6204623", "GSM6758284", "GSM6758285"]
slice_index_list = list(range(len(slice_name_list)))

data_dir = '../../data/spMOdata/EpiTran_HumanMouse_Zhang2023/preprocessed_from_fragments/'
save_dir = f'../../results/HumanMouse_Zhang2023/mb/'

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
sc.tl.louvain(result, random_state=1234)
sc.tl.leiden(result, random_state=1234)
for i in range(len(cas_list)):
    cas_list[i].obs['louvain'] = result.obs['louvain'][spots_count[i]:spots_count[i + 1]].copy()
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
    with open(save_dir + 'sp_embeddings_raw.csv', 'w', newline='') as file:
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
    save_path = save_dir + f"raw_slices_umap.pdf"
    plt.savefig(save_path)

# integrated
sp_embedding = reducer.fit_transform(result.obsm['INSTINCT_latent'])
if save:
    with open(save_dir + 'sp_embeddings_integrated.csv', 'w', newline='') as file:
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
    save_path = save_dir + f"integrated_slices_umap.pdf"
    plt.savefig(save_path)


unique_labels = result.obs[method].unique()
# color_palette = sns.color_palette("tab20", n_colors=len(unique_labels))
# color_list = [color_palette[i] for i in range(len(unique_labels))]
if method == 'louvain':
    color_palette = ['gold', 'dodgerblue', 'orange', 'deepskyblue',
                     'g', 'limegreen', 'gainsboro', 'y',
                     'darkorange', 'darkgray', 'saddlebrown', 'chocolate']
elif method == 'leiden':
    color_palette = ['dodgerblue', 'orange', 'gold', 'g', 'limegreen',
                     'darkorange', 'deepskyblue', 'y', 'wheat',
                     'darkgray', 'chocolate', 'saddlebrown', 'violet']
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
    save_path = save_dir + f"{method}_identified_clusters_umap.pdf"
    plt.savefig(save_path)

# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle(f'Clustering Results', fontsize=16)
for i in range(len(cas_list)):
    cluster_colors = list(cas_list[i].obs[method].astype('str').map(color_dict))
    if i == 2:
        s = 10
    else:
        s = 40
    axs[i].scatter(cas_list[i].obsm['spatial'][:, 1], cas_list[i].obsm['spatial'][:, 0], linewidth=1, s=s,
                   marker=".", color=cluster_colors, alpha=0.9)
    if i != 1:
        axs[i].invert_yaxis()
    if i != 0:
        axs[i].invert_xaxis()
    axs[i].set_title(f'{label_list[i]}', size=12)
    axs[i].axis('off')
legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8)
                   for label, color in zip(list(range(len(unique_labels))), color_list)]
axs[len(cas_list)-1].legend(handles=legend_elements,
              fontsize=8, title='Clusters', title_fontsize=10, bbox_to_anchor=(1, 1))
plt.gcf().subplots_adjust(left=0.05, top=0.8, bottom=0.05, right=0.90)
if save:
    save_path = save_dir + f'{method}_clustering_results.pdf'
    plt.savefig(save_path)
plt.show()

