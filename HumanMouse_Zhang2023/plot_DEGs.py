import os
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

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
rna_slice_name_list = ["GSM6204636_MouseBrain_20um", "GSM6753041_MouseBrain_20um_repATAC", "GSM6753043_MouseBrain_20um_100barcodes_ATAC"]
label_list = ["GSM6204623", "GSM6758284", "GSM6758285"]
slice_index_list = list(range(len(slice_name_list)))

data_dir = '../../data/spMOdata/EpiTran_HumanMouse_Zhang2023/preprocessed_from_fragments/'
save_dir = f'../../results/HumanMouse_Zhang2023/mb/'

marker_list = ['Rgs9', 'Pde10a', 'Gng7', 'Bcl11b', 'Foxp1',
               'Plp1', 'Mbp', 'Tspan2',
               'Dgkg',
               'Sox4', 'Dlx1',  # 'Zbtb20',
               'Isl1', 'Rreb1',
               'Mef2c',
               ]

marker = 'Plp1'
if not os.path.exists(save_dir + f'{marker}/') and save:
    os.makedirs(save_dir + f'{marker}/')

# read the filtered and annotated CAS data
cas_list = [ad.read_h5ad(save_dir + f'clustered_{sample}.h5ad') for sample in slice_name_list]
cas_concat = ad.concat(cas_list, label='slice_name', keys=slice_name_list)
cas_concat.obsm['X_pca'] = np.load(save_dir + f'input_matrix.npy')
cas_concat.obsm['INSTINCT_latent'] = pd.read_csv(save_dir + f'INSTINCT_embed.csv', header=None).values
cas_concat.obsm['raw_emb'] = pd.read_csv(save_dir + f'sp_embeddings_raw.csv', header=None).values
cas_concat.obsm['inte_emb'] = pd.read_csv(save_dir + f'sp_embeddings_integrated.csv', header=None).values

# read the raw RNA data
rna_list = [ad.read_h5ad(data_dir + f'{sample}.h5ad') for sample in rna_slice_name_list]
for j in range(len(rna_list)):
    rna_list[j].obs_names = [x + '-1_' + slice_name_list[j] for x in rna_list[j].obs_names]
    print(rna_list[j].shape)

# filter and reorder spots in rna slices
for i in range(len(slice_name_list)):
    obs_list = [obs_name for obs_name in cas_list[i].obs_names if obs_name in rna_list[i].obs_names]
    cas_list[i] = cas_list[i][obs_list, :]
    rna_list[i] = rna_list[i][obs_list, :]
    print(rna_list[i].shape)

# concatenate the rna slices
rna_concat = ad.concat(rna_list, label='slice_name', keys=slice_name_list)
sc.pp.filter_genes(rna_concat, min_cells=500)
sc.pp.normalize_total(rna_concat, target_sum=1e4)
sc.pp.log1p(rna_concat)
print(rna_concat.shape)
cas_concat = cas_concat[rna_concat.obs_names, :]
rna_concat.obsm['X_pca'] = cas_concat.obsm['X_pca']
rna_concat.obsm['INSTINCT_latent'] = cas_concat.obsm['INSTINCT_latent']
rna_concat.obsm['raw_emb'] = cas_concat.obsm['raw_emb']
rna_concat.obsm['inte_emb'] = cas_concat.obsm['inte_emb']

data_column = rna_concat[:, marker].copy()

# raw
sp_embedding = rna_concat.obsm['raw_emb'].copy()
n_spots = rna_concat.shape[0]
size = 10000 / n_spots
order = np.arange(n_spots)
plt.figure(figsize=(6, 5))
plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=data_column.X.toarray(), cmap='viridis')
plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                labelleft=False, labelbottom=False, grid_alpha=0)
plt.colorbar(label='Color')

plt.title(f'Raw', fontsize=16)
if save:
    save_path = save_dir + f"{marker}/{marker}_raw_umap.pdf"
    plt.savefig(save_path)

# integrated
sp_embedding = rna_concat.obsm['inte_emb'].copy()
n_spots = rna_concat.shape[0]
size = 10000 / n_spots
order = np.arange(n_spots)
plt.figure(figsize=(6, 5))
plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=data_column.X.toarray(), cmap='viridis')
plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                labelleft=False, labelbottom=False, grid_alpha=0)
plt.colorbar(label='Color')

plt.title(f'Integrated', fontsize=16)
if save:
    save_path = save_dir + f"{marker}/{marker}_integrated_umap.pdf"
    plt.savefig(save_path)

scalar_mappable = plt.cm.ScalarMappable(cmap='viridis')
colors = scalar_mappable.to_rgba(data_column.X.toarray())

spots_count = [0]
n = 0
for sample in rna_list:
    num = sample.shape[0]
    n += num
    spots_count.append(n)

# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for i in range(len(rna_list)):
    cluster_colors = list(colors[spots_count[i]: spots_count[i+1]])
    if i == 2:
        s = 10
    else:
        s = 40
    axs[i].scatter(rna_list[i].obsm['spatial'][:, 1], rna_list[i].obsm['spatial'][:, 0], linewidth=1, s=s,
                   marker=".", color=cluster_colors, alpha=0.9)
    if i != 1:
        axs[i].invert_yaxis()
    if i != 0:
        axs[i].invert_xaxis()
    axs[i].set_title(f'{label_list[i]}', size=12)
    axs[i].axis('off')
# plt.colorbar(label='Color')
plt.gcf().subplots_adjust(left=0.05, top=0.8, bottom=0.05, right=0.90)
if save:
    save_path = save_dir + f'{marker}/{marker}_slices.pdf'
    plt.savefig(save_path)
plt.show()
