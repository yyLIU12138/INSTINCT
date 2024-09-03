import os
import numpy as np
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


save = False
method = 'leiden'
file_format = 'png'
cmap = 'bwr'

data_dir = '../../data/spCASdata/HumanMouse_Deng2022/preprocessed/'
save_dir = '../../results/HumanMouse_Deng2022/'
slice_name_list = ['GSM5238385_ME11_50um', 'GSM5238386_ME13_50um', 'GSM5238387_ME13_50um_2']
label_list = ['GSM5238385', 'GSM5238386', 'GSM5238387']
slice_used = [0, 1, 2]
slice_name_list = [slice_name_list[i] for i in slice_used]
label_list = [label_list[i] for i in slice_used]
slice_index_list = list(range(len(slice_name_list)))

save_dir = f'../../results/HumanMouse_Deng2022/{slice_used}/'

motif_list = ['Gata1', 'Gata4', 'Gabpa', 'Klf1', 'Klf12', 'FOS::JUN', 'Nfe2l2',
              'Pou5f1::Sox2', 'Rfx1', 'Pou2f3', 'Sox3', 'Sox6', 'Sox2']
save_name_list = ['Gata1', 'Gata4', 'Gabpa', 'Klf1', 'Klf12', 'FOS_JUN', 'Nfe2l2',
                  'Pou5f1_Sox2', 'Rfx1', 'Pou2f3', 'Sox3', 'Sox6', 'Sox2']

motif_scores = pd.read_csv(save_dir + f'concat/motif_enrichment_analysis/sorted_devs.csv', index_col=0, header=0)

for k, motif in enumerate(motif_list):

    print(motif)

    if not os.path.exists(save_dir + f'concat/motif_enrichment_analysis/{save_name_list[k]}/') and save:
        os.makedirs(save_dir + f'concat/motif_enrichment_analysis/{save_name_list[k]}/')

    adata_concat = ad.read_h5ad(save_dir + f'concat/selected_concat.h5ad')
    print(adata_concat.shape)

    adata_concat.obsm['raw_emb'] = pd.read_csv(save_dir + f'sp_embeddings_raw.csv', header=None).values
    adata_concat.obsm['inte_emb'] = pd.read_csv(save_dir + f'sp_embeddings_integrated.csv', header=None).values

    adata_concat = adata_concat[adata_concat.obs_names.isin(motif_scores.index)]
    adata_concat.obs[motif] = motif_scores.loc[adata_concat.obs_names, motif].values
    adata_concat = adata_concat[~adata_concat.obs[motif].isna()]
    print(adata_concat.shape)

    data_column = adata_concat.obs[motif].copy()

    # raw
    sp_embedding = adata_concat.obsm['raw_emb'].copy()
    n_spots = adata_concat.shape[0]
    size = 10000 / n_spots
    order = np.arange(n_spots)
    plt.figure(figsize=(6, 5))
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=data_column, cmap=cmap)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    plt.colorbar(label='Color')

    plt.title(f'Raw ({motif})', fontsize=14)
    if save:
        save_path = save_dir + f"concat/motif_enrichment_analysis/{save_name_list[k]}/{save_name_list[k]}_raw_umap.{file_format}"
        plt.savefig(save_path)

    # integrated
    sp_embedding = adata_concat.obsm['inte_emb'].copy()
    n_spots = adata_concat.shape[0]
    size = 10000 / n_spots
    order = np.arange(n_spots)
    plt.figure(figsize=(6, 5))
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=data_column, cmap=cmap)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    plt.colorbar(label='Color')

    plt.title(f'Integrated ({motif})', fontsize=14)
    if save:
        save_path = save_dir + f"concat/motif_enrichment_analysis/{save_name_list[k]}/{save_name_list[k]}_integrated_umap.{file_format}"
        plt.savefig(save_path)

    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap)
    colors = scalar_mappable.to_rgba(data_column)

    spots_count = [0]
    n = 0
    for label in label_list:
        filtered_rows = adata_concat.obs[adata_concat.obs['slice_name'] == label]
        num = filtered_rows.shape[0]
        n += num
        spots_count.append(n)

    if len(slice_name_list) == 2:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    elif len(slice_name_list) == 3:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(len(slice_name_list)):
        cluster_colors = list(colors[spots_count[i]: spots_count[i + 1]])
        axs[i].scatter(adata_concat[spots_count[i]: spots_count[i + 1]].obsm['spatial'][:, 0],
                       adata_concat[spots_count[i]: spots_count[i + 1]].obsm['spatial'][:, 1],
                       linewidth=1, s=40, marker=".", color=cluster_colors, alpha=0.9)
        axs[i].invert_yaxis()
        axs[i].set_title(f'{label_list[i]} ({motif})', size=12)
        axs[i].axis('off')
    plt.gcf().subplots_adjust(left=0.05, top=0.8, bottom=0.05, right=0.90)
    if save:
        save_path = save_dir + f'concat/motif_enrichment_analysis/{save_name_list[k]}/{save_name_list[k]}_spatial.{file_format}'
        plt.savefig(save_path)
    plt.show()

