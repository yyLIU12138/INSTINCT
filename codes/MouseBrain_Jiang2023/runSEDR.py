import os
import csv
import torch
import random
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
from codes.INSTINCT_main.utils import preprocess_CAS, peak_sets_alignment

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

num_iters = 8

# mouse brain
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

save_dir = '../../results/MouseBrain_Jiang2023/'
if not os.path.exists(save_dir + 'comparison/SEDR/'):
    os.makedirs(save_dir + 'comparison/SEDR/')

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

# preprocess CAS data
print('Start preprocessing')
preprocess_CAS(cas_list, adata_concat, use_fragment_count=True, min_cells_rate=0.03)
print('Done!')

for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_merged_{slice_name_list[i]}_atac.h5ad")

# SEDR
print('----------SEDR----------')

import SEDR
import harmonypy as hm

for j in range(num_iters):

    print(f'Iteration {j}')

    torch.manual_seed(1234+j)
    np.random.seed(1234+j)
    random.seed(1234+j)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234+j)
    torch.backends.cudnn.benchmark = True

    cas_list = []

    for proj_name in tqdm(slice_name_list):
        adata_tmp = ad.read_h5ad(save_dir + f"filtered_merged_{proj_name}_atac.h5ad")

        adata_tmp.obs['batch_name'] = proj_name
        graph_dict_tmp = SEDR.graph_construction(adata_tmp, 12)

        if proj_name == slice_name_list[0]:
            graph_dict = graph_dict_tmp
            adata_tmp.obs['proj_name'] = proj_name
            cas_list.append(adata_tmp)
        else:
            graph_dict = SEDR.combine_graph_dict(graph_dict, graph_dict_tmp)
            adata_tmp.obs['proj_name'] = proj_name
            cas_list.append(adata_tmp)

    adata = ad.concat(cas_list, label='batch', keys=slice_name_list)

    # preprocessing
    # adata.layers['count'] = adata.X.toarray()
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X

    sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
    using_dec = False
    if using_dec:
        sedr_net.train_with_dec()
    else:
        sedr_net.train_without_dec()
    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm['SEDR'] = sedr_feat

    meta_data = adata.obs[['batch']]
    # print(meta_data)
    data_mat = adata.obsm['SEDR']
    vars_use = ['batch']
    ho = hm.run_harmony(data_mat, meta_data, vars_use, random_state=1234+j)

    # res = pd.DataFrame(ho.Z_corr).T
    # res_df = pd.DataFrame(data=res.values, columns=['X{}'.format(i + 1) for i in range(res.shape[1])],
    #                       index=adata.obs.index)
    # adata.obsm[f'SEDR.Harmony'] = res_df
    # print(adata.obsm[f'SEDR.Harmony'].shape)

    with open(save_dir + f'comparison/SEDR/SEDR_embed_{j}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(ho.Z_corr.T)

print('----------Done----------\n')
