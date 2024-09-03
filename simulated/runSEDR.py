import os
import csv
import torch
import random
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA

from ..INSTINCT import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

num_clusters = 5
num_iters = 8

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

if not os.path.exists(save_dir + 'comparison/SEDR/'):
    os.makedirs(save_dir + 'comparison/SEDR/')

cas_list = [ad.read_h5ad(f"../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
for j in range(len(cas_list)):
    cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]
adata_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)
preprocess_CAS(cas_list, adata_concat)

for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_spot_level_slice_{slice_name_list[i]}.h5ad")

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
        adata_tmp = ad.read_h5ad(save_dir + f"filtered_spot_level_slice_{proj_name}.h5ad")

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
