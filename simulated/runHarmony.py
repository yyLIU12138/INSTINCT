import os
import csv
import torch
import anndata as ad
import numpy as np
import scanpy as sc
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

if not os.path.exists(save_dir + 'comparison/Harmony/'):
    os.makedirs(save_dir + 'comparison/Harmony/')

cas_list = [ad.read_h5ad(f"../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
for j in range(len(cas_list)):
    cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]
adata_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)
preprocess_CAS(cas_list, adata_concat)

adata_concat.write_h5ad(save_dir + f"preprocessed_concat.h5ad")
for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_spot_level_slice_{slice_name_list[i]}.h5ad")

# Harmony
print('----------Harmony----------')

import harmonypy as hm

for j in range(num_iters):

    print(f'Iteration {j}')

    cas_list = [ad.read_h5ad(save_dir + f"filtered_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
    adata_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)

    sc.pp.normalize_total(adata_concat, target_sum=1e4)
    sc.pp.log1p(adata_concat)
    highly_variable_peaks = set()
    for adata in cas_list:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        highly_variable_peaks.update(adata.var[adata.var['highly_variable']].index)
    highly_variable_mask = np.isin(adata_concat.var.index, list(highly_variable_peaks))
    adata_concat = adata_concat[:, highly_variable_mask]
    sc.pp.scale(adata_concat)

    pca = PCA(n_components=100, random_state=1234)
    data_mat = pca.fit_transform(adata_concat.X)
    meta_data = adata_concat.obs[['slice_index']]
    vars_use = ['slice_index']

    ho = hm.run_harmony(data_mat, meta_data, vars_use, random_state=1234+j)

    with open(save_dir + f'comparison/Harmony/Harmony_embed_{j}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(ho.Z_corr.T)

print('----------Done----------\n')


if not os.path.exists(save_dir + 'comparison/Harmony_same_input/'):
    os.makedirs(save_dir + 'comparison/Harmony_same_input/')

# Harmony (same input with INSTINCT)
print('----------Harmony (same input as INSTINCT)----------')

for j in range(num_iters):

    print(f'Iteration {j}')

    cas_list = [ad.read_h5ad(save_dir + f"filtered_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
    adata_concat = ad.read_h5ad(save_dir + f"preprocessed_concat.h5ad")

    print(f'Applying PCA to reduce the feature dimension to 100 ...')
    pca = PCA(n_components=100, random_state=1234)
    data_mat = pca.fit_transform(adata_concat.X.toarray())
    np.save(save_dir + f'input_matrix.npy', data_mat)
    print('Done !')

    data_mat = np.load(save_dir + f'input_matrix.npy')
    meta_data = adata_concat.obs[['slice_index']]
    vars_use = ['slice_index']

    ho = hm.run_harmony(data_mat, meta_data, vars_use, random_state=1234+j)

    with open(save_dir + f'comparison/Harmony_same_input/Harmony_same_input_embed_{j}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(ho.Z_corr.T)

print('----------Done----------\n')
