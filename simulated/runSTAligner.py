import os
import csv
import torch
import scipy
import numpy as np
import scanpy as sc
import anndata as ad

from INSTINCT import *

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

if not os.path.exists(save_dir + 'comparison/STAligner/'):
    os.makedirs(save_dir + 'comparison/STAligner/')

cas_list = [ad.read_h5ad(f"../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
for j in range(len(cas_list)):
    cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]
adata_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)
preprocess_CAS(cas_list, adata_concat)

for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_spot_level_slice_{slice_name_list[i]}.h5ad")

# STAligner
print('----------STAligner----------')

import STAligner

for j in range(num_iters):

    print(f'Iteration {j}')

    cas_list = []
    adj_list = []

    for mode in slice_name_list:

        print(mode)

        adata = ad.read_h5ad(save_dir + f"filtered_spot_level_slice_{mode}.h5ad")

        # Constructing the spatial network
        STAligner.Cal_Spatial_Net(adata, rad_cutoff=4.5)  # the spatial network are saved in adata.uns[‘adj’]

        # Normalization
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]

        adj_list.append(adata.uns['adj'])
        cas_list.append(adata)

    adata_concat = ad.concat(cas_list, label="slice_index", keys=slice_index_list)
    adata_concat.obs['real_spot_clusters'] = adata_concat.obs['real_spot_clusters'].astype('category')
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_index"].astype(str).astype('category')
    print('adata_concat.shape: ', adata_concat.shape)

    adj_concat = np.asarray(adj_list[0].todense())
    for batch_id in range(1, len(slice_name_list)):
        adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
    adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

    adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=50,
                                             device=device, random_seed=1234+j)

    with open(save_dir + f'comparison/STAligner/STAligner_embed_{j}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(adata_concat.obsm['STAligner'])

print('----------Done----------\n')
