import os
import csv
import torch
import scipy
import numpy as np
import anndata as ad
import scanpy as sc

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

num_iters = 8

# mouse embryo
data_dir = '../../data/spCASdata/MouseEmbryo_Llorens-Bobadilla2023/spATAC/'
save_dir = '../../results/MouseEmbryo_Llorens-Bobadilla2023/all/'

slice_name_list = ['E12_5-S1', 'E12_5-S2', 'E13_5-S1', 'E13_5-S2', 'E15_5-S1', 'E15_5-S2']
slice_index_list = list(range(len(slice_name_list)))

if not os.path.exists(save_dir + 'comparison/STAligner/'):
    os.makedirs(save_dir + 'comparison/STAligner/')

# STAligner
print('----------STAligner----------')

import STAligner

for j in range(num_iters):

    print(f'Iteration {j}')

    cas_list = []
    adj_list = []

    for i, sample in enumerate(slice_name_list):

        print(sample)

        adata = ad.read_h5ad(save_dir + f"filtered_{sample}.h5ad")

        # Constructing the spatial network
        STAligner.Cal_Spatial_Net(adata, rad_cutoff=150)  # the spatial network are saved in adata.uns[‘adj’]

        # Normalization
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]

        adj_list.append(adata.uns['adj'])
        cas_list.append(adata)

    adata_concat = ad.concat(cas_list, label="slice_name", keys=slice_name_list)
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype(str).astype('category')
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

