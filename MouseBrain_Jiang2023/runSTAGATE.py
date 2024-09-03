import os
import csv
import torch
import anndata as ad
import scanpy as sc

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

num_iters = 8

# mouse brain
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

slice_index_list = list(range(len(slice_name_list)))

save_dir = '../../results/MouseBrain_Jiang2023/single/'
if not os.path.exists(save_dir + 'STAGATE/'):
    os.makedirs(save_dir + 'STAGATE/')

# STAGATE
print('----------STAGATE----------')

import STAGATE_pyG

for i in range(num_iters):

    print(f'Iteration {i}')

    # load raw data
    cas_list = []
    for sample in slice_name_list:
        sample_data = ad.read_h5ad(data_dir + sample + '_atac.h5ad')

        if 'insertion' in sample_data.obsm:
            del sample_data.obsm['insertion']

        cas_list.append(sample_data)

    for j in range(len(cas_list)):

        adata = cas_list[j]

        # sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=1.5)
        STAGATE_pyG.Stats_Spatial_Net(adata)

        adata = STAGATE_pyG.train_STAGATE(adata, random_seed=1234+i, device=device)
        print(adata.obsm['STAGATE'].shape)

        with open(save_dir + f'STAGATE/STAGATE_embed_{i}_{slice_name_list[j]}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(adata.obsm['STAGATE'])

print('----------Done----------\n')

