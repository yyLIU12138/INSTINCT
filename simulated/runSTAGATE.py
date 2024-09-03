import os
import csv
import torch
import scanpy as sc
import anndata as ad

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

save_dir = f'../../results/simulated/scenario_{scenario}/single/'
if not os.path.exists(save_dir + 'STAGATE/'):
    os.makedirs(save_dir + 'STAGATE/')

# STAGATE
print('----------STAGATE----------')

import STAGATE_pyG

for i in range(num_iters):

    print(f'Iteration {i}')

    cas_list = [ad.read_h5ad(f"../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad") for mode in
                slice_name_list]
    # for j in range(len(cas_list)):
    #     cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]

    for j in range(len(cas_list)):

        adata = cas_list[j]

        # sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=4.5)
        STAGATE_pyG.Stats_Spatial_Net(adata)

        adata = STAGATE_pyG.train_STAGATE(adata, random_seed=1234+i, device=device)
        print(adata.obsm['STAGATE'].shape)

        with open(save_dir + f'STAGATE/STAGATE_embed_{i}_{slice_name_list[j]}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(adata.obsm['STAGATE'])

print('----------Done----------\n')

