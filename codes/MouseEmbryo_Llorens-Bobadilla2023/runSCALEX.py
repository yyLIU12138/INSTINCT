import os
import csv
import torch
import anndata as ad

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

if not os.path.exists(save_dir + 'comparison/SCALEX/'):
    os.makedirs(save_dir + 'comparison/SCALEX/')

# SCALE
print('----------SCALEX----------')

from scalex import SCALEX

for j in range(num_iters):

    print(f'Iteration {j}')

    cas_list = [ad.read_h5ad(save_dir + f"filtered_{sample}.h5ad") for sample in slice_name_list]
    for i, adata in enumerate(cas_list):
        adata.obs['batch'] = slice_name_list[i]

    # during interation 4, the random seed of SCALEX was change from 1234+4 to 1234+8
    # since error occurs when setting the seed to 1234+4
    if j == 4:
        adata_concat = SCALEX(cas_list, batch_key='batch', profile='ATAC', min_features=1, n_top_features=30000,
                              seed=1242, target_sum=None, ignore_umap=True, outdir=save_dir + 'comparison/SCALEX/')
    else:
        adata_concat = SCALEX(cas_list, batch_key='batch', profile='ATAC', min_features=1, n_top_features=30000,
                              seed=1234 + j, target_sum=None, ignore_umap=True, outdir=save_dir + 'comparison/SCALEX/')
    # print(adata_concat)

    with open(save_dir + f'comparison/SCALEX/SCALEX_embed_{j}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(adata_concat.obsm['latent'])

print('----------Done----------\n')
