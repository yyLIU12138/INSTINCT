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
if not os.path.exists(save_dir + 'comparison/Scanorama/'):
    os.makedirs(save_dir + 'comparison/Scanorama/')

# Scanorama
print('----------Scanorama----------')

import scanorama

for j in range(num_iters):

    print(f'Iteration {j}')

    cas_list = [ad.read_h5ad(save_dir + f"filtered_{sample}.h5ad") for sample in slice_name_list]
    for adata in cas_list:
        adata.X = adata.X.tocsr()

    corrected = scanorama.correct_scanpy(cas_list, return_dimred=True, seed=1234+j, hvg=5000)

    adata_concat = ad.concat(corrected, label='slice_name', keys=slice_name_list)
    # print(adata_concat.obsm['X_scanorama'].shape)

    with open(save_dir + f'comparison/Scanorama/Scanorama_embed_{j}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(adata_concat.obsm['X_scanorama'])

print('----------Done----------\n')
