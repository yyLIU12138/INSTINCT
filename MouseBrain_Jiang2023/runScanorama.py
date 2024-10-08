import os
import csv
import torch
import anndata as ad

from ..INSTINCT import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

num_iters = 8

# mouse brain
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

save_dir = '../../results/MouseBrain_Jiang2023/'
if not os.path.exists(save_dir + 'comparison/Scanorama/'):
    os.makedirs(save_dir + 'comparison/Scanorama/')

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

# Harmony
print('----------Scanorama----------')

import scanorama

for j in range(num_iters):

    print(f'Iteration {j}')

    cas_list = [ad.read_h5ad(save_dir + f"filtered_merged_{sample}_atac.h5ad") for sample in slice_name_list]
    for adata in cas_list:
        adata.X = adata.X.tocsr()

    corrected = scanorama.correct_scanpy(cas_list, return_dimred=True, seed=1234+j, hvg=5000)

    adata_concat = ad.concat(corrected, label='slice_name', keys=slice_name_list)
    print(adata_concat.obsm['X_scanorama'].shape)

    with open(save_dir + f'comparison/Scanorama/Scanorama_embed_{j}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(adata_concat.obsm['X_scanorama'])

print('----------Done----------\n')
