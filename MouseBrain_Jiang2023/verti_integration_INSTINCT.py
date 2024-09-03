import os
import csv
import torch
import numpy as np
import anndata as ad

from sklearn.decomposition import PCA

from ..INSTINCT import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

save = False
mode_index = 3
mode_list = ['E11_0', 'E13_5', 'E15_5', 'E18_5']
mode = mode_list[mode_index]

# mouse brain
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
if not os.path.exists(data_dir + f'{mode}/'):
    os.makedirs(data_dir + f'{mode}/')
save_dir = f'../../results/MouseBrain_Jiang2023/vertical/{mode}/'
if not os.path.exists(save_dir + 'INSTINCT/'):
    os.makedirs(save_dir + 'INSTINCT/')
slice_name_list = [f'{mode}-S1', f'{mode}-S2']

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
    adata.write_h5ad(f'{data_dir}{mode}/merged_{slice_name_list[idx]}_atac.h5ad')

# load the merged data
cas_list = [ad.read_h5ad(data_dir + mode + '/merged_' + sample + '_atac.h5ad') for sample in slice_name_list]
for j in range(len(cas_list)):
    cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]

# concatenation
adata_concat = ad.concat(cas_list, label="slice_name", keys=slice_name_list)
# adata_concat.obs_names_make_unique()

# preprocess CAS data
print('Start preprocessing')
preprocess_CAS(cas_list, adata_concat, use_fragment_count=True, min_cells_rate=0.03)
print(adata_concat.shape)
print('Done!')

adata_concat.write_h5ad(save_dir + f"{mode}_preprocessed_concat_atac.h5ad")
for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_merged_{slice_name_list[i]}_atac.h5ad")

cas_list = [ad.read_h5ad(save_dir + f"filtered_merged_{sample}_atac.h5ad") for sample in slice_name_list]
# origin_concat = ad.concat(cas_list, label="slice_name", keys=slice_name_list)
adata_concat = ad.read_h5ad(save_dir + f"{mode}_preprocessed_concat_atac.h5ad")

print(f'Applying PCA to reduce the feature dimension to 100 ...')
pca = PCA(n_components=100, random_state=1234)
input_matrix = pca.fit_transform(adata_concat.X.toarray())
np.save(save_dir + f'{mode}_input_matrix_atac.npy', input_matrix)
print('Done !')

input_matrix = np.load(save_dir + f'{mode}_input_matrix_atac.npy')
adata_concat.obsm['X_pca'] = input_matrix

# calculate the spatial graph
create_neighbor_graph(cas_list, adata_concat)

INSTINCT_model = INSTINCT_Model(cas_list, adata_concat, device=device)

INSTINCT_model.train(report_loss=True, report_interval=100)

INSTINCT_model.eval(cas_list)

result = ad.concat(cas_list, label="slice_name", keys=slice_name_list)

if save:
    with open(save_dir + f'INSTINCT/{mode}_INSTINCT_embed.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result.obsm['INSTINCT_latent'])

    with open(save_dir + f'INSTINCT/{mode}_INSTINCT_noise_embed.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result.obsm['INSTINCT_latent_noise'])

