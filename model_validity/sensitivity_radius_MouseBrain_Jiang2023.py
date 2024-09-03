import os
import csv
import torch
import anndata as ad
import numpy as np
from sklearn.decomposition import PCA

from ..INSTINCT import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

num_iters = 8

# parameters
radius_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]

# mouse brain
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

save_dir = '../../results/MouseBrain_Jiang2023/'
if not os.path.exists('../../results/model_validity/MouseBrain_Jiang2023/sensitivity/radius/'):
    os.makedirs('../../results/model_validity/MouseBrain_Jiang2023/sensitivity/radius/')

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
preprocess_CAS(cas_list, adata_concat, use_fragment_count=True)
print('Done!')

adata_concat.write_h5ad(save_dir + f"preprocessed_concat_atac.h5ad")
for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_merged_{slice_name_list[i]}_atac.h5ad")

cas_list = [ad.read_h5ad(save_dir + f"filtered_merged_{sample}_atac.h5ad") for sample in slice_name_list]
adata_concat = ad.read_h5ad(save_dir + f"preprocessed_concat_atac.h5ad")

print(f'Applying PCA to reduce the feature dimension to 100 ...')
pca = PCA(n_components=100, random_state=1234)
input_matrix = pca.fit_transform(adata_concat.X.toarray())
print('Done !')

adata_concat.obsm['X_pca'] = input_matrix

n_neighbors_list = []

for i in range(len(radius_list)):

    print(f'The radius rate is {radius_list[i]}\n')

    if 'graph_list' in adata_concat.uns:
        del adata_concat.uns['graph_list']

    # calculate the spatial graph
    create_neighbor_graph(cas_list, adata_concat, rad_coef=radius_list[i])

    # calculate n_neighbors
    n_neighbors = []
    for g in adata_concat.uns['graph_list']:
        n_neighbors.append(np.mean(np.sum(g, axis=1)))
    n_neighbors_list.append(n_neighbors)

    for k in range(num_iters):

        print(f'Iteration {k}')

        INSTINCT_model = INSTINCT_Model(cas_list, adata_concat, seed=1236+k, device=device)

        INSTINCT_model.train(report_loss=False)

        INSTINCT_model.eval(cas_list)

        result = ad.concat(cas_list, label="slice_name", keys=slice_name_list)

        with open(f'../../results/model_validity/MouseBrain_Jiang2023/sensitivity/radius/rad_coef={radius_list[i]}_embed_{k}.csv',
                  'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(result.obsm['INSTINCT_latent'])

with open('../../results/model_validity/MouseBrain_Jiang2023/sensitivity/radius/n_neighbors.txt', 'w') as f:
    for sublist in n_neighbors_list:
        formatted_sublist = ["{:.2f}".format(item) for item in sublist]
        f.write(" ".join(formatted_sublist) + "\n")
