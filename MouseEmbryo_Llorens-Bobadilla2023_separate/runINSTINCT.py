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

num_iters = 8

# mouse embryo
data_dir = '../../data/spCASdata/MouseEmbryo_Llorens-Bobadilla2023/spATAC/'
save_dir = '../../results/MouseEmbryo_Llorens-Bobadilla2023/separate/'

mode = 'S1'
slice_name_list = [f'E12_5-{mode}', f'E13_5-{mode}', f'E15_5-{mode}']
slice_index_list = list(range(len(slice_name_list)))

if not os.path.exists(save_dir + f'{mode}/comparison/INSTINCT/'):
    os.makedirs(save_dir + f'{mode}/comparison/INSTINCT/')

# load dataset
cas_list = [ad.read_h5ad(data_dir + sample + '.h5ad') for sample in slice_name_list]
for i in range(len(cas_list)):
    cas_list[i].obs_names = [x + '_' + slice_name_list[i] for x in cas_list[i].obs_names]

# concatenation
adata_concat = ad.concat(cas_list, label="slice_name", keys=slice_name_list)

# preprocess CAS data
# peaks are already merged and fragment counts are stored in the data matrices
print('Start preprocessing')
preprocess_CAS(cas_list, adata_concat, min_cells_rate=0.003)
print('Done!')
print(adata_concat)

adata_concat.write_h5ad(save_dir + f"preprocessed_concat_{mode}.h5ad")
for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_{slice_name_list[i]}.h5ad")

cas_list = [ad.read_h5ad(save_dir + f"filtered_{sample}.h5ad") for sample in slice_name_list]
adata_concat = ad.read_h5ad(save_dir + f"preprocessed_concat_{mode}.h5ad")

print(f'Applying PCA to reduce the feature dimension to 100 ...')
pca = PCA(n_components=100, random_state=1234)
input_matrix = pca.fit_transform(adata_concat.X.toarray())
np.save(save_dir + f'input_matrix_{mode}.npy', input_matrix)
print('Done !')

input_matrix = np.load(save_dir + f'input_matrix_{mode}.npy')
adata_concat.obsm['X_pca'] = input_matrix

# calculate the spatial graph
create_neighbor_graph(cas_list, adata_concat)

# INSTINCT
print('----------INSTINCT----------')

for j in range(num_iters):

    print(f'Iteration {j}')

    INSTINCT_model = INSTINCT_Model(cas_list,
                                    adata_concat,
                                    seed=1234+j,
                                    device=device)

    INSTINCT_model.train(report_loss=True, report_interval=100)

    INSTINCT_model.eval(cas_list)

    result = ad.concat(cas_list, label="slice_idx", keys=slice_index_list)

    with open(save_dir + f'{mode}/comparison/INSTINCT/INSTINCT_embed_{j}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result.obsm['INSTINCT_latent'])

print('----------Done----------\n')
