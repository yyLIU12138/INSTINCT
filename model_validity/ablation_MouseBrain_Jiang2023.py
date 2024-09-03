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

# mouse brain
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

save_dir = '../../results/MouseBrain_Jiang2023/'
if not os.path.exists('../../results/model_validity/MouseBrain_Jiang2023/'):
    os.makedirs('../../results/model_validity/MouseBrain_Jiang2023/')

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
np.save(save_dir + 'input_matrix_atac.npy', input_matrix)
print('Done !')

input_matrix = np.load(save_dir + 'input_matrix_atac.npy')
adata_concat.obsm['X_pca'] = input_matrix

# calculate the spatial graph
create_neighbor_graph(cas_list, adata_concat)

# in order to test the model without discriminator and noise generator, the model should be changed manually
titles = ['complete', 'without_Loss_adv', 'without_Loss_cls',
          'without_Loss_la', 'without_Loss_rec', 'without_D',
          'without_D_NG', 'use_euclidean', 'without_clamp']
params = [[1, 10, 20, 10, True, 10], [0, 10, 20, 10, True, 10], [1, 0, 20, 10, True, 10],
          [1, 10, 0, 10, True, 10], [1, 10, 20, 0, True, 10], [0, 0, 20, 10, True, 10],
          [0, 0, 20, 10, True, 10], [1, 10, 20, 10, False, 10], [1, 10, 20, 10, True, None]]

title = None

for k in range(len(params)):

    if title == 'without_D_NG' and k != titles.index('without_D_NG'):
        continue
    elif title != 'without_D_NG' and k == titles.index('without_D_NG'):
        continue

    if not os.path.exists(f'../../results/model_validity/MouseBrain_Jiang2023/{titles[k]}/'):
        os.makedirs(f'../../results/model_validity/MouseBrain_Jiang2023/{titles[k]}/')

    for j in range(num_iters):

        print(f'{titles[k]} Round {j}')
        print(params[k])

        INSTINCT_model = INSTINCT_Model(cas_list,
                                        adata_concat,
                                        lambda_adv=params[k][0],  # hyperparameter for the adversarial loss
                                        lambda_cls=params[k][1],  # hyperparameter for the classification loss
                                        lambda_la=params[k][2],  # hyperparameter for the latent loss
                                        lambda_rec=params[k][3],  # hyperparameter for the reconstruction loss
                                        seed=1234+j,  # random seed
                                        use_cos=params[k][4],  # use cosine similarity to find the nearest neighbors
                                        margin=params[k][5],  # the margin of latent loss
                                        device=device)

        INSTINCT_model.train(report_loss=False)

        INSTINCT_model.eval(cas_list)

        result = ad.concat(cas_list, label="slice_name", keys=slice_name_list)

        with open(f'../../results/model_validity/MouseBrain_Jiang2023/{titles[k]}/{titles[k]}_embed_{j}.csv',
                  'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(result.obsm['INSTINCT_latent'])

