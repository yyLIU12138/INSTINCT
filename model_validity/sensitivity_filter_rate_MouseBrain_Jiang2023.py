import os
import csv
import torch
import anndata as ad
from sklearn.decomposition import PCA

from ..INSTINCT import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

num_iters = 8

# parameters
filter_rate_list = [0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30]#[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

# mouse brain
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

save_dir = '../../results/model_validity/MouseBrain_Jiang2023/sensitivity/filter_rate/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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

for i in range(len(filter_rate_list)):

    print(f'The filter rate is {filter_rate_list[i]}\n')

    # load the merged data
    cas_list = [ad.read_h5ad(data_dir + 'merged_' + sample + '_atac.h5ad') for sample in slice_name_list]
    for j in range(len(cas_list)):
        cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]

    # concatenation
    adata_concat = ad.concat(cas_list, label="slice_name", keys=slice_name_list)
    # adata_concat.obs_names_make_unique()

    # preprocess CAS data
    print('Start preprocessing')
    preprocess_CAS(cas_list, adata_concat, use_fragment_count=True, min_cells_rate=filter_rate_list[i])
    print('Done!')

    print(adata_concat.shape[1])

    print(f'Applying PCA to reduce the feature dimension to 100 ...')
    pca = PCA(n_components=100, random_state=1234)
    input_matrix = pca.fit_transform(adata_concat.X.toarray())
    print('Done !')

    adata_concat.obsm['X_pca'] = input_matrix

    # calculate the spatial graph
    create_neighbor_graph(cas_list, adata_concat)

    for k in range(num_iters):

        print(f'Iteration {k}')

        INSTINCT_model = INSTINCT_Model(cas_list, adata_concat, seed=1236+k, device=device)

        INSTINCT_model.train(report_loss=False)

        INSTINCT_model.eval(cas_list)

        result = ad.concat(cas_list, label="slice_name", keys=slice_name_list)

        with open(save_dir + f'min_cells_rate={filter_rate_list[i]}_embed_{k}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(result.obsm['INSTINCT_latent'])

    with open(save_dir + 'n_features.txt', 'a' if os.path.isfile(save_dir + 'n_features.txt') else 'w') as f:
        f.write(str(adata_concat.shape[1])+'\n')
    f.close()




