import os
import csv
import torch
import numpy as np
import anndata as ad
from sklearn.decomposition import PCA

import INSTINCT

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

name_concat = slice_name_list[0]
for mode in slice_name_list[1:]:
    name_concat = name_concat + '_' + mode

save_dir = f'../../results/simulated/scenario_{scenario}/T_' + name_concat + '/'

if not os.path.exists(save_dir + 'comparison/INSTINCT/'):
    os.makedirs(save_dir + 'comparison/INSTINCT/')

cas_list = [ad.read_h5ad(f"../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
for j in range(len(cas_list)):
    print(cas_list[j].shape)
    cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]
adata_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)
preprocess_CAS(cas_list, adata_concat)

adata_concat.write_h5ad(save_dir + f"preprocessed_concat.h5ad")
for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_spot_level_slice_{slice_name_list[i]}.h5ad")

cas_list = [ad.read_h5ad(save_dir + f"filtered_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
adata_concat = ad.read_h5ad(save_dir + f"preprocessed_concat.h5ad")
print(adata_concat.shape)

print(f'Applying PCA to reduce the feature dimension to 100 ...')

pca = PCA(n_components=100, random_state=1234)
input_matrix = pca.fit_transform(adata_concat.X.toarray())
np.save(save_dir + f'input_matrix.npy', input_matrix)

print('Done !')

input_matrix = np.load(save_dir + f'input_matrix.npy')
adata_concat.obsm['X_pca'] = input_matrix

# calculate the spatial graph
create_neighbor_graph(cas_list, adata_concat)

# INSTINCT
print('----------INSTINCT----------')

for j in range(num_iters):

    print(f'Iteration {j}')

    INSTINCT_model = INSTINCT_Model(cas_list,
                                    adata_concat,
                                    input_mat_key='X_pca',  # the key of the input matrix in adata_concat.obsm
                                    input_dim=100,  # the input dimension
                                    hidden_dims_G=[50],  # hidden dimensions of the encoder and the decoder
                                    latent_dim=30,  # the dimension of latent space
                                    hidden_dims_D=[50],  # hidden dimensions of the discriminator
                                    lambda_adv=1,  # hyperparameter for the adversarial loss
                                    lambda_cls=10,  # hyperparameter for the classification loss
                                    lambda_la=20,  # hyperparameter for the latent loss
                                    lambda_rec=10,  # hyperparameter for the reconstruction loss
                                    seed=1234+j,  # random seed
                                    learn_rates=[1e-3, 5e-4],  # learning rate
                                    training_steps=[500, 500],  # training_steps
                                    early_stop=False,  # use the latent loss to control the number of training steps
                                    min_steps=500,  # the least number of steps when training the whole model
                                    use_cos=True,  # use cosine similarity to find the nearest neighbors
                                    margin=10,  # the margin of latent loss
                                    alpha=1,  # the hyperparameter for triplet loss
                                    k=50,  # the amount of neighbors to find
                                    device=device)

    INSTINCT_model.train(report_loss=False, report_interval=100)

    INSTINCT_model.eval(cas_list)

    result = ad.concat(cas_list, label="slice_index", keys=slice_index_list)

    with open(save_dir + f'comparison/INSTINCT/INSTINCT_embed_{j}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result.obsm['INSTINCT_latent'])

print('----------Done----------\n')
