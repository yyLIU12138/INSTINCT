import os
import anndata as ad
import numpy as np
import torch
import csv

from sklearn.decomposition import PCA
from INSTINCT import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CAS_samples = ["GSM6206884_HumanBrain_50um", "GSM6801813_ME13_50um", "GSM6204624_ME13_100barcodes_25um",
               "GSM6204623_MouseBrain_20um", "GSM6758284_MouseBrain_20um_repATAC", "GSM6758285_MouseBrain_20um_100barcodes_ATAC",
               "GSM6204621_MouseBrain_20um_H3K27ac", "GSM6704977_MouseBrain_20um_rep_H3K27ac", "GSM6704978_MouseBrain_20um_100barcodes_H3K27me3",
               "GSM6704979_MouseBrain_20um_100barcodes_H3K27ac", "GSM6704980_MouseBrain_20um_100barcodes_H3K4me3"]
RNA_samples = ["GSM6206885_HumanBrain_50um", "GSM6799937_ME13_50um", "GSM6204637_ME13_100barcodes_25um",
               "GSM6204636_MouseBrain_20um", "GSM6753041_MouseBrain_20um_repATAC", "GSM6753043_MouseBrain_20um_100barcodes_ATAC",
               "GSM6204635_MouseBrain_20um_H3K27ac", "GSM6753042_MouseBrain_20um_repH3K27ac", "GSM6753044_MouseBrain_20um_100barcodes_H3K27me3",
               "GSM6753045_MouseBrain_20um_100barcodes_H3K27ac", "GSM6753046_MouseBrain_20um_100barcodes_H3K4me3"]

save = False
slice_name_list = ["GSM6204623_MouseBrain_20um", "GSM6758284_MouseBrain_20um_repATAC", "GSM6758285_MouseBrain_20um_100barcodes_ATAC"]
rna_slice_name_list = ["GSM6204636_MouseBrain_20um", "GSM6753041_MouseBrain_20um_repATAC", "GSM6753043_MouseBrain_20um_100barcodes_ATAC"]
slice_index_list = list(range(len(slice_name_list)))

data_dir = '../../data/spMOdata/EpiTran_HumanMouse_Zhang2023/preprocessed_from_fragments/'
save_dir = f'../../results/HumanMouse_Zhang2023/mb/'

if not os.path.exists(data_dir + f'mb_merged/'):
    os.makedirs(data_dir + f'mb_merged/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# load raw data
cas_list = []
for sample in slice_name_list:
    sample_data = ad.read_h5ad(data_dir + sample + '.h5ad')

    if 'insertion' in sample_data.obsm:
        del sample_data.obsm['insertion']

    cas_list.append(sample_data)

# merge peaks
cas_list = peak_sets_alignment(cas_list)

# save the merged data
for idx, adata in enumerate(cas_list):
    adata.write_h5ad(data_dir + f'mb_merged/merged_{slice_name_list[idx]}.h5ad')

# load the merged data
cas_list = [ad.read_h5ad(data_dir + f'mb_merged/merged_{sample}.h5ad') for sample in slice_name_list]
for j in range(len(cas_list)):
    cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]

# read the raw RNA data
rna_list = [ad.read_h5ad(data_dir + f'{sample}.h5ad') for sample in rna_slice_name_list]
for j in range(len(rna_list)):
    rna_list[j].obs_names = [x + '-1_' + slice_name_list[j] for x in rna_list[j].obs_names]
    print(rna_list[j].shape)

# filter spots that is not tissue
for i in range(len(slice_name_list)):
    obs_list = [obs_name for obs_name in cas_list[i].obs_names if obs_name in rna_list[i].obs_names]
    cas_list[i] = cas_list[i][obs_list, :]
    print(cas_list[i].shape)

# concatenation
adata_concat = ad.concat(cas_list, label="slice_name", keys=slice_name_list)
# adata_concat.obs_names_make_unique()
print(adata_concat.shape)

# preprocess CAS data
print('Start preprocessing')
preprocess_CAS(cas_list, adata_concat, use_fragment_count=True, min_cells_rate=0.02)
print(adata_concat.shape)
print('Done!')

adata_concat.write_h5ad(save_dir + f"preprocessed_concat.h5ad")
for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_merged_{slice_name_list[i]}.h5ad")

cas_list = [ad.read_h5ad(save_dir + f"filtered_merged_{sample}.h5ad") for sample in slice_name_list]
origin_concat = ad.concat(cas_list, label="slice_idx", keys=slice_index_list)
adata_concat = ad.read_h5ad(save_dir + f"preprocessed_concat.h5ad")

print(f'Applying PCA to reduce the feature dimension to 100 ...')
pca = PCA(n_components=100, random_state=1234)
input_matrix = pca.fit_transform(adata_concat.X.toarray())
np.save(save_dir + 'input_matrix.npy', input_matrix)
print('Done !')

input_matrix = np.load(save_dir + 'input_matrix.npy')
adata_concat.obsm['X_pca'] = input_matrix

# calculate the spatial graph
create_neighbor_graph(cas_list, adata_concat)

spots_count = [0]
n = 0
for sample in cas_list:
    num = sample.shape[0]
    n += num
    spots_count.append(n)

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
                                seed=1236,  # random seed
                                learn_rates=[1e-3, 5e-4],  # learning rate
                                training_steps=[500, 500],  # training_steps
                                early_stop=False,  # use the latent loss to control the number of training steps
                                min_steps=500,  # the least number of steps when training the whole model
                                use_cos=True,  # use cosine similarity to find the nearest neighbors
                                margin=10,  # the margin of latent loss
                                alpha=1,  # the hyperparameter for triplet loss
                                k=50,  # the amount of neighbors to find
                                device=device)

INSTINCT_model.train(report_loss=True, report_interval=100)

INSTINCT_model.eval(cas_list)

result = ad.concat(cas_list, label="slice_idx", keys=slice_index_list)

if save:
    with open(save_dir + 'INSTINCT_embed.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result.obsm['INSTINCT_latent'])

    with open(save_dir + 'INSTINCT_noise_embed.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result.obsm['INSTINCT_latent_noise'])
