import os
import csv
import torch
import scipy
import numpy as np
import anndata as ad
import scanpy as sc

from codes.INSTINCT_main.utils import preprocess_CAS, peak_sets_alignment

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

num_iters = 8

# mouse brain
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

save_dir = '../../results/MouseBrain_Jiang2023/'
if not os.path.exists(save_dir + 'comparison/STAligner/'):
    os.makedirs(save_dir + 'comparison/STAligner/')

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

# STAligner
print('----------STAligner----------')

import STAligner

for j in range(num_iters):

    print(f'Iteration {j}')

    cas_list = []
    adj_list = []

    for i, sample in enumerate(slice_name_list):

        print(sample)

        adata = ad.read_h5ad(save_dir + f"filtered_merged_{sample}_atac.h5ad")

        # Constructing the spatial network
        STAligner.Cal_Spatial_Net(adata, rad_cutoff=150)  # the spatial network are saved in adata.uns[‘adj’]

        # Normalization
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]

        adj_list.append(adata.uns['adj'])
        cas_list.append(adata)

    adata_concat = ad.concat(cas_list, label="slice_name", keys=slice_name_list)
    # adata_concat.obs['Annotation_for_Combined'] = adata_concat.obs['Annotation_for_Combined'].astype('category')
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype(str).astype('category')
    print('adata_concat.shape: ', adata_concat.shape)

    adj_concat = np.asarray(adj_list[0].todense())
    for batch_id in range(1, len(slice_name_list)):
        adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
    adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

    adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=50,
                                             device=device, random_seed=1234+j)

    with open(save_dir + f'comparison/STAligner/STAligner_embed_{j}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(adata_concat.obsm['STAligner'])

print('----------Done----------\n')

