import os
import csv
import torch
import scipy
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

num_iters = 8

# DLPFC
data_dir = '../../data/STdata/10xVisium/DLPFC_Maynard2021/'
sample_group_list = [['151507', '151508', '151509', '151510'],
                     ['151669', '151670', '151671', '151672'],
                     ['151673', '151674', '151675', '151676']]

save_dir = '../../results/DLPFC_Maynard2021/'
if not os.path.exists(save_dir + 'comparison/STAligner/'):
    os.makedirs(save_dir + 'comparison/STAligner/')

# STAligner
print('----------STAligner----------')

import STAligner

for idx in range(len(sample_group_list)):

    print(f'Group {idx}')

    # load data
    slice_name_list = sample_group_list[idx]
    slice_index_list = list(range(len(slice_name_list)))

    for j in range(num_iters):

        print(f'Iteration {j}')

        rna_list = []
        adj_list = []

        for i, sample in enumerate(slice_name_list):

            print(sample)

            adata = sc.read_visium(path=data_dir + f'{sample}/', count_file=sample + '_filtered_feature_bc_matrix.h5')
            adata.var_names_make_unique()

            # read the annotation
            Ann_df = pd.read_csv(data_dir + f'{sample}/meta_data.csv', sep=',', index_col=0)

            if not all(Ann_df.index.isin(adata.obs_names)):
                raise ValueError("Some rows in the annotation file are not present in the adata.obs_names")

            adata.obs['image_row'] = Ann_df.loc[adata.obs_names, 'imagerow']
            adata.obs['image_col'] = Ann_df.loc[adata.obs_names, 'imagecol']
            adata.obs['Manual_Annotation'] = Ann_df.loc[adata.obs_names, 'ManualAnnotation']

            adata.obs_names = [x + '_' + sample for x in adata.obs_names]

            # Constructing the spatial network
            STAligner.Cal_Spatial_Net(adata, rad_cutoff=150)  # the spatial network are saved in adata.uns[‘adj’]

            # Normalization
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            adata = adata[:, adata.var['highly_variable']]

            adj_list.append(adata.uns['adj'])
            rna_list.append(adata)

        adata_concat = ad.concat(rna_list, label="slice_name", keys=slice_name_list)
        adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype(str).astype('category')
        print('adata_concat.shape: ', adata_concat.shape)

        adj_concat = np.asarray(adj_list[0].todense())
        for batch_id in range(1, len(slice_name_list)):
            adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
        adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

        adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=50,
                                                 device=device, random_seed=1234+j)

        with open(save_dir + f'comparison/STAligner/STAligner_group{idx}_embed_{j}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(adata_concat.obsm['STAligner'])

print('----------Done----------\n')

