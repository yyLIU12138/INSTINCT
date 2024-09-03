import os
import csv
import torch
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd

from sklearn.decomposition import PCA

from ..INSTINCT import *

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
if not os.path.exists(save_dir + 'comparison/INSTINCT/'):
    os.makedirs(save_dir + 'comparison/INSTINCT/')

# INSTINCT
print('----------INSTINCT----------')

for idx in range(len(sample_group_list)):

    print(f'Group {idx}')

    # load data
    slice_name_list = sample_group_list[idx]
    slice_index_list = list(range(len(slice_name_list)))

    rna_list = []
    for sample in slice_name_list:
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
        rna_list.append(adata)
        # print(adata.shape)

    # concatenation
    adata_concat = ad.concat(rna_list, label="slice_name", keys=slice_name_list)
    # adata_concat.obs_names_make_unique()

    # preprocess SRT data
    print('Start preprocessing')
    rna_list, adata_concat = preprocess_SRT(rna_list, adata_concat, n_top_genes=5000)
    print(adata_concat.shape)
    print('Done!')

    origin_concat = ad.concat(rna_list, label="slice_name", keys=slice_index_list)

    print(f'Applying PCA to reduce the feature dimension to 100 ...')
    pca = PCA(n_components=100, random_state=1234)
    input_matrix = pca.fit_transform(adata_concat.X.toarray())
    np.save(save_dir + f'input_matrix_group{idx}.npy', input_matrix)
    print('Done !')

    input_matrix = np.load(save_dir + f'input_matrix_group{idx}.npy')
    adata_concat.obsm['X_pca'] = input_matrix

    # calculate the spatial graph
    create_neighbor_graph(rna_list, adata_concat)

    spots_count = [0]
    n = 0
    for sample in rna_list:
        num = sample.shape[0]
        n += num
        spots_count.append(n)

    for j in range(num_iters):

        print(f'Iteration {j}')

        INSTINCT_model = INSTINCT_Model(rna_list, adata_concat, seed=1234+j, device=device)

        INSTINCT_model.train(report_loss=False)

        INSTINCT_model.eval(rna_list)

        result = ad.concat(rna_list, label="slice_name", keys=slice_index_list)

        with open(save_dir + f'comparison/INSTINCT/INSTINCT_group{idx}_embed_{j}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(result.obsm['INSTINCT_latent'])

print('----------Done----------\n')
