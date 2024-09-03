import os
import csv
import torch
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from ..INSTINCT import *
from ..evaluation_utils import cluster_metrics, bio_conservation_metrics, batch_correction_metrics, knn_cross_validation, match_cluster_labels

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# DLPFC
data_dir = '../../data/STdata/10xVisium/DLPFC_Maynard2021/'
sample_group_list = [['151507', '151508', '151509', '151510'],
                     ['151669', '151670', '151671', '151672'],
                     ['151673', '151674', '151675', '151676']]
n_cluster_list = [7, 5, 7]

save_dir = '../../results/DLPFC_Maynard2021/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for idx in range(len(sample_group_list)):

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

    INSTINCT_model = INSTINCT_Model(rna_list, adata_concat, device=device)

    INSTINCT_model.train(report_loss=True, report_interval=100)

    INSTINCT_model.eval(rna_list)

    result = ad.concat(rna_list, label="slice_name", keys=slice_index_list)

    with open(save_dir + f'INSTINCT_embed_group{idx}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result.obsm['INSTINCT_latent'])

    with open(save_dir + f'INSTINCT_noise_embed_group{idx}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result.obsm['INSTINCT_latent_noise'])

    gm = GaussianMixture(n_components=n_cluster_list[idx], covariance_type='tied', random_state=1234)
    y = gm.fit_predict(result.obsm['INSTINCT_latent'], y=None)
    result.obs["gm_clusters"] = pd.Series(y, index=result.obs.index, dtype='category')
    result.obs['matched_clusters'] = pd.Series(match_cluster_labels(result.obs['Manual_Annotation'],
                                                                    result.obs["gm_clusters"]),
                                               index=result.obs.index, dtype='category')

    ari, ami, nmi, fmi, comp, homo = cluster_metrics(result.obs['Manual_Annotation'],
                                                     result.obs['matched_clusters'].tolist())
    map, c_asw, i_asw, i_f1 = bio_conservation_metrics(result, use_rep='INSTINCT_latent',
                                                       label_key='Manual_Annotation', batch_key='slice_name')
    b_asw, b_pcr, kbet, g_conn = batch_correction_metrics(result, origin_concat, use_rep='INSTINCT_latent',
                                                          label_key='Manual_Annotation',
                                                          batch_key='slice_name')
    accu, kappa, mf1, wf1 = knn_cross_validation(result.obsm['INSTINCT_latent'], result.obs['Manual_Annotation'],
                                                 k=20, batch_idx=result.obs['slice_name'])




