import os
import csv
import time
import torch
import anndata as ad
import scanpy as sc
import pandas as pd

from sklearn.decomposition import PCA

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
if not os.path.exists(save_dir + 'comparison/GraphST/'):
    os.makedirs(save_dir + 'comparison/GraphST/')

# GraphST
print('----------GraphST----------')

from GraphST import GraphST
import paste as pst
import ot

for idx in range(len(sample_group_list)):

    print(f'Group {idx}')

    # load data
    slice_name_list = sample_group_list[idx]
    slice_index_list = list(range(len(slice_name_list)))

    for j in range(num_iters):

        print(f'Iteration {j}')

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

        # Pairwise align the slices
        start = time.time()
        pis = []
        for i in range(len(rna_list) - 1):
            pi0 = pst.match_spots_using_spatial_heuristic(rna_list[i].obsm['spatial'], rna_list[i + 1].obsm['spatial'],
                                                          use_ot=True)
            pi = pst.pairwise_align(rna_list[i], rna_list[i + 1], G_init=pi0, norm=True, verbose=False,
                                    backend=ot.backend.TorchBackend(), use_gpu=True)
            pis.append(pi)
        print('Alignment Runtime: ' + str(time.time() - start))

        # To visualize the alignment you can stack the slices
        # according to the alignment pi
        rna_list = pst.stack_slices_pairwise(rna_list, pis)

        adata_concat = ad.concat(rna_list, label='slice_name', keys=slice_name_list)

        # define model
        model = GraphST.GraphST(adata_concat, device=device, random_seed=1234+j)

        # run model
        result = model.train()
        # print(result)

        pca = PCA(n_components=20, random_state=1234)
        embedding = pca.fit_transform(result.obsm['emb'].copy())
        result.obsm['emb_pca'] = embedding

        with open(save_dir + f'comparison/GraphST/GraphST_group{idx}_embed_{j}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(result.obsm['emb_pca'])

print('----------Done----------\n')
