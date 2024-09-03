import os
import csv
import torch
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from tqdm import tqdm

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
if not os.path.exists(save_dir + 'comparison/SEDR/'):
    os.makedirs(save_dir + 'comparison/SEDR/')

# SEDR
print('----------SEDR----------')

import SEDR
import harmonypy as hm

for idx in range(len(sample_group_list)):

    print(f'Group {idx}')

    # load data
    slice_name_list = sample_group_list[idx]
    slice_index_list = list(range(len(slice_name_list)))

    for j in range(num_iters):

        print(f'Iteration {j}')

        random_seed = 1234 + j
        SEDR.fix_seed(random_seed)

        rna_list = []
        for proj_name in tqdm(slice_name_list):
            adata_tmp = sc.read_visium(path=data_dir + f'{proj_name}/', count_file=proj_name + '_filtered_feature_bc_matrix.h5')
            adata_tmp.var_names_make_unique()

            adata_tmp.obs['batch_name'] = proj_name
            graph_dict_tmp = SEDR.graph_construction(adata_tmp, 12)

            # read the annotation
            Ann_df = pd.read_csv(data_dir + f'{proj_name}/meta_data.csv', sep=',', index_col=0)

            if not all(Ann_df.index.isin(adata_tmp.obs_names)):
                raise ValueError("Some rows in the annotation file are not present in the adata.obs_names")

            adata_tmp.obs['image_row'] = Ann_df.loc[adata_tmp.obs_names, 'imagerow']
            adata_tmp.obs['image_col'] = Ann_df.loc[adata_tmp.obs_names, 'imagecol']
            adata_tmp.obs['Manual_Annotation'] = Ann_df.loc[adata_tmp.obs_names, 'ManualAnnotation']

            adata_tmp.obs_names = [x + '_' + proj_name for x in adata_tmp.obs_names]

            if proj_name == slice_name_list[0]:
                adata = adata_tmp
                graph_dict = graph_dict_tmp
                name = proj_name
                adata.obs['proj_name'] = proj_name
            else:
                var_names = adata.var_names.intersection(adata_tmp.var_names)
                adata = adata[:, var_names]
                adata_tmp = adata_tmp[:, var_names]
                adata_tmp.obs['proj_name'] = proj_name

                adata = adata.concatenate(adata_tmp)
                graph_dict = SEDR.combine_graph_dict(graph_dict, graph_dict_tmp)
                name = name + '_' + proj_name

        # preprocessing
        adata.layers['count'] = adata.X.toarray()
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
        adata = adata[:, adata.var['highly_variable'] == True]
        sc.pp.scale(adata)

        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

        sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
        using_dec = False
        if using_dec:
            sedr_net.train_with_dec()
        else:
            sedr_net.train_without_dec()
        sedr_feat, _, _, _ = sedr_net.process()
        adata.obsm['SEDR'] = sedr_feat

        meta_data = adata.obs[['batch']]
        data_mat = adata.obsm['SEDR']
        vars_use = ['batch']
        ho = hm.run_harmony(data_mat, meta_data, vars_use, random_state=1234+j)
        print(ho.Z_corr.T.shape)

        with open(save_dir + f'comparison/SEDR/SEDR_group{idx}_embed_{j}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(ho.Z_corr.T)

print('----------Done----------\n')
