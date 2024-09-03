import os
import pickle
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from ..evaluation_utils import knn_cross_validation

import warnings
warnings.filterwarnings("ignore")

num_iters = 8

# DLPFC
data_dir = '../../data/STdata/10xVisium/DLPFC_Maynard2021/'
sample_group_list = [['151507', '151508', '151509', '151510'],
                     ['151669', '151670', '151671', '151672'],
                     ['151673', '151674', '151675', '151676']]
cls_list = ['Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'WM']
num_clusters_list = [7, 5, 7]

save_dir = '../../results/DLPFC_Maynard2021/'

models = ['INSTINCT', 'INSTINCT_cas', 'SEDR', 'STAligner', 'GraphST']

# clustering and calculating scores
for i in range(len(models)):

    if not os.path.exists(save_dir + f'annotation/{models[i]}/'):
        os.makedirs(save_dir + f'annotation/{models[i]}/')

    for idx in range(len(sample_group_list)):

        # load data
        slice_name_list = sample_group_list[idx]
        slice_index_list = list(range(len(slice_name_list)))

        if not os.path.exists(save_dir + f'annotation/{models[i]}/{models[i]}_group{idx}_results_dict.pkl'):
            accus = np.zeros((num_iters,), dtype=float)
            kappas = np.zeros((num_iters,), dtype=float)
            mf1s = np.zeros((num_iters,), dtype=float)
            wf1s = np.zeros((num_iters,), dtype=float)

            results_dict = {'Accuracies': accus, 'Kappas': kappas, 'mF1s': mf1s, 'wF1s': wf1s}
            with open(save_dir + f'annotation/{models[i]}/{models[i]}_group{idx}_results_dict.pkl', 'wb') as file:
                pickle.dump(results_dict, file)

        for j in range(num_iters):

            print(f'{models[i]} group {idx} iteration {j}')

            rna_list = []
            for sample in slice_name_list:
                adata = sc.read_visium(path=data_dir + f'{sample}/',
                                       count_file=sample + '_filtered_feature_bc_matrix.h5')
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

            # concatenation
            adata_concat = ad.concat(rna_list, label="slice_name", keys=slice_name_list)

            # preprocess SRT data
            sc.pp.filter_genes(adata_concat, min_cells=1)
            sc.pp.filter_cells(adata_concat, min_genes=1)

            embed = pd.read_csv(save_dir + f'comparison/{models[i]}/{models[i]}_group{idx}_embed_{j}.csv', header=None).values
            adata_concat.obsm['latent'] = embed
            adata_concat = adata_concat[~adata_concat.obs['Manual_Annotation'].isna(), :]

            accu, kappa, mf1, wf1 = knn_cross_validation(adata_concat.obsm['latent'].copy(), adata_concat.obs['Manual_Annotation'].copy(),
                                                         k=20, batch_idx=adata_concat.obs['slice_name'].copy())

            with open(save_dir + f'annotation/{models[i]}/{models[i]}_group{idx}_results_dict.pkl', 'rb') as file:
                results_dict = pickle.load(file)

            results_dict['Accuracies'][j] = accu
            results_dict['Kappas'][j] = kappa
            results_dict['mF1s'][j] = mf1
            results_dict['wF1s'][j] = wf1

            with open(save_dir + f'annotation/{models[i]}/{models[i]}_group{idx}_results_dict.pkl', 'wb') as file:
                pickle.dump(results_dict, file)
