import os
import pickle
import numpy as np
import anndata as ad
import pandas as pd
import seaborn as sns
import scanpy as sc

from sklearn.mixture import GaussianMixture

from ..evaluation_utils import cluster_metrics, rep_metrics, match_cluster_labels

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

layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]

save_dir = '../../results/DLPFC_Maynard2021/'

models = ['INSTINCT', 'INSTINCT_cas', 'SEDR', 'STAligner', 'GraphST']

# clustering and calculating scores
for i in range(len(models)):

    for idx in range(len(sample_group_list)):

        slice_name_list = sample_group_list[idx]
        slice_index_list = list(range(len(slice_name_list)))

        if not os.path.exists(save_dir + f'comparison/{models[i]}/{models[i]}_group{idx}_results_dict.pkl'):
            aris = np.zeros((num_iters,), dtype=float)
            amis = np.zeros((num_iters,), dtype=float)
            nmis = np.zeros((num_iters,), dtype=float)
            fmis = np.zeros((num_iters,), dtype=float)
            comps = np.zeros((num_iters,), dtype=float)
            homos = np.zeros((num_iters,), dtype=float)
            maps = np.zeros((num_iters,), dtype=float)
            c_asws = np.zeros((num_iters,), dtype=float)
            b_asws = np.zeros((num_iters,), dtype=float)
            b_pcrs = np.zeros((num_iters,), dtype=float)
            kbets = np.zeros((num_iters,), dtype=float)
            g_conns = np.zeros((num_iters,), dtype=float)

            results_dict = {'ARIs': aris, 'AMIs': amis, 'NMIs': nmis, 'FMIs': fmis, 'COMPs': comps, 'HOMOs': homos,
                            'mAPs': maps, 'Cell_type_ASWs': c_asws, 'Batch_ASWs': b_asws, 'Batch_PCRs': b_pcrs,
                            'kBETs': kbets, 'Graph_connectivities': g_conns}
            with open(save_dir + f'comparison/{models[i]}/{models[i]}_group{idx}_results_dict.pkl', 'wb') as file:
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

            origin_concat = adata_concat.copy()

            embed = pd.read_csv(save_dir + f'comparison/{models[i]}/{models[i]}_group{idx}_embed_{j}.csv', header=None).values
            adata_concat.obsm['latent'] = embed

            gm = GaussianMixture(n_components=num_clusters_list[idx], covariance_type='tied', random_state=1234)
            y = gm.fit_predict(adata_concat.obsm['latent'], y=None)

            adata_concat.obs["gm_clusters"] = pd.Series(y, index=adata_concat.obs.index, dtype='category')
            adata_concat = adata_concat[~adata_concat.obs['Manual_Annotation'].isna(), :]
            origin_concat = origin_concat[~origin_concat.obs['Manual_Annotation'].isna(), :]
            print(adata_concat.shape)
            adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(adata_concat.obs['Manual_Annotation'],
                                                                                  adata_concat.obs["gm_clusters"]),
                                                             index=adata_concat.obs.index, dtype='category')

            ari, ami, nmi, fmi, comp, homo = cluster_metrics(adata_concat.obs['Manual_Annotation'],
                                                             adata_concat.obs['matched_clusters'].tolist())
            map, c_asw, b_asw, b_pcr, kbet, g_conn = rep_metrics(adata_concat, origin_concat, use_rep='latent',
                                                                 label_key='Manual_Annotation', batch_key='slice_name')

            with open(save_dir + f'comparison/{models[i]}/{models[i]}_group{idx}_results_dict.pkl', 'rb') as file:
                results_dict = pickle.load(file)

            results_dict['ARIs'][j] = ari
            results_dict['AMIs'][j] = ami
            results_dict['NMIs'][j] = nmi
            results_dict['FMIs'][j] = fmi
            results_dict['COMPs'][j] = comp
            results_dict['HOMOs'][j] = homo
            results_dict['mAPs'][j] = map
            results_dict['Cell_type_ASWs'][j] = c_asw
            results_dict['Batch_ASWs'][j] = b_asw
            results_dict['Batch_PCRs'][j] = b_pcr
            results_dict['kBETs'][j] = kbet
            results_dict['Graph_connectivities'][j] = g_conn

            with open(save_dir + f'comparison/{models[i]}/{models[i]}_group{idx}_results_dict.pkl', 'wb') as file:
                pickle.dump(results_dict, file)
