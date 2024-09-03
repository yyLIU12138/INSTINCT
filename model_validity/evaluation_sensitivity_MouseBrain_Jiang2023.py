import os
import pickle
import pandas as pd
import anndata as ad
import numpy as np
from sklearn.mixture import GaussianMixture

from ..evaluation_utils import match_cluster_labels, cluster_metrics, bio_conservation_metrics, batch_correction_metrics

import warnings
warnings.filterwarnings("ignore")

# mouse brain
num_iters = 8
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']
cls_list = ['Primary_brain_1', 'Primary_brain_2', 'Midbrain',  'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
            'Subpallium_1', 'Subpallium_2', 'Cartilage_1', 'Cartilage_2', 'Cartilage_3', 'Cartilage_4',
            'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv']

save_dir = '../../results/model_validity/MouseBrain_Jiang2023/sensitivity/'
cas_list = [ad.read_h5ad(f"../../results/MouseBrain_Jiang2023/filtered_merged_{sample}_atac.h5ad") for sample in slice_name_list]
adata_concat = ad.concat(cas_list, label='slice_name', keys=slice_name_list)
origin_concat = adata_concat.copy()

titles = ['clamp_margin', 'training_epoch', 'filter_rate', 'losses_hyper', 'k_neighbors', 'radius']
parameters_dict = {
    'clamp_margin': [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'training_epoch': [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'filter_rate': [0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30],#[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    'losses_hyper': [1, 5, 10, 15, 20, 25, 30],
    'k_neighbors': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'radius': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
}
file_name_dict = {
    'clamp_margin': ['margin'],
    'training_epoch': ['stage1', 'stage2'],
    'filter_rate': ['min_cells_rate'],
    'losses_hyper': ['lambda_cls', 'lambda_la', 'lambda_rec'],
    'k_neighbors': ['k'],
    'radius': ['rad_coef']
}

for i, title in enumerate(titles):

    tmp_dir = save_dir + f'{title}/'
    parameter_list = parameters_dict[title]
    file_name_list = file_name_dict[title]

    for file_name in file_name_list:

        if not os.path.exists(f'{tmp_dir}/{file_name}_results_dict.pkl'):

            aris = np.zeros((len(parameter_list), 8), dtype=float)
            amis = np.zeros((len(parameter_list), 8), dtype=float)
            nmis = np.zeros((len(parameter_list), 8), dtype=float)
            fmis = np.zeros((len(parameter_list), 8), dtype=float)
            comps = np.zeros((len(parameter_list), 8), dtype=float)
            homos = np.zeros((len(parameter_list), 8), dtype=float)
            maps = np.zeros((len(parameter_list), 8), dtype=float)
            c_asws = np.zeros((len(parameter_list), 8), dtype=float)
            i_asws = np.zeros((len(parameter_list), 8), dtype=float)
            i_f1s = np.zeros((len(parameter_list), 8), dtype=float)
            b_asws = np.zeros((len(parameter_list), 8), dtype=float)
            b_pcrs = np.zeros((len(parameter_list), 8), dtype=float)
            kbets = np.zeros((len(parameter_list), 8), dtype=float)
            g_conns = np.zeros((len(parameter_list), 8), dtype=float)

            results_dict = {'ARIs': aris, 'AMIs': amis, 'NMIs': nmis, 'FMIs': fmis, 'COMPs': comps, 'HOMOs': homos,
                            'mAPs': maps, 'Cell_type_ASWs': c_asws, 'Isolated_label_ASWs': i_asws,
                            'Isolated_label_F1s': i_f1s,
                            'Batch_ASWs': b_asws, 'Batch_PCRs': b_pcrs, 'kBETs': kbets, 'Graph_connectivities': g_conns}
            with open(f'{tmp_dir}/{file_name}_results_dict.pkl', 'wb') as file:
                pickle.dump(results_dict, file)

        for j, param in enumerate(parameter_list):

            for k in range(num_iters):

                print(f'{title}  {file_name}={param}  Iteration {k}')

                embed = pd.read_csv(f'{tmp_dir}/{file_name}={param}_embed_{k}.csv', header=None).values
                adata_concat.obsm['latent'] = embed

                gm = GaussianMixture(n_components=len(cls_list), covariance_type='tied', random_state=1234)
                y = gm.fit_predict(adata_concat.obsm['latent'], y=None)
                adata_concat.obs["gm_clusters"] = pd.Series(y, index=adata_concat.obs.index, dtype='category')
                adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(
                    adata_concat.obs['Annotation_for_Combined'], adata_concat.obs["gm_clusters"]),
                    index=adata_concat.obs.index, dtype='category')

                ari, ami, nmi, fmi, comp, homo = cluster_metrics(adata_concat.obs['Annotation_for_Combined'],
                                                                 adata_concat.obs['matched_clusters'].tolist())
                map, c_asw, i_asw, i_f1 = bio_conservation_metrics(adata_concat, use_rep='latent',
                                                                   label_key='Annotation_for_Combined',
                                                                   batch_key='slice_name')
                b_asw, b_pcr, kbet, g_conn = batch_correction_metrics(adata_concat, origin_concat, use_rep='latent',
                                                                      label_key='Annotation_for_Combined',
                                                                      batch_key='slice_name')

                with open(f'{tmp_dir}/{file_name}_results_dict.pkl', 'rb') as file:
                    results_dict = pickle.load(file)

                results_dict['ARIs'][j][k] = ari
                results_dict['AMIs'][j][k] = ami
                results_dict['NMIs'][j][k] = nmi
                results_dict['FMIs'][j][k] = fmi
                results_dict['COMPs'][j][k] = comp
                results_dict['HOMOs'][j][k] = homo
                results_dict['mAPs'][j][k] = map
                results_dict['Cell_type_ASWs'][j][k] = c_asw
                results_dict['Isolated_label_ASWs'][j][k] = i_asw
                results_dict['Isolated_label_F1s'][j][k] = i_f1
                results_dict['Batch_ASWs'][j][k] = b_asw
                results_dict['Batch_PCRs'][j][k] = b_pcr
                results_dict['kBETs'][j][k] = kbet
                results_dict['Graph_connectivities'][j][k] = g_conn

                with open(f'{tmp_dir}/{file_name}_results_dict.pkl', 'wb') as file:
                    pickle.dump(results_dict, file)






