import os
import pickle
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from codes.evaluation_utils import cluster_metrics, batch_correction_metrics, bio_conservation_metrics, match_cluster_labels

import warnings
warnings.filterwarnings("ignore")

# mouse brain
num_iters = 8
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']
cls_list = ['Primary_brain_1', 'Primary_brain_2', 'Midbrain',  'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
            'Subpallium_1', 'Subpallium_2', 'Cartilage_1', 'Cartilage_2', 'Cartilage_3', 'Cartilage_4',
            'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv']

save_dir = '../../results/MouseBrain_Jiang2023/'
cas_list = [ad.read_h5ad(save_dir + f"filtered_merged_{sample}_atac.h5ad") for sample in slice_name_list]
origin_concat = ad.concat(cas_list, label='slice_name', keys=slice_name_list)
adata_concat = origin_concat.copy()

spots_count = [0]
n = 0
for sample in cas_list:
    num = sample.shape[0]
    n += num
    spots_count.append(n)

models = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']

# clustering and calculating scores
for i in range(len(models)):

    if not os.path.exists(save_dir + f'comparison/{models[i]}/{models[i]}_results_dict.pkl'):
        aris = np.zeros((num_iters,), dtype=float)
        amis = np.zeros((num_iters,), dtype=float)
        nmis = np.zeros((num_iters,), dtype=float)
        fmis = np.zeros((num_iters,), dtype=float)
        comps = np.zeros((num_iters,), dtype=float)
        homos = np.zeros((num_iters,), dtype=float)
        maps = np.zeros((num_iters,), dtype=float)
        c_asws = np.zeros((num_iters,), dtype=float)
        i_asws = np.zeros((num_iters,), dtype=float)
        i_f1s = np.zeros((num_iters,), dtype=float)
        b_asws = np.zeros((num_iters,), dtype=float)
        b_pcrs = np.zeros((num_iters,), dtype=float)
        kbets = np.zeros((num_iters,), dtype=float)
        g_conns = np.zeros((num_iters,), dtype=float)

        results_dict = {'ARIs': aris, 'AMIs': amis, 'NMIs': nmis, 'FMIs': fmis, 'COMPs': comps, 'HOMOs': homos,
                        'mAPs': maps, 'Cell_type_ASWs': c_asws, 'Isolated_label_ASWs': i_asws, 'Isolated_label_F1s': i_f1s,
                        'Batch_ASWs': b_asws, 'Batch_PCRs': b_pcrs, 'kBETs': kbets, 'Graph_connectivities': g_conns}
        with open(save_dir + f'comparison/{models[i]}/{models[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)

    for j in range(num_iters):

        print(f'{models[i]} iteration {j}')

        embed = pd.read_csv(save_dir + f'comparison/{models[i]}/{models[i]}_embed_{j}.csv', header=None).values
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
                                                           label_key='Annotation_for_Combined', batch_key='slice_name')
        b_asw, b_pcr, kbet, g_conn = batch_correction_metrics(adata_concat, origin_concat, use_rep='latent',
                                                              label_key='Annotation_for_Combined',
                                                              batch_key='slice_name')

        with open(save_dir + f'comparison/{models[i]}/{models[i]}_results_dict.pkl', 'rb') as file:
            results_dict = pickle.load(file)

        results_dict['ARIs'][j] = ari
        results_dict['AMIs'][j] = ami
        results_dict['NMIs'][j] = nmi
        results_dict['FMIs'][j] = fmi
        results_dict['COMPs'][j] = comp
        results_dict['HOMOs'][j] = homo
        results_dict['mAPs'][j] = map
        results_dict['Cell_type_ASWs'][j] = c_asw
        results_dict['Isolated_label_ASWs'][j] = i_asw
        results_dict['Isolated_label_F1s'][j] = i_f1
        results_dict['Batch_ASWs'][j] = b_asw
        results_dict['Batch_PCRs'][j] = b_pcr
        results_dict['kBETs'][j] = kbet
        results_dict['Graph_connectivities'][j] = g_conn

        with open(save_dir + f'comparison/{models[i]}/{models[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)


# frag/ bina/ read
num_iters = 8

cas_list = [ad.read_h5ad(save_dir + f"filtered_merged_{sample}_atac.h5ad") for sample in slice_name_list]
origin_concat = ad.concat(cas_list, label='slice_name', keys=slice_name_list)
adata_concat = origin_concat.copy()

titles = ['frag', 'bina', 'read']

# clustering and calculate scores
for i in range(len(titles)):

    if not os.path.exists(save_dir + f'comparison/INSTINCT_{titles[i]}/INSTINCT_{titles[i]}_results_dict.pkl'):
        aris = np.zeros((num_iters,), dtype=float)
        amis = np.zeros((num_iters,), dtype=float)
        nmis = np.zeros((num_iters,), dtype=float)
        fmis = np.zeros((num_iters,), dtype=float)
        comps = np.zeros((num_iters,), dtype=float)
        homos = np.zeros((num_iters,), dtype=float)
        maps = np.zeros((num_iters,), dtype=float)
        c_asws = np.zeros((num_iters,), dtype=float)
        i_asws = np.zeros((num_iters,), dtype=float)
        i_f1s = np.zeros((num_iters,), dtype=float)
        b_asws = np.zeros((num_iters,), dtype=float)
        b_pcrs = np.zeros((num_iters,), dtype=float)
        kbets = np.zeros((num_iters,), dtype=float)
        g_conns = np.zeros((num_iters,), dtype=float)

        results_dict = {'ARIs': aris, 'AMIs': amis, 'NMIs': nmis, 'FMIs': fmis, 'COMPs': comps, 'HOMOs': homos,
                        'mAPs': maps, 'Cell_type_ASWs': c_asws, 'Isolated_label_ASWs': i_asws, 'Isolated_label_F1s': i_f1s,
                        'Batch_ASWs': b_asws, 'Batch_PCRs': b_pcrs, 'kBETs': kbets, 'Graph_connectivities': g_conns}
        with open(save_dir + f'comparison/INSTINCT_{titles[i]}/INSTINCT_{titles[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)

    for j in range(num_iters):

        print(f'{titles[i]} iteration {j}')

        embed = pd.read_csv(save_dir + f'comparison/INSTINCT_{titles[i]}/INSTINCT_embed_{j}_{titles[i]}.csv', header=None).values
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
                                                           label_key='Annotation_for_Combined', batch_key='slice_name')
        b_asw, b_pcr, kbet, g_conn = batch_correction_metrics(adata_concat, origin_concat, use_rep='latent',
                                                              label_key='Annotation_for_Combined',
                                                              batch_key='slice_name')

        with open(save_dir + f'comparison/INSTINCT_{titles[i]}/INSTINCT_{titles[i]}_results_dict.pkl', 'rb') as file:
            results_dict = pickle.load(file)

        results_dict['ARIs'][j] = ari
        results_dict['AMIs'][j] = ami
        results_dict['NMIs'][j] = nmi
        results_dict['FMIs'][j] = fmi
        results_dict['COMPs'][j] = comp
        results_dict['HOMOs'][j] = homo
        results_dict['mAPs'][j] = map
        results_dict['Cell_type_ASWs'][j] = c_asw
        results_dict['Isolated_label_ASWs'][j] = i_asw
        results_dict['Isolated_label_F1s'][j] = i_f1
        results_dict['Batch_ASWs'][j] = b_asw
        results_dict['Batch_PCRs'][j] = b_pcr
        results_dict['kBETs'][j] = kbet
        results_dict['Graph_connectivities'][j] = g_conn

        with open(save_dir + f'comparison/INSTINCT_{titles[i]}/INSTINCT_{titles[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)
