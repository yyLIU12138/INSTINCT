import os
import pickle
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from codes.evaluation_utils import cluster_metrics, rep_metrics, match_cluster_labels

import warnings
warnings.filterwarnings("ignore")

# mouse embryo
num_iters = 8
mode = 'S1'
slice_name_list = [f'E12_5-{mode}', f'E13_5-{mode}', f'E15_5-{mode}']
cluster_list = ['Forebrain', 'Midbrain', 'Hindbrain', 'Periventricular', 'Meningeal_PNS_1', 'Meningeal_PNS_2',
                'Internal', 'Facial_bone', 'Muscle_heart', 'Limb', 'Liver']

save_dir = '../../results/MouseEmbryo_Llorens-Bobadilla2023/separate/'
cas_list = [ad.read_h5ad(save_dir + f"filtered_{sample}.h5ad") for sample in slice_name_list]
origin_concat = ad.concat(cas_list, label='slice_name', keys=slice_name_list)
print(origin_concat.shape)
adata_concat = origin_concat.copy()

models = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']

# clustering and calculating scores
for i in range(len(models)):

    if not os.path.exists(save_dir + f'{mode}/comparison/{models[i]}/{models[i]}_results_dict.pkl'):
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
                        'mAPs': maps, 'Cell_type_ASWs': c_asws,
                        'Batch_ASWs': b_asws, 'Batch_PCRs': b_pcrs, 'kBETs': kbets, 'Graph_connectivities': g_conns}
        with open(save_dir + f'{mode}/comparison/{models[i]}/{models[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)

    for j in range(num_iters):

        print(f'{models[i]} iteration {j}')

        embed = pd.read_csv(save_dir + f'{mode}/comparison/{models[i]}/{models[i]}_embed_{j}.csv', header=None).values
        adata_concat.obsm['latent'] = embed

        gm = GaussianMixture(n_components=len(cluster_list), covariance_type='tied', random_state=1234)
        y = gm.fit_predict(adata_concat.obsm['latent'], y=None)
        adata_concat.obs["gm_clusters"] = pd.Series(y, index=adata_concat.obs.index, dtype='category')
        adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(
            adata_concat.obs['clusters'], adata_concat.obs["gm_clusters"]),
            index=adata_concat.obs.index, dtype='category')

        ari, ami, nmi, fmi, comp, homo = cluster_metrics(adata_concat.obs['clusters'],
                                                         adata_concat.obs['matched_clusters'].tolist())
        map, c_asw, b_asw, b_pcr, kbet, g_conn = rep_metrics(adata_concat, origin_concat, use_rep='latent',
                                                             label_key='clusters', batch_key='slice_name')

        with open(save_dir + f'{mode}/comparison/{models[i]}/{models[i]}_results_dict.pkl', 'rb') as file:
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

        with open(save_dir + f'{mode}/comparison/{models[i]}/{models[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)


# frag/ bina
num_iters = 8

cas_list = [ad.read_h5ad(save_dir + f"filtered_{sample}.h5ad") for sample in slice_name_list]
origin_concat = ad.concat(cas_list, label='slice_name', keys=slice_name_list)
adata_concat = origin_concat.copy()

titles = ['frag', 'bina']

# clustering and calculate scores
for i in range(len(titles)):

    if not os.path.exists(save_dir + f'{mode}/comparison/INSTINCT_{titles[i]}/INSTINCT_{titles[i]}_results_dict.pkl'):
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
                        'mAPs': maps, 'Cell_type_ASWs': c_asws,
                        'Batch_ASWs': b_asws, 'Batch_PCRs': b_pcrs, 'kBETs': kbets, 'Graph_connectivities': g_conns}
        with open(save_dir + f'{mode}/comparison/INSTINCT_{titles[i]}/INSTINCT_{titles[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)

    for j in range(num_iters):

        print(f'{titles[i]} iteration {j}')

        embed = pd.read_csv(save_dir + f'{mode}/comparison/INSTINCT_{titles[i]}/INSTINCT_embed_{j}_{titles[i]}.csv', header=None).values
        adata_concat.obsm['latent'] = embed

        gm = GaussianMixture(n_components=len(cluster_list), covariance_type='tied', random_state=1234)
        y = gm.fit_predict(adata_concat.obsm['latent'], y=None)
        adata_concat.obs["gm_clusters"] = pd.Series(y, index=adata_concat.obs.index, dtype='category')
        adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(
            adata_concat.obs['clusters'], adata_concat.obs["gm_clusters"]),
            index=adata_concat.obs.index, dtype='category')

        ari, ami, nmi, fmi, comp, homo = cluster_metrics(adata_concat.obs['clusters'],
                                                         adata_concat.obs['matched_clusters'].tolist())
        map, c_asw, b_asw, b_pcr, kbet, g_conn = rep_metrics(adata_concat, origin_concat, use_rep='latent',
                                                             label_key='clusters', batch_key='slice_name')

        with open(save_dir + f'{mode}/comparison/INSTINCT_{titles[i]}/INSTINCT_{titles[i]}_results_dict.pkl', 'rb') as file:
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

        with open(save_dir + f'{mode}/comparison/INSTINCT_{titles[i]}/INSTINCT_{titles[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)
