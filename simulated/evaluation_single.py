import os
import pickle
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc

from sklearn.mixture import GaussianMixture

from ..evaluation_utils import cluster_metrics, rep_metrics, match_cluster_labels, bio_conservation_metrics_single

import warnings
warnings.filterwarnings("ignore")

scenario = 1

slice_name_list = [
    'Tech_0_0_Bio_0_0.5',
    'Tech_0_0.1_Bio_0_0',
    'Tech_0_0.1_Bio_0_0.5',
             ]

slice_index_list = list(range(len(slice_name_list)))

save_dir = f'../../results/simulated/scenario_{scenario}/single/'

num_iters = 8
num_clusters = 5

models = ['SCALE', 'STAGATE']

cas_list = [ad.read_h5ad(f"../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad") for mode in
            slice_name_list]
origin_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)
adata_concat = origin_concat.copy()

# joint clustering
for i in range(len(models)):

    if not os.path.exists(save_dir + f'{models[i]}/{models[i]}_results_dict.pkl'):
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
        with open(save_dir + f'{models[i]}/{models[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)

    for j in range(num_iters):

        print(f'{models[i]} iteration {j}')

        embed = [pd.read_csv(save_dir + f'{models[i]}/{models[i]}_embed_{j}_{name}.csv', header=None).values
                 for name in slice_name_list]
        embed = np.vstack(embed)
        adata_concat.obsm['latent'] = embed

        gm = GaussianMixture(n_components=num_clusters, covariance_type='tied', random_state=1234)
        y = gm.fit_predict(adata_concat.obsm['latent'], y=None)
        adata_concat.obs["gm_clusters"] = pd.Series(y, index=adata_concat.obs.index, dtype='category')
        adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(adata_concat.obs['real_spot_clusters'],
                                                                              adata_concat.obs["gm_clusters"]),
                                                         index=adata_concat.obs.index, dtype='category')

        ari, ami, nmi, fmi, comp, homo = cluster_metrics(adata_concat.obs['real_spot_clusters'],
                                                         adata_concat.obs['matched_clusters'].tolist())
        map, c_asw, b_asw, b_pcr, kbet, g_conn = rep_metrics(adata_concat, origin_concat, use_rep='latent',
                                                             label_key='real_spot_clusters', batch_key='slice_index')

        with open(save_dir + f'{models[i]}/{models[i]}_results_dict.pkl', 'rb') as file:
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

        with open(save_dir + f'{models[i]}/{models[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)


models = ['INSTINCT', 'SCALE', 'STAGATE']
name_concat = slice_name_list[0]
for mode in slice_name_list[1:]:
    name_concat = name_concat + '_' + mode
cas_list = [ad.read_h5ad(f"../../results/simulated/scenario_{scenario}/T_{name_concat}/filtered_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
spots_count = [0]
n = 0
for sample in cas_list:
    num = sample.shape[0]
    n += num
    spots_count.append(n)

# single clustering
for i in range(len(models)):

    for k, slice_name in enumerate(slice_name_list):

        if not os.path.exists(save_dir + f'{models[i]}/{models[i]}_results_dict_{slice_name}.pkl'):
            aris = np.zeros((num_iters,), dtype=float)
            amis = np.zeros((num_iters,), dtype=float)
            nmis = np.zeros((num_iters,), dtype=float)
            fmis = np.zeros((num_iters,), dtype=float)
            comps = np.zeros((num_iters,), dtype=float)
            homos = np.zeros((num_iters,), dtype=float)
            maps = np.zeros((num_iters,), dtype=float)
            c_asws = np.zeros((num_iters,), dtype=float)

            results_dict = {'ARIs': aris, 'AMIs': amis, 'NMIs': nmis, 'FMIs': fmis, 'COMPs': comps, 'HOMOs': homos,
                            'mAPs': maps, 'Cell_type_ASWs': c_asws}
            with open(save_dir + f'{models[i]}/{models[i]}_results_dict_{slice_name}.pkl', 'wb') as file:
                pickle.dump(results_dict, file)

        if models[i] != 'INSTINCT':
            adata = ad.read_h5ad(f"../../data/simulated/{slice_name}/{scenario}_spot_level_slice_{slice_name}.h5ad")
            if models[i] == 'SCALE':
                sc.pp.filter_cells(adata, min_genes=100)
                sc.pp.filter_genes(adata, min_cells=3)
        else:
            adata = cas_list[i]

        for j in range(num_iters):

            print(f'{models[i]} iteration {j}')

            if models[i] != 'INSTINCT':
                embed = pd.read_csv(save_dir + f'{models[i]}/{models[i]}_embed_{j}_{slice_name}.csv',
                                    header=None).values
                adata.obsm['latent'] = embed

                gm = GaussianMixture(n_components=num_clusters, covariance_type='tied', random_state=1234)
                y = gm.fit_predict(adata.obsm['latent'], y=None)
                adata.obs["gm_clusters"] = pd.Series(y, index=adata.obs.index, dtype='category')
            else:
                embed = pd.read_csv(save_dir + f'{models[i]}/{models[i]}_embed_{j}.csv', header=None).values
                adata.obsm['latent'] = embed[spots_count[k]: spots_count[k + 1], :]

                gm = GaussianMixture(n_components=num_clusters, covariance_type='tied', random_state=1234)
                y = gm.fit_predict(embed, y=None)
                adata.obs["gm_clusters"] = pd.Series(y[spots_count[k]: spots_count[k + 1]], index=adata.obs.index,
                                                     dtype='category')

            adata.obs['matched_clusters'] = pd.Series(match_cluster_labels(adata.obs['real_spot_clusters'],
                                                                           adata.obs["gm_clusters"]),
                                                      index=adata.obs.index, dtype='category')

            ari, ami, nmi, fmi, comp, homo = cluster_metrics(adata.obs['real_spot_clusters'],
                                                             adata.obs['matched_clusters'].tolist())
            map, c_asw = bio_conservation_metrics_single(adata, use_rep='latent', label_key='real_spot_clusters')

            with open(save_dir + f'{models[i]}/{models[i]}_results_dict_{slice_name}.pkl', 'rb') as file:
                results_dict = pickle.load(file)

            results_dict['ARIs'][j] = ari
            results_dict['AMIs'][j] = ami
            results_dict['NMIs'][j] = nmi
            results_dict['FMIs'][j] = fmi
            results_dict['COMPs'][j] = comp
            results_dict['HOMOs'][j] = homo
            results_dict['mAPs'][j] = map
            results_dict['Cell_type_ASWs'][j] = c_asw

            with open(save_dir + f'{models[i]}/{models[i]}_results_dict_{slice_name}.pkl', 'wb') as file:
                pickle.dump(results_dict, file)
