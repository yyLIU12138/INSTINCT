import os
import pickle
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc

from sklearn.mixture import GaussianMixture

from ..evaluation_utils import cluster_metrics, match_cluster_labels, bio_conservation_metrics, \
    batch_correction_metrics, bio_conservation_metrics_single

import warnings
warnings.filterwarnings("ignore")

# mouse brain
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

cls_list_1 = ['Primary_brain_1', "Primary_brain_2", 'Midbrain', 'Diencephalon_and_hindbrain',
              'Cartilage_2',
              'Mesenchyme', 'Muscle', 'DPallm']
cls_list_2 = ['Primary_brain_1', 'Midbrain', 'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
              'Cartilage_1', 'Cartilage_2', 'Cartilage_3', 'Cartilage_4',
              'Mesenchyme', 'Muscle', 'DPallm', 'DPallv']
cls_list_3 = ['Primary_brain_1', 'Midbrain', 'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
              'Subpallium_2', 'Cartilage_2', 'Cartilage_3',
              'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv']
cls_list_4 = ['Midbrain', 'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
              "Subpallium_1", 'Subpallium_2', 'Cartilage_1', 'Cartilage_2', 'Cartilage_3', 'Cartilage_4',
              'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv']
cls_list_all = [cls_list_1, cls_list_2, cls_list_3, cls_list_4]

cls_list = ['Primary_brain_1', 'Primary_brain_2', 'Midbrain', 'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
            'Subpallium_1', 'Subpallium_2', 'Cartilage_1', 'Cartilage_2', 'Cartilage_3', 'Cartilage_4',
            'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv']

save_dir = '../../results/MouseBrain_Jiang2023/single/'

num_iters = 8

models = ['SCALE', 'STAGATE']

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
        i_asws = np.zeros((num_iters,), dtype=float)
        i_f1s = np.zeros((num_iters,), dtype=float)
        b_asws = np.zeros((num_iters,), dtype=float)
        b_pcrs = np.zeros((num_iters,), dtype=float)
        kbets = np.zeros((num_iters,), dtype=float)
        g_conns = np.zeros((num_iters,), dtype=float)

        results_dict = {'ARIs': aris, 'AMIs': amis, 'NMIs': nmis, 'FMIs': fmis, 'COMPs': comps, 'HOMOs': homos,
                        'mAPs': maps, 'Cell_type_ASWs': c_asws, 'Isolated_label_ASWs': i_asws,
                        'Isolated_label_F1s': i_f1s, 'Batch_ASWs': b_asws, 'Batch_PCRs': b_pcrs,
                        'kBETs': kbets, 'Graph_connectivities': g_conns}
        with open(save_dir + f'{models[i]}/{models[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)

    # load raw data
    cas_list = []
    for sample in slice_name_list:
        sample_data = ad.read_h5ad(data_dir + sample + '_atac.h5ad')

        if 'insertion' in sample_data.obsm:
            del sample_data.obsm['insertion']

        if models[i] == 'SCALE':
            sc.pp.filter_cells(sample_data, min_genes=100)
            sc.pp.filter_genes(sample_data, min_cells=3)

        cas_list.append(sample_data)
    origin_concat = ad.concat(cas_list, label='slice_name', keys=slice_name_list)
    adata_concat = origin_concat.copy()

    for j in range(num_iters):

        print(f'{models[i]} iteration {j}')

        embed = [pd.read_csv(save_dir + f'{models[i]}/{models[i]}_embed_{j}_{name}.csv', header=None).values
                 for name in slice_name_list]
        embed = np.vstack(embed)
        adata_concat.obsm['latent'] = embed

        gm = GaussianMixture(n_components=len(cls_list), covariance_type='tied', random_state=1234)
        y = gm.fit_predict(adata_concat.obsm['latent'], y=None)
        adata_concat.obs["gm_clusters"] = pd.Series(y, index=adata_concat.obs.index, dtype='category')
        adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(adata_concat.obs['Annotation_for_Combined'],
                                                                              adata_concat.obs["gm_clusters"]),
                                                         index=adata_concat.obs.index, dtype='category')

        ari, ami, nmi, fmi, comp, homo = cluster_metrics(adata_concat.obs['Annotation_for_Combined'],
                                                         adata_concat.obs['matched_clusters'].tolist())
        map, c_asw, i_asw, i_f1 = bio_conservation_metrics(adata_concat, use_rep='latent',
                                                           label_key='Annotation_for_Combined', batch_key='slice_name')
        b_asw, b_pcr, kbet, g_conn = batch_correction_metrics(adata_concat, origin_concat, use_rep='latent',
                                                              label_key='Annotation_for_Combined',
                                                              batch_key='slice_name')

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
        results_dict['Isolated_label_ASWs'][j] = i_asw
        results_dict['Isolated_label_F1s'][j] = i_f1
        results_dict['Batch_ASWs'][j] = b_asw
        results_dict['Batch_PCRs'][j] = b_pcr
        results_dict['kBETs'][j] = kbet
        results_dict['Graph_connectivities'][j] = g_conn

        with open(save_dir + f'{models[i]}/{models[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)


models = ['INSTINCT', 'SCALE', 'STAGATE']
cas_list = [ad.read_h5ad(f"../../results/MouseBrain_Jiang2023/filtered_merged_{sample}_atac.h5ad") for sample in slice_name_list]
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
            adata = ad.read_h5ad(data_dir + slice_name + '_atac.h5ad')
            if 'insertion' in adata.obsm:
                del adata.obsm['insertion']
            if models[i] == 'SCALE':
                sc.pp.filter_cells(adata, min_genes=100)
                sc.pp.filter_genes(adata, min_cells=3)
        else:
            adata = cas_list[k]

        for j in range(num_iters):

            print(f'{models[i]} iteration {j}')

            if models[i] != 'INSTINCT':
                embed = pd.read_csv(save_dir + f'{models[i]}/{models[i]}_embed_{j}_{slice_name}.csv', header=None).values
                adata.obsm['latent'] = embed

                gm = GaussianMixture(n_components=len(cls_list_all[k]), covariance_type='tied', random_state=1234)
                y = gm.fit_predict(adata.obsm['latent'], y=None)
                adata.obs["gm_clusters"] = pd.Series(y, index=adata.obs.index, dtype='category')
            else:
                embed = pd.read_csv(save_dir + f'{models[i]}/{models[i]}_embed_{j}.csv', header=None).values
                adata.obsm['latent'] = embed[spots_count[k]: spots_count[k+1], :]

                gm = GaussianMixture(n_components=len(cls_list), covariance_type='tied', random_state=1234)
                y = gm.fit_predict(embed, y=None)
                adata.obs["gm_clusters"] = pd.Series(y[spots_count[k]: spots_count[k+1]], index=adata.obs.index, dtype='category')

            adata.obs['matched_clusters'] = pd.Series(match_cluster_labels(adata.obs['Annotation_for_Combined'],
                                                                           adata.obs["gm_clusters"]),
                                                      index=adata.obs.index, dtype='category')

            ari, ami, nmi, fmi, comp, homo = cluster_metrics(adata.obs['Annotation_for_Combined'],
                                                             adata.obs['matched_clusters'].tolist())
            map, c_asw = bio_conservation_metrics_single(adata, use_rep='latent', label_key='Annotation_for_Combined')

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

