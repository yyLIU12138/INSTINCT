import os
import pickle
import numpy as np
import anndata as ad
import pandas as pd

from ..evaluation_utils import batch_correction_metrics

import warnings
warnings.filterwarnings("ignore")

scenario = 1
num_clusters = 5
num_iters = 8

slice_name_list = [
    'Tech_0_0_Bio_0_0.5',
    'Tech_0_0.1_Bio_0_0',
    'Tech_0_0.1_Bio_0_0.5',
             ]

slice_index_list = list(range(len(slice_name_list)))

name_concat = slice_name_list[0]
for mode in slice_name_list[1:]:
    name_concat = name_concat + '_' + mode

save_dir = f'../../results/simulated/scenario_{scenario}/T_' + name_concat + '/'

slice_index_list = [str(i) for i in range(len(slice_name_list))]

models = ['INSTINCT', 'SCALEX']

spot_rows = 40
spot_columns = 50
cell_type_template = np.zeros((3 * spot_columns, 3 * spot_rows), dtype=int)
coords = [[x, y] for x in range(3*20, 3*40) for y in range(3*15, 3*30)]  # cluster 0
cell_type_template[tuple(zip(*coords))] = 0
coords = [[x, y] for x in range(3*0, 3*10) for y in range(3*0, 3*40)] + \
         [[x, y] for x in range(3*10, 3*20) for y in range(3*0, 3*10)]  # cluster 1
cell_type_template[tuple(zip(*coords))] = 1
coords = [[x, y] for x in range(3*40, 3*50) for y in range(3*20, 3*40)] + \
         [[x, y] for x in range(3*10, 3*40) for y in range(3*35, 3*40)]  # cluster 2
cell_type_template[tuple(zip(*coords))] = 2
coords = [[x, y] for x in range(3*20, 3*50) for y in range(3*0, 3*10)] + \
         [[x, y] for x in range(3*40, 3*50) for y in range(3*10, 3*20)]  # cluster 3
cell_type_template[tuple(zip(*coords))] = 3
coords = [[x, y] for x in range(3*10, 3*20) for y in range(3*10, 3*35)] + \
         [[x, y] for x in range(3*20, 3*40) for y in range(3*10, 3*15)] + \
         [[x, y] for x in range(3*20, 3*40) for y in range(3*30, 3*35)]  # cluster 4
cell_type_template[tuple(zip(*coords))] = 4

cas_list = [ad.read_h5ad(save_dir + f"filtered_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
origin_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)

template_values = cell_type_template[origin_concat.obsm['spatial'][:, 0], origin_concat.obsm['spatial'][:, 1]]
spot_type_values = list(origin_concat.obs['real_spot_clusters'])
matching_indices = np.where(spot_type_values == template_values.astype(str))[0]

origin_concat = origin_concat[matching_indices, :]
origin_concat = ad.AnnData(origin_concat.X, obs=origin_concat.obs, var=origin_concat.var,
                           obsm=origin_concat.obsm, dtype='float')

# clustering and calculating scores
for i in range(len(models)):

    if not os.path.exists(save_dir + f'comparison/{models[i]}/{models[i]}_results_dict_batch_corr_del.pkl'):

        b_asws = np.zeros((num_iters,), dtype=float)
        b_pcrs = np.zeros((num_iters,), dtype=float)
        kbets = np.zeros((num_iters,), dtype=float)
        g_conns = np.zeros((num_iters,), dtype=float)

        results_dict = {'Batch_ASWs': b_asws, 'Batch_PCRs': b_pcrs,
                        'kBETs': kbets, 'Graph_connectivities': g_conns}
        with open(save_dir + f'comparison/{models[i]}/{models[i]}_results_dict_batch_corr_del.pkl', 'wb') as file:
            pickle.dump(results_dict, file)

    for j in range(num_iters):

        print(f'{models[i]} iteration {j}')

        adata_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)

        embed = pd.read_csv(save_dir + f'comparison/{models[i]}/{models[i]}_embed_{j}.csv', header=None).values
        adata_concat.obsm['latent'] = embed

        adata_concat = adata_concat[matching_indices, :]

        print(adata_concat.shape)

        b_asw, b_pcr, kbet, g_conn = batch_correction_metrics(adata_concat, origin_concat, use_rep='latent',
                                                              label_key='real_spot_clusters', batch_key='slice_index')

        with open(save_dir + f'comparison/{models[i]}/{models[i]}_results_dict_batch_corr_del.pkl', 'rb') as file:
            results_dict = pickle.load(file)

        results_dict['Batch_ASWs'][j] = b_asw
        results_dict['Batch_PCRs'][j] = b_pcr
        results_dict['kBETs'][j] = kbet
        results_dict['Graph_connectivities'][j] = g_conn

        with open(save_dir + f'comparison/{models[i]}/{models[i]}_results_dict_batch_corr_del.pkl', 'wb') as file:
            pickle.dump(results_dict, file)

