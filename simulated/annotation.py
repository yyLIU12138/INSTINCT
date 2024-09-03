import os
import pickle
import numpy as np
import pandas as pd
import anndata as ad

from ..evaluation_utils import knn_cross_validation

import warnings
warnings.filterwarnings("ignore")

scenario = 1

slice_name_list = [
    'Tech_0_0_Bio_0_0.5',
    'Tech_0_0.1_Bio_0_0',
    'Tech_0_0.1_Bio_0_0.5',
             ]

name_concat = slice_name_list[0]
for mode in slice_name_list[1:]:
    name_concat = name_concat + '_' + mode

save_dir = f'../../results/simulated/scenario_{scenario}/T_' + name_concat + '/'

slice_index_list = [str(i) for i in range(len(slice_name_list))]
num_iters = 8

models = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']

cas_list = [ad.read_h5ad(save_dir + f"filtered_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
adata_concat = ad.concat(cas_list, label='slice_idx', keys=slice_index_list)


# clustering and calculating scores
for i in range(len(models)):

    if not os.path.exists(save_dir + f'annotation/{models[i]}/'):
        os.makedirs(save_dir + f'annotation/{models[i]}/')

    if not os.path.exists(save_dir + f'annotation/{models[i]}/{models[i]}_results_dict.pkl'):
        accus = np.zeros((num_iters,), dtype=float)
        kappas = np.zeros((num_iters,), dtype=float)
        mf1s = np.zeros((num_iters,), dtype=float)
        wf1s = np.zeros((num_iters,), dtype=float)

        results_dict = {'Accuracies': accus, 'Kappas': kappas, 'mF1s': mf1s, 'wF1s': wf1s}
        with open(save_dir + f'annotation/{models[i]}/{models[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)

    for j in range(num_iters):

        print(f'{models[i]} iteration {j}')

        embed = pd.read_csv(save_dir + f'comparison/{models[i]}/{models[i]}_embed_{j}.csv', header=None).values

        accu, kappa, mf1, wf1 = knn_cross_validation(embed, adata_concat.obs['real_spot_clusters'].copy(),
                                                     k=20, batch_idx=adata_concat.obs['slice_idx'].copy())

        with open(save_dir + f'annotation/{models[i]}/{models[i]}_results_dict.pkl', 'rb') as file:
            results_dict = pickle.load(file)

        results_dict['Accuracies'][j] = accu
        results_dict['Kappas'][j] = kappa
        results_dict['mF1s'][j] = mf1
        results_dict['wF1s'][j] = wf1

        with open(save_dir + f'annotation/{models[i]}/{models[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)
