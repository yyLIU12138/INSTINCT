import os
import pickle
import numpy as np
import anndata as ad
import pandas as pd

from codes.evaluation_utils import knn_cross_validation

import warnings
warnings.filterwarnings("ignore")

# mouse embryo
num_iters = 8
slice_name_list = ['E12_5-S1', 'E12_5-S2', 'E13_5-S1', 'E13_5-S2', 'E15_5-S1', 'E15_5-S2']
slice_index_list = list(range(len(slice_name_list)))
cluster_list = ['Forebrain', 'Midbrain', 'Hindbrain', 'Periventricular', 'Meningeal_PNS_1', 'Meningeal_PNS_2',
                'Internal', 'Facial_bone', 'Muscle_heart', 'Limb', 'Liver']

save_dir = '../../results/MouseEmbryo_Llorens-Bobadilla2023/all/'
cas_list = [ad.read_h5ad(save_dir + f"filtered_{sample}.h5ad") for sample in slice_name_list]
origin_concat = ad.concat(cas_list, label='slice_idx', keys=slice_index_list)
print(origin_concat.shape)
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

        accu, kappa, mf1, wf1 = knn_cross_validation(embed, adata_concat.obs['clusters'].copy(),
                                                     k=20, batch_idx=adata_concat.obs['slice_idx'].copy())

        with open(save_dir + f'annotation/{models[i]}/{models[i]}_results_dict.pkl', 'rb') as file:
            results_dict = pickle.load(file)

        results_dict['Accuracies'][j] = accu
        results_dict['Kappas'][j] = kappa
        results_dict['mF1s'][j] = mf1
        results_dict['wF1s'][j] = wf1

        with open(save_dir + f'annotation/{models[i]}/{models[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)
