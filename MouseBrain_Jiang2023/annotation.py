import os
import pickle
import numpy as np
import pandas as pd
import anndata as ad

from ..evaluation_utils import knn_cross_validation

import warnings
warnings.filterwarnings("ignore")

# mouse brain
num_iters = 8
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']
slice_index_list = list(range(len(slice_name_list)))
cls_list = ['Primary_brain_1', 'Primary_brain_2', 'Midbrain',  'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
            'Subpallium_1', 'Subpallium_2', 'Cartilage_1', 'Cartilage_2', 'Cartilage_3', 'Cartilage_4',
            'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv']

save_dir = '../../results/MouseBrain_Jiang2023/'
cas_list = [ad.read_h5ad(save_dir + f"filtered_merged_{sample}_atac.h5ad") for sample in slice_name_list]
adata_concat = ad.concat(cas_list, label='slice_idx', keys=slice_index_list)

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

        accu, kappa, mf1, wf1 = knn_cross_validation(embed, adata_concat.obs['Annotation_for_Combined'].copy(),
                                                     k=20, batch_idx=adata_concat.obs['slice_idx'].copy())

        with open(save_dir + f'annotation/{models[i]}/{models[i]}_results_dict.pkl', 'rb') as file:
            results_dict = pickle.load(file)

        results_dict['Accuracies'][j] = accu
        results_dict['Kappas'][j] = kappa
        results_dict['mF1s'][j] = mf1
        results_dict['wF1s'][j] = wf1

        with open(save_dir + f'annotation/{models[i]}/{models[i]}_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)
