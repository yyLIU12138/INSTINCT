import os
import csv
import torch

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

num_iters = 8

# mouse brain
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

slice_index_list = list(range(len(slice_name_list)))

save_dir = '../../results/MouseBrain_Jiang2023/single/'
if not os.path.exists(save_dir + 'SCALE/'):
    os.makedirs(save_dir + 'SCALE/')

# SCALE
print('----------SCALE----------')

from scale import SCALE_function

for i in range(num_iters):

    print(f'Iteration {i}')

    for sample in slice_name_list:

        adata = SCALE_function([data_dir+sample+'_atac.h5ad'], outdir=save_dir+'SCALE/', seed=1234+i)
        print(adata.obsm['latent'].shape)

        with open(save_dir + f'SCALE/SCALE_embed_{i}_{sample}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(adata.obsm['latent'])

print('----------Done----------\n')

