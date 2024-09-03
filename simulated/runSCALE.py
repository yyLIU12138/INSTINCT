import os
import csv
import torch

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

num_clusters = 5
num_iters = 8

scenario = 1

slice_name_list = [
    'Tech_0_0_Bio_0_0.5',
    'Tech_0_0.1_Bio_0_0',
    'Tech_0_0.1_Bio_0_0.5',
             ]

slice_index_list = list(range(len(slice_name_list)))

save_dir = f'../../results/simulated/scenario_{scenario}/single/'
if not os.path.exists(save_dir + 'SCALE/'):
    os.makedirs(save_dir + 'SCALE/')

# SCALE
print('----------SCALE----------')

from scale import SCALE_function

for i in range(num_iters):

    print(f'Iteration {i}')

    for mode in slice_name_list:

        adata = SCALE_function([f"../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad"],
                               outdir=save_dir+'SCALE/', gpu=device, seed=1234+i)
        print(adata.obsm['latent'].shape)

        with open(save_dir + f'SCALE/SCALE_embed_{i}_{mode}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(adata.obsm['latent'])

print('----------Done----------\n')

