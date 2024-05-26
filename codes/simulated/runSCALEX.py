import os
import csv
import torch
import anndata as ad

from codes.INSTINCT_main.utils import preprocess_CAS

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

name_concat = slice_name_list[0]
for mode in slice_name_list[1:]:
    name_concat = name_concat + '_' + mode

save_dir = f'../../results/simulated/scenario_{scenario}/T_' + name_concat + '/'

if not os.path.exists(save_dir + 'comparison/SCALEX/'):
    os.makedirs(save_dir + 'comparison/SCALEX/')

cas_list = [ad.read_h5ad(f"../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
for j in range(len(cas_list)):
    cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]
adata_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)
preprocess_CAS(cas_list, adata_concat)

for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_spot_level_slice_{slice_name_list[i]}.h5ad")

# SCALE
print('----------SCALEX----------')

from scalex import SCALEX

for j in range(num_iters):

    print(f'Iteration {j}')

    cas_list = [ad.read_h5ad(save_dir + f"filtered_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
    for i, adata in enumerate(cas_list):
        adata.obs['batch'] = str(slice_index_list[i])

    adata_concat = SCALEX(cas_list, batch_key='batch', profile='ATAC', min_features=1, n_top_features=30000,
                          seed=1234 + j, target_sum=None, ignore_umap=True, outdir=save_dir+'comparison/SCALEX/')
    # print(adata_concat)

    with open(save_dir + f'comparison/SCALEX/SCALEX_embed_{j}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(adata_concat.obsm['latent'])

print('----------Done----------\n')
