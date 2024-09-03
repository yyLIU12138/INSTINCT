import os
import csv
import time
import torch
import anndata as ad
from sklearn.decomposition import PCA

from ..INSTINCT import *

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

if not os.path.exists(save_dir + 'comparison/GraphST/'):
    os.makedirs(save_dir + 'comparison/GraphST/')

cas_list = [ad.read_h5ad(f"../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
for j in range(len(cas_list)):
    cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]
adata_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)
preprocess_CAS(cas_list, adata_concat)

for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_spot_level_slice_{slice_name_list[i]}.h5ad")

# GraphST
print('----------GraphST----------')

from GraphST import GraphST
import paste as pst
import ot

for j in range(num_iters):

    print(f'Iteration {j}')

    if j == 0:

        cas_list = [ad.read_h5ad(save_dir + f"filtered_spot_level_slice_{sample}.h5ad") for sample in slice_name_list]

        # Pairwise align the slices
        start = time.time()
        pis = []
        for i in range(len(cas_list) - 1):
            pi0 = pst.match_spots_using_spatial_heuristic(cas_list[i].obsm['spatial'], cas_list[i + 1].obsm['spatial'],
                                                          use_ot=True)
            pi = pst.pairwise_align(cas_list[i], cas_list[i + 1], G_init=pi0, norm=True, verbose=False,
                                    backend=ot.backend.TorchBackend(), use_gpu=True)
            pis.append(pi)
        print('Alignment Runtime: ' + str(time.time() - start))

        # To visualize the alignment you can stack the slices
        # according to the alignment pi
        cas_list = pst.stack_slices_pairwise(cas_list, pis)

        for i in range(len(cas_list)):
            cas_list[i].write_h5ad(
                save_dir + f'comparison/GraphST/filtered_spot_level_slice_{slice_name_list[i]}_aligned.h5ad')

    cas_list = [ad.read_h5ad(save_dir + f'comparison/GraphST/filtered_spot_level_slice_{sample}_aligned.h5ad')
                for sample in slice_name_list]
    adata_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)

    # define model
    model = GraphST.GraphST(adata_concat, device=device, random_seed=1234+j)

    # run model
    result = model.train()
    # print(result)

    pca = PCA(n_components=20, random_state=1234)
    emb_pca = pca.fit_transform(result.obsm['emb'].copy())

    with open(save_dir + f'comparison/GraphST/GraphST_embed_{j}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(emb_pca)

print('----------Done----------\n')
