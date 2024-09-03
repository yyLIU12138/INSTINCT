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

num_iters = 8

# mouse brain
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

save_dir = '../../results/MouseBrain_Jiang2023/'
if not os.path.exists(save_dir + 'comparison/GraphST/'):
    os.makedirs(save_dir + 'comparison/GraphST/')

# load raw data
cas_dict = {}
for sample in slice_name_list:
    sample_data = ad.read_h5ad(data_dir + sample + '_atac.h5ad')

    if 'insertion' in sample_data.obsm:
        del sample_data.obsm['insertion']

    cas_dict[sample] = sample_data
cas_list = [cas_dict[sample] for sample in slice_name_list]

# merge peaks
cas_list = peak_sets_alignment(cas_list)

# save the merged data
for idx, adata in enumerate(cas_list):
    adata.write_h5ad(f'{data_dir}merged_{slice_name_list[idx]}_atac.h5ad')

# load the merged data
cas_list = [ad.read_h5ad(data_dir + 'merged_' + sample + '_atac.h5ad') for sample in slice_name_list]
for j in range(len(cas_list)):
    cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]

# concatenation
adata_concat = ad.concat(cas_list, label="slice_name", keys=slice_name_list)
# adata_concat.obs_names_make_unique()

# preprocess CAS data
print('Start preprocessing')
preprocess_CAS(cas_list, adata_concat, use_fragment_count=True, min_cells_rate=0.03)
print('Done!')

for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_merged_{slice_name_list[i]}_atac.h5ad")

# GraphST
print('----------GraphST----------')

from GraphST import GraphST
import paste as pst
import ot

for j in range(num_iters):

    print(f'Iteration {j}')

    if j == 0:

        cas_list = [ad.read_h5ad(save_dir + f"filtered_merged_{sample}_atac.h5ad") for sample in slice_name_list]

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
            cas_list[i].write_h5ad(save_dir + f'comparison/GraphST/filtered_merged_{slice_name_list[i]}_atac_aligned.h5ad')

    cas_list = [ad.read_h5ad(save_dir + f'comparison/GraphST/filtered_merged_{sample}_atac_aligned.h5ad')
                for sample in slice_name_list]
    adata_concat = ad.concat(cas_list, label='slice_name', keys=slice_name_list)

    # define model
    model = GraphST.GraphST(adata_concat, device=device, random_seed=1234+j)

    # run model
    result = model.train()
    # print(result)

    pca = PCA(n_components=20, random_state=1234)
    embedding = pca.fit_transform(result.obsm['emb'].copy())
    result.obsm['emb_pca'] = embedding

    with open(save_dir + f'comparison/GraphST/GraphST_embed_{j}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result.obsm['emb_pca'])

print('----------Done----------\n')
