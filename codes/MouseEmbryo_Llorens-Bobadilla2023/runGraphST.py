import os
import csv
import time
import torch
import anndata as ad

from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

num_iters = 8

# mouse embryo
data_dir = '../../data/spCASdata/MouseEmbryo_Llorens-Bobadilla2023/spATAC/'
save_dir = '../../results/MouseEmbryo_Llorens-Bobadilla2023/all/'

slice_name_list = ['E12_5-S1', 'E12_5-S2', 'E13_5-S1', 'E13_5-S2', 'E15_5-S1', 'E15_5-S2']
slice_index_list = list(range(len(slice_name_list)))
if not os.path.exists(save_dir + 'comparison/GraphST/'):
    os.makedirs(save_dir + 'comparison/GraphST/')

# GraphST
print('----------GraphST----------')

from GraphST import GraphST
import paste as pst
import ot

for j in range(num_iters):

    print(f'Iteration {j}')

    if j == 0:

        cas_list = [ad.read_h5ad(save_dir + f"filtered_{sample}.h5ad") for sample in slice_name_list]

        # Pairwise align the slices
        start = time.time()
        pis = []
        for i in range(len(cas_list) - 1):
            pi0 = pst.match_spots_using_spatial_heuristic(cas_list[i].obsm['spatial'], cas_list[i + 1].obsm['spatial'],
                                                          use_ot=True)
            pi = pst.pairwise_align(cas_list[i], cas_list[i + 1], G_init=pi0, norm=True, verbose=False,
                                    backend=ot.backend.TorchBackend(), use_gpu=False)
            pis.append(pi)
        print('Alignment Runtime: ' + str(time.time() - start))

        # To visualize the alignment you can stack the slices
        # according to the alignment pi
        cas_list = pst.stack_slices_pairwise(cas_list, pis)

        for i in range(len(cas_list)):
            cas_list[i].write_h5ad(save_dir + f'comparison/GraphST/filtered_{slice_name_list[i]}_aligned.h5ad')

    cas_list = [ad.read_h5ad(save_dir + f'comparison/GraphST/filtered_{sample}_aligned.h5ad')
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
