import os
import csv
import torch
import anndata as ad
import scanpy as sc
import episcanpy as epi

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
if not os.path.exists(save_dir + 'comparison/PeakVI/'):
    os.makedirs(save_dir + 'comparison/PeakVI/')

# PeakVI
print('----------PeakVI----------')

import scvi

for j in range(num_iters):

    print(f'Iteration {j}')

    scvi.settings.seed = 1234+j

    cas_list = [ad.read_h5ad(save_dir + f"filtered_{sample}.h5ad") for sample in slice_name_list]
    adata_concat = ad.concat(cas_list, label='slice_name', keys=slice_name_list)

    print("# regions before filtering:", adata_concat.shape[-1])

    min_cells = int(adata_concat.shape[0] * 0.05)
    sc.pp.filter_genes(adata_concat, min_cells=min_cells)

    print("# regions after filtering:", adata_concat.shape[-1])

    epi.pp.binarize(adata_concat)
    scvi.model.PEAKVI.setup_anndata(adata_concat, batch_key='slice_name')
    model = scvi.model.PEAKVI(adata_concat)
    model.train(use_gpu=True)
    # print(adata_concat)

    # if j == 0:
    #     if not os.path.exists(save_dir + 'comparison/PeakVI/model/'):
    #         os.makedirs(save_dir + 'comparison/PeakVI/model/')
    #     model.save(save_dir + 'comparison/PeakVI/model/', overwrite=True)
    #
    # model = scvi.model.PEAKVI.load(save_dir + 'comparison/PeakVI/model/', adata=adata_concat)

    latent = model.get_latent_representation()
    adata_concat.obsm['PeakVI'] = latent

    with open(save_dir + f'comparison/PeakVI/PeakVI_embed_{j}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(adata_concat.obsm['PeakVI'])

print('----------Done----------\n')
