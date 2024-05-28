import os
import csv
import torch
import episcanpy as epi
import anndata as ad
import scanpy as sc

import INSTINCT

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

if not os.path.exists(save_dir + 'comparison/PeakVI/'):
    os.makedirs(save_dir + 'comparison/PeakVI/')

cas_list = [ad.read_h5ad(f"../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
for j in range(len(cas_list)):
    cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]
adata_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)
preprocess_CAS(cas_list, adata_concat)

for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_spot_level_slice_{slice_name_list[i]}.h5ad")

# PeakVI
print('----------PeakVI----------')

import scvi

for j in range(num_iters):

    print(f'Iteration {j}')

    scvi.settings.seed = 1234+j

    cas_list = [ad.read_h5ad(save_dir + f"filtered_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
    adata_concat = ad.concat(cas_list, label='slice_index', keys=slice_index_list)

    print("# regions before filtering:", adata_concat.shape[-1])

    min_cells = int(adata_concat.shape[0] * 0.05)
    sc.pp.filter_genes(adata_concat, min_cells=min_cells)

    print("# regions after filtering:", adata_concat.shape[-1])

    epi.pp.binarize(adata_concat)
    scvi.model.PEAKVI.setup_anndata(adata_concat, batch_key='slice_index')
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
