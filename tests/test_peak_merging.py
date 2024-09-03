import os
import time
import anndata as ad

from ..INSTINCT import peak_sets_alignment

import warnings
warnings.filterwarnings("ignore")

save_dir = '../../results/tests/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

slice_name_list = ['../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/E11_0-S1_atac.h5ad',
                   '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/E13_5-S1_atac.h5ad',
                   '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/E15_5-S1_atac.h5ad',
                   '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/E18_5-S1_atac.h5ad',
                   "../../data/spMOdata/EpiTran_HumanMouse_Zhang2023/preprocessed_from_fragments/GSM6204623_MouseBrain_20um.h5ad",
                   "../../data/spMOdata/EpiTran_HumanMouse_Zhang2023/preprocessed_from_fragments/GSM6758284_MouseBrain_20um_repATAC.h5ad",
                   "../../data/spMOdata/EpiTran_HumanMouse_Zhang2023/preprocessed_from_fragments/GSM6758285_MouseBrain_20um_100barcodes_ATAC.h5ad",
                   '../../data/spCASdata/HumanMouse_Deng2022/preprocessed/GSM5238385_ME11_50um.h5ad',
                   '../../data/spCASdata/HumanMouse_Deng2022/preprocessed/GSM5238386_ME13_50um.h5ad',
                   '../../data/spCASdata/HumanMouse_Deng2022/preprocessed/GSM5238387_ME13_50um_2.h5ad',
                   ]

for i in range(1, len(slice_name_list)):

    slice_used = [slice_name_list[j] for j in range(i+1)]
    print(len(slice_used))

    # load raw data
    cas_list = []
    for sample in slice_used:
        sample_data = ad.read_h5ad(sample)

        if 'insertion' in sample_data.obsm:
            del sample_data.obsm['insertion']

        cas_list.append(sample_data)

    start_time = time.time()

    # merge peaks
    cas_list = peak_sets_alignment(cas_list)

    end_time = time.time()

    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)

    n_spots = 0
    for j in range(len(cas_list)):
        n_spots += cas_list[j].shape[0]

    with open(save_dir + "peak_merging_log.txt", "a") as file:
        file.write(f"n_spots: {n_spots}, n_merged_peaks: {cas_list[0].shape[1]}, {minutes}min{seconds}sec\n")

    del cas_list



