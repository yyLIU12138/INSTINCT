import os
import matplotlib.pyplot as plt
import anndata as ad
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

save = False

CAS_samples = ["GSM6206884_HumanBrain_50um", "GSM6801813_ME13_50um", "GSM6204624_ME13_100barcodes_25um",
               "GSM6204623_MouseBrain_20um", "GSM6758284_MouseBrain_20um_repATAC", "GSM6758285_MouseBrain_20um_100barcodes_ATAC",
               "GSM6204621_MouseBrain_20um_H3K27ac", "GSM6704977_MouseBrain_20um_rep_H3K27ac", "GSM6704978_MouseBrain_20um_100barcodes_H3K27me3",
               "GSM6704979_MouseBrain_20um_100barcodes_H3K27ac", "GSM6704980_MouseBrain_20um_100barcodes_H3K4me3"]
RNA_samples = ["GSM6206885_HumanBrain_50um", "GSM6799937_ME13_50um", "GSM6204637_ME13_100barcodes_25um",
               "GSM6204636_MouseBrain_20um", "GSM6753041_MouseBrain_20um_repATAC", "GSM6753043_MouseBrain_20um_100barcodes_ATAC",
               "GSM6204635_MouseBrain_20um_H3K27ac", "GSM6753042_MouseBrain_20um_repH3K27ac", "GSM6753044_MouseBrain_20um_100barcodes_H3K27me3",
               "GSM6753045_MouseBrain_20um_100barcodes_H3K27ac", "GSM6753046_MouseBrain_20um_100barcodes_H3K4me3"]

data_dir = '../../data/spMOdata/EpiTran_HumanMouse_Zhang2023/preprocessed_from_fragments/'
save_dir = '../../results/HumanMouse_Zhang2023/'
slice_name_list = ["GSM6204623_MouseBrain_20um", "GSM6758284_MouseBrain_20um_repATAC", "GSM6758285_MouseBrain_20um_100barcodes_ATAC"]
slice_index_list = list(range(len(slice_name_list)))
if not os.path.exists(save_dir + 'counts/'):
    os.makedirs(save_dir + 'counts/')

# read counts & fragment counts
for i in range(len(slice_name_list)):

    adata = ad.read_h5ad(data_dir + slice_name_list[i] + '.h5ad')
    if 'insertion' in adata.obsm:
        del adata.obsm['insertion']

    fig, axs = plt.subplots(figsize=(10, 5))

    element_counts = Counter(adata.X.toarray().flatten())

    element_counts[-1] = sum(count for value, count in element_counts.items() if value > 60)
    elements, counts = zip(*element_counts.items())
    elements, counts = zip(*[(elem, count) for elem, count in zip(elements, counts) if elem <= 60])
    elements = [elem if elem != -1 else 61 for elem in elements]

    labels = [str(j) for j in range(0, 58, 3)]
    labels.append('>60')
    labels_pos = [3 * i - 0.1 for i in range(len(labels) - 1)]
    labels_pos.append(3.9 + 3 * (len(labels) - 2))
    axs.set_xticks(labels_pos)
    axs.set_xticklabels(labels, fontsize=10)

    plt.bar(elements, counts, color='royalblue', log=True)
    plt.xlabel('Reads', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title(f'Raw Data {slice_name_list[i]}', fontsize=14)

    if save:
        save_path = save_dir + f"counts/raw_read_counts_{slice_name_list[i]}.png"
        plt.savefig(save_path)
