import os
import matplotlib.pyplot as plt
import anndata as ad
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

save = False

data_dir = '../../data/spCASdata/HumanMouse_Deng2022/preprocessed/'
save_dir = '../../results/HumanMouse_Deng2022/'
slice_name_list = ['GSM5238385_ME11_50um', 'GSM5238386_ME13_50um', 'GSM5238387_ME13_50um_2']
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
