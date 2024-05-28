import os
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1234)

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

if not os.path.exists(save_dir + 'counts/'):
    os.makedirs(save_dir + 'counts/')

# fragment counts
for i in range(len(slice_name_list)):

    adata = ad.read_h5ad(f"../../data/simulated/{slice_name_list[i]}/sc_simulated_{slice_name_list[i]}.h5ad")

    fig, axs = plt.subplots(figsize=(10, 5))

    element_counts = Counter(adata.X.toarray().flatten())

    element_counts[-1] = sum(count for value, count in element_counts.items() if value > 80)
    elements, counts = zip(*element_counts.items())
    elements, counts = zip(*[(elem, count) for elem, count in zip(elements, counts) if elem <= 80])
    elements = [elem if elem != -1 else 81 for elem in elements]

    labels = [str(j) for j in range(0, 81, 3)]
    labels.append('>80')
    labels_pos = [3 * i - 0.1 for i in range(len(labels) - 1)]
    labels_pos.append(2.9 + 3 * (len(labels) - 2))
    axs.set_xticks(labels_pos)
    axs.set_xticklabels(labels, fontsize=10)

    plt.bar(elements, counts, color='royalblue', log=True)
    plt.xlabel('Reads', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title(f'{slice_name_list[i]}', fontsize=14)

    # save_path = save_dir + f"counts/sc_dataset_fragments_counts_{slice_name_list[i]}.png"
    # plt.savefig(save_path)


    adata = ad.read_h5ad(f"../../data/simulated/{slice_name_list[i]}/spot_level_slice_{slice_name_list[i]}_{scenario}.h5ad")

    fig, axs = plt.subplots(figsize=(10, 5))

    element_counts = Counter(adata.X.toarray().flatten())

    element_counts[-1] = sum(count for value, count in element_counts.items() if value > 80)
    elements, counts = zip(*element_counts.items())
    elements, counts = zip(*[(elem, count) for elem, count in zip(elements, counts) if elem <= 80])
    elements = [elem if elem != -1 else 81 for elem in elements]

    labels = [str(j) for j in range(0, 81, 3)]
    labels.append('>80')
    labels_pos = [3 * i - 0.1 for i in range(len(labels) - 1)]
    labels_pos.append(2.9 + 3 * (len(labels) - 2))
    axs.set_xticks(labels_pos)
    axs.set_xticklabels(labels, fontsize=10)

    plt.bar(elements, counts, color='royalblue', log=True)
    plt.xlabel('Reads', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title(f'{slice_name_list[i]}', fontsize=14)

    # save_path = save_dir + f"counts/sp_slice_fragments_counts_{slice_name_list[i]}.png"
    # plt.savefig(save_path)
    plt.show()


# mean-variance
for i in range(len(slice_name_list)):

    fig, axs = plt.subplots(figsize=(6, 6))

    adata = ad.read_h5ad(f"../../data/simulated/{slice_name_list[i]}/sc_simulated_{slice_name_list[i]}.h5ad")
    X_array = adata.X.toarray()

    means = np.mean(X_array, axis=0)
    variances = np.var(X_array, axis=0)

    plt.scatter(means, variances, color='royalblue', edgecolors='white')
    plt.plot([min(means), max(variances)], [min(means), max(variances)], color='black',
             linestyle='--', label='$\sigma^2 = \mu$ (Poisson limit)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Read Mean $\mu$', fontsize=12)
    plt.ylabel('Read Variance $\sigma^2$', fontsize=12)
    plt.title('Single Cell Dataset', fontsize=14)
    plt.legend(fontsize=12)

    # save_path = save_dir + f"counts/sc_dataset_fragments_mv_{slice_name_list[i]}.png"
    # plt.savefig(save_path, dpi=300)
    # plt.show()

    del adata, X_array, means, variances


    fig, axs = plt.subplots(figsize=(6, 6))

    adata = ad.read_h5ad(f"../../data/simulated/{slice_name_list[i]}/spot_level_slice_{slice_name_list[i]}_{scenario}.h5ad")
    X_array = adata.X.toarray()

    means = np.mean(X_array, axis=0)
    variances = np.var(X_array, axis=0)

    plt.scatter(means, variances, color='royalblue', edgecolors='white')
    plt.plot([min(means), max(variances)], [min(means), max(variances)], color='black',
             linestyle='--', label='$\sigma^2 = \mu$ (Poisson limit)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Read Mean $\mu$', fontsize=12)
    plt.ylabel('Read Variance $\sigma^2$', fontsize=12)
    plt.title('Spatial Slice', fontsize=14)
    plt.legend(fontsize=12)

    # save_path = save_dir + f"counts/sp_slice_fragments_mv_{slice_name_list[i]}.png"
    # plt.savefig(save_path, dpi=300)
    # plt.show()

    del adata, X_array, means, variances
    plt.show()


