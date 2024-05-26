import os
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

save = False

data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
save_dir = '../../results/MouseBrain_Jiang2023/'
if not os.path.exists(save_dir + 'counts/'):
    os.makedirs(save_dir + 'counts/')

slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

# read counts & fragment counts
for i in range(len(slice_name_list)):

    adata = ad.read_h5ad(data_dir + slice_name_list[i] + '_atac.h5ad')
    if 'insertion' in adata.obsm:
        del adata.obsm['insertion']

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
    plt.title(f'Raw Data {slice_name_list[i]}', fontsize=14)

    if save:
        save_path = save_dir + f"counts/raw_read_counts_{slice_name_list[i]}.png"
        plt.savefig(save_path)

    fig, axs = plt.subplots(figsize=(10, 5))

    element_counts = Counter(np.ceil((adata.X / 2).toarray().flatten()))

    element_counts[-1] = sum(count for value, count in element_counts.items() if value > 40)
    elements, counts = zip(*element_counts.items())
    elements, counts = zip(*[(elem, count) for elem, count in zip(elements, counts) if elem <= 40])
    elements = [elem if elem != -1 else 41 for elem in elements]

    labels = [str(j) for j in range(0, 41, 3)]
    labels.append('>40')
    labels_pos = [3 * i - 0.1 for i in range(len(labels) - 1)]
    labels_pos.append(1.9 + 3 * (len(labels) - 2))
    axs.set_xticks(labels_pos)
    axs.set_xticklabels(labels, fontsize=10)

    plt.bar(elements, counts, color='royalblue', log=True)
    plt.xlabel('Fragments', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.title(f'Raw Data {slice_name_list[i]}', fontsize=16)

    if save:
        save_path = save_dir + f"counts/raw_fragment_counts_{slice_name_list[i]}.png"
        plt.savefig(save_path)

    adata = ad.read_h5ad(data_dir + 'merged_' + slice_name_list[i] + '_atac.h5ad')

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
    plt.title(f'Merged Data {slice_name_list[i]}', fontsize=14)

    if save:
        save_path = save_dir + f"counts/merged_read_counts_{slice_name_list[i]}.png"
        plt.savefig(save_path)

    fig, axs = plt.subplots(figsize=(10, 5))

    element_counts = Counter(np.ceil((adata.X / 2).toarray().flatten()))

    element_counts[-1] = sum(count for value, count in element_counts.items() if value > 40)
    elements, counts = zip(*element_counts.items())
    elements, counts = zip(*[(elem, count) for elem, count in zip(elements, counts) if elem <= 40])
    elements = [elem if elem != -1 else 41 for elem in elements]

    labels = [str(j) for j in range(0, 41, 3)]
    labels.append('>40')
    labels_pos = [3 * i - 0.1 for i in range(len(labels) - 1)]
    labels_pos.append(1.9 + 3 * (len(labels) - 2))
    axs.set_xticks(labels_pos)
    axs.set_xticklabels(labels, fontsize=10)

    plt.bar(elements, counts, color='royalblue', log=True)
    plt.xlabel('Fragments', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.title(f'Merged Data {slice_name_list[i]}', fontsize=16)

    if save:
        save_path = save_dir + f"counts/merged_fragment_counts_{slice_name_list[i]}.png"
        plt.savefig(save_path)
    plt.show()


# mean-variance
for i in range(len(slice_name_list)):

    fig, axs = plt.subplots(figsize=(6, 6))

    adata = ad.read_h5ad(data_dir + slice_name_list[i] + '_atac.h5ad')
    if 'insertion' in adata.obsm:
        del adata.obsm['insertion']
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
    plt.title(f'Raw Data {slice_name_list[i]}', fontsize=14)
    plt.legend(fontsize=12)

    if save:
        save_path = save_dir + f"counts/raw_read_mv_{slice_name_list[i]}.png"
        plt.savefig(save_path, dpi=300)

    del adata, X_array, means, variances

    fig, axs = plt.subplots(figsize=(6, 6))

    adata = ad.read_h5ad(data_dir + slice_name_list[i] + '_atac.h5ad')
    if 'insertion' in adata.obsm:
        del adata.obsm['insertion']
    X_array = np.ceil(adata.X.toarray() / 2)

    means = np.mean(X_array, axis=0)
    variances = np.var(X_array, axis=0)

    plt.scatter(means, variances, color='royalblue', edgecolors='white')
    plt.plot([min(means), max(variances)], [min(means), max(variances)], color='black',
             linestyle='--', label='$\sigma^2 = \mu$ (Poisson limit)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Fragment Mean $\mu$', fontsize=12)
    plt.ylabel('Fragment Variance $\sigma^2$', fontsize=12)
    plt.title(f'Raw Data {slice_name_list[i]}', fontsize=14)
    plt.legend(fontsize=12)

    if save:
        save_path = save_dir + f"counts/raw_fragment_mv_{slice_name_list[i]}.png"
        plt.savefig(save_path, dpi=300)

    del adata, X_array, means, variances

    fig, axs = plt.subplots(figsize=(6, 6))

    adata = ad.read_h5ad(data_dir + 'merged_' + slice_name_list[i] + '_atac.h5ad')
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
    plt.title(f'Merged Data {slice_name_list[i]}', fontsize=14)
    plt.legend(fontsize=12)

    if save:
        save_path = save_dir + f"counts/merged_read_mv_{slice_name_list[i]}.png"
        plt.savefig(save_path, dpi=300)

    del adata, X_array, means, variances

    fig, axs = plt.subplots(figsize=(6, 6))

    adata = ad.read_h5ad(data_dir + 'merged_' + slice_name_list[i] + '_atac.h5ad')
    X_array = np.ceil(adata.X.toarray() / 2)

    means = np.mean(X_array, axis=0)
    variances = np.var(X_array, axis=0)

    plt.scatter(means, variances, color='royalblue', edgecolors='white')
    plt.plot([min(means), max(variances)], [min(means), max(variances)], color='black',
             linestyle='--', label='$\sigma^2 = \mu$ (Poisson limit)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Fragment Mean $\mu$', fontsize=12)
    plt.ylabel('Fragment Variance $\sigma^2$', fontsize=12)
    plt.title(f'Merged Data {slice_name_list[i]}', fontsize=14)
    plt.legend(fontsize=12)

    if save:
        save_path = save_dir + f"counts/merged_fragment_mv_{slice_name_list[i]}.png"
        plt.savefig(save_path, dpi=300)
    plt.show()

    del adata, X_array, means, variances