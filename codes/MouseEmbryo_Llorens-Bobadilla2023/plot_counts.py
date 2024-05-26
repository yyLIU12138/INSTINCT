import os
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

save = False

data_dir = '../../data/spCASdata/MouseEmbryo_Llorens-Bobadilla2023/spATAC/'
save_dir = '../../results/MouseEmbryo_Llorens-Bobadilla2023/'
if not os.path.exists(save_dir + 'counts/'):
    os.makedirs(save_dir + 'counts/')

slice_name_list = ['E12_5-S1', 'E12_5-S2', 'E13_5-S1', 'E13_5-S2', 'E15_5-S1', 'E15_5-S2']

# fragment counts
for i in range(len(slice_name_list)):

    adata = ad.read_h5ad(data_dir + slice_name_list[i] + '.h5ad')

    fig, axs = plt.subplots(figsize=(7, 5))

    element_counts = Counter(adata.X.toarray().flatten())

    elements, counts = zip(*element_counts.items())

    max_element = int(max(elements))
    x_ticks = list(range(max_element + 1))

    sorted_elements, sorted_counts = zip(*sorted(zip(elements, counts), key=lambda x: x[0]))

    axs.set_xticks(x_ticks)
    axs.set_xticklabels(x_ticks, fontsize=10)

    plt.bar(sorted_elements, sorted_counts, color='royalblue', log=True)
    plt.xlabel('Fragments', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title(f'{slice_name_list[i]}', fontsize=14)

    if save:
        save_path = save_dir + f"counts/fragment_counts_{slice_name_list[i]}.png"
        plt.savefig(save_path)


# mean-variance
for i in range(len(slice_name_list)):

    fig, axs = plt.subplots(figsize=(6, 6))

    adata = ad.read_h5ad(data_dir + slice_name_list[i] + '.h5ad')
    X_array = adata.X.toarray()

    means = np.mean(X_array, axis=0)
    variances = np.var(X_array, axis=0)

    plt.scatter(means, variances, color='royalblue', edgecolors='white')
    plt.plot([min(means), max(variances)], [min(means), max(variances)], color='black',
             linestyle='--', label='$\sigma^2 = \mu$ (Poisson limit)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Fragment Mean $\mu$', fontsize=12)
    plt.ylabel('Fragment Variance $\sigma^2$', fontsize=12)
    plt.title(f'{slice_name_list[i]}', fontsize=14)
    plt.legend(fontsize=12)

    if save:
        save_path = save_dir + f"counts/fragment_mv_{slice_name_list[i]}.png"
        plt.savefig(save_path, dpi=300)

    del adata, X_array, means, variances

