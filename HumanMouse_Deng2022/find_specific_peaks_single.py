import os
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
import episcanpy as epi
import scipy

from ..INSTINCT import *

import warnings
warnings.filterwarnings("ignore")

save = False
n_peaks = 1000
model = 'INSTINCT'
method = 'leiden'

data_dir = '../../data/spCASdata/HumanMouse_Deng2022/preprocessed/'
save_dir = '../../results/HumanMouse_Deng2022/'
slice_name_list = ['GSM5238385_ME11_50um', 'GSM5238386_ME13_50um', 'GSM5238387_ME13_50um_2']
label_list = ['GSM5238385', 'GSM5238386', 'GSM5238387']
slice_used = [0, 1, 2]
slice_name_list = [slice_name_list[i] for i in slice_used]
label_list = [label_list[i] for i in slice_used]
slice_index_list = list(range(len(slice_name_list)))

save_dir = f'../../results/HumanMouse_Deng2022/{slice_used}/'

filter_rate_list = [0.03, 0.07, 0.03]
for i in range(len(slice_name_list)):

    if not os.path.exists(save_dir + f'{label_list[i]}/peaks/'):
        os.makedirs(save_dir + f'{label_list[i]}/peaks/')

    sample = ad.read_h5ad(data_dir + slice_name_list[i] + '.h5ad')
    if 'insertion' in sample.obsm:
        del sample.obsm['insertion']

    sample.obs_names = [x + '_' + slice_name_list[i] for x in sample.obs_names]

    clustered_adata = ad.read_h5ad(save_dir + f'clustered_{slice_name_list[i]}.h5ad')

    sample = sample[clustered_adata.obs_names, :]
    sample.obs[method] = clustered_adata.obs[method]

    # filter
    category_counts = sample.obs[method].value_counts()
    valid_categories = category_counts[category_counts >= 10].index
    sample = sample[sample.obs[method].isin(valid_categories), :]

    epi.pp.filter_features(sample, min_cells=int(filter_rate_list[i] * sample.shape[0]))
    epi.pp.filter_cells(sample, min_features=1)
    print(sample.shape)
    sample.X = scipy.sparse.csr_matrix(np.ceil((sample.X / 2).toarray()))
    sample.X = TFIDF(sample.X.T, type_=2).T.copy()

    # find cluster specific peaks
    group_list = list(set(sample.obs[method]))
    print(group_list)

    sc.tl.rank_genes_groups(sample, method, groups=group_list, method='wilcoxon')
    peaks_list = []
    seen_peaks = set()

    cas_peaks = pd.DataFrame(sample.uns["rank_genes_groups"]["names"])
    cas_logfoldchanges = pd.DataFrame(sample.uns["rank_genes_groups"]["logfoldchanges"])
    cas_pvals_adj = pd.DataFrame(sample.uns["rank_genes_groups"]["pvals_adj"])

    for col in list(cas_peaks.columns):

        peaks = cas_peaks[col].tolist()
        logfoldchanges = cas_logfoldchanges[col].tolist()
        pvals_adj = cas_pvals_adj[col].tolist()

        peaks_filtered = [peaks[j] for j in range(len(peaks)) if logfoldchanges[j] > 1]
        pvals_adj_filtered = [pvals_adj[j] for j in range(len(pvals_adj)) if logfoldchanges[j] > 1]
        print(len(peaks_filtered))

        if len(peaks_filtered) <= n_peaks:
            selected_peaks = peaks_filtered
        else:
            min_indices = np.argsort(pvals_adj_filtered)[:n_peaks]
            selected_peaks = [peaks_filtered[j] for j in min_indices]
        # save peaks
        if save:
            if not selected_peaks:
                with open(save_dir + f'{label_list[i]}/peaks/{col}_specific_peaks.txt', 'w') as f:
                    pass
            else:
                with open(save_dir + f'{label_list[i]}/peaks/{col}_specific_peaks.txt', 'w') as f:
                    for item in selected_peaks:
                        f.write(item + '\n')
        for peak in selected_peaks:
            if peak not in seen_peaks:
                peaks_list.append(peak)
                seen_peaks.add(peak)
        print(f"Label: {col}, Number of specific peaks: {len(selected_peaks)}")

    sorted_peaks = [peak for peak in sample.var_names if peak in peaks_list]
    print(len(sorted_peaks))
    if save:
        with open(save_dir + f'{label_list[i]}/peaks/all_specific_peaks.txt', 'w') as f:
            for item in sorted_peaks:
                f.write(item + '\n')
        selected_adata = ad.read_h5ad(data_dir + slice_name_list[i] + '.h5ad')
        if 'insertion' in selected_adata.obsm:
            del selected_adata.obsm['insertion']
        selected_adata.obs_names = [x + '_' + slice_name_list[i] for x in selected_adata.obs_names]
        selected_adata = selected_adata[sample.obs_names, sorted_peaks]
        selected_adata.obs[method] = sample.obs[method]
        selected_adata.X = scipy.sparse.csr_matrix(np.ceil((selected_adata.X / 2).toarray()))
        selected_adata.X = TFIDF(selected_adata.X.T, type_=2).T.copy()
        print(selected_adata.shape)
        selected_adata.write_h5ad(save_dir + f'{label_list[i]}/selected_' + slice_name_list[i] + '.h5ad')

