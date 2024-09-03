import os
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
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

if not os.path.exists(save_dir + f'concat/peaks/'):
    os.makedirs(save_dir + f'concat/peaks/')

cas_list = [ad.read_h5ad(save_dir + f'clustered_{sample}.h5ad') for i, sample in enumerate(slice_name_list)]
adata_concat = ad.concat(cas_list, label="slice_name", keys=label_list)
print(adata_concat.shape)
adata_concat.X = scipy.sparse.csr_matrix(np.ceil((adata_concat.X / 2).toarray()))
adata_concat.X = TFIDF(adata_concat.X.T, type_=2).T.copy()

# find cluster specific peaks
group_list = list(set(adata_concat.obs[method]))
print(group_list)

sc.tl.rank_genes_groups(adata_concat, method, groups=group_list, method='wilcoxon')
peaks_list = []
seen_peaks = set()

cas_peaks = pd.DataFrame(adata_concat.uns["rank_genes_groups"]["names"])
cas_logfoldchanges = pd.DataFrame(adata_concat.uns["rank_genes_groups"]["logfoldchanges"])
cas_pvals_adj = pd.DataFrame(adata_concat.uns["rank_genes_groups"]["pvals_adj"])

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
            with open(save_dir + f'concat/peaks/{col}_specific_peaks.txt', 'w') as f:
                pass
        else:
            with open(save_dir + f'concat/peaks/{col}_specific_peaks.txt', 'w') as f:
                for item in selected_peaks:
                    f.write(item + '\n')
    for peak in selected_peaks:
        if peak not in seen_peaks:
            peaks_list.append(peak)
            seen_peaks.add(peak)
    print(f"Label: {col}, Number of specific peaks: {len(selected_peaks)}")

sorted_peaks = [peak for peak in adata_concat.var_names if peak in peaks_list]
print(len(sorted_peaks))
if save:
    with open(save_dir + f'concat/peaks/all_specific_peaks.txt', 'w') as f:
        for item in sorted_peaks:
            f.write(item + '\n')
    adata_concat = ad.concat(cas_list, label="slice_name", keys=label_list)
    adata_concat = adata_concat[adata_concat.obs_names, sorted_peaks]
    adata_concat.X = scipy.sparse.csr_matrix(np.ceil((adata_concat.X / 2).toarray()))
    adata_concat.X = TFIDF(adata_concat.X.T, type_=2).T.copy()
    print(adata_concat.shape)
    adata_concat.write_h5ad(save_dir + f'concat/selected_concat.h5ad')
