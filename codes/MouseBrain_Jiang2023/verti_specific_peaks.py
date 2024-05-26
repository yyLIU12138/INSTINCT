import os
import scipy
import numpy as np
import anndata as ad
import scanpy as sc
import episcanpy as epi
import pandas as pd

from codes.INSTINCT_main.utils import TFIDF

import warnings
warnings.filterwarnings("ignore")

save = False
n_peaks = 300
model = 'INSTINCT'

mode_list = ['E11_0', 'E13_5', 'E15_5', 'E18_5']
mode_index = 3
mode = mode_list[mode_index]

data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
# sc.settings.set_figure_params(dpi=300, facecolor="white")

save_dir = f'../../results/MouseBrain_Jiang2023/vertical/{mode}/'
# if not os.path.exists(save_dir + f'{model}/peaks/S1/'):
#     os.makedirs(save_dir + f'{model}/peaks/S1/')
if not os.path.exists(save_dir + f'{model}/peaks/S2/'):
    os.makedirs(save_dir + f'{model}/peaks/S2/')
slice_name_list = [f'{mode}-S1', f'{mode}-S2']

# read the filtered and annotated CAS data
cas_s1 = ad.read_h5ad(save_dir + f'filtered_merged_{slice_name_list[0]}_atac.h5ad')
cas_s2 = ad.read_h5ad(save_dir + f'{model}/annotated_{slice_name_list[1]}_atac.h5ad')

# read the raw CAS data
raw_list = [ad.read_h5ad(data_dir + f'{sample}_atac.h5ad') for sample in slice_name_list]
for j in range(len(raw_list)):
    raw_list[j].obs_names = [x + '_' + slice_name_list[j] for x in raw_list[j].obs_names]

# filter and reorder spots in cas slices
epi.pp.filter_features(raw_list[0], min_cells=int(0.03 * raw_list[0].shape[0]))
epi.pp.filter_cells(raw_list[0], min_features=1)
obs_list = [obs_name for obs_name in cas_s1.obs_names if obs_name in raw_list[0].obs_names]
cas_s1 = cas_s1[obs_list, :]
raw_list[0] = raw_list[0][obs_list, :]

epi.pp.filter_features(raw_list[1], min_cells=int(0.03 * raw_list[1].shape[0]))
epi.pp.filter_cells(raw_list[1], min_features=1)
obs_list = [obs_name for obs_name in cas_s2.obs_names if obs_name in raw_list[1].obs_names]
cas_s2 = cas_s2[obs_list, :]
raw_list[1] = raw_list[1][obs_list, :]

raw_s1, raw_s2 = raw_list
print(raw_s1.shape, raw_s2.shape)

# transfer the annotated labels to raw s2 slice
raw_s2.obs['predicted_labels'] = cas_s2.obs['predicted_labels'].copy()

# change the read count matrix to fragment count matrix
raw_s1.X = scipy.sparse.csr_matrix(np.ceil((raw_s1.X / 2).toarray()))
raw_s2.X = scipy.sparse.csr_matrix(np.ceil((raw_s2.X / 2).toarray()))

raw_s1 = ad.AnnData(raw_s1.X, obs=raw_s1.obs, var=raw_s1.var, obsm=raw_s1.obsm, dtype=float)

# TFIDF transformation
raw_s1.X = TFIDF(raw_s1.X.T, type_=2).T.copy()
raw_s2.X = TFIDF(raw_s2.X.T, type_=2).T.copy()

# find domain specific peaks
group_list = list(set(raw_s2.obs['predicted_labels']))
print(group_list)

print('S2')
cas_s2_peaks_list = []
bg_peak_list = []

sc.tl.rank_genes_groups(raw_s2, "predicted_labels", groups=group_list, method='wilcoxon')

cas_s2_peaks = pd.DataFrame(raw_s2.uns["rank_genes_groups"]["names"])
cas_s2_logfoldchanges = pd.DataFrame(raw_s2.uns["rank_genes_groups"]["logfoldchanges"])
cas_s2_pvals_adj = pd.DataFrame(raw_s2.uns["rank_genes_groups"]["pvals_adj"])

for col in list(cas_s2_peaks.columns):

    s2_peaks = cas_s2_peaks[col].tolist()
    s2_logfoldchanges = cas_s2_logfoldchanges[col].tolist()
    s2_pvals_adj = cas_s2_pvals_adj[col].tolist()

    s2_peaks_filtered = [s2_peaks[i] for i in range(len(s2_peaks)) if s2_logfoldchanges[i] > 0.2]
    s2_pvals_adj_filtered = [s2_pvals_adj[i] for i in range(len(s2_pvals_adj)) if s2_logfoldchanges[i] > 0.2]
    print(len(s2_peaks_filtered))

    if len(s2_peaks_filtered) <= n_peaks:
        selected_peaks = s2_peaks_filtered
    else:
        min_indices = np.argsort(s2_pvals_adj_filtered)[:n_peaks]
        selected_peaks = [s2_peaks_filtered[i] for i in min_indices]
    cas_s2_peaks_list.append(selected_peaks)
    # save peaks
    if save:
        if not selected_peaks:
            with open(save_dir + f'{model}/peaks/S2/{col}_specific_peaks.txt', 'w') as f:
                pass
        else:
            with open(save_dir + f'{model}/peaks/S2/{col}_specific_peaks.txt', 'w') as f:
                for item in selected_peaks:
                    f.write(item + '\n')
    print(f"Label: {col}, Number of specific peaks: {len(selected_peaks)}")

    max_indices = np.argsort(s2_pvals_adj)[-n_peaks:]
    bg_peaks = [s2_peaks[i] for i in max_indices]
    bg_peak_list.append(bg_peaks)


# background peaks
s2_union_set = set()
for lst in cas_s2_peaks_list:
    s2_union_set.update(lst)
s2_union_list = list(s2_union_set)
print(len(s2_union_list))
s2_bg_peaks = [peak for peak in raw_s2.var_names.tolist() if peak not in s2_union_list]
if save:
    with open(save_dir + f'{model}/peaks/S2/bg_peaks_all.txt', 'w') as f:
        for item in s2_bg_peaks:
            f.write(item + '\n')

bg_union_set = set()
for lst in bg_peak_list:
    bg_union_set.update(lst)
bg_union_list = list(bg_union_set)
print(f"Number of background peaks: {len(bg_union_list)}")
if save:
    with open(save_dir + f'{model}/peaks/S2/bg_peaks_union.txt', 'w') as f:
        for item in bg_union_list:
            f.write(item + '\n')





