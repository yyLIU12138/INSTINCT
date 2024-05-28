import os
import anndata as ad
import scanpy as sc
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

CAS_samples = ["GSM6206884_HumanBrain_50um", "GSM6801813_ME13_50um", "GSM6204624_ME13_100barcodes_25um",
               "GSM6204623_MouseBrain_20um", "GSM6758284_MouseBrain_20um_repATAC", "GSM6758285_MouseBrain_20um_100barcodes_ATAC",
               "GSM6204621_MouseBrain_20um_H3K27ac", "GSM6704977_MouseBrain_20um_rep_H3K27ac", "GSM6704978_MouseBrain_20um_100barcodes_H3K27me3",
               "GSM6704979_MouseBrain_20um_100barcodes_H3K27ac", "GSM6704980_MouseBrain_20um_100barcodes_H3K4me3"]
RNA_samples = ["GSM6206885_HumanBrain_50um", "GSM6799937_ME13_50um", "GSM6204637_ME13_100barcodes_25um",
               "GSM6204636_MouseBrain_20um", "GSM6753041_MouseBrain_20um_repATAC", "GSM6753043_MouseBrain_20um_100barcodes_ATAC",
               "GSM6204635_MouseBrain_20um_H3K27ac", "GSM6753042_MouseBrain_20um_repH3K27ac", "GSM6753044_MouseBrain_20um_100barcodes_H3K27me3",
               "GSM6753045_MouseBrain_20um_100barcodes_H3K27ac", "GSM6753046_MouseBrain_20um_100barcodes_H3K4me3"]

save = False
slice_name_list = ["GSM6204623_MouseBrain_20um", "GSM6758284_MouseBrain_20um_repATAC", "GSM6758285_MouseBrain_20um_100barcodes_ATAC"]
rna_slice_name_list = ["GSM6204636_MouseBrain_20um", "GSM6753041_MouseBrain_20um_repATAC", "GSM6753043_MouseBrain_20um_100barcodes_ATAC"]
label_list = ["GSM6204623", "GSM6758284", "GSM6758285"]
slice_index_list = list(range(len(slice_name_list)))

data_dir = '../../data/spMOdata/EpiTran_HumanMouse_Zhang2023/preprocessed_from_fragments/'
save_dir = f'../../results/HumanMouse_Zhang2023/mb/'

method = 'leiden'

if not os.path.exists(save_dir + f'DEGs_concat/'):
    os.makedirs(save_dir + f'DEGs_concat/')
for i in range(len(slice_name_list)):
    if not os.path.exists(save_dir + f'DEGs_slice{i}/'):
        os.makedirs(save_dir + f'DEGs_slice{i}/')

# read the filtered and annotated CAS data
cas_list = [ad.read_h5ad(save_dir + f'clustered_{sample}.h5ad') for sample in slice_name_list]

# read the raw RNA data
rna_list = [ad.read_h5ad(data_dir + f'{sample}.h5ad') for sample in rna_slice_name_list]
for j in range(len(rna_list)):
    rna_list[j].obs_names = [x + '-1_' + slice_name_list[j] for x in rna_list[j].obs_names]
    print(rna_list[j].shape)

# filter and reorder spots in rna slices
for i in range(len(slice_name_list)):
    obs_list = [obs_name for obs_name in cas_list[i].obs_names if obs_name in rna_list[i].obs_names]
    cas_list[i] = cas_list[i][obs_list, :]
    rna_list[i] = rna_list[i][obs_list, :]
    # transfer the cluster labels from cas slices to rna slice
    rna_list[i].obs[method] = cas_list[i].obs[method].copy()
    print(rna_list[i].shape)

# normalization
for i in range(len(slice_name_list)):
    sc.pp.normalize_total(rna_list[i], target_sum=1e4)
    sc.pp.log1p(rna_list[i])
    rna_list[i].var_names_make_unique()

# concatenate the rna slices
rna_concat = ad.concat(rna_list, label='slice_name', keys=slice_name_list)
print(rna_concat.shape)

# find DEGs
group_list = list(set(rna_concat.obs[method]))
print(group_list)

rna_concat_degs_list = []
sc.tl.rank_genes_groups(rna_concat, method, groups=group_list, method='wilcoxon')
rna_concat_genes = pd.DataFrame(rna_concat.uns["rank_genes_groups"]["names"])
rna_concat_logfoldchanges = pd.DataFrame(rna_concat.uns["rank_genes_groups"]["logfoldchanges"])
rna_concat_pvals_adj = pd.DataFrame(rna_concat.uns["rank_genes_groups"]["pvals_adj"])
for col in list(rna_concat_genes.columns):
    concat_genes = rna_concat_genes[col].tolist()
    concat_logfoldchanges = rna_concat_logfoldchanges[col].tolist()
    concat_pvals_adj = rna_concat_pvals_adj[col].tolist()
    concat_degs_list = [concat_genes[i] for i in range(len(concat_genes)) if concat_logfoldchanges[i] > 1 and concat_pvals_adj[i] < 0.01]
    rna_concat_degs_list.append(concat_degs_list)
    # save DEGs
    if save:
        if not concat_degs_list:
            with open(save_dir + f'DEGs_concat/{col}_DEGs.txt', 'w') as f:
                pass
        else:
            with open(save_dir + f'DEGs_concat/{col}_DEGs.txt', 'w') as f:
                # f.write(','.join(concat_degs_list))
                for item in concat_degs_list:
                    f.write(item + '\n')
    print(f"Label: {col}, Number of DEGs: {len(concat_degs_list)}")


# find DEGs slice specific
for i in range(len(slice_name_list)):

    group_list = list(set(rna_list[i].obs[method]))
    print(group_list)

    rna_degs_list = []
    sc.tl.rank_genes_groups(rna_list[i], method, groups=group_list, method='wilcoxon')
    rna_genes = pd.DataFrame(rna_list[i].uns["rank_genes_groups"]["names"])
    rna_logfoldchanges = pd.DataFrame(rna_list[i].uns["rank_genes_groups"]["logfoldchanges"])
    rna_pvals_adj = pd.DataFrame(rna_list[i].uns["rank_genes_groups"]["pvals_adj"])
    for col in list(rna_genes.columns):
        genes = rna_genes[col].tolist()
        logfoldchanges = rna_logfoldchanges[col].tolist()
        pvals_adj = rna_pvals_adj[col].tolist()
        degs_list = [genes[j] for j in range(len(genes)) if logfoldchanges[j] > 1 and pvals_adj[j] < 0.05]
        rna_degs_list.append(degs_list)
        # save DEGs
        if save:
            if not degs_list:
                with open(save_dir + f'DEGs_slice{i}/{col}_DEGs.txt', 'w') as f:
                    pass
            else:
                with open(save_dir + f'DEGs_slice{i}/{col}_DEGs.txt', 'w') as f:
                    # f.write(','.join(degs_list))
                    for item in degs_list:
                        f.write(item + '\n')
        print(f"Label: {col}, Number of DEGs: {len(degs_list)}")
