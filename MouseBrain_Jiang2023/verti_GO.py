import os
import anndata as ad
import scanpy as sc
import pandas as pd
import gseapy as gp

import warnings
warnings.filterwarnings("ignore")

save = False
model = 'INSTINCT'

mode_list = ['E11_0', 'E13_5', 'E15_5', 'E18_5']
mode_index = 3
mode = mode_list[mode_index]

data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
# sc.settings.set_figure_params(dpi=300, facecolor="white")

save_dir = f'../../results/MouseBrain_Jiang2023/vertical/{mode}/'
if not os.path.exists(save_dir + f'{model}/DEGs/S1/'):
    os.makedirs(save_dir + f'{model}/DEGs/S1/')
if not os.path.exists(save_dir + f'{model}/DEGs/S2/'):
    os.makedirs(save_dir + f'{model}/DEGs/S2/')
slice_name_list = [f'{mode}-S1', f'{mode}-S2']

# read the filtered and annotated CAS data
cas_s1 = ad.read_h5ad(save_dir + f'filtered_merged_{slice_name_list[0]}_atac.h5ad')
cas_s2 = ad.read_h5ad(save_dir + f'{model}/annotated_{slice_name_list[1]}_atac.h5ad')
cas_list = [cas_s1, cas_s2]

# read the raw RNA data
rna_list = [ad.read_h5ad(data_dir + f'{sample}_expr.h5ad') for sample in slice_name_list]
for j in range(len(rna_list)):
    rna_list[j].obs_names = [x + '_' + slice_name_list[j] for x in rna_list[j].obs_names]

# filter and reorder spots in rna slices
obs_list = [obs_name for obs_name in cas_list[0].obs_names if obs_name in rna_list[0].obs_names]
cas_list[0] = cas_list[0][obs_list, :]
rna_list[0] = rna_list[0][obs_list, :]
# rename the obs in CAS s2
cas_list[1].obs_names = [obs_name.split('_')[0] + f'-1_{slice_name_list[1]}' for obs_name in cas_list[1].obs_names]
obs_list = [obs_name for obs_name in cas_list[1].obs_names if obs_name in rna_list[1].obs_names]
cas_list[1] = cas_list[1][obs_list, :]
rna_list[1] = rna_list[1][obs_list, :]

# transfer the annotated labels from cas s2 slice to rna s2 slice
rna_list[1].obs['predicted_labels'] = cas_list[1].obs['predicted_labels'].copy()

# [cas_s1, cas_s2] = cas_list
[rna_s1, rna_s2] = rna_list

# normalization
sc.pp.normalize_total(rna_s1, target_sum=1e4)
sc.pp.log1p(rna_s1)
rna_s1.var_names_make_unique()
sc.pp.normalize_total(rna_s2, target_sum=1e4)
sc.pp.log1p(rna_s2)
rna_s2.var_names_make_unique()


# find DEGs
group_list = list(set(rna_s2.obs['predicted_labels']))
print(group_list)
# predicted_labels = cas_s2.obs['predicted_labels']
# label_counts = predicted_labels.value_counts()
# labels_with_30plus_points = label_counts[label_counts > 30].index.tolist()
# print("Labels with more than 30 spots:")
# print(labels_with_30plus_points)

print('S2')
rna_s2_degs_list = []
sc.tl.rank_genes_groups(rna_s2, "predicted_labels", groups=group_list, method='wilcoxon')
rna_s2_genes = pd.DataFrame(rna_s2.uns["rank_genes_groups"]["names"])
rna_s2_logfoldchanges = pd.DataFrame(rna_s2.uns["rank_genes_groups"]["logfoldchanges"])
rna_s2_pvals_adj = pd.DataFrame(rna_s2.uns["rank_genes_groups"]["pvals_adj"])
for col in list(rna_s2_genes.columns):
    s2_genes = rna_s2_genes[col].tolist()
    s2_logfoldchanges = rna_s2_logfoldchanges[col].tolist()
    s2_pvals_adj = rna_s2_pvals_adj[col].tolist()
    s2_degs_list = [s2_genes[i] for i in range(len(s2_genes)) if s2_logfoldchanges[i] > 0.2 and s2_pvals_adj[i] < 0.05]
    rna_s2_degs_list.append(s2_degs_list)
    # save DEGs
    if save:
        if not s2_degs_list:
            with open(save_dir + f'{model}/DEGs/S2/{col}_DEGs.txt', 'w') as f:
                pass
        else:
            with open(save_dir + f'{model}/DEGs/S2/{col}_DEGs.txt', 'w') as f:
                for item in s2_degs_list:
                    f.write(item + '\n')
    print(f"Label: {col}, Number of DEGs: {len(s2_degs_list)}")

print('\nS1')
rna_s1_degs_list = []
sc.tl.rank_genes_groups(rna_s1, "Annotation_for_Combined", groups=group_list, method='wilcoxon')
rna_s1_genes = pd.DataFrame(rna_s1.uns["rank_genes_groups"]["names"])
rna_s1_logfoldchanges = pd.DataFrame(rna_s1.uns["rank_genes_groups"]["logfoldchanges"])
rna_s1_pvals_adj = pd.DataFrame(rna_s1.uns["rank_genes_groups"]["pvals_adj"])
for col in list(rna_s1_genes.columns):
    s1_genes = rna_s1_genes[col].tolist()
    s1_logfoldchanges = rna_s1_logfoldchanges[col].tolist()
    s1_pvals_adj = rna_s1_pvals_adj[col].tolist()
    s1_degs_list = [s1_genes[i] for i in range(len(s1_genes)) if s1_logfoldchanges[i] > 0.2 and s1_pvals_adj[i] < 0.05]
    rna_s1_degs_list.append(s1_degs_list)
    # save DEGs
    if save:
        if not s1_degs_list:
            with open(save_dir + f'{model}/DEGs/S1/{col}_DEGs.txt', 'w') as f:
                pass
        else:
            with open(save_dir + f'{model}/DEGs/S1/{col}_DEGs.txt', 'w') as f:
                for item in s1_degs_list:
                    f.write(item + '\n')
    print(f"Label: {col}, Number of DEGs: {len(s1_degs_list)}")


# GO analysis
# names = gp.get_library_name(organism='mouse')
for i, col in enumerate(list(rna_s2_genes.columns)):
    if not rna_s2_degs_list[i]:
        print(f'{col} Skipped')
        continue
    print(col)
    enr = gp.enrichr(gene_list=rna_s2_degs_list[i], gene_sets='GO_Biological_Process_2021', organism='mouse',
                     outdir=save_dir + f'{model}/GO_analysis/{col}/')


# compute overlap
print('\nCompute overlap')
for i, col in enumerate(list(rna_s2_genes.columns)):
    genes_s1 = set(rna_s1_degs_list[i])
    genes_s2 = set(rna_s2_degs_list[i])
    if not genes_s1:
        continue
    overlap_count = len(genes_s1.intersection(genes_s2))
    print(f"Label: {col}, Overlap Count: {overlap_count}, Overlap Percentage: {(overlap_count / len(list(genes_s1)) * 100): .2f}%")

