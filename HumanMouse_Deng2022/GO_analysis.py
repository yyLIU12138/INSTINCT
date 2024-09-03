import os
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
import gseapy as gp

import warnings
warnings.filterwarnings("ignore")

save = False
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

# read the gene score data
filter_list = [['4', '5', '12'], ['5', '7', '12', '14'], ['5']]
for i in range(len(slice_name_list)):

    if not os.path.exists(save_dir + f'{label_list[i]}/DEGs'):
        os.makedirs(save_dir + f'{label_list[i]}/DEGs')

    sample_adata = ad.read_h5ad(save_dir + f'{label_list[i]}/clustered_{slice_name_list[i]}_rna.h5ad')

    # filter
    category_counts = sample_adata.obs[method].value_counts()
    valid_categories = category_counts[category_counts >= 10].index
    sample_adata = sample_adata[sample_adata.obs[method].isin(valid_categories), :]

    sc.pp.filter_genes(sample_adata, min_cells=1)
    sc.pp.filter_cells(sample_adata, min_genes=1)
    print(sample_adata.shape)

    # sc.pp.highly_variable_genes(sample_adata, flavor="seurat_v3", n_top_genes=5000)
    # rna_concat = sample_adata[:, sample_adata.var['highly_variable']]
    sc.pp.log1p(sample_adata)

    # find DEGs
    group_list = list(set(sample_adata.obs[method]))
    print(group_list)

    rna_degs_list = []
    sc.tl.rank_genes_groups(sample_adata, method, groups=group_list, method='wilcoxon')
    rna_genes = pd.DataFrame(sample_adata.uns["rank_genes_groups"]["names"])
    rna_logfoldchanges = pd.DataFrame(sample_adata.uns["rank_genes_groups"]["logfoldchanges"])
    rna_pvals_adj = pd.DataFrame(sample_adata.uns["rank_genes_groups"]["pvals_adj"])
    for col in list(rna_genes.columns):
        genes = rna_genes[col].tolist()
        logfoldchanges = rna_logfoldchanges[col].tolist()
        pvals_adj = rna_pvals_adj[col].tolist()

        degs_list = [genes[k] for k in range(len(genes)) if logfoldchanges[k] > 0.2 and pvals_adj[k] < 0.05]

        rna_degs_list.append(degs_list)

        # save DEGs
        if save:
            if not degs_list:
                with open(save_dir + f'{label_list[i]}/DEGs/{col}_DEGs.txt', 'w') as f:
                    pass
            else:
                with open(save_dir + f'{label_list[i]}/DEGs/{col}_DEGs.txt', 'w') as f:
                    for item in degs_list:
                        f.write(item + '\n')
        print(f"Label: {col}, Number of DEGs: {len(degs_list)}")

    if save:
        # GO analysis
        # names = gp.get_library_name(organism='mouse')
        for j, col in enumerate(list(rna_genes.columns)):
            if not rna_degs_list[j] or col in filter_list[i]:
                print(f'{col} Skipped')
                continue
            print(col)
            enr = gp.enrichr(gene_list=rna_degs_list[j], gene_sets='GO_Biological_Process_2021', organism='mouse',
                             outdir=save_dir + f'{label_list[i]}/GO_analysis/{col}/')


rna_list = [ad.read_h5ad(save_dir + f'{label_list[i]}/clustered_{sample}_rna.h5ad') for i, sample in enumerate(slice_name_list)]
rna_concat = ad.concat(rna_list, label='slice_name', keys=slice_name_list)
rna_concat.var_names_make_unique()

filter_list = ['5', '8', '12', '14']
if not os.path.exists(save_dir + 'concat/DEGs/'):
    os.makedirs(save_dir + 'concat/DEGs/')

# normalization
sc.pp.filter_genes(rna_concat, min_cells=1)
sc.pp.filter_cells(rna_concat, min_genes=1)
print(rna_concat.shape)
# sc.pp.highly_variable_genes(rna_concat, flavor="seurat_v3", n_top_genes=5000)
# rna_concat = rna_concat[:, rna_concat.var['highly_variable']]
sc.pp.log1p(rna_concat)

# find DEGs
group_list = list(set(rna_concat.obs[method]))
print(group_list)

rna_degs_list = []
sc.tl.rank_genes_groups(rna_concat, method, groups=group_list, method='wilcoxon')
rna_genes = pd.DataFrame(rna_concat.uns["rank_genes_groups"]["names"])
rna_logfoldchanges = pd.DataFrame(rna_concat.uns["rank_genes_groups"]["logfoldchanges"])
rna_pvals_adj = pd.DataFrame(rna_concat.uns["rank_genes_groups"]["pvals_adj"])
for col in list(rna_genes.columns):
    genes = rna_genes[col].tolist()
    logfoldchanges = rna_logfoldchanges[col].tolist()
    pvals_adj = rna_pvals_adj[col].tolist()

    degs_list = [genes[i] for i in range(len(genes)) if logfoldchanges[i] > 0.2 and pvals_adj[i] < 0.05]
    rna_degs_list.append(degs_list)

    # save DEGs
    if save:
        if not degs_list:
            with open(save_dir + f'concat/DEGs/{col}_DEGs.txt', 'w') as f:
                pass
        else:
            with open(save_dir + f'concat/DEGs/{col}_DEGs.txt', 'w') as f:
                for item in degs_list:
                    f.write(item + '\n')
    print(f"Label: {col}, Number of DEGs: {len(degs_list)}")

if save:
    # GO analysis
    # names = gp.get_library_name(organism='mouse')
    for i, col in enumerate(list(rna_genes.columns)):
        if not rna_degs_list[i] or col in filter_list:
            print(f'{col} Skipped')
            continue
        print(col)
        enr = gp.enrichr(gene_list=rna_degs_list[i], gene_sets='GO_Biological_Process_2021', organism='mouse',
                         outdir=save_dir + f'concat/GO_analysis/{col}/')


