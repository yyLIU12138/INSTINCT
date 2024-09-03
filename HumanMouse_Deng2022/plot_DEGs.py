import os
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


save = False
method = 'leiden'
file_format = 'png'
cmap = 'hot'

data_dir = '../../data/spCASdata/HumanMouse_Deng2022/preprocessed/'
save_dir = '../../results/HumanMouse_Deng2022/'
slice_name_list = ['GSM5238385_ME11_50um', 'GSM5238386_ME13_50um', 'GSM5238387_ME13_50um_2']
label_list = ['GSM5238385', 'GSM5238386', 'GSM5238387']
slice_used = [0, 1, 2]
slice_name_list = [slice_name_list[i] for i in slice_used]
label_list = [label_list[i] for i in slice_used]
slice_index_list = list(range(len(slice_name_list)))

save_dir = f'../../results/HumanMouse_Deng2022/{slice_used}/'

marker_list = ['Il1rapl1', 'Ptprd', 'Cntn4', 'C130071C03Rik', 'Sox1', 'Sptb',
               'Slc30a10', 'Snrpn', 'Col2a1', 'Mir140', 'Fgfr2', 'Notch1']

rna_list = [ad.read_h5ad(save_dir + f'{label_list[i]}/clustered_{slice_name_list[i]}_rna.h5ad') for i in range(len(slice_name_list))]

# normalization
for i in range(len(rna_list)):
    sc.pp.normalize_total(rna_list[i], target_sum=1e4)
    sc.pp.log1p(rna_list[i])

for k, marker in enumerate(marker_list):

    if not os.path.exists(save_dir + f'marker_genes/{marker}/') and save:
        os.makedirs(save_dir + f'marker_genes/{marker}/')

    if len(slice_name_list) == 2:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    elif len(slice_name_list) == 3:
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for i in range(len(rna_list)):
        scatter = axs[i].scatter(rna_list[i].obsm['spatial'][:, 0], rna_list[i].obsm['spatial'][:, 1], linewidth=1, s=40,
                       marker=".", c=rna_list[i][:, marker].X.toarray(), cmap=cmap,  alpha=0.9)
        axs[i].invert_yaxis()
        axs[i].set_title(f'{label_list[i]} ({marker})', size=12)
        axs[i].axis('off')
        cbar = fig.colorbar(scatter, ax=axs[i], orientation='vertical')
        cbar.set_label(marker)
    plt.gcf().subplots_adjust(left=0.05, top=0.8, bottom=0.05, right=0.90)
    if save:
        save_path = save_dir + f'marker_genes/{marker}/{marker}_spatial.{file_format}'
        plt.savefig(save_path)
    plt.show()
