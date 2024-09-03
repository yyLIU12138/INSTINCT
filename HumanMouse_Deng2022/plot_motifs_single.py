import os
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


save = False
method = 'leiden'
file_format = 'png'
cmap = 'bwr'

data_dir = '../../data/spCASdata/HumanMouse_Deng2022/preprocessed/'
save_dir = '../../results/HumanMouse_Deng2022/'
slice_name_list = ['GSM5238385_ME11_50um', 'GSM5238386_ME13_50um', 'GSM5238387_ME13_50um_2']
label_list = ['GSM5238385', 'GSM5238386', 'GSM5238387']
slice_used = [0, 1, 2]
slice_name_list = [slice_name_list[i] for i in slice_used]
label_list = [label_list[i] for i in slice_used]
slice_index_list = list(range(len(slice_name_list)))

save_dir = f'../../results/HumanMouse_Deng2022/{slice_used}/'

motif_list = ['Gata1', 'Gata4', 'Sox3', 'Sox6', 'Sox17', 'Rfx1', 'Pou5f1::Sox2', 'Sox2', 'Pou2f3',]
save_name_list = ['Gata1', 'Gata4', 'Sox3', 'Sox6', 'Sox17', 'Rfx1', 'Pou5f1_Sox2', 'Sox2', 'Pou2f3',]

motif_scores_list = [pd.read_csv(save_dir + f'{label_list[idx]}/motif_enrichment_analysis/sorted_devs.csv',
                                 index_col=0, header=0) for idx in range(len(label_list))]

for k, motif in enumerate(motif_list):

    print(motif)

    if not os.path.exists(save_dir + f'intersect/motif_enrichment_analysis/{save_name_list[k]}/') and save:
        os.makedirs(save_dir + f'intersect/motif_enrichment_analysis/{save_name_list[k]}/')

    cas_list = [ad.read_h5ad(save_dir + f'{label_list[idx]}/selected_{slice_name_list[idx]}.h5ad')
                for idx in range(len(label_list))]

    for i in range(len(cas_list)):
        cas_list[i] = cas_list[i][cas_list[i].obs_names.isin(motif_scores_list[i].index)]
        cas_list[i].obs[motif] = motif_scores_list[i].loc[cas_list[i].obs_names, motif].values
        cas_list[i] = cas_list[i][~cas_list[i].obs[motif].isna()]

    if len(slice_name_list) == 2:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    elif len(slice_name_list) == 3:
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for i in range(len(slice_name_list)):
        scatter = axs[i].scatter(cas_list[i].obsm['spatial'][:, 0], cas_list[i].obsm['spatial'][:, 1],
                                 linewidth=1, s=40, marker=".", c=cas_list[i].obs[motif], cmap=cmap, alpha=0.9)
        axs[i].invert_yaxis()
        axs[i].set_title(f'{label_list[i]}', size=12)
        axs[i].axis('off')
        cbar = fig.colorbar(scatter, ax=axs[i], orientation='vertical')
        cbar.set_label(motif)
    plt.gcf().subplots_adjust(left=0.05, top=0.8, bottom=0.05, right=0.95)
    if save:
        save_path = save_dir + f'intersect/motif_enrichment_analysis/{save_name_list[k]}/{save_name_list[k]}_spatial.{file_format}'
        plt.savefig(save_path)
    plt.show()