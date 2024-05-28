import os
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA

from INSTINCT import *

from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

save = False

# mouse embryo
data_dir = '../../data/spCASdata/MouseEmbryo_Llorens-Bobadilla2023/spATAC/'
save_dir = '../../results/MouseEmbryo_Llorens-Bobadilla2023/all/'

slice_name_list = ['E12_5-S1', 'E12_5-S2', 'E13_5-S1', 'E13_5-S2', 'E15_5-S1', 'E15_5-S2']

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cluster_list = ['Forebrain', 'Midbrain', 'Hindbrain', 'Periventricular', 'Meningeal_PNS_1', 'Meningeal_PNS_2',
                'Internal', 'Facial_bone', 'Muscle_heart', 'Limb', 'Liver']

label_list = ['Forebrain', 'Midbrain', 'Hindbrain', 'Periventricular', 'Meningeal/PNS_1', 'Meningeal/PNS_2',
              'Internal', 'Facial/bone', 'Muscle/heart', 'Limb', 'Liver']

color_list = ['royalblue', 'dodgerblue', 'deepskyblue', 'forestgreen', 'yellowgreen', 'y',
              'grey', 'crimson', 'deeppink', 'orchid', 'orange']

order_list = [1, 8, 2, 10, 6, 7, 3, 0, 9, 4, 5]

cluster_to_color_map = {cluster: color for cluster, color in zip(cluster_list, color_list)}
order_to_cluster_map = {order: cluster for order, cluster in zip(order_list, cluster_list)}

# load dataset
cas_list = [ad.read_h5ad(data_dir + sample + '.h5ad') for sample in slice_name_list]
for i in range(len(cas_list)):
    cas_list[i].obs_names = [x + '_' + slice_name_list[i] for x in cas_list[i].obs_names]

# concatenation
adata_concat = ad.concat(cas_list, label="slice_name", keys=slice_name_list)

# preprocess CAS data
# peaks are already merged and fragment counts are stored in the data matrices
print('Start preprocessing')
preprocess_CAS(cas_list, adata_concat, min_cells_rate=0.003)
print('Done!')
print(adata_concat)

adata_concat.write_h5ad(save_dir + f"preprocessed_concat.h5ad")
for i in range(len(slice_name_list)):
    cas_list[i].write_h5ad(save_dir + f"filtered_{slice_name_list[i]}.h5ad")

cas_list = [ad.read_h5ad(save_dir + f"filtered_{sample}.h5ad") for sample in slice_name_list]
adata_concat = ad.read_h5ad(save_dir + f"preprocessed_concat.h5ad")

print(f'Applying PCA to reduce the feature dimension to 100 ...')
pca = PCA(n_components=100, random_state=1234)
input_matrix = pca.fit_transform(adata_concat.X.toarray())
np.save(save_dir + f'input_matrix.npy', input_matrix)
print('Done !')

fig, axs = plt.subplots(2, 3, figsize=(10, 6))

for i in range(len(cas_list)):
    fig.suptitle(f'Mouse Embryo Dataset', fontsize=16)
    real_colors = list(cas_list[i].obs['clusters'].astype('str').map(cluster_to_color_map))
    if slice_name_list[i] == 'E12_5-S1' or slice_name_list[i] == 'E12_5-S2':
        size = 20
    else:
        size = 15
    axs[int(i % 2), int(i / 2)].scatter(cas_list[i].obsm['spatial'][:, 1], cas_list[i].obsm['spatial'][:, 0],
                                        linewidth=0.5, s=size, marker=".", color=real_colors, alpha=0.9)
    axs[int(i % 2), int(i / 2)].set_title(f'{slice_name_list[i]}', size=12)
    if slice_name_list[i] == 'E15_5-S1':
        axs[int(i % 2), int(i / 2)].invert_xaxis()
        axs[int(i % 2), int(i / 2)].invert_yaxis()
    axs[int(i % 2), int(i / 2)].axis('off')

legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=cluster_to_color_map[cluster],
               label=label) for cluster, label in zip(cluster_list, label_list)
    ]
axs[0, 2].legend(handles=legend_handles, fontsize=8, title='Spot-types',
                 title_fontsize=10, bbox_to_anchor=(1, 1))
plt.gcf().subplots_adjust(left=0.05, top=None, bottom=None, right=0.85)
if save:
    save_path = save_dir + f'mouse_embryo_dataset.pdf'
    plt.savefig(save_path, dpi=500)
plt.show()






