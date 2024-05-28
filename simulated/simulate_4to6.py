import os
import random
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from umap.umap_ import UMAP

from INSTINCT import *

from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1234)
random.seed(1234)

ori_scenario = 1
scenario = 4

# parameters
num_clusters = 5
clusters_name = ['0', '1', '2', '3', '4']
# num_spots = 2000
# num_cells = 9 * num_spots
# num_spots_list = [300, 350, 400, 450, 500]
# num_cells_list = [9 * i for i in num_spots_list]
spot_rows = 40
spot_columns = 50

# assign cell type
coords_list_4 = []
coords_list_5 = []
coords_list_6 = []
coords0 = [[x, y] for x in range(3*20, 3*40) for y in range(3*15, 3*30)]  # cluster 0
coords1 = [[x, y] for x in range(3*0, 3*10) for y in range(3*0, 3*40)] + \
          [[x, y] for x in range(3*10, 3*20) for y in range(3*0, 3*10)]  # cluster 1
coords2 = [[x, y] for x in range(3*40, 3*50) for y in range(3*20, 3*40)] + \
          [[x, y] for x in range(3*10, 3*40) for y in range(3*35, 3*40)]  # cluster 2
coords3 = [[x, y] for x in range(3*20, 3*50) for y in range(3*0, 3*10)] + \
          [[x, y] for x in range(3*40, 3*50) for y in range(3*10, 3*20)]  # cluster 3
coords4 = [[x, y] for x in range(3*10, 3*20) for y in range(3*10, 3*35)] + \
          [[x, y] for x in range(3*20, 3*40) for y in range(3*10, 3*15)] + \
          [[x, y] for x in range(3*20, 3*40) for y in range(3*30, 3*35)]  # cluster 4

coords_list_4.append([[x, y] for x in range(3*0, 3*10) for y in range(3*0, 3*40)])
coords_list_4.append([[x, y] for x in range(3*0, 3*5) for y in range(3*0, 3*40)] +
                     [[x, y] for x in range(3*45, 3*50) for y in range(3*0, 3*40)])
coords_list_4.append([[x, y] for x in range(3*40, 3*50) for y in range(3*0, 3*40)])

coords_list_5.append([])
coords_list_5.append([])
coords_list_5.append(coords3)

coords_list_6.append([[x, y] for x in range(3*0, 3*10) for y in range(3*0, 3*40)] +
                     [[x, y] for x in range(3*10, 3*20) for y in range(3*0, 3*5)] +
                     [[x, y] for x in range(3*10, 3*15) for y in range(3*5, 3*10)])
coords_list_6.append(coords1)
coords_list_6.append(coords1)

pca = PCA(n_components=100, random_state=1234)

reducer = UMAP(n_neighbors=30, n_components=2, metric="correlation", n_epochs=None, learning_rate=1.0,
               min_dist=0.3, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=1, repulsion_strength=1,
               negative_sample_rate=5, a=None, b=None, random_state=1234, metric_kwds=None,
               angular_rp_forest=False, verbose=True)

mode_list = [
    'Tech_0_0_Bio_0_0.5',
    'Tech_0_0.1_Bio_0_0',
    'Tech_0_0.1_Bio_0_0.5',
             ]

if scenario == 4:
    coords_list = coords_list_4
elif scenario == 5:
    coords_list = coords_list_5
elif scenario == 6:
    coords_list = coords_list_6

for j, mode in enumerate(mode_list):

    read_dir = f'../../data/simulated/{mode}/'
    save_dir = f"../../results/simulated/scenario_{scenario}/{mode}/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    adata = ad.read_h5ad(read_dir + f'{ori_scenario}_spot_level_slice_{mode}.h5ad')

    coords_to_remove = coords_list[j]

    spatial_coords = adata.obsm['spatial']

    rows_to_remove = []
    for i, coord in enumerate(spatial_coords):
        if list(coord) in coords_to_remove:
            rows_to_remove.append(i)

    adata = adata[~np.isin(np.arange(len(adata)), rows_to_remove)]

    adata.write_h5ad(f'../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad')

    adata = ad.read_h5ad(f'../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad')

    sp_cmap = {clusters_name[i]: sns.color_palette()[i] for i in range(num_clusters)}

    # plot slices
    sp_colors = list(adata.obs['real_spot_clusters'].astype('str').map(sp_cmap))

    plt.figure()
    plt.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], color=sp_colors, marker='o', s=25)
    plt.title(f'Spot Level Slice {mode}', fontsize=16)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.xlim(-5, 155)
    plt.ylim(-5, 125)
    plt.gca().invert_yaxis()
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=sp_cmap[f'{i}'], label=f"{i}")
        for i in range(num_clusters)
    ]
    plt.legend(handles=legend_handles, fontsize=8, title='Spot-types',
               title_fontsize=10, bbox_to_anchor=(1, 1), loc=1)
    save_path = save_dir + f'sp_slice_{mode}.pdf'
    plt.savefig(save_path)
    # plt.show()

    # single slice umap
    print("Start preprocessing ...")

    preprocess_CAS([adata.copy()], adata)

    print("Done !")

    sp_cmap = {clusters_name[i]: sns.color_palette()[i] for i in range(num_clusters)}

    print('Applying PCA to reduce the dimension to 100 ...')

    sp_X_pca = pca.fit_transform(adata.X.toarray())

    print('Done ! ')

    sp_embedding = reducer.fit_transform(sp_X_pca)
    adata.obsm["X_umap"] = sp_embedding
    pd.DataFrame(adata.obsm["X_umap"]).to_csv(save_dir + f"sp_slice_{mode}_umap.csv")

    sp_embedding = sc.read_csv(save_dir + f"sp_slice_{mode}_umap.csv").X[1:]
    adata.obsm["X_umap"] = sp_embedding

    n_spots = sp_embedding.shape[0]
    size = 10000 / n_spots

    order = np.arange(n_spots)

    sp_colors = list(adata.obs['real_spot_clusters'].astype('str').map(sp_cmap))

    plt.figure(figsize=(5, 5))
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=sp_colors)
    plt.title(f"Spot Level Slice {mode}", fontsize=16)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=sp_cmap[f'{i}'], label=f"{i}")
        for i in range(num_clusters)
    ]
    plt.legend(handles=legend_handles, fontsize=8, title='Spot-types',
               title_fontsize=10, bbox_to_anchor=(1, 1), loc=1)

    save_path = save_dir + f"sp_slice_{mode}_umap.pdf"
    plt.savefig(save_path)
    # plt.show()


mode_list = [
    'Tech_0_0_Bio_0_0.5',
    'Tech_0_0.1_Bio_0_0',
    'Tech_0_0.1_Bio_0_0.5',
             ]

slice_index_list = list(range(len(mode_list)))

name_concat = mode_list[0]
for mode in mode_list[1:]:
    name_concat = name_concat + '_' + mode

save_dir = f'../../results/simulated/scenario_{scenario}/T_' + name_concat + '/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# preprocessing
print("Start preprocessing ...")

sp_adata_list = [ad.read_h5ad(f"../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad") for mode in mode_list]
for j in range(len(sp_adata_list)):
    sp_adata_list[j].obs_names = [x + '_' + mode_list[j] for x in sp_adata_list[j].obs_names]
sp_adata_concat = ad.concat(sp_adata_list, label='slice_index', keys=slice_index_list)
preprocess_CAS(sp_adata_list, sp_adata_concat)
sp_adata_concat.write_h5ad(save_dir + f"preprocessed_concat.h5ad")
for i in range(len(mode_list)):
    sp_adata_list[i].write_h5ad(save_dir + f"filtered_spot_level_slice_{mode_list[i]}.h5ad")
# sp_adata_concat = ad.read_h5ad(save_dir + "preprocessed_concat.h5ad")

print("Done !")

print('Applying PCA to reduce the dimension to 100 ...')

sp_X_pca = pca.fit_transform(sp_adata_concat.X.toarray())
np.save(save_dir + 'input_matrix.npy', sp_X_pca)
# sp_X_pca = np.load(save_dir + 'input_matrix.npy')

print('Done ! ')

sp_embedding = reducer.fit_transform(sp_X_pca)
sp_adata_concat.obsm["X_umap"] = sp_embedding
pd.DataFrame(sp_adata_concat.obsm["X_umap"]).to_csv(save_dir + f"sp_umap_{name_concat}.csv")

sp_embedding = sc.read_csv(save_dir + f"sp_umap_{name_concat}.csv").X[1:]
sp_adata_concat.obsm["X_umap"] = sp_embedding

n_spots = sp_embedding.shape[0]
size = 10000 / n_spots

order = np.arange(n_spots)

color_list = [[0.2298057, 0.29871797, 0.75368315],
              [0.70567316, 0.01555616, 0.15023281],
              [0.2298057, 0.70567316, 0.15023281]]
slice_cmap = {f'{i}': color_list[i] for i in range(len(mode_list))}
colors = list(sp_adata_concat.obs['slice_index'].astype('str').map(slice_cmap))

plt.figure(figsize=(5, 5))
plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
plt.title('Spot Level Slices', fontsize=16)
plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                labelleft=False, labelbottom=False, grid_alpha=0)
legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=slice_cmap[f'{i}'], label=f"{i}")
        for i in range(len(mode_list))
    ]
plt.legend(handles=legend_handles,
           fontsize=8, title='Slices', title_fontsize=10, bbox_to_anchor=(1, 1), loc=1)

save_path = save_dir + f"sp_slices_umap_{name_concat}.pdf"
plt.savefig(save_path)

sp_cmap = {clusters_name[i]: sns.color_palette()[i] for i in range(num_clusters)}
colors = list(sp_adata_concat.obs['real_spot_clusters'].astype('str').map(sp_cmap))

plt.figure(figsize=(5, 5))
plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
plt.title('Spot Level Slices', fontsize=16)
plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                labelleft=False, labelbottom=False, grid_alpha=0)
legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=sp_cmap[f'{i}'], label=f"{i}")
        for i in range(num_clusters)
    ]
plt.legend(handles=legend_handles,
           fontsize=8, title='Spot-types', title_fontsize=10, bbox_to_anchor=(1, 1), loc=1)

save_path = save_dir + f"sp_clusters_umap_{name_concat}.pdf"
plt.savefig(save_path)
# plt.show()



