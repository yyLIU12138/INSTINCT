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
from scipy.sparse import csr_matrix, vstack

from INSTINCT import *

from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1234)
random.seed(1234)

scenario = 1

# parameters
num_clusters = 5
clusters_name = ['0', '1', '2', '3', '4']
# num_spots = 2000
# num_cells = 9 * num_spots
# num_spots_list = [300, 350, 400, 450, 500]
# num_cells_list = [9 * i for i in num_spots_list]
spot_rows = 40
spot_columns = 50
cell_type_template = np.zeros((3 * spot_columns, 3 * spot_rows), dtype=int)

main_type_percentage_list = [0.8, 0.7, 0.6, 0.8]
main_type_percentage = main_type_percentage_list[scenario-1]

# assign cell type
coords = [[x, y] for x in range(3*20, 3*40) for y in range(3*15, 3*30)]  # cluster 0
cell_type_template[tuple(zip(*coords))] = 0
coords = [[x, y] for x in range(3*0, 3*10) for y in range(3*0, 3*40)] + \
         [[x, y] for x in range(3*10, 3*20) for y in range(3*0, 3*10)]  # cluster 1
cell_type_template[tuple(zip(*coords))] = 1
coords = [[x, y] for x in range(3*40, 3*50) for y in range(3*20, 3*40)] + \
         [[x, y] for x in range(3*10, 3*40) for y in range(3*35, 3*40)]  # cluster 2
cell_type_template[tuple(zip(*coords))] = 2
coords = [[x, y] for x in range(3*20, 3*50) for y in range(3*0, 3*10)] + \
         [[x, y] for x in range(3*40, 3*50) for y in range(3*10, 3*20)]  # cluster 3
cell_type_template[tuple(zip(*coords))] = 3
coords = [[x, y] for x in range(3*10, 3*20) for y in range(3*10, 3*35)] + \
         [[x, y] for x in range(3*20, 3*40) for y in range(3*10, 3*15)] + \
         [[x, y] for x in range(3*20, 3*40) for y in range(3*30, 3*35)]  # cluster 4
cell_type_template[tuple(zip(*coords))] = 4

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

for mode in mode_list:

    print(mode)

    data_dir = f'../../data/simulated/{mode}/sc_simulated_{mode}.h5ad'
    save_dir = f"../../results/simulated/scenario_{scenario}/{mode}/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # read single cell dataset
    sc_dataset = ad.read_h5ad(data_dir)

    sc_type = sc_dataset.obs['celltype']
    indices_by_celltype = [sc_type.index[sc_type == i].tolist() for i in clusters_name]

    print('Start Preparing Cell Level Slice ...')

    sc_slice_X = []
    sc_slice_types = []
    sc_slice_coords = []

    for x in range(3 * spot_columns):

        for y in range(3 * spot_rows):

            # set percentages for cell types
            pre_assigned_type = cell_type_template[x][y]
            cell_type_percentages = [(1 - main_type_percentage) / (num_clusters - 1) for i in range(num_clusters)]
            cell_type_percentages[pre_assigned_type] = main_type_percentage

            # randomly select the cell type based on the set percentages
            assigned_type = random.choices([i for i in range(num_clusters)], cell_type_percentages)[0]

            # randomly select the cell based on the selected type
            assigned_cell_idx = random.choices(indices_by_celltype[assigned_type])[0]

            sc_slice_X.append(sc_dataset.X[int(assigned_cell_idx)])
            sc_slice_types.append(sc_dataset.obs['celltype'][assigned_cell_idx])
            sc_slice_coords.append([x, y])

    sc_slice_X = vstack(sc_slice_X)
    sc_slice_types = np.array(sc_slice_types)
    sc_slice_coords = np.array(sc_slice_coords)

    print('Done !')

    print('Start Preparing Spot Level Slice ...')

    sp_slice_X = []
    sp_slice_types = []
    sp_slice_coords = []

    for x in range(spot_columns):

        for y in range(spot_rows):

            # cell selection
            indices = []
            target_coords = [[x * 3 + m, y * 3 + n] for m in range(3) for n in range(3)]
            for coords in target_coords:
                idx = np.where((sc_slice_coords == coords).all(axis=1))[0][0]
                indices.append(idx)

            spot_features = csr_matrix(np.sum(sc_slice_X[indices], axis=0))
            unique_elements, counts = np.unique(sc_slice_types[indices], return_counts=True)
            spot_type = unique_elements[np.argmax(counts)]

            sp_slice_X.append(spot_features)
            sp_slice_types.append(spot_type)
            sp_slice_coords.append([x * 3 + 1, y * 3 + 1])

    sp_slice_X = vstack(sp_slice_X)
    sp_slice_types = np.array(sp_slice_types)
    sp_slice_coords = np.array(sp_slice_coords)

    print('Done !')

    print(f'Constructing Anndata Slice ...')

    sc_adata = ad.AnnData(sc_slice_X)
    sc_adata.var_names = [f'Feature_{i}' for i in range(sc_adata.X.shape[1])]
    sc_adata.obs_names = [f'Cell_{i}' for i in range(sc_adata.X.shape[0])]
    sc_adata.obs['real_cell_types'] = sc_slice_types
    sc_adata.obsm['spatial'] = sc_slice_coords

    sp_adata = ad.AnnData(sp_slice_X)
    sp_adata.var_names = [f'Feature_{i}' for i in range(sp_adata.X.shape[1])]
    sp_adata.obs_names = [f'Spot_{i}' for i in range(sp_adata.X.shape[0])]
    sp_adata.obs['real_spot_clusters'] = sp_slice_types
    sp_adata.obsm['spatial'] = sp_slice_coords

    print('Done !')

    # save simulated slices
    sc_adata.write_h5ad(f'../../data/simulated/{mode}/{scenario}_cell_level_slice_{mode}.h5ad')
    sp_adata.write_h5ad(f'../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad')

    # load simulated slices
    sc_adata = ad.read_h5ad(f'../../data/simulated/{mode}/{scenario}_cell_level_slice_{mode}.h5ad')
    sp_adata = ad.read_h5ad(f'../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad')

    sc_cmap = {clusters_name[i]: sns.color_palette()[i] for i in range(num_clusters)}
    sp_cmap = {clusters_name[i]: sns.color_palette()[i] for i in range(num_clusters)}

    # plot slices
    sc_colors = list(sc_adata.obs['real_cell_types'].astype('str').map(sc_cmap))

    plt.figure()
    plt.scatter(sc_adata.obsm['spatial'][:, 0], sc_adata.obsm['spatial'][:, 1], color=sc_colors, marker='o', s=1)
    plt.title(f'Cell Level Slice {mode}', fontsize=16)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.gca().invert_yaxis()
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=sc_cmap[f'{i}'], label=f"{i}")
        for i in range(num_clusters)
    ]
    plt.legend(handles=legend_handles, fontsize=8, title='Cell-types',
               title_fontsize=10, bbox_to_anchor=(1, 1), loc=1)
    save_path = save_dir + f'sc_slice_{mode}.pdf'
    plt.savefig(save_path)

    sp_colors = list(sp_adata.obs['real_spot_clusters'].astype('str').map(sp_cmap))

    plt.figure()
    plt.scatter(sp_adata.obsm['spatial'][:, 0], sp_adata.obsm['spatial'][:, 1], color=sp_colors, marker='o', s=25)
    plt.title(f'Spot Level Slice {mode}', fontsize=16)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
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

    # preprocess_CAS([sc_adata.copy()], sc_adata)
    preprocess_CAS([sp_adata.copy()], sp_adata)

    print("Done !")

    sc_cmap = {clusters_name[i]: sns.color_palette()[i] for i in range(num_clusters)}
    sp_cmap = {clusters_name[i]: sns.color_palette()[i] for i in range(num_clusters)}

    # print('Applying PCA to reduce the dimension to 100 ...')
    #
    # sc_X_pca = pca.fit_transform(sc_adata.X.toarray())
    #
    # print('Done ! ')
    #
    # # embedding
    # sc_embedding = reducer.fit_transform(sc_X_pca)
    # sc_adata.obsm["X_umap"] = sc_embedding
    # pd.DataFrame(sc_adata.obsm["X_umap"]).to_csv(save_dir + f"sc_slice_{mode}_umap.csv")
    #
    # # plot umaps
    # sc_embedding = sc.read_csv(save_dir + f"sc_slice_{mode}_umap.csv").X[1:]
    # sc_adata.obsm["X_umap"] = sc_embedding
    #
    # n_spots = sc_embedding.shape[0]
    # size = 10000 / n_spots
    #
    # order = np.arange(n_spots)
    #
    # sc_colors = list(sc_adata.obs['real_cell_types'].astype('str').map(sc_cmap))
    #
    # plt.figure(figsize=(5, 5))
    # plt.scatter(sc_embedding[order, 0], sc_embedding[order, 1], s=size, c=sc_colors)
    # plt.title(f"Cell Level Slice {mode}", fontsize=16)
    # plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
    #                 labelleft=False, labelbottom=False, grid_alpha=0)
    # plt.legend(handles=[mpatches.Patch(color=sc_cmap[clusters_name[i]], label=clusters_name[i])
    #                     for i in range(num_clusters)], fontsize=8, title='Types',
    #            title_fontsize=10, bbox_to_anchor=(1, 1), loc=1)
    #
    # save_path = save_dir + f"sc_slice_{mode}_umap.png"
    # plt.savefig(save_path)

    print('Applying PCA to reduce the dimension to 100 ...')

    sp_X_pca = pca.fit_transform(sp_adata.X.toarray())

    print('Done ! ')

    sp_embedding = reducer.fit_transform(sp_X_pca)
    sp_adata.obsm["X_umap"] = sp_embedding
    pd.DataFrame(sp_adata.obsm["X_umap"]).to_csv(save_dir + f"sp_slice_{mode}_umap.csv")

    sp_embedding = sc.read_csv(save_dir + f"sp_slice_{mode}_umap.csv").X[1:]
    sp_adata.obsm["X_umap"] = sp_embedding

    n_spots = sp_embedding.shape[0]
    size = 10000 / n_spots

    order = np.arange(n_spots)

    sp_colors = list(sp_adata.obs['real_spot_clusters'].astype('str').map(sp_cmap))

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

    del sc_dataset, sc_slice_types, sc_slice_coords, sc_slice_X, sp_slice_types, sp_slice_coords, sp_slice_X

# multi-slices umap
mode_list = [
    'Tech_0_0_Bio_0_0.5',
    'Tech_0_0.1_Bio_0_0',
    'Tech_0_0.1_Bio_0_0.5',
             ]

slice_index_list = list(range(len(mode_list)))

name_concat = mode_list[0]
for mode in mode_list[1:]:
    name_concat = name_concat + '_' + mode

if len(mode_list) == 2:
    save_dir = f'../../results/simulated/scenario_{scenario}/D_' + name_concat + '/'
elif len(mode_list) == 3:
    save_dir = f'../../results/simulated/scenario_{scenario}/T_' + name_concat + '/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# preprocessing
print("Start preprocessing ...")

# sc_adata_list = [ad.read_h5ad(f"../../data/simulated/{mode}/{scenario}_cell_level_slice_{mode}.h5ad") for mode in mode_list]
# for j in range(len(sc_adata_list)):
#     sc_adata_list[j].obs_names = [x + '_' + mode_list[j] for x in sc_adata_list[j].obs_names]
# sc_adata_concat = ad.concat(sc_adata_list, label='slice_index', keys=slice_index_list)
# sc_adata_concat.obs_names_make_unique()
# preprocess_CAS(sc_adata_list, sc_adata_concat)

sp_adata_list = [ad.read_h5ad(f"../../data/simulated/{mode}/{scenario}_spot_level_slice_{mode}.h5ad") for mode in mode_list]
for j in range(len(sp_adata_list)):
    sp_adata_list[j].obs_names = [x + '_' + mode_list[j] for x in sp_adata_list[j].obs_names]
sp_adata_concat = ad.concat(sp_adata_list, label='slice_index', keys=slice_index_list)
preprocess_CAS(sp_adata_list, sp_adata_concat)
sp_adata_concat.write_h5ad(save_dir + f"preprocessed_concat.h5ad")
for i in range(len(mode_list)):
    sp_adata_list[i].write_h5ad(save_dir + f"filtered_spot_level_slice_{mode_list[i]}.h5ad")
# sp_adata_concat = ad.read_h5ad(save_dir + "preprocessed_concat.h5ad")
print(sp_adata_concat.shape)

print("Done !")

# print('Applying PCA to reduce the dimension to 100 ...')
#
# sc_X_pca = pca.fit_transform(sc_adata_concat.X.toarray())
#
# print('Done ! ')
#
# sc_embedding = reducer.fit_transform(sc_X_pca)
# sc_adata_concat.obsm["X_umap"] = sc_embedding
# pd.DataFrame(sc_adata_concat.obsm["X_umap"]).to_csv(save_dir + f"sc_umap_{name_concat}.csv")
#
# sc_embedding = sc.read_csv(save_dir + f"sc_umap_{name_concat}.csv").X[1:]
# sc_adata_concat.obsm["X_umap"] = sc_embedding
#
# n_spots = sc_embedding.shape[0]
# size = 10000 / n_spots
#
# order = np.arange(n_spots)
#
# color_list = [[0.2298057, 0.29871797, 0.75368315],
#               [0.70567316, 0.01555616, 0.15023281],
#               [0.2298057, 0.70567316, 0.15023281]]
# cmap = {f'{i}': color_list[i] for i in range(len(mode_list))}
# colors = list(sc_adata_concat.obs['slice_index'].astype('str').map(cmap))
#
# plt.figure(figsize=(5, 5))
# plt.scatter(sc_embedding[order, 0], sc_embedding[order, 1], s=size, c=colors)
# plt.title('Cell Level Slices', fontsize=16)
# plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
#                 labelleft=False, labelbottom=False, grid_alpha=0)
# plt.legend(handles=[mpatches.Patch(color=cmap[f'{i}'], label=f"{i}") for i in range(len(mode_list))],
#            fontsize=8, title='Slices', title_fontsize=10, bbox_to_anchor=(1, 1), loc=1)
#
# save_path = save_dir + f"sc_umap_{name_concat}.png"
# plt.savefig(save_path)

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

