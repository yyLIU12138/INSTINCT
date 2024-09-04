import numpy as np
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from umap.umap_ import UMAP
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from ..INSTINCT import preprocess_SRT
from .plot_utils import plot_DLPFC

from ..evaluation_utils import match_cluster_labels

from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

save_dir = '../../results/DLPFC_Maynard2021/'
save = False

# DLPFC
data_dir = '../../data/STdata/10xVisium/DLPFC_Maynard2021/'
sample_group_list = [['151507', '151508', '151509', '151510'],
                     ['151669', '151670', '151671', '151672'],
                     ['151673', '151674', '151675', '151676']]
cls_list = ['Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'WM']
num_clusters_list = [7, 5, 7]
samples = ['A', 'B', 'C']

file_format = 'pdf'

layer_to_color_map = {'Layer{0}'.format(i+1): sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]
matched_to_color_map = {i+1: sns.color_palette()[i] for i in range(7)}

reducer = UMAP(n_neighbors=30, n_components=2, metric="correlation", n_epochs=None, learning_rate=1.0,
               min_dist=0.3, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=1, repulsion_strength=1,
               negative_sample_rate=5, a=None, b=None, random_state=1234, metric_kwds=None,
               angular_rp_forest=False, verbose=False)

models = ['INSTINCT', 'INSTINCT_cas', 'SEDR', 'STAligner', 'GraphST']
colors_for_labels = ['darkviolet', 'violet', 'darkslategray', 'c', 'cyan']

for idx in range(len(sample_group_list)):

    slice_name_list = sample_group_list[idx]
    slice_index_list = list(range(len(slice_name_list)))

    rna_list = []
    for sample in slice_name_list:
        adata = sc.read_visium(path=data_dir + f'{sample}/',
                               count_file=sample + '_filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()

        # read the annotation
        Ann_df = pd.read_csv(data_dir + f'{sample}/meta_data.csv', sep=',', index_col=0)

        if not all(Ann_df.index.isin(adata.obs_names)):
            raise ValueError("Some rows in the annotation file are not present in the adata.obs_names")

        adata.obs['image_row'] = Ann_df.loc[adata.obs_names, 'imagerow']
        adata.obs['image_col'] = Ann_df.loc[adata.obs_names, 'imagecol']
        adata.obs['Manual_Annotation'] = Ann_df.loc[adata.obs_names, 'ManualAnnotation']

        adata.obs_names = [x + '_' + sample for x in adata.obs_names]
        rna_list.append(adata)

    # concatenation
    adata_concat = ad.concat(rna_list, label="slice_name", keys=slice_name_list)

    # preprocess SRT data
    _, adata_concat = preprocess_SRT(rna_list, adata_concat, n_top_genes=5000)
    adata_concat = adata_concat[~adata_concat.obs['Manual_Annotation'].isna(), :]

    pca = PCA(n_components=100, random_state=1234)
    raw_pca = pca.fit_transform(adata_concat.X.toarray())
    sp_embedding = reducer.fit_transform(raw_pca)

    n_spots = adata_concat.shape[0]
    size = 10000 / n_spots
    order = np.arange(n_spots)
    colors_for_slices = [[0.2298057, 0.29871797, 0.75368315],
                         [0.70567316, 0.01555616, 0.15023281],
                         [0.2298057, 0.70567316, 0.15023281],
                         [0.5830223, 0.59200322, 0.12993134]]
    slice_cmap = {slice_name_list[i]: colors_for_slices[i] for i in range(len(slice_name_list))}
    colors = list(adata_concat.obs['slice_name'].astype('str').map(slice_cmap))
    plt.figure(figsize=(5, 5))
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    legend_handles = [
            Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=slice_cmap[slice_name_list[i]], label=slice_name_list[i])
            for i in range(len(slice_name_list))
        ]
    plt.legend(handles=legend_handles, fontsize=8, title='Slices', title_fontsize=10,
               loc='upper left')
    plt.title(f'Slices (Raw / Sample {samples[idx]})', fontsize=16)
    if save:
        save_path = save_dir + f"/group{idx}_raw_slices_umap.pdf"
        plt.savefig(save_path)

    colors = list(adata_concat.obs['Manual_Annotation'].astype('str').map(layer_to_color_map))
    plt.figure(figsize=(5, 5))
    plt.scatter(sp_embedding[order, 0], sp_embedding[order, 1], s=size, c=colors)
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False, grid_alpha=0)
    plt.title(f'Annotated Spot-types (Raw / Sample {samples[idx]})', fontsize=16)
    if save:
        save_path = save_dir + f"/group{idx}_raw_annotated_clusters_umap.pdf"
        plt.savefig(save_path)
    plt.show()

    for j, model in enumerate(models):

        rna_list = []
        for sample in slice_name_list:
            adata = sc.read_visium(path=data_dir + f'{sample}/',
                                   count_file=sample + '_filtered_feature_bc_matrix.h5')
            adata.var_names_make_unique()

            # read the annotation
            Ann_df = pd.read_csv(data_dir + f'{sample}/meta_data.csv', sep=',', index_col=0)

            if not all(Ann_df.index.isin(adata.obs_names)):
                raise ValueError("Some rows in the annotation file are not present in the adata.obs_names")

            adata.obs['image_row'] = Ann_df.loc[adata.obs_names, 'imagerow']
            adata.obs['image_col'] = Ann_df.loc[adata.obs_names, 'imagecol']
            adata.obs['Manual_Annotation'] = Ann_df.loc[adata.obs_names, 'ManualAnnotation']

            adata.obs_names = [x + '_' + sample for x in adata.obs_names]
            rna_list.append(adata)

        # concatenation
        adata_concat = ad.concat(rna_list, label="slice_name", keys=slice_name_list)

        # plot clustering results
        embed = pd.read_csv(save_dir + f'comparison/{model}/{model}_group{idx}_embed_2.csv', header=None).values
        adata_concat.obsm['latent'] = embed
        gm = GaussianMixture(n_components=num_clusters_list[idx], covariance_type='tied', random_state=1234)
        y = gm.fit_predict(adata_concat.obsm['latent'], y=None)
        adata_concat.obs["gm_clusters"] = pd.Series(y, index=adata_concat.obs.index, dtype='category')

        adata_concat = adata_concat[~adata_concat.obs['Manual_Annotation'].isna(), :]
        spots_count = [0]
        n = 0
        for k in range(len(rna_list)):
            rna_list[k] = rna_list[k][~rna_list[k].obs['Manual_Annotation'].isna(), :]
            num = rna_list[k].shape[0]
            n += num
            spots_count.append(n)

        if idx != 1:
            adata_concat.obs['matched_clusters'] = list(pd.Series(1 + match_cluster_labels(
                adata_concat.obs['Manual_Annotation'], adata_concat.obs["gm_clusters"]),
                                                                  index=adata_concat.obs.index, dtype='category'))
        else:
            adata_concat.obs['matched_clusters'] = list(pd.Series(3 + match_cluster_labels(
                adata_concat.obs['Manual_Annotation'], adata_concat.obs["gm_clusters"]),
                                                                  index=adata_concat.obs.index, dtype='category'))
        my_clusters = np.sort(list(set(adata_concat.obs['matched_clusters'])))

        for i in range(len(rna_list)):
            rna_list[i].obs['matched_clusters'] = list(adata_concat.obs['matched_clusters'][spots_count[i]:spots_count[i+1]])

        sp_embedding = reducer.fit_transform(adata_concat.obsm['latent'])

        plot_DLPFC(rna_list, adata_concat, 'Manual_Annotation', 'matched_clusters', model, idx, layer_to_color_map,
                   matched_to_color_map, my_clusters, slice_name_list, cls_list, sp_embedding,
                   save_root=save_dir+'comparison/', frame_color=colors_for_labels[j], file_format=file_format,
                   save=save, plot=True)






