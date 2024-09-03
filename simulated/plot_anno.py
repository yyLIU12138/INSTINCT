import pickle
import numpy as np
import anndata as ad
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from ..evaluation_utils import knn_label_translation

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

scenario = 1

slice_name_list = [
    'Tech_0_0_Bio_0_0.5',
    'Tech_0_0.1_Bio_0_0',
    'Tech_0_0.1_Bio_0_0.5',
             ]

slice_index_list = list(range(len(slice_name_list)))

name_concat = slice_name_list[0]
for mode in slice_name_list[1:]:
    name_concat = name_concat + '_' + mode

save_dir = f'../../results/simulated/scenario_{scenario}/T_' + name_concat + '/'

save = False
file_format = 'png'

models = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']
labels = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']
color_list = ['darkviolet', 'chocolate', 'sandybrown', 'peachpuff', 'darkslategray', 'c', 'cyan']
metric_list = ['Accuracy', 'Kappa', 'mF1', 'wF1']

# cross validation scores
accus, kappas, mf1s, wf1s = [], [], [], []

for model in models:

    with open(save_dir + f'annotation/{model}/{model}_results_dict.pkl', 'rb') as file:
        results_dict = pickle.load(file)

    accus.append(results_dict['Accuracies'])
    kappas.append(results_dict['Kappas'])
    mf1s.append(results_dict['mF1s'])
    wf1s.append(results_dict['wF1s'])

accus = np.array(accus)
kappas = np.array(kappas)
mf1s = np.array(mf1s)
wf1s = np.array(wf1s)

stacked_matrix = np.stack([accus, kappas, mf1s, wf1s], axis=-1)

# separate
fig, axs = plt.subplots(figsize=(10, 3))

for i, model in enumerate(models):

    positions = np.arange(4) * (len(models) + 1) + i
    boxplot = axs.boxplot(stacked_matrix[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list, showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list[i])

plt.title(f'Simulated (Secnario {scenario})', fontsize=12)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([3 + 8 * i for i in range(4)])
axs.set_xticklabels(metric_list, fontsize=10)
axs.tick_params(axis='y', labelsize=10)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list[i], label=label)
           for i, label in enumerate(labels)]
axs.legend(handles=handles, loc=(1.01, 0.35), fontsize=8)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.85)

if save:
    save_path = save_dir + f"annotation/cross_validation_separate_scores_box_plot.{file_format}"
    plt.savefig(save_path)
plt.show()


# separate bar
fig, axs = plt.subplots(figsize=(8, 5))

for i, model in enumerate(models):

    means = np.mean(stacked_matrix[i], axis=0)
    std_devs = np.std(stacked_matrix[i], axis=0)

    positions = np.arange(4) * (len(models) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list[i], capsize=5)

axs.set_title(f'Simulated (Secnario {scenario})', fontsize=12)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([3 + 8 * i for i in range(4)])
axs.set_xticklabels(metric_list, fontsize=10)
axs.tick_params(axis='y', labelsize=10)
axs.set_ylim(0.5, 1)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list[i], label=label)
           for i, label in enumerate(labels)]
axs.legend(handles=handles, loc=(1.01, 0.35), fontsize=8)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.85)

if save:
    save_path = save_dir + f"annotation/cross_validation_separate_scores_bar_plot.{file_format}"
    plt.savefig(save_path)
plt.show()


# overall
overall_scores = []
for i in range(len(models)):
    score = np.mean(stacked_matrix[i], axis=1)
    overall_scores.append(score)
overall_scores = np.array(overall_scores)

fig, axs = plt.subplots(figsize=(6, 6))

positions = np.arange(len(models))
for i, box in enumerate(axs.boxplot(overall_scores.T, positions=positions, patch_artist=True, labels=labels,
                                    widths=0.8, showfliers=False, zorder=0)['boxes']):
    box.set_facecolor(color_list[i])

axs.set_xticklabels(labels, rotation=30)
axs.tick_params(axis='x', labelsize=12)
axs.tick_params(axis='y', labelsize=12)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

for j, d in enumerate(overall_scores):
    y = np.random.normal(positions[j], 0.1, size=len(d))
    plt.scatter(y, d, alpha=1, color='white', s=20, zorder=1, edgecolors='black')

axs.set_title(f'Overall Score (Secnario {scenario})', fontsize=14)
plt.gcf().subplots_adjust(left=0.15, top=None, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"annotation/cross_validation_overall_score_box_plot.{file_format}"
    plt.savefig(save_path, dpi=300)
plt.show()


# confusion matrix
slice_name_list = [
    'Tech_0_0_Bio_0_0.5',
    'Tech_0_0.1_Bio_0_0',
    'Tech_0_0.1_Bio_0_0.5',
             ]
slice_index_list = list(range(len(slice_name_list)))
cls_list = ['0', '1', '2', '3', '4']

cas_list = [ad.read_h5ad(save_dir + f"filtered_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
adata_concat = ad.concat(cas_list, label='slice_idx', keys=slice_index_list)
embed = pd.read_csv(save_dir + f'comparison/INSTINCT/INSTINCT_embed_2.csv', header=None).values
adata_concat.obsm['latent'] = embed

spots_count = [0]
n = 0
for sample in cas_list:
    num = sample.shape[0]
    n += num
    spots_count.append(n)

for j in range(len(cas_list)):
    cas_list[j].obsm['latent'] = embed[spots_count[j]:spots_count[j+1]]

for j in range(len(cas_list)):
    cas_list[j].obs['predicted_labels'] = knn_label_translation(adata_concat[adata_concat.obs['slice_idx'] != j].obsm['latent'].copy(),
                                                                adata_concat[adata_concat.obs['slice_idx'] != j].obs['real_spot_clusters'].copy(),
                                                                adata_concat[adata_concat.obs['slice_idx'] == j].obsm['latent'].copy(), k=20)

    true_labels = cas_list[j].obs['real_spot_clusters']
    predicted_labels = cas_list[j].obs['predicted_labels']
    cm = confusion_matrix(true_labels, predicted_labels, labels=cls_list)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Greens', square=True, cbar=False,
                xticklabels=cls_list, yticklabels=cls_list)

    plt.title(f'Slice {j} Confusion Matrix', fontsize=12)
    plt.xlabel('Predicted Labels', fontsize=10)
    plt.ylabel('True Labels', fontsize=10)
    plt.gcf().subplots_adjust(left=0.15, top=None, bottom=0.15, right=None)

    if save:
        save_path = save_dir + f"annotation/INSTINCT_confusion_matrix_slice{j}.{file_format}"
        plt.savefig(save_path, dpi=300)
plt.show()


# Sankey diagram
from pyecharts.charts import Sankey
from pyecharts import options as opts


slice_name_list = [
    'Tech_0_0_Bio_0_0.5',
    'Tech_0_0.1_Bio_0_0',
    'Tech_0_0.1_Bio_0_0.5',
             ]
slice_index_list = list(range(len(slice_name_list)))
cls_list = ['0', '1', '2', '3', '4']
colors_for_clusters = ['blue', 'orange', 'red', 'green', 'purple']
color_dict = {}
for i in range(len(cls_list)):
    color_dict['T_' + cls_list[i]] = colors_for_clusters[i]
    color_dict['P_' + cls_list[i]] = colors_for_clusters[i]

cas_list = [ad.read_h5ad(save_dir + f"filtered_spot_level_slice_{mode}.h5ad") for mode in slice_name_list]
adata_concat = ad.concat(cas_list, label='slice_idx', keys=slice_index_list)
embed = pd.read_csv(save_dir + f'comparison/INSTINCT/INSTINCT_embed_2.csv', header=None).values
adata_concat.obsm['latent'] = embed

spots_count = [0]
n = 0
for sample in cas_list:
    num = sample.shape[0]
    n += num
    spots_count.append(n)

for j in range(len(cas_list)):
    cas_list[j].obsm['latent'] = embed[spots_count[j]:spots_count[j+1]]

for j in range(len(cas_list)):
    cas_list[j].obs['predicted_labels'] = knn_label_translation(adata_concat[adata_concat.obs['slice_idx'] != j].obsm['latent'].copy(),
                                                                adata_concat[adata_concat.obs['slice_idx'] != j].obs['real_spot_clusters'].copy(),
                                                                adata_concat[adata_concat.obs['slice_idx'] == j].obsm['latent'].copy(), k=20)

    true_labels = cas_list[j].obs['real_spot_clusters']
    predicted_labels = cas_list[j].obs['predicted_labels']

    true_labels = 'T_' + true_labels.astype(str)
    predicted_labels = 'P_' + predicted_labels.astype(str)

    data = pd.DataFrame({'source': true_labels, 'target': predicted_labels})
    nodes_list = list(set(data['source'])) + list(set(data['target']))
    nodes = []
    for node in nodes_list:
        dic = {}
        dic['name'] = node
        dic['itemStyle'] = {'color': color_dict[node]}
        nodes.append(dic)
    links_df = data.groupby(['source', 'target']).size().reset_index(name='value')
    links_df = links_df[links_df['value'] != 0]
    links = []
    for i in links_df.values:
        dic = {}
        dic['source'] = i[0]
        dic['target'] = i[1]
        dic['value'] = i[2]
        links.append(dic)

    # print(nodes)
    # print(links)

    sankey_base = (
        Sankey(init_opts=opts.InitOpts(width='1100px', height='600px', bg_color='white'))
        .add(
            series_name=slice_name_list[j],
            nodes=nodes,
            links=links,
            pos_left="20%",
            pos_top="10%",
            linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
            label_opts=opts.LabelOpts(position="right"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"Slice {j}", pos_left="20%"))
    )
    if save:
        sankey_base.render(save_dir + f"annotation/INSTINCT/sankey_diagram_slice{j}.html")



