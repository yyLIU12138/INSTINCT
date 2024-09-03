import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1234)

save = False

scenario = 1
file_format = 'png'

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

slice_index_list = [str(i) for i in range(len(slice_name_list))]
num_iters = 8
num_clusters = 5

models = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']
labels1 = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']
color_list_1 = ['darkviolet', 'chocolate', 'sandybrown', 'peachpuff', 'darkslategray', 'c', 'cyan']
metric_list = ['ARI', 'AMI', 'NMI', 'FMI', 'Comp', 'Homo', 'mAP', 'Spot-type ASW',
               'Batch ASW', 'Batch PCR', 'kBET', 'Graph connectivity']
metric_groups = ['Clustering Performance', 'Representation Quality', 'Batch Correction']


# load results
aris = []
amis = []
nmis = []
fmis = []
comps = []
homos = []
maps = []
c_asws = []
b_asws = []
b_pcrs = []
kbets = []
g_conns = []

for model_name in models:

    with open(save_dir + f'comparison/{model_name}/{model_name}_results_dict.pkl', 'rb') as file:
        results_dict = pickle.load(file)

    aris.append(results_dict['ARIs'])
    amis.append(results_dict['AMIs'])
    nmis.append(results_dict['NMIs'])
    fmis.append(results_dict['FMIs'])
    comps.append(results_dict['COMPs'])
    homos.append(results_dict['HOMOs'])
    maps.append(results_dict['mAPs'])
    c_asws.append(results_dict['Cell_type_ASWs'])
    b_asws.append(results_dict['Batch_ASWs'])
    b_pcrs.append(results_dict['Batch_PCRs'])
    kbets.append(results_dict['kBETs'])
    g_conns.append(results_dict['Graph_connectivities'])

aris = np.array(aris)
amis = np.array(amis)
nmis = np.array(nmis)
fmis = np.array(fmis)
comps = np.array(comps)
homos = np.array(homos)
maps = np.array(maps)
c_asws = np.array(c_asws)
b_asws = np.array(b_asws)
b_pcrs = np.array(b_pcrs)
kbets = np.array(kbets)
g_conns = np.array(g_conns)

cluster_stacked = np.stack([aris, amis, nmis, fmis, comps, homos], axis=-1)
bio_conserve_stacked = np.stack([maps, c_asws], axis=-1)
batch_corr_stacked = np.stack([b_asws, b_pcrs, kbets, g_conns], axis=-1)
stacked_matrix = np.stack([aris, amis, nmis, fmis, comps, homos, maps, c_asws, b_asws, b_pcrs, kbets, g_conns], axis=-1)


# separate scores box plot
fig, axs = plt.subplots(figsize=(10, 4))

for i, model in enumerate(models):

    positions = np.arange(6) * (len(models) + 1) + i
    boxplot = axs.boxplot(cluster_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[0:6], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_1[i])

axs.set_title('Clustering Performance', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([(len(models)-1)/2 + (len(models)+1) * i for i in range(6)])
axs.set_xticklabels(metric_list[0:6], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
           for i, label in enumerate(labels1)]
axs.legend(handles=handles, loc=(1, 0.55), fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.85)

if save:
    save_path = save_dir + f"comparison/methods_clustering_performance_separate_scores_box_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
plt.show()

fig, axs = plt.subplots(figsize=(6, 4))

for i, model in enumerate(models):

    positions = np.arange(2) * (len(models) + 1) + i
    boxplot = axs.boxplot(bio_conserve_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[6:8], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_1[i])

plt.title('Representation Quality', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([(len(models)-1)/2 + (len(models)+1) * i for i in range(2)])
axs.set_xticklabels(metric_list[6:8], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
           for i, label in enumerate(labels1)]
axs.legend(handles=handles, loc='lower left', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"comparison/methods_representation_quality_separate_scores_box_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
plt.show()

fig, axs = plt.subplots(figsize=(8, 4))

for i, model in enumerate(models):

    positions = np.arange(4) * (len(models) + 1) + i
    boxplot = axs.boxplot(batch_corr_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[8:12], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_1[i])

plt.title('Batch Correction', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([(len(models)-1)/2 + (len(models)+1) * i for i in range(4)])
axs.set_xticklabels(metric_list[8:12], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
           for i, label in enumerate(labels1)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"comparison/methods_batch_correction_separate_scores_box_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
plt.show()


# separate scores bar plot
fig, axs = plt.subplots(figsize=(10, 4))

for i, model in enumerate(models):

    means = np.mean(cluster_stacked[i], axis=0)
    std_devs = np.std(cluster_stacked[i], axis=0)

    positions = np.arange(6) * (len(models) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_1[i], capsize=5)

axs.set_title('Clustering Performance', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([(len(models)-1)/2 + (len(models)+1) * i for i in range(6)])
axs.set_xticklabels(metric_list[0:6], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
           for i, label in enumerate(labels1)]
axs.legend(handles=handles, loc=(1, 0.55), fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.85)

if save:
    save_path = save_dir + f"comparison/methods_clustering_performance_separate_scores_bar_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
plt.show()

fig, axs = plt.subplots(figsize=(6, 4))

for i, model in enumerate(models):

    means = np.mean(bio_conserve_stacked[i], axis=0)
    std_devs = np.std(bio_conserve_stacked[i], axis=0)

    positions = np.arange(2) * (len(models) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_1[i], capsize=5)

plt.title('Representation Quality', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([(len(models)-1)/2 + (len(models)+1) * i for i in range(2)])
axs.set_xticklabels(metric_list[6:8], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
           for i, label in enumerate(labels1)]
axs.legend(handles=handles, loc='lower left', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"comparison/methods_representation_quality_separate_scores_bar_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
plt.show()

fig, axs = plt.subplots(figsize=(8, 4))

for i, model in enumerate(models):

    means = np.mean(batch_corr_stacked[i], axis=0)
    std_devs = np.std(batch_corr_stacked[i], axis=0)

    positions = np.arange(4) * (len(models) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_1[i], capsize=5)

plt.title('Batch Correction', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([(len(models)-1)/2 + (len(models)+1) * i for i in range(4)])
axs.set_xticklabels(metric_list[8:12], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
           for i, label in enumerate(labels1)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"comparison/methods_batch_correction_separate_scores_bar_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
plt.show()


# separate score rank plot
metric_list = ['ARI', 'AMI', 'NMI', 'FMI', 'Comp', 'Homo', 'mAP', 'Spot-type\nASW',
               'Batch\nASW', 'Batch\nPCR', 'kBET', 'Graph\nconnectivity']

mean_scores = []
for i in range(len(models)):
    score = np.mean(stacked_matrix[i], axis=0)
    mean_scores.append(score)
mean_scores = np.array(mean_scores) + 0.1

median_scores = []
for i in range(len(models)):
    score = np.median(stacked_matrix[i], axis=0)
    median_scores.append(score)
median_scores = np.array(median_scores) + 0.1

mtx_list = [mean_scores, median_scores]
type_list = ['mean', 'median']

for k in range(len(mtx_list)):

    matrix = mtx_list[k]

    if save:
        overview_matrix = np.zeros((7, len(metric_list)+4), dtype=float)
        if type_list[k] == 'mean':
            overview_matrix[:, 0:6] = np.mean(cluster_stacked, axis=1)
            overview_matrix[:, 6] = np.mean(np.mean(cluster_stacked, axis=-1), axis=-1)
            overview_matrix[:, 7:9] = np.mean(bio_conserve_stacked, axis=1)
            overview_matrix[:, 9] = np.mean(np.mean(bio_conserve_stacked, axis=-1), axis=-1)
            overview_matrix[:, 10:14] = np.mean(batch_corr_stacked, axis=1)
            overview_matrix[:, 14] = np.mean(np.mean(batch_corr_stacked, axis=-1), axis=-1)
            overview_matrix[:, 15] = np.mean((np.mean(cluster_stacked, axis=-1) +
                                              np.mean(bio_conserve_stacked, axis=-1) +
                                              np.mean(batch_corr_stacked, axis=-1)) / 3, axis=-1)
        elif type_list[k] == 'median':
            overview_matrix[:, 0:6] = np.median(cluster_stacked, axis=1)
            overview_matrix[:, 6] = np.median(np.mean(cluster_stacked, axis=-1), axis=-1)
            overview_matrix[:, 7:9] = np.median(bio_conserve_stacked, axis=1)
            overview_matrix[:, 9] = np.median(np.mean(bio_conserve_stacked, axis=-1), axis=-1)
            overview_matrix[:, 10:14] = np.median(batch_corr_stacked, axis=1)
            overview_matrix[:, 14] = np.median(np.mean(batch_corr_stacked, axis=-1), axis=-1)
            overview_matrix[:, 15] = np.median((np.mean(cluster_stacked, axis=-1) +
                                                np.mean(bio_conserve_stacked, axis=-1) +
                                                np.mean(batch_corr_stacked, axis=-1)) / 3, axis=-1)
        np.savetxt(save_dir + f'comparison/methods_{type_list[k]}_separate_scores.txt', overview_matrix)

    sorted_indices = np.argsort(matrix, axis=0)
    first_indices = sorted_indices[-1, :]
    second_indices = sorted_indices[-2, :]
    print(first_indices)
    print(second_indices)

    y_positions, x_positions = np.where(np.ones_like(matrix))

    colors = []
    for i in range(len(models)):
        colors += [color_list_1[i]] * len(metric_list)

    circle_sizes = (np.square(matrix * 25)).flatten()

    fig, axs = plt.subplots(figsize=(8.5, 6.5))
    plt.axis('off')

    gold_sizes = []
    silver_sizes = []
    for i in range(len(metric_list)):
        gold_sizes.append(np.square(matrix[first_indices[i]][i] * 25 + 5))
        silver_sizes.append(np.square(matrix[second_indices[i]][i] * 25 + 5))

    plt.scatter(np.arange(len(metric_list)), first_indices, c='gold', s=gold_sizes, alpha=1, edgecolors='black')
    plt.scatter(np.arange(len(metric_list)), second_indices, c='silver', s=silver_sizes, alpha=1, edgecolors='black')
    plt.scatter(x_positions, y_positions, c=colors, s=circle_sizes, alpha=1, edgecolors='black')
    for i in range(len(metric_list)):
        text = plt.text(i, first_indices[i], '1', ha='center', va='center',
                        fontdict={'size': 8, 'weight': 'bold', 'color': 'gold'})
        text.set_path_effects([withStroke(linewidth=1.5, foreground='black')])
        text = plt.text(i, second_indices[i], '2', ha='center', va='center',
                        fontdict={'size': 8, 'weight': 'bold', 'color': 'silver'})
        text.set_path_effects([withStroke(linewidth=1.5, foreground='black')])

    for i in range(len(models)):
        text = plt.text(-1.1, i, models[i], ha='center', va='center',
                        fontdict={'size': 12})
    for i in range(len(metric_list)):
        text = plt.text(i, 6.9, metric_list[i], ha='center', va='center', rotation=30,
                        fontdict={'size': 11})

    plt.ylim(-1.5, 8.5)
    plt.gca().invert_yaxis()
    plt.gcf().subplots_adjust(left=None, top=0.9, bottom=0.1, right=None)

    if save:
        save_path = save_dir + f"comparison/methods_{type_list[k]}_separate_scores_rank_plot.{file_format}"
        plt.savefig(save_path, dpi=500)
    plt.show()


# group mean score box plot
for k in range(len(metric_groups)):

    overall_scores = []
    for i in range(len(models)):
        if metric_groups[k] == 'Clustering Performance':
            score = np.mean(stacked_matrix[i][:, 0:6], axis=1)
        elif metric_groups[k] == 'Representation Quality':
            score = np.mean(stacked_matrix[i][:, 6:8], axis=1)
        elif metric_groups[k] == 'Batch Correction':
            score = np.mean(stacked_matrix[i][:, 8:12], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(6, 6))

    positions = np.arange(len(models))
    for i, box in enumerate(axs.boxplot(overall_scores.T, positions=positions, patch_artist=True, labels=models,
                                        widths=0.8, showfliers=False, zorder=0)['boxes']):
        box.set_facecolor(color_list_1[i])

    axs.set_xticklabels(models, rotation=30)
    axs.tick_params(axis='x', labelsize=12)
    axs.tick_params(axis='y', labelsize=12)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_linewidth(1)
    axs.spines['bottom'].set_linewidth(1)

    for j, d in enumerate(overall_scores):
        y = np.random.normal(positions[j], 0.1, size=len(d))
        plt.scatter(y, d, alpha=1, color='white', s=20, zorder=1, edgecolors='black')

    axs.set_title(f'{metric_groups[k]}', fontsize=16)

    if save:
        save_path = save_dir + f"comparison/methods_{metric_groups[k]}_score_box_plot.{file_format}"
        plt.savefig(save_path, dpi=300)
plt.show()


# group mean score bar plot
for k in range(len(metric_groups)):

    overall_scores = []
    for i in range(len(models)):
        if metric_groups[k] == 'Clustering Performance':
            score = np.mean(stacked_matrix[i][:, 0:6], axis=1)
        elif metric_groups[k] == 'Representation Quality':
            score = np.mean(stacked_matrix[i][:, 6:8], axis=1)
        elif metric_groups[k] == 'Batch Correction':
            score = np.mean(stacked_matrix[i][:, 8:12], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(6, 6))

    means = np.mean(overall_scores, axis=1)
    std_devs = np.std(overall_scores, axis=1)

    positions = np.arange(len(models))
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_1, capsize=5)

    axs.set_xticks(positions)
    axs.set_xticklabels(models, rotation=30)
    axs.tick_params(axis='x', labelsize=12)
    axs.tick_params(axis='y', labelsize=12)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_linewidth(1)
    axs.spines['bottom'].set_linewidth(1)

    # if metric_groups[k] == 'Biological Conservation':
    #     axs.set_ylim(bottom=0.7)

    for j, d in enumerate(overall_scores):
        y = np.random.normal(positions[j], 0.1, size=len(d))
        plt.scatter(y, d, alpha=1, color='white', s=20, zorder=1, edgecolors='black')

    axs.set_title(f'{metric_groups[k]}', fontsize=16)

    if save:
        save_path = save_dir + f"comparison/methods_{metric_groups[k]}_score_bar_plot.{file_format}"
        plt.savefig(save_path, dpi=300)
plt.show()


# overall score box plot
overall_scores = []
for i in range(len(models)):
    score = (np.mean(cluster_stacked[i], axis=-1) +
             np.mean(bio_conserve_stacked[i], axis=-1) +
             np.mean(batch_corr_stacked[i], axis=-1)) / 3
    overall_scores.append(score)
overall_scores = np.array(overall_scores)

fig, axs = plt.subplots(figsize=(6, 6))

positions = np.arange(len(models))
for i, box in enumerate(axs.boxplot(overall_scores.T, positions=positions, patch_artist=True, labels=models,
                                    widths=0.8, showfliers=False, zorder=0)['boxes']):
    box.set_facecolor(color_list_1[i])

axs.set_xticklabels(models, rotation=30)
axs.tick_params(axis='x', labelsize=12)
axs.tick_params(axis='y', labelsize=12)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

for j, d in enumerate(overall_scores):
    y = np.random.normal(positions[j], 0.1, size=len(d))
    plt.scatter(y, d, alpha=1, color='white', s=20, zorder=1, edgecolors='black')

axs.set_title(f'Overall Score', fontsize=16)

if save:
    save_path = save_dir + f"comparison/methods_overall_score_box_plot.{file_format}"
    plt.savefig(save_path, dpi=300)
plt.show()
