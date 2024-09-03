import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

save_dir = '../../results/MouseBrain_Jiang2023/comparison/Harmony_Seurat/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save = False
file_format = 'png'

slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']

models = ['INSTINCT', 'Harmony', 'Seurat']
labels1 = ['INSTINCT', 'Harmony', 'Seurat']
color_list_1 = ['darkviolet', 'sienna', 'orangered']
# models = ['INSTINCT', 'INSTINCT_MLP']
# labels1 = ['INSTINCT', 'INSTINCT_MLP']
# color_list_1 = ['darkviolet', 'yellowgreen']

metric_list = ['ARI', 'AMI', 'NMI', 'FMI', 'Comp', 'Homo', 'mAP', 'Spot-type\nASW', 'Isolated\nlabel ASW',
               'Isolated\nlabel F1', 'Batch\nASW', 'Batch\nPCR', 'kBET', 'Graph\nconnectivity']
metric_groups = ['Clustering Performance', 'Representation Quality', 'Batch Correction']

# comparison between methods
aris, amis, nmis, fmis, comps, homos, maps, c_asws = [], [], [], [], [], [], [], []
i_asws, i_f1s, b_asws, b_pcrs, kbets, g_conns = [], [], [], [], [], []

for model in models:

    with open(f'../../results/MouseBrain_Jiang2023/comparison/{model}/{model}_results_dict.pkl', 'rb') as file:
        results_dict = pickle.load(file)

    aris.append(results_dict['ARIs'])
    amis.append(results_dict['AMIs'])
    nmis.append(results_dict['NMIs'])
    fmis.append(results_dict['FMIs'])
    comps.append(results_dict['COMPs'])
    homos.append(results_dict['HOMOs'])
    maps.append(results_dict['mAPs'])
    c_asws.append(results_dict['Cell_type_ASWs'])
    i_asws.append(results_dict['Isolated_label_ASWs'])
    i_f1s.append(results_dict['Isolated_label_F1s'])
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
i_asws = np.array(i_asws)
i_f1s = np.array(i_f1s)
b_asws = np.array(b_asws)
b_pcrs = np.array(b_pcrs)
kbets = np.array(kbets)
g_conns = np.array(g_conns)

cluster_stacked = np.stack([aris, amis, nmis, fmis, comps, homos], axis=-1)
bio_conserve_stacked = np.stack([maps, c_asws, i_asws, i_f1s], axis=-1)
batch_corr_stacked = np.stack([b_asws, b_pcrs, kbets, g_conns], axis=-1)
stacked_matrix = np.stack([aris, amis, nmis, fmis, comps, homos, maps, c_asws,
                           i_asws, i_f1s, b_asws, b_pcrs, kbets, g_conns], axis=-1)


# separate scores box plot
fig, axs = plt.subplots(figsize=(1+len(models)*3, len(models)+1))

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
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.85)

if save:
    save_path = save_dir + f"methods_clustering_performance_separate_scores_box_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
# plt.show()

fig, axs = plt.subplots(figsize=(2+len(models)*2, len(models)+1))

for i, model in enumerate(models):

    positions = np.arange(4) * (len(models) + 1) + i
    boxplot = axs.boxplot(bio_conserve_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[6:10], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_1[i])

plt.title('Representation Quality', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([(len(models)-1)/2 + (len(models)+1) * i for i in range(4)])
axs.set_xticklabels(metric_list[6:10], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

# handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
#            for i, label in enumerate(labels1)]
# axs.legend(handles=handles, loc='lower left', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"methods_representation_quality_separate_scores_box_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
# plt.show()

fig, axs = plt.subplots(figsize=(2+len(models)*2, len(models)+1))

for i, model in enumerate(models):

    positions = np.arange(4) * (len(models) + 1) + i
    boxplot = axs.boxplot(batch_corr_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[10:14], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_1[i])

plt.title('Batch Correction', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([(len(models)-1)/2 + (len(models)+1) * i for i in range(4)])
axs.set_xticklabels(metric_list[10:14], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

# handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
#            for i, label in enumerate(labels1)]
# axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"methods_batch_correction_separate_scores_box_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
plt.show()


# separate scores bar plot
fig, axs = plt.subplots(figsize=(1+len(models)*3, len(models)+1))

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
    save_path = save_dir + f"methods_clustering_performance_separate_scores_bar_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
# plt.show()

fig, axs = plt.subplots(figsize=(2+len(models)*2, len(models)+1))

for i, model in enumerate(models):

    means = np.mean(bio_conserve_stacked[i], axis=0)
    std_devs = np.std(bio_conserve_stacked[i], axis=0)

    positions = np.arange(4) * (len(models) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_1[i], capsize=5)

plt.title('Representation Quality', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([(len(models)-1)/2 + (len(models)+1) * i for i in range(4)])
axs.set_xticklabels(metric_list[6:10], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

# handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
#            for i, label in enumerate(labels1)]
# axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"methods_representation_quality_separate_scores_bar_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
# plt.show()

fig, axs = plt.subplots(figsize=(2+len(models)*2, len(models)+1))

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
axs.set_xticklabels(metric_list[10:14], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

# handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
#            for i, label in enumerate(labels1)]
# axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"methods_batch_correction_separate_scores_bar_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
plt.show()


# group mean score box plot
for k in range(len(metric_groups)):

    overall_scores = []
    for i in range(len(models)):
        if metric_groups[k] == 'Clustering Performance':
            score = np.mean(cluster_stacked[i], axis=1)
        elif metric_groups[k] == 'Representation Quality':
            score = np.mean(bio_conserve_stacked[i], axis=1)
        elif metric_groups[k] == 'Batch Correction':
            score = np.mean(batch_corr_stacked[i], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(len(models)+1, len(models)+3))

    positions = np.arange(len(models))
    for i, box in enumerate(axs.boxplot(overall_scores.T, positions=positions, patch_artist=True, labels=labels1,
                                        widths=0.7, showfliers=False, zorder=0)['boxes']):
        box.set_facecolor(color_list_1[i])

    axs.set_xticklabels(labels1, rotation=30)
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
    plt.gcf().subplots_adjust(left=0.15, top=None, bottom=0.15, right=None)

    if save:
        save_path = save_dir + f"methods_{metric_groups[k]}_score_box_plot.{file_format}"
        plt.savefig(save_path, dpi=300)
plt.show()


# group mean score bar plot
for k in range(len(metric_groups)):

    overall_scores = []
    for i in range(len(models)):
        if metric_groups[k] == 'Clustering Performance':
            score = np.mean(cluster_stacked[i], axis=1)
        elif metric_groups[k] == 'Representation Quality':
            score = np.mean(bio_conserve_stacked[i], axis=1)
        elif metric_groups[k] == 'Batch Correction':
            score = np.mean(batch_corr_stacked[i], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(len(models)+1, len(models)+3))

    means = np.mean(overall_scores, axis=1)
    std_devs = np.std(overall_scores, axis=1)

    positions = np.arange(len(models))
    axs.bar(positions, means, yerr=std_devs, width=0.7, color=color_list_1, capsize=5)

    axs.set_xticks(positions)
    axs.set_xticklabels(labels1, rotation=30)
    axs.tick_params(axis='x', labelsize=12)
    axs.tick_params(axis='y', labelsize=12)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_linewidth(1)
    axs.spines['bottom'].set_linewidth(1)

    # if metric_groups[k] == 'Clustering Performance':
    #     axs.set_ylim(bottom=0.2)
    # elif metric_groups[k] == 'Biological Conservation':
    #     axs.set_ylim(bottom=0.5)
    # elif metric_groups[k] == 'Batch Correction':
    #     axs.set_ylim(bottom=0.3)

    for j, d in enumerate(overall_scores):
        y = np.random.normal(positions[j], 0.1, size=len(d))
        plt.scatter(y, d, alpha=1, color='white', s=20, zorder=1, edgecolors='black')

    axs.set_title(f'{metric_groups[k]}', fontsize=16)
    plt.gcf().subplots_adjust(left=0.15, top=None, bottom=0.15, right=None)

    if save:
        save_path = save_dir + f"methods_{metric_groups[k]}_score_bar_plot.{file_format}"
        plt.savefig(save_path, dpi=300)
plt.show()


# overall score box plot
overall_scores = []
for i in range(len(models)):
    score = (np.mean(cluster_stacked[i], axis=1) +
             np.mean(bio_conserve_stacked[i], axis=1) +
             np.mean(batch_corr_stacked[i], axis=1)) / 3
    overall_scores.append(score)
overall_scores = np.array(overall_scores)

fig, axs = plt.subplots(figsize=(len(models)+1, len(models)+3))

positions = np.arange(len(models))
for i, box in enumerate(axs.boxplot(overall_scores.T, positions=positions, patch_artist=True, labels=labels1,
                                    widths=0.8, showfliers=False, zorder=0)['boxes']):
    box.set_facecolor(color_list_1[i])

axs.set_xticklabels(labels1, rotation=30)
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
plt.gcf().subplots_adjust(left=0.15, top=None, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"methods_overall_score_box_plot.{file_format}"
    plt.savefig(save_path, dpi=300)
plt.show()

