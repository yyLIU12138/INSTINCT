import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

save_dir = '../../results/model_validity/MouseBrain_Jiang2023/'
save = False

if save:
    if not os.path.exists(save_dir + 'loss_functions/'):
        os.makedirs(save_dir + 'loss_functions/')
    if not os.path.exists(save_dir + 'model_structures/'):
        os.makedirs(save_dir + 'model_structures/')
    if not os.path.exists(save_dir + 'knn_strategy/'):
        os.makedirs(save_dir + 'knn_strategy/')
    if not os.path.exists(save_dir + 'clamp/'):
        os.makedirs(save_dir + 'clamp/')

titles = ['complete', 'without_Loss_adv', 'without_Loss_cls',
          'without_Loss_la', 'without_Loss_rec', 'without_D',
          'without_D_NG', 'use_euclidean', 'without_clamp']

titles1 = ['complete', 'without_Loss_adv', 'without_Loss_cls', 'without_Loss_la', 'without_Loss_rec']
labels1 = ['Complete', 'w/o $L_{adv}$', 'w/o $L_{cls}$', 'w/o $L_{la}$', 'w/o $L_{rec}$']
color_list_1 = ['darkviolet', 'mediumblue', 'b', 'darkslateblue', 'slateblue', 'mediumslateblue']

titles2 = ['complete', 'without_D', 'without_D_NG']
labels2 = ['Complete', 'w/o D', 'w/o D & NG']
color_list_2 = ['darkviolet', 'lightseagreen', 'turquoise']

titles3 = ['complete', 'use_euclidean']
labels3 = ['Cosine\nSimilarity', 'Euclidean\nDistance']
color_list_3 = ['darkviolet', 'darkred']

titles4 = ['complete', 'without_clamp']
labels4 = ['w/ clamp', 'w/o clamp']
color_list_4 = ['darkviolet', 'steelblue']

metric_list = ['ARI', 'AMI', 'NMI', 'FMI', 'Comp', 'Homo', 'mAP', 'Spot-type\nASW', 'Isolated label\nASW',
               'Isolated label\nF1', 'Batch\nASW', 'Batch\nPCR', 'kBET', 'Graph\nconnectivity']
metric_groups = ['Clustering Performance', 'Representation Quality', 'Batch Correction']


# loss functions
aris, amis, nmis, fmis, comps, homos, maps, c_asws = [], [], [], [], [], [], [], []
i_asws, i_f1s, b_asws, b_pcrs, kbets, g_conns = [], [], [], [], [], []

for title in titles1:

    with open(save_dir + f'{title}/{title}_results_dict.pkl', 'rb') as file:
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

# separate scores box plot
fig, axs = plt.subplots(figsize=(10, 5))

for i, titles in enumerate(titles1):

    positions = np.arange(6) * (len(titles1) + 1) + i
    boxplot = axs.boxplot(cluster_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[0:6], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_1[i])

axs.set_title('Clustering Performance', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([2 + 6 * i for i in range(6)])
axs.set_xticklabels(metric_list[0:6], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
           for i, label in enumerate(labels1)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"loss_functions/loss_functions_clustering_performance_separate_scores_box_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(10, 5))

for i, title in enumerate(titles1):

    positions = np.arange(4) * (len(titles1) + 1) + i
    boxplot = axs.boxplot(bio_conserve_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[6:10], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_1[i])

plt.title('Representation Quality', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([2 + 6 * i for i in range(4)])
axs.set_xticklabels(metric_list[6:10], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
           for i, label in enumerate(labels1)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"loss_functions/loss_functions_representation_quality_separate_scores_box_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(10, 5))

for i, title in enumerate(titles1):

    positions = np.arange(4) * (len(titles1) + 1) + i
    boxplot = axs.boxplot(batch_corr_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[10:14], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_1[i])

plt.title('Batch Correction', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([2 + 6 * i for i in range(4)])
axs.set_xticklabels(metric_list[10:14], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
           for i, label in enumerate(labels1)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"loss_functions/loss_functions_batch_correction_separate_scores_box_plot.pdf"
    plt.savefig(save_path)
plt.show()


# separate scores bar plot
fig, axs = plt.subplots(figsize=(10, 5))

for i, title in enumerate(titles1):

    means = np.mean(cluster_stacked[i], axis=0)
    std_devs = np.std(cluster_stacked[i], axis=0)

    positions = np.arange(6) * (len(titles1) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_1[i], capsize=5)

axs.set_title('Clustering Performance', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([2 + 6 * i for i in range(6)])
axs.set_xticklabels(metric_list[0:6], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
           for i, label in enumerate(labels1)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"loss_functions/loss_functions_clustering_performance_separate_scores_bar_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(10, 5))

for i, title in enumerate(titles1):

    means = np.mean(bio_conserve_stacked[i], axis=0)
    std_devs = np.std(bio_conserve_stacked[i], axis=0)

    positions = np.arange(4) * (len(titles1) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_1[i], capsize=5)

plt.title('Representation Quality', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([2 + 6 * i for i in range(4)])
axs.set_xticklabels(metric_list[6:10], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
           for i, label in enumerate(labels1)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"loss_functions/loss_functions_representation_quality_separate_scores_bar_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(10, 5))

for i, title in enumerate(titles1):

    means = np.mean(batch_corr_stacked[i], axis=0)
    std_devs = np.std(batch_corr_stacked[i], axis=0)

    positions = np.arange(4) * (len(titles1) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_1[i], capsize=5)

plt.title('Batch Correction', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([2 + 6 * i for i in range(4)])
axs.set_xticklabels(metric_list[10:14], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
           for i, label in enumerate(labels1)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"loss_functions/loss_functions_batch_correction_separate_scores_bar_plot.pdf"
    plt.savefig(save_path)
plt.show()


# group mean score box plot
for k in range(len(metric_groups)):

    overall_scores = []
    for i in range(len(titles1)):
        if metric_groups[k] == 'Clustering Performance':
            score = np.mean(cluster_stacked[i], axis=1)
        elif metric_groups[k] == 'Representation Quality':
            score = np.mean(bio_conserve_stacked[i], axis=1)
        elif metric_groups[k] == 'Batch Correction':
            score = np.mean(batch_corr_stacked[i], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(6, 6))

    positions = np.arange(len(titles1))
    for i, box in enumerate(axs.boxplot(overall_scores.T, positions=positions, patch_artist=True, labels=labels1,
                                        widths=0.7, showfliers=False, zorder=0)['boxes']):
        box.set_facecolor(color_list_1[i])

    axs.set_xticklabels(labels1)
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
        save_path = save_dir + f"loss_functions/loss_functions_{metric_groups[k]}_score_box_plot.pdf"
        plt.savefig(save_path)
plt.show()


# group mean score bar plot
for k in range(len(metric_groups)):

    overall_scores = []
    for i in range(len(titles1)):
        if metric_groups[k] == 'Clustering Performance':
            score = np.mean(cluster_stacked[i], axis=1)
        elif metric_groups[k] == 'Representation Quality':
            score = np.mean(bio_conserve_stacked[i], axis=1)
        elif metric_groups[k] == 'Batch Correction':
            score = np.mean(batch_corr_stacked[i], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(6, 6))

    means = np.mean(overall_scores, axis=1)
    std_devs = np.std(overall_scores, axis=1)

    positions = np.arange(len(titles1))
    axs.bar(positions, means, yerr=std_devs, width=0.7, color=color_list_1, capsize=5)

    axs.set_xticks(positions)
    axs.set_xticklabels(labels1)
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

    if save:
        save_path = save_dir + f"loss_functions/loss_functions_{metric_groups[k]}_score_bar_plot.pdf"
        plt.savefig(save_path)
plt.show()


# overall score box plot
overall_scores = []
for i in range(len(titles1)):
    score = (np.mean(cluster_stacked[i], axis=1) +
             np.mean(bio_conserve_stacked[i], axis=1) +
             np.mean(batch_corr_stacked[i], axis=1)) / 3
    overall_scores.append(score)
overall_scores = np.array(overall_scores)

fig, axs = plt.subplots(figsize=(6, 6))

positions = np.arange(len(titles1))
for i, box in enumerate(axs.boxplot(overall_scores.T, positions=positions, patch_artist=True, labels=labels1,
                                    widths=0.8, showfliers=False, zorder=0)['boxes']):
    box.set_facecolor(color_list_1[i])

axs.set_xticklabels(labels1)
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
    save_path = save_dir + f"loss_functions/loss_functions_overall_score_box_plot.pdf"
    plt.savefig(save_path)
plt.show()


# model structures
aris, amis, nmis, fmis, comps, homos, maps, c_asws = [], [], [], [], [], [], [], []
i_asws, i_f1s, b_asws, b_pcrs, kbets, g_conns = [], [], [], [], [], []

for title in titles2:

    with open(save_dir + f'{title}/{title}_results_dict.pkl', 'rb') as file:
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

# separate scores box plot
fig, axs = plt.subplots(figsize=(8, 3))

for i, titles in enumerate(titles2):

    positions = np.arange(6) * (len(titles2) + 1) + i
    boxplot = axs.boxplot(cluster_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[0:6], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_2[i])

axs.set_title('Clustering Performance', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([1 + 4 * i for i in range(6)])
axs.set_xticklabels(metric_list[0:6], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_2[i], label=label)
           for i, label in enumerate(labels2)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"model_structures/model_structures_clustering_performance_separate_scores_box_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles2):

    positions = np.arange(4) * (len(titles2) + 1) + i
    boxplot = axs.boxplot(bio_conserve_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[6:10], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_2[i])

plt.title('Representation Quality', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([1 + 4 * i for i in range(4)])
axs.set_xticklabels(metric_list[6:10], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_2[i], label=label)
           for i, label in enumerate(labels2)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"model_structures/model_structures_representation_quality_separate_scores_box_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles2):

    positions = np.arange(4) * (len(titles2) + 1) + i
    boxplot = axs.boxplot(batch_corr_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[10:14], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_2[i])

plt.title('Batch Correction', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([1 + 4 * i for i in range(4)])
axs.set_xticklabels(metric_list[10:14], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_2[i], label=label)
           for i, label in enumerate(labels2)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"model_structures/model_structures_batch_correction_separate_scores_box_plot.pdf"
    plt.savefig(save_path)
plt.show()


# separate scores bar plot
fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles2):

    means = np.mean(cluster_stacked[i], axis=0)
    std_devs = np.std(cluster_stacked[i], axis=0)

    positions = np.arange(6) * (len(titles2) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_2[i], capsize=5)

axs.set_title('Clustering Performance', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([1 + 4 * i for i in range(6)])
axs.set_xticklabels(metric_list[0:6], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_2[i], label=label)
           for i, label in enumerate(labels2)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"model_structures/model_structures_clustering_performance_separate_scores_bar_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles2):

    means = np.mean(bio_conserve_stacked[i], axis=0)
    std_devs = np.std(bio_conserve_stacked[i], axis=0)

    positions = np.arange(4) * (len(titles2) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_2[i], capsize=5)

axs.set_title('Representation Quality', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([1 + 4 * i for i in range(4)])
axs.set_xticklabels(metric_list[6:10], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_2[i], label=label)
           for i, label in enumerate(labels2)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"model_structures/model_structures_representation_quality_separate_scores_bar_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles2):

    means = np.mean(batch_corr_stacked[i], axis=0)
    std_devs = np.std(batch_corr_stacked[i], axis=0)

    positions = np.arange(4) * (len(titles2) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_2[i], capsize=5)

plt.title('Batch Correction', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([1 + 4 * i for i in range(4)])
axs.set_xticklabels(metric_list[10:14], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_2[i], label=label)
           for i, label in enumerate(labels2)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"model_structures/model_structures_batch_correction_separate_scores_bar_plot.pdf"
    plt.savefig(save_path)
plt.show()


# group mean score box plot
for k in range(len(metric_groups)):

    overall_scores = []
    for i in range(len(titles2)):
        if metric_groups[k] == 'Clustering Performance':
            score = np.mean(cluster_stacked[i], axis=1)
        elif metric_groups[k] == 'Representation Quality':
            score = np.mean(bio_conserve_stacked[i], axis=1)
        elif metric_groups[k] == 'Batch Correction':
            score = np.mean(batch_corr_stacked[i], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(4, 6))

    positions = np.arange(len(titles2))
    for i, box in enumerate(axs.boxplot(overall_scores.T, positions=positions, patch_artist=True, labels=labels2,
                                        widths=0.7, showfliers=False, zorder=0)['boxes']):
        box.set_facecolor(color_list_2[i])

    axs.set_xticklabels(labels2)
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
    plt.gcf().subplots_adjust(left=0.2, top=None, bottom=None, right=None)

    if save:
        save_path = save_dir + f"model_structures/model_structures_{metric_groups[k]}_score_box_plot.pdf"
        plt.savefig(save_path)
plt.show()


# group mean score bar plot
for k in range(len(metric_groups)):

    overall_scores = []
    for i in range(len(titles2)):
        if metric_groups[k] == 'Clustering Performance':
            score = np.mean(cluster_stacked[i], axis=1)
        elif metric_groups[k] == 'Representation Quality':
            score = np.mean(bio_conserve_stacked[i], axis=1)
        elif metric_groups[k] == 'Batch Correction':
            score = np.mean(batch_corr_stacked[i], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(4, 6))

    means = np.mean(overall_scores, axis=1)
    std_devs = np.std(overall_scores, axis=1)

    positions = np.arange(len(titles2))
    axs.bar(positions, means, yerr=std_devs, width=0.7, color=color_list_2, capsize=5)

    axs.set_xticks(positions)
    axs.set_xticklabels(labels2)
    axs.tick_params(axis='x', labelsize=12)
    axs.tick_params(axis='y', labelsize=12)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_linewidth(1)
    axs.spines['bottom'].set_linewidth(1)

    # if metric_groups[k] == 'Clustering Performance':
    #     axs.set_ylim(bottom=0.4)
    # elif metric_groups[k] == 'Biological Conservation':
    #     axs.set_ylim(bottom=0.55)
    # elif metric_groups[k] == 'Batch Correction':
    #     axs.set_ylim(bottom=0.7)

    for j, d in enumerate(overall_scores):
        y = np.random.normal(positions[j], 0.1, size=len(d))
        plt.scatter(y, d, alpha=1, color='white', s=20, zorder=1, edgecolors='black')

    axs.set_title(f'{metric_groups[k]}', fontsize=16)
    plt.gcf().subplots_adjust(left=0.2, top=None, bottom=None, right=None)

    if save:
        save_path = save_dir + f"model_structures/model_structures_{metric_groups[k]}_score_bar_plot.pdf"
        plt.savefig(save_path)
plt.show()


# overall score box plot
overall_scores = []
for i in range(len(titles2)):
    score = (np.mean(cluster_stacked[i], axis=1) +
             np.mean(bio_conserve_stacked[i], axis=1) +
             np.mean(batch_corr_stacked[i], axis=1)) / 3
    overall_scores.append(score)
overall_scores = np.array(overall_scores)

fig, axs = plt.subplots(figsize=(4, 6))

positions = np.arange(len(titles2))
for i, box in enumerate(axs.boxplot(overall_scores.T, positions=positions, patch_artist=True, labels=labels2,
                                    widths=0.7, showfliers=False, zorder=0)['boxes']):
    box.set_facecolor(color_list_2[i])

axs.set_xticklabels(labels2)
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
plt.gcf().subplots_adjust(left=0.2, top=None, bottom=None, right=None)

if save:
    save_path = save_dir + f"model_structures/model_structures_overall_score_box_plot.pdf"
    plt.savefig(save_path)
plt.show()


# knn strategy
aris, amis, nmis, fmis, comps, homos, maps, c_asws = [], [], [], [], [], [], [], []
i_asws, i_f1s, b_asws, b_pcrs, kbets, g_conns = [], [], [], [], [], []

for title in titles3:

    with open(save_dir + f'{title}/{title}_results_dict.pkl', 'rb') as file:
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

# separate scores box plot
fig, axs = plt.subplots(figsize=(8, 3))

for i, titles in enumerate(titles3):

    positions = np.arange(6) * (len(titles3) + 1) + i
    boxplot = axs.boxplot(cluster_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[0:6], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_3[i])

axs.set_title('Clustering Performance', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([0.5 + 3 * i for i in range(6)])
axs.set_xticklabels(metric_list[0:6], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_3[i], label=label)
           for i, label in enumerate(labels3)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"knn_strategy/knn_strategy_clustering_performance_separate_scores_box_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles3):

    positions = np.arange(4) * (len(titles3) + 1) + i
    boxplot = axs.boxplot(bio_conserve_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[6:10], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_3[i])

plt.title('Representation Quality', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([0.5 + 3 * i for i in range(4)])
axs.set_xticklabels(metric_list[6:10], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_3[i], label=label)
           for i, label in enumerate(labels3)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"knn_strategy/knn_strategy_representation_quality_separate_scores_box_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles3):

    positions = np.arange(4) * (len(titles3) + 1) + i
    boxplot = axs.boxplot(batch_corr_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[10:14], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_3[i])

plt.title('Batch Correction', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([0.5 + 3 * i for i in range(4)])
axs.set_xticklabels(metric_list[10:14], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_3[i], label=label)
           for i, label in enumerate(labels3)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"knn_strategy/knn_strategy_batch_correction_separate_scores_box_plot.pdf"
    plt.savefig(save_path)
plt.show()


# separate scores bar plot
fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles3):

    means = np.mean(cluster_stacked[i], axis=0)
    std_devs = np.std(cluster_stacked[i], axis=0)

    positions = np.arange(6) * (len(titles3) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_3[i], capsize=5)

axs.set_title('Clustering Performance', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([0.5 + 3 * i for i in range(6)])
axs.set_xticklabels(metric_list[0:6], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_3[i], label=label)
           for i, label in enumerate(labels3)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"knn_strategy/knn_strategy_clustering_performance_separate_scores_bar_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles3):

    means = np.mean(bio_conserve_stacked[i], axis=0)
    std_devs = np.std(bio_conserve_stacked[i], axis=0)

    positions = np.arange(4) * (len(titles3) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_3[i], capsize=5)

axs.set_title('Representation Quality', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([0.5 + 3 * i for i in range(4)])
axs.set_xticklabels(metric_list[6:10], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_3[i], label=label)
           for i, label in enumerate(labels3)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"knn_strategy/knn_strategy_representation_quality_separate_scores_bar_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles3):

    means = np.mean(batch_corr_stacked[i], axis=0)
    std_devs = np.std(batch_corr_stacked[i], axis=0)

    positions = np.arange(4) * (len(titles3) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_3[i], capsize=5)

plt.title('Batch Correction', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([0.5 + 3 * i for i in range(4)])
axs.set_xticklabels(metric_list[10:14], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_3[i], label=label)
           for i, label in enumerate(labels3)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"knn_strategy/knn_strategy_batch_correction_separate_scores_bar_plot.pdf"
    plt.savefig(save_path)
plt.show()


# group mean score box plot
for k in range(len(metric_groups)):

    overall_scores = []
    for i in range(len(titles3)):
        if metric_groups[k] == 'Clustering Performance':
            score = np.mean(cluster_stacked[i], axis=1)
        elif metric_groups[k] == 'Representation Quality':
            score = np.mean(bio_conserve_stacked[i], axis=1)
        elif metric_groups[k] == 'Batch Correction':
            score = np.mean(batch_corr_stacked[i], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(4, 6))

    positions = np.arange(len(titles3))
    for i, box in enumerate(axs.boxplot(overall_scores.T, positions=positions, patch_artist=True, labels=labels3,
                                        widths=0.7, showfliers=False, zorder=0)['boxes']):
        box.set_facecolor(color_list_3[i])

    axs.set_xticklabels(labels3)
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
    plt.gcf().subplots_adjust(left=0.2, top=None, bottom=None, right=None)

    if save:
        save_path = save_dir + f"knn_strategy/knn_strategy_{metric_groups[k]}_score_box_plot.pdf"
        plt.savefig(save_path)
plt.show()


# group mean score bar plot
for k in range(len(metric_groups)):

    overall_scores = []
    for i in range(len(titles3)):
        if metric_groups[k] == 'Clustering Performance':
            score = np.mean(cluster_stacked[i], axis=1)
        elif metric_groups[k] == 'Representation Quality':
            score = np.mean(bio_conserve_stacked[i], axis=1)
        elif metric_groups[k] == 'Batch Correction':
            score = np.mean(batch_corr_stacked[i], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(4, 6))

    means = np.mean(overall_scores, axis=1)
    std_devs = np.std(overall_scores, axis=1)

    positions = np.arange(len(titles3))
    axs.bar(positions, means, yerr=std_devs, width=0.7, color=color_list_3, capsize=5)

    axs.set_xticks(positions)
    axs.set_xticklabels(labels3)
    axs.tick_params(axis='x', labelsize=12)
    axs.tick_params(axis='y', labelsize=12)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_linewidth(1)
    axs.spines['bottom'].set_linewidth(1)

    # if metric_groups[k] == 'Clustering Performance':
    #     axs.set_ylim(bottom=0.35)
    # elif metric_groups[k] == 'Biological Conservation':
    #     axs.set_ylim(bottom=0.55)
    # elif metric_groups[k] == 'Batch Correction':
    #     axs.set_ylim(bottom=0.70)

    for j, d in enumerate(overall_scores):
        y = np.random.normal(positions[j], 0.1, size=len(d))
        plt.scatter(y, d, alpha=1, color='white', s=20, zorder=1, edgecolors='black')

    axs.set_title(f'{metric_groups[k]}', fontsize=16)
    plt.gcf().subplots_adjust(left=0.2, top=None, bottom=None, right=None)

    if save:
        save_path = save_dir + f"knn_strategy/knn_strategy_{metric_groups[k]}_score_bar_plot.pdf"
        plt.savefig(save_path)
plt.show()


# overall score box plot
overall_scores = []
for i in range(len(titles3)):
    score = (np.mean(cluster_stacked[i], axis=1) +
             np.mean(bio_conserve_stacked[i], axis=1) +
             np.mean(batch_corr_stacked[i], axis=1)) / 3
    overall_scores.append(score)
overall_scores = np.array(overall_scores)

fig, axs = plt.subplots(figsize=(4, 6))

positions = np.arange(len(titles3))
for i, box in enumerate(axs.boxplot(overall_scores.T, positions=positions, patch_artist=True, labels=labels3,
                                    widths=0.7, showfliers=False, zorder=0)['boxes']):
    box.set_facecolor(color_list_3[i])

axs.set_xticklabels(labels3)
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
plt.gcf().subplots_adjust(left=0.2, top=None, bottom=None, right=None)

if save:
    save_path = save_dir + f"knn_strategy/knn_strategy_overall_score_box_plot.pdf"
    plt.savefig(save_path)
plt.show()


# clamp
aris, amis, nmis, fmis, comps, homos, maps, c_asws,  = [], [], [], [], [], [], [], []
i_asws, i_f1s, b_asws, b_pcrs, kbets, g_conns = [], [], [], [], [], []

for title in titles4:

    with open(save_dir + f'{title}/{title}_results_dict.pkl', 'rb') as file:
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

# separate scores box plot
fig, axs = plt.subplots(figsize=(8, 3))

for i, titles in enumerate(titles4):

    positions = np.arange(6) * (len(titles4) + 1) + i
    boxplot = axs.boxplot(cluster_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[0:6], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_4[i])

axs.set_title('Clustering Performance', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([0.5 + 3 * i for i in range(6)])
axs.set_xticklabels(metric_list[0:6], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_4[i], label=label)
           for i, label in enumerate(labels4)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"clamp/clamp_clustering_performance_separate_scores_box_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles4):

    positions = np.arange(4) * (len(titles4) + 1) + i
    boxplot = axs.boxplot(bio_conserve_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[6:10], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_4[i])

plt.title('Representation Quality', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([0.5 + 3 * i for i in range(4)])
axs.set_xticklabels(metric_list[6:10], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_4[i], label=label)
           for i, label in enumerate(labels4)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"clamp/clamp_representation_quality_separate_scores_box_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles4):

    positions = np.arange(4) * (len(titles4) + 1) + i
    boxplot = axs.boxplot(batch_corr_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[10:14], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_4[i])

plt.title('Batch Correction', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([0.5 + 3 * i for i in range(4)])
axs.set_xticklabels(metric_list[10:14], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_4[i], label=label)
           for i, label in enumerate(labels4)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"clamp/clamp_batch_correction_separate_scores_box_plot.pdf"
    plt.savefig(save_path)
plt.show()


# separate scores bar plot
fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles4):

    means = np.mean(cluster_stacked[i], axis=0)
    std_devs = np.std(cluster_stacked[i], axis=0)

    positions = np.arange(6) * (len(titles4) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_4[i], capsize=5)

axs.set_title('Clustering Performance', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([0.5 + 3 * i for i in range(6)])
axs.set_xticklabels(metric_list[0:6], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_4[i], label=label)
           for i, label in enumerate(labels4)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"clamp/clamp_clustering_performance_separate_scores_bar_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles4):

    means = np.mean(bio_conserve_stacked[i], axis=0)
    std_devs = np.std(bio_conserve_stacked[i], axis=0)

    positions = np.arange(4) * (len(titles4) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_4[i], capsize=5)

axs.set_title('Representation Quality', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([0.5 + 3 * i for i in range(4)])
axs.set_xticklabels(metric_list[6:10], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_4[i], label=label)
           for i, label in enumerate(labels4)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"clamp/clamp_representation_quality_separate_scores_bar_plot.pdf"
    plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(figsize=(8, 3))

for i, title in enumerate(titles4):

    means = np.mean(batch_corr_stacked[i], axis=0)
    std_devs = np.std(batch_corr_stacked[i], axis=0)

    positions = np.arange(4) * (len(titles4) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_4[i], capsize=5)

plt.title('Batch Correction', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([0.5 + 3 * i for i in range(4)])
axs.set_xticklabels(metric_list[10:14], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_4[i], label=label)
           for i, label in enumerate(labels4)]
axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"clamp/clamp_batch_correction_separate_scores_bar_plot.pdf"
    plt.savefig(save_path)
plt.show()


# group mean score box plot
for k in range(len(metric_groups)):

    overall_scores = []
    for i in range(len(titles4)):
        if metric_groups[k] == 'Clustering Performance':
            score = np.mean(cluster_stacked[i], axis=1)
        elif metric_groups[k] == 'Representation Quality':
            score = np.mean(bio_conserve_stacked[i], axis=1)
        elif metric_groups[k] == 'Batch Correction':
            score = np.mean(batch_corr_stacked[i], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(4, 6))

    positions = np.arange(len(titles4))
    for i, box in enumerate(axs.boxplot(overall_scores.T, positions=positions, patch_artist=True, labels=labels4,
                                        widths=0.7, showfliers=False, zorder=0)['boxes']):
        box.set_facecolor(color_list_4[i])

    axs.set_xticklabels(labels4)
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
    plt.gcf().subplots_adjust(left=0.2, top=None, bottom=None, right=None)

    if save:
        save_path = save_dir + f"clamp/clamp_{metric_groups[k]}_score_box_plot.pdf"
        plt.savefig(save_path)
plt.show()


# group mean score bar plot
for k in range(len(metric_groups)):

    overall_scores = []
    for i in range(len(titles4)):
        if metric_groups[k] == 'Clustering Performance':
            score = np.mean(cluster_stacked[i], axis=1)
        elif metric_groups[k] == 'Representation Quality':
            score = np.mean(bio_conserve_stacked[i], axis=1)
        elif metric_groups[k] == 'Batch Correction':
            score = np.mean(batch_corr_stacked[i], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(4, 6))

    means = np.mean(overall_scores, axis=1)
    std_devs = np.std(overall_scores, axis=1)

    positions = np.arange(len(titles4))
    axs.bar(positions, means, yerr=std_devs, width=0.7, color=color_list_4, capsize=5)

    axs.set_xticks(positions)
    axs.set_xticklabels(labels4)
    axs.tick_params(axis='x', labelsize=12)
    axs.tick_params(axis='y', labelsize=12)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_linewidth(1)
    axs.spines['bottom'].set_linewidth(1)

    # if metric_groups[k] == 'Clustering Performance':
    #     axs.set_ylim(bottom=0.35)
    # elif metric_groups[k] == 'Biological Conservation':
    #     axs.set_ylim(bottom=0.55)
    # elif metric_groups[k] == 'Batch Correction':
    #     axs.set_ylim(bottom=0.70)

    for j, d in enumerate(overall_scores):
        y = np.random.normal(positions[j], 0.1, size=len(d))
        plt.scatter(y, d, alpha=1, color='white', s=20, zorder=1, edgecolors='black')

    axs.set_title(f'{metric_groups[k]}', fontsize=16)
    plt.gcf().subplots_adjust(left=0.2, top=None, bottom=None, right=None)

    if save:
        save_path = save_dir + f"clamp/clamp_{metric_groups[k]}_score_bar_plot.pdf"
        plt.savefig(save_path)
plt.show()


# overall score box plot
overall_scores = []
for i in range(len(titles4)):
    score = (np.mean(cluster_stacked[i], axis=1) +
             np.mean(bio_conserve_stacked[i], axis=1) +
             np.mean(batch_corr_stacked[i], axis=1)) / 3
    overall_scores.append(score)
overall_scores = np.array(overall_scores)

fig, axs = plt.subplots(figsize=(4, 6))

positions = np.arange(len(titles4))
for i, box in enumerate(axs.boxplot(overall_scores.T, positions=positions, patch_artist=True, labels=labels4,
                                    widths=0.7, showfliers=False, zorder=0)['boxes']):
    box.set_facecolor(color_list_4[i])

axs.set_xticklabels(labels4)
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
plt.gcf().subplots_adjust(left=0.2, top=None, bottom=None, right=None)

if save:
    save_path = save_dir + f"clamp/clamp_overall_score_box_plot.pdf"
    plt.savefig(save_path, dpi=300)
plt.show()
