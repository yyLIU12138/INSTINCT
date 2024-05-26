import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

save_dir = '../../results/MouseBrain_Jiang2023/'
save = False

models = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']
labels1 = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']
color_list_1 = ['darkviolet', 'chocolate', 'sandybrown', 'peachpuff', 'darkslategray', 'c', 'cyan']

input_types = ['frag', 'bina', 'read']
labels2 = ['Fragment Count', 'Binary', 'Read Count']
color_list_2 = ['lawngreen', 'yellow', 'dodgerblue']

metric_list = ['ARI', 'AMI', 'NMI', 'FMI', 'Comp', 'Homo', 'mAP', 'Spot-type\nASW', 'Isolated\nlabel ASW',
               'Isolated\nlabel F1', 'Batch\nASW', 'Batch\nPCR', 'kBET', 'Graph\nconnectivity']
metric_groups = ['Clustering Performance', 'Representation Quality', 'Batch Correction']

# comparison between methods
aris, amis, nmis, fmis, comps, homos, maps, c_asws = [], [], [], [], [], [], [], []
i_asws, i_f1s, b_asws, b_pcrs, kbets, g_conns = [], [], [], [], [], []

for model in models:

    with open(save_dir + f'comparison/{model}/{model}_results_dict.pkl', 'rb') as file:
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

axs.set_xticks([3 + 8 * i for i in range(6)])
axs.set_xticklabels(metric_list[0:6], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
           for i, label in enumerate(labels1)]
axs.legend(handles=handles, loc=(1, 0.55), fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.85)

if save:
    save_path = save_dir + f"comparison/methods_clustering_performance_separate_scores_box_plot.pdf"
    plt.savefig(save_path, dpi=500)
# plt.show()

fig, axs = plt.subplots(figsize=(8, 4))

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

axs.set_xticks([3 + 8 * i for i in range(4)])
axs.set_xticklabels(metric_list[6:10], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

# handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
#            for i, label in enumerate(labels1)]
# axs.legend(handles=handles, loc='lower left', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"comparison/methods_representation_quality_separate_scores_box_plot.pdf"
    plt.savefig(save_path, dpi=500)
# plt.show()

fig, axs = plt.subplots(figsize=(8, 4))

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

axs.set_xticks([3 + 8 * i for i in range(4)])
axs.set_xticklabels(metric_list[10:14], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

# handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
#            for i, label in enumerate(labels1)]
# axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"comparison/methods_batch_correction_separate_scores_box_plot.pdf"
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

axs.set_xticks([3 + 8 * i for i in range(6)])
axs.set_xticklabels(metric_list[0:6], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
           for i, label in enumerate(labels1)]
axs.legend(handles=handles, loc=(1, 0.55), fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.85)

if save:
    save_path = save_dir + f"comparison/methods_clustering_performance_separate_scores_bar_plot.pdf"
    plt.savefig(save_path, dpi=500)
# plt.show()

fig, axs = plt.subplots(figsize=(8, 4))

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

axs.set_xticks([3 + 8 * i for i in range(4)])
axs.set_xticklabels(metric_list[6:10], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

# handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
#            for i, label in enumerate(labels1)]
# axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"comparison/methods_representation_quality_separate_scores_bar_plot.pdf"
    plt.savefig(save_path, dpi=500)
# plt.show()

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

axs.set_xticks([3 + 8 * i for i in range(4)])
axs.set_xticklabels(metric_list[10:14], fontsize=12)
axs.tick_params(axis='y', labelsize=12)

# handles = [plt.Rectangle((0, 0), 1, 1, color=color_list_1[i], label=label)
#            for i, label in enumerate(labels1)]
# axs.legend(handles=handles, loc='lower right', fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=None)

if save:
    save_path = save_dir + f"comparison/methods_batch_correction_separate_scores_bar_plot.pdf"
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

    fig, axs = plt.subplots(figsize=(6, 6))

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
        save_path = save_dir + f"comparison/methods_{metric_groups[k]}_score_box_plot.pdf"
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

    fig, axs = plt.subplots(figsize=(6, 6))

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
        save_path = save_dir + f"comparison/methods_{metric_groups[k]}_score_bar_plot.pdf"
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

fig, axs = plt.subplots(figsize=(6, 6))

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
    save_path = save_dir + f"comparison/methods_overall_score_box_plot.pdf"
    plt.savefig(save_path, dpi=300)
plt.show()


# separate score rank plot
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
            overview_matrix[:, 7:11] = np.mean(bio_conserve_stacked, axis=1)
            overview_matrix[:, 11] = np.mean(np.mean(bio_conserve_stacked, axis=-1), axis=-1)
            overview_matrix[:, 12:16] = np.mean(batch_corr_stacked, axis=1)
            overview_matrix[:, 16] = np.mean(np.mean(batch_corr_stacked, axis=-1), axis=-1)
            overview_matrix[:, 17] = np.mean((np.mean(cluster_stacked, axis=-1) +
                                              np.mean(bio_conserve_stacked, axis=-1) +
                                              np.mean(batch_corr_stacked, axis=-1)) / 3, axis=-1)
        elif type_list[k] == 'median':
            overview_matrix[:, 0:6] = np.median(cluster_stacked, axis=1)
            overview_matrix[:, 6] = np.median(np.mean(cluster_stacked, axis=-1), axis=-1)
            overview_matrix[:, 7:11] = np.median(bio_conserve_stacked, axis=1)
            overview_matrix[:, 11] = np.median(np.mean(bio_conserve_stacked, axis=-1), axis=-1)
            overview_matrix[:, 12:16] = np.median(batch_corr_stacked, axis=1)
            overview_matrix[:, 16] = np.median(np.mean(batch_corr_stacked, axis=-1), axis=-1)
            overview_matrix[:, 17] = np.median((np.mean(cluster_stacked, axis=-1) +
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

    fig, axs = plt.subplots(figsize=(10, 6.5))
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
                        fontdict={'size': 11})
    for i in range(len(metric_list)):
        text = plt.text(i, 6.8, metric_list[i], ha='center', va='center', rotation=30,
                        fontdict={'size': 9})

    plt.ylim(-1.5, 8.5)
    plt.gca().invert_yaxis()
    plt.gcf().subplots_adjust(left=None, top=0.9, bottom=0.1, right=None)

    if save:
        save_path = save_dir + f"comparison/methods_{type_list[k]}_separate_scores_rank_plot.pdf"
        plt.savefig(save_path, dpi=500)
    plt.show()


# comparison between input types
aris, amis, nmis, fmis, comps, homos, maps, c_asws,  = [], [], [], [], [], [], [], []
i_asws, i_f1s, b_asws, b_pcrs, kbets, g_conns = [], [], [], [], [], []

for type in input_types:

    with open(save_dir + f'comparison/INSTINCT_{type}/INSTINCT_{type}_results_dict.pkl', 'rb') as file:
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
fig, axs = plt.subplots(figsize=(8, 4))

for i, input_type in enumerate(input_types):

    positions = np.arange(6) * (len(input_types) + 1) + i
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
    save_path = save_dir + f"comparison/input_types_clustering_performance_separate_scores_box_plot.pdf"
    plt.savefig(save_path, dpi=500)
plt.show()

fig, axs = plt.subplots(figsize=(8, 4))

for i, input_type in enumerate(input_types):

    positions = np.arange(4) * (len(input_types) + 1) + i
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
    save_path = save_dir + f"comparison/input_types_representation_quality_separate_scores_box_plot.pdf"
    plt.savefig(save_path, dpi=500)
plt.show()

fig, axs = plt.subplots(figsize=(8, 4))

for i, input_type in enumerate(input_types):

    positions = np.arange(4) * (len(input_types) + 1) + i
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
    save_path = save_dir + f"comparison/input_types_batch_correction_separate_scores_box_plot.pdf"
    plt.savefig(save_path, dpi=500)
plt.show()


# separate scores bar plot
fig, axs = plt.subplots(figsize=(8, 4))

for i, input_type in enumerate(input_types):

    means = np.mean(cluster_stacked[i], axis=0)
    std_devs = np.std(cluster_stacked[i], axis=0)

    positions = np.arange(6) * (len(input_types) + 1) + i
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
    save_path = save_dir + f"comparison/input_types_clustering_performance_separate_scores_bar_plot.pdf"
    plt.savefig(save_path, dpi=500)
plt.show()

fig, axs = plt.subplots(figsize=(8, 4))

for i, input_type in enumerate(input_types):

    means = np.mean(bio_conserve_stacked[i], axis=0)
    std_devs = np.std(bio_conserve_stacked[i], axis=0)

    positions = np.arange(4) * (len(input_types) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_2[i], capsize=5)

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
    save_path = save_dir + f"comparison/input_types_representation_quality_separate_scores_bar_plot.pdf"
    plt.savefig(save_path, dpi=500)
plt.show()

fig, axs = plt.subplots(figsize=(8, 4))

for i, input_type in enumerate(input_types):

    means = np.mean(batch_corr_stacked[i], axis=0)
    std_devs = np.std(batch_corr_stacked[i], axis=0)

    positions = np.arange(4) * (len(input_types) + 1) + i
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
    save_path = save_dir + f"comparison/input_types_batch_correction_separate_scores_bar_plot.pdf"
    plt.savefig(save_path, dpi=500)
plt.show()


# group mean score box plot
for k in range(len(metric_groups)):

    overall_scores = []
    for i in range(len(input_types)):
        if metric_groups[k] == 'Clustering Performance':
            score = np.mean(cluster_stacked[i], axis=1)
        elif metric_groups[k] == 'Representation Quality':
            score = np.mean(bio_conserve_stacked[i], axis=1)
        elif metric_groups[k] == 'Batch Correction':
            score = np.mean(batch_corr_stacked[i], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(4.5, 6))

    positions = np.arange(len(input_types))
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
    plt.gcf().subplots_adjust(left=0.15, top=None, bottom=None, right=None)

    if save:
        save_path = save_dir + f"comparison/input_types_{metric_groups[k]}_score_box_plot.pdf"
        plt.savefig(save_path, dpi=300)
plt.show()


# group mean score bar plot
for k in range(len(metric_groups)):

    overall_scores = []
    for i in range(len(input_types)):
        if metric_groups[k] == 'Clustering Performance':
            score = np.mean(cluster_stacked[i], axis=1)
        elif metric_groups[k] == 'Representation Quality':
            score = np.mean(bio_conserve_stacked[i], axis=1)
        elif metric_groups[k] == 'Batch Correction':
            score = np.mean(batch_corr_stacked[i], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(4.5, 6))

    means = np.mean(overall_scores, axis=1)
    std_devs = np.std(overall_scores, axis=1)

    positions = np.arange(len(input_types))
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
    #     axs.set_ylim(bottom=0.2)
    # elif metric_groups[k] == 'Biological Conservation':
    #     axs.set_ylim(bottom=0.5)
    # elif metric_groups[k] == 'Batch Correction':
    #     axs.set_ylim(bottom=0.3)

    for j, d in enumerate(overall_scores):
        y = np.random.normal(positions[j], 0.1, size=len(d))
        plt.scatter(y, d, alpha=1, color='white', s=20, zorder=1, edgecolors='black')

    axs.set_title(f'{metric_groups[k]}', fontsize=16)
    plt.gcf().subplots_adjust(left=0.15, top=None, bottom=None, right=None)

    if save:
        save_path = save_dir + f"comparison/input_types_{metric_groups[k]}_score_bar_plot.pdf"
        plt.savefig(save_path, dpi=300)
plt.show()


# overall score box plot
overall_scores = []
for i in range(len(input_types)):
    score = (np.mean(cluster_stacked[i], axis=1) +
             np.mean(bio_conserve_stacked[i], axis=1) +
             np.mean(batch_corr_stacked[i], axis=1)) / 3
    overall_scores.append(score)
overall_scores = np.array(overall_scores)

fig, axs = plt.subplots(figsize=(4.5, 6))

positions = np.arange(len(input_types))
for i, box in enumerate(axs.boxplot(overall_scores.T, positions=positions, patch_artist=True, labels=labels2,
                                    widths=0.8, showfliers=False, zorder=0)['boxes']):
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
plt.gcf().subplots_adjust(left=0.15, top=None, bottom=None, right=None)

if save:
    save_path = save_dir + f"comparison/input_types_overall_score_box_plot.pdf"
    plt.savefig(save_path, dpi=300)
plt.show()













