import pickle
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1234)

save = False

scenario = 1
file_format = 'pdf'

slice_name_list = [
    'Tech_0_0_Bio_0_0.5',
    'Tech_0_0.1_Bio_0_0',
    'Tech_0_0.1_Bio_0_0.5',
             ]

slice_index_list = list(range(len(slice_name_list)))

name_concat = slice_name_list[0]
for mode in slice_name_list[1:]:
    name_concat = name_concat + '_' + mode

save_dir = f'../../results/simulated/scenario_{scenario}/single/'

slice_index_list = [str(i) for i in range(len(slice_name_list))]
num_iters = 8
num_clusters = 5

models = ['INSTINCT', 'SCALE', 'STAGATE']
labels1 = ['INSTINCT', 'SCALE', 'STAGATE']
color_list_1 = ['darkviolet', 'darkgoldenrod', 'steelblue']
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

    with open(save_dir + f'{model_name}/{model_name}_results_dict.pkl', 'rb') as file:
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

axs.set_title('Clustering Performance (Joint)', fontsize=16)

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
    save_path = save_dir + f"methods_clustering_performance_separate_scores_box_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
plt.show()

fig, axs = plt.subplots(figsize=(6, 4))

for i, model in enumerate(models):

    positions = np.arange(2) * (len(models) + 1) + i
    boxplot = axs.boxplot(bio_conserve_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[6:8], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_1[i])

plt.title('Representation Quality (Joint)', fontsize=16)

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
    save_path = save_dir + f"methods_representation_quality_separate_scores_box_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
plt.show()

fig, axs = plt.subplots(figsize=(8, 4))

for i, model in enumerate(models):

    positions = np.arange(4) * (len(models) + 1) + i
    boxplot = axs.boxplot(batch_corr_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                          labels=metric_list[8:12], showfliers=False, zorder=1)

    for patch in boxplot['boxes']:
        patch.set_facecolor(color_list_1[i])

plt.title('Batch Correction (Joint)', fontsize=16)

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
    save_path = save_dir + f"methods_batch_correction_separate_scores_box_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
plt.show()


# separate scores bar plot
fig, axs = plt.subplots(figsize=(10, 4))

for i, model in enumerate(models):

    means = np.mean(cluster_stacked[i], axis=0)
    std_devs = np.std(cluster_stacked[i], axis=0)

    positions = np.arange(6) * (len(models) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_1[i], capsize=5)

axs.set_title('Clustering Performance (Joint)', fontsize=16)

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
plt.show()

fig, axs = plt.subplots(figsize=(6, 4))

for i, model in enumerate(models):

    means = np.mean(bio_conserve_stacked[i], axis=0)
    std_devs = np.std(bio_conserve_stacked[i], axis=0)

    positions = np.arange(2) * (len(models) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_1[i], capsize=5)

plt.title('Representation Quality (Joint)', fontsize=16)

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
    save_path = save_dir + f"methods_representation_quality_separate_scores_bar_plot.{file_format}"
    plt.savefig(save_path, dpi=500)
plt.show()

fig, axs = plt.subplots(figsize=(8, 4))

for i, model in enumerate(models):

    means = np.mean(batch_corr_stacked[i], axis=0)
    std_devs = np.std(batch_corr_stacked[i], axis=0)

    positions = np.arange(4) * (len(models) + 1) + i
    axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_1[i], capsize=5)

plt.title('Batch Correction (Joint)', fontsize=16)

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
    save_path = save_dir + f"methods_batch_correction_separate_scores_bar_plot.{file_format}"
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

    fig, axs = plt.subplots(figsize=(4, 6))

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

    axs.set_title(f'{metric_groups[k]} (Joint)', fontsize=16)

    if save:
        save_path = save_dir + f"methods_{metric_groups[k]}_score_box_plot.{file_format}"
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

    fig, axs = plt.subplots(figsize=(4, 6))

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

    axs.set_title(f'{metric_groups[k]} (Joint)', fontsize=16)

    if save:
        save_path = save_dir + f"methods_{metric_groups[k]}_score_bar_plot.{file_format}"
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

fig, axs = plt.subplots(figsize=(4, 6))

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

axs.set_title(f'Overall Score (Joint)', fontsize=16)

if save:
    save_path = save_dir + f"methods_overall_score_box_plot.{file_format}"
    plt.savefig(save_path, dpi=300)
plt.show()


metric_list = ['ARI', 'AMI', 'NMI', 'FMI', 'Comp', 'Homo', 'mAP', 'Spot-type ASW']
metric_groups = ['Clustering Performance', 'Representation Quality']

for idx, slice_name in enumerate(slice_name_list):

    # load results
    aris = []
    amis = []
    nmis = []
    fmis = []
    comps = []
    homos = []
    maps = []
    c_asws = []

    for model_name in models:

        with open(save_dir + f'{model_name}/{model_name}_results_dict_{slice_name}.pkl', 'rb') as file:
            results_dict = pickle.load(file)

        aris.append(results_dict['ARIs'])
        amis.append(results_dict['AMIs'])
        nmis.append(results_dict['NMIs'])
        fmis.append(results_dict['FMIs'])
        comps.append(results_dict['COMPs'])
        homos.append(results_dict['HOMOs'])
        maps.append(results_dict['mAPs'])
        c_asws.append(results_dict['Cell_type_ASWs'])

    aris = np.array(aris)
    amis = np.array(amis)
    nmis = np.array(nmis)
    fmis = np.array(fmis)
    comps = np.array(comps)
    homos = np.array(homos)
    maps = np.array(maps)
    c_asws = np.array(c_asws)

    cluster_stacked = np.stack([aris, amis, nmis, fmis, comps, homos], axis=-1)
    bio_conserve_stacked = np.stack([maps, c_asws], axis=-1)
    stacked_matrix = np.stack([aris, amis, nmis, fmis, comps, homos, maps, c_asws], axis=-1)


    # separate scores box plot
    fig, axs = plt.subplots(figsize=(10, 4))

    for i, model in enumerate(models):

        positions = np.arange(6) * (len(models) + 1) + i
        boxplot = axs.boxplot(cluster_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                              labels=metric_list[0:6], showfliers=False, zorder=1)

        for patch in boxplot['boxes']:
            patch.set_facecolor(color_list_1[i])

    axs.set_title(f'Clustering Performance (Slice {idx})', fontsize=16)

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
        save_path = save_dir + f"{idx}_methods_clustering_performance_separate_scores_box_plot.{file_format}"
        plt.savefig(save_path, dpi=500)
    plt.show()

    fig, axs = plt.subplots(figsize=(6, 4))

    for i, model in enumerate(models):

        positions = np.arange(2) * (len(models) + 1) + i
        boxplot = axs.boxplot(bio_conserve_stacked[i], positions=positions, patch_artist=True, widths=0.75,
                              labels=metric_list[6:8], showfliers=False, zorder=1)

        for patch in boxplot['boxes']:
            patch.set_facecolor(color_list_1[i])

    plt.title(f'Representation Quality (Slice {idx})', fontsize=16)

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
        save_path = save_dir + f"{idx}_methods_representation_quality_separate_scores_box_plot.{file_format}"
        plt.savefig(save_path, dpi=500)
    plt.show()


    # separate scores bar plot
    fig, axs = plt.subplots(figsize=(10, 4))

    for i, model in enumerate(models):

        means = np.mean(cluster_stacked[i], axis=0)
        std_devs = np.std(cluster_stacked[i], axis=0)

        positions = np.arange(6) * (len(models) + 1) + i
        axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_1[i], capsize=5)

    axs.set_title(f'Clustering Performance (Slice {idx})', fontsize=16)

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
        save_path = save_dir + f"{idx}_methods_clustering_performance_separate_scores_bar_plot.{file_format}"
        plt.savefig(save_path, dpi=500)
    plt.show()

    fig, axs = plt.subplots(figsize=(6, 4))

    for i, model in enumerate(models):

        means = np.mean(bio_conserve_stacked[i], axis=0)
        std_devs = np.std(bio_conserve_stacked[i], axis=0)

        positions = np.arange(2) * (len(models) + 1) + i
        axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list_1[i], capsize=5)

    plt.title(f'Representation Quality (Slice {idx})', fontsize=16)

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
        save_path = save_dir + f"{idx}_methods_representation_quality_separate_scores_bar_plot.{file_format}"
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
            overall_scores.append(score)
        overall_scores = np.array(overall_scores)

        fig, axs = plt.subplots(figsize=(4, 6))

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

        axs.set_title(f'{metric_groups[k]} (Slice {idx})', fontsize=16)

        if save:
            save_path = save_dir + f"{idx}_methods_{metric_groups[k]}_score_box_plot.{file_format}"
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
            overall_scores.append(score)
        overall_scores = np.array(overall_scores)

        fig, axs = plt.subplots(figsize=(4, 6))

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

        axs.set_title(f'{metric_groups[k]} (Slice {idx})', fontsize=16)

        if save:
            save_path = save_dir + f"{idx}_methods_{metric_groups[k]}_score_bar_plot.{file_format}"
            plt.savefig(save_path, dpi=300)
    plt.show()


    # overall score box plot
    overall_scores = []
    for i in range(len(models)):
        score = (np.mean(cluster_stacked[i], axis=-1) +
                 np.mean(bio_conserve_stacked[i], axis=-1)) / 2
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(4, 6))

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

    axs.set_title(f'Overall Score (Slice {idx})', fontsize=16)

    if save:
        save_path = save_dir + f"{idx}_methods_overall_score_box_plot.{file_format}"
        plt.savefig(save_path, dpi=300)
    plt.show()