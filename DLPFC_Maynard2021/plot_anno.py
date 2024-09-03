import pickle
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

data_dir = '../../data/STdata/10xVisium/DLPFC_Maynard2021/'
sample_group_list = [['151507', '151508', '151509', '151510'],
                     ['151669', '151670', '151671', '151672'],
                     ['151673', '151674', '151675', '151676']]
cls_list = ['Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'WM']
num_clusters_list = [7, 5, 7]

save_dir = '../../results/DLPFC_Maynard2021/'
save = False

file_format = 'pdf'

models = ['INSTINCT', 'INSTINCT_cas', 'SEDR', 'STAligner', 'GraphST']
labels = ['INSTINCT_RNA', 'INSTINCT_ATAC', 'SEDR', 'STAligner', 'GraphST']
color_list = ['darkviolet', 'violet', 'darkslategray', 'c', 'cyan']
metric_list = ['Accuracy', 'Kappa', 'mF1', 'wF1']

group_to_sample = {0: 'A', 1: 'B', 2: 'C'}

# cross validation scores
for idx in range(len(sample_group_list)):

    accus, kappas, mf1s, wf1s = [], [], [], []

    for model in models:

        with open(save_dir + f'annotation/{model}/{model}_group{idx}_results_dict.pkl', 'rb') as file:
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
    fig, axs = plt.subplots(figsize=(8, 3))

    for i, model in enumerate(models):

        positions = np.arange(4) * (len(models) + 1) + i
        boxplot = axs.boxplot(stacked_matrix[i], positions=positions, patch_artist=True, widths=0.75,
                              labels=metric_list, showfliers=False, zorder=1)

        for patch in boxplot['boxes']:
            patch.set_facecolor(color_list[i])

    plt.title(f'DLPFC (Sample {group_to_sample[idx]})', fontsize=12)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_linewidth(1)
    axs.spines['bottom'].set_linewidth(1)

    axs.set_xticks([(len(models)-1)/2 + (len(models)+1) * i for i in range(4)])
    axs.set_xticklabels(metric_list, fontsize=10)
    axs.tick_params(axis='y', labelsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_list[i], label=label)
               for i, label in enumerate(labels)]
    axs.legend(handles=handles, loc=(1.01, 0.35), fontsize=8)
    plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.83)

    if save:
        save_path = save_dir + f"annotation/group{idx}_cross_validation_separate_scores_box_plot.{file_format}"
        plt.savefig(save_path)
    plt.show()


    # separate bar
    fig, axs = plt.subplots(figsize=(8, 3))

    for i, model in enumerate(models):

        means = np.mean(stacked_matrix[i], axis=0)
        std_devs = np.std(stacked_matrix[i], axis=0)

        positions = np.arange(4) * (len(models) + 1) + i
        axs.bar(positions, means, yerr=std_devs, width=0.8, color=color_list[i], capsize=5)

    axs.set_title(f'DLPFC (Sample {group_to_sample[idx]})', fontsize=12)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_linewidth(1)
    axs.spines['bottom'].set_linewidth(1)

    axs.set_xticks([(len(models)-1)/2 + (len(models)+1) * i for i in range(4)])
    axs.set_xticklabels(metric_list, fontsize=10)
    axs.tick_params(axis='y', labelsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_list[i], label=label)
               for i, label in enumerate(labels)]
    axs.legend(handles=handles, loc=(1.01, 0.35), fontsize=8)
    plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.83)

    if save:
        save_path = save_dir + f"annotation/group{idx}_cross_validation_separate_scores_bar_plot.{file_format}"
        plt.savefig(save_path)
    plt.show()


    # overall
    overall_scores = []
    for i in range(len(models)):
        score = np.mean(stacked_matrix[i], axis=1)
        overall_scores.append(score)
    overall_scores = np.array(overall_scores)

    fig, axs = plt.subplots(figsize=(4.5, 6))

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

    axs.set_title(f'Overall Score (Sample {group_to_sample[idx]})', fontsize=16)
    plt.gcf().subplots_adjust(left=0.15, top=None, bottom=0.15, right=None)

    if save:
        save_path = save_dir + f"annotation/group{idx}_cross_validation_overall_score_box_plot.{file_format}"
        plt.savefig(save_path, dpi=300)
    plt.show()

