import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

save = False

file_format = 'png'

# scenario 1 to 3
slice_name_list = [
    'Tech_0_0_Bio_0_0.5',
    'Tech_0_0.1_Bio_0_0',
    'Tech_0_0.1_Bio_0_0.5',
             ]

slice_index_list = list(range(len(slice_name_list)))

name_concat = slice_name_list[0]
for mode in slice_name_list[1:]:
    name_concat = name_concat + '_' + mode

save_dir = f'../../results/simulated/annotation/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

slice_index_list = [str(i) for i in range(len(slice_name_list))]
num_iters = 8
num_clusters = 5

models = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']
legend = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']
color_list = ['darkviolet', 'chocolate', 'sandybrown', 'peachpuff', 'darkslategray', 'c', 'cyan']
metric_groups = ['Clustering Performance', 'Representation Quality', 'Batch Correction']


# load results
accus_mean = []
kappas_mean = []
mf1s_mean = []
wf1s_mean = []
overall_mean = []
accus_median = []
kappas_median = []
mf1s_median = []
wf1s_median = []
overall_median = []

for scenario in range(1, 4):

    accus, kappas, mf1s, wf1s = [], [], [], []

    for model_name in models:

        with open(f'../../results/simulated/scenario_{scenario}/T_{name_concat}/annotation/'
                  f'{model_name}/{model_name}_results_dict.pkl', 'rb') as file:
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

    accus_mean.append(np.mean(accus, axis=-1))
    kappas_mean.append(np.mean(kappas, axis=-1))
    mf1s_mean.append(np.mean(mf1s, axis=-1))
    wf1s_mean.append(np.mean(wf1s, axis=-1))
    overall_mean.append(np.mean(np.mean(stacked_matrix, axis=-1), axis=-1))

    accus_median.append(np.median(accus, axis=-1))
    kappas_median.append(np.median(kappas, axis=-1))
    mf1s_median.append(np.median(mf1s, axis=-1))
    wf1s_median.append(np.median(wf1s, axis=-1))
    overall_median.append(np.median(np.mean(stacked_matrix, axis=-1), axis=-1))

accus_mean = np.array(accus_mean)
kappas_mean = np.array(kappas_mean)
mf1s_mean = np.array(mf1s_mean)
wf1s_mean = np.array(wf1s_mean)
overall_mean = np.array(overall_mean)
accus_median = np.array(accus_median)
kappas_median = np.array(kappas_median)
mf1s_median = np.array(mf1s_median)
wf1s_median = np.array(wf1s_median)
overall_median = np.array(overall_median)

summary = [accus_mean, kappas_mean, mf1s_mean, wf1s_mean, overall_mean,
           accus_median, kappas_median, mf1s_median, wf1s_median, overall_median]
titles = ['Accuracy', 'Kappa', 'mF1', 'wF1', 'Overall',
          'Accuracy', 'Kappa', 'mF1', 'wF1', 'Overall']
mode_list = ['mean', 'mean', 'mean', 'mean', 'mean', 'median', 'median', 'median', 'median', 'median']

for j in range(len(summary)):

    mtx = summary[j]

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, model in enumerate(models):
        ax.plot(['Scenario 1', 'Scenario 2', 'Scenario 3'], mtx[:, i], label=legend[i],
                color=color_list[i], marker='o', markersize=6)

    ax.legend(loc=(1.05, 0.5), fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    ax.set_title(titles[j], fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.gcf().subplots_adjust(left=0.07, top=0.9, bottom=0.15, right=0.77)

    if save:
        save_path = save_dir + f"{mode_list[j]}_scenario_1to3_line_chart_{titles[j]}.{file_format}"
        plt.savefig(save_path)
    plt.show()
