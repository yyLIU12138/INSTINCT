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

save_dir = f'../../results/simulated/comparison_with_SCALEX/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

slice_index_list = [str(i) for i in range(len(slice_name_list))]
num_iters = 8
num_clusters = 5

models = ['INSTINCT', 'SCALEX']
legend = ['INSTINCT', 'SCALEX']
color_list = ['darkviolet', 'sandybrown']
metric_groups = ['Batch Correction']


# load results
B_ASW = []
B_PCR = []
kBET = []
G_conn = []
batch_corr = []

for scenario in range(1, 4):

    b_asws = []
    b_pcrs = []
    kbets = []
    g_conns = []

    for model_name in models:

        with open(f'../../results/simulated/scenario_{scenario}/T_{name_concat}/comparison/'
                  f'{model_name}/{model_name}_results_dict.pkl', 'rb') as file:
            results_dict = pickle.load(file)

        b_asws.append(results_dict['Batch_ASWs'])
        b_pcrs.append(results_dict['Batch_PCRs'])
        kbets.append(results_dict['kBETs'])
        g_conns.append(results_dict['Graph_connectivities'])

    b_asws = np.array(b_asws)
    b_pcrs = np.array(b_pcrs)
    kbets = np.array(kbets)
    g_conns = np.array(g_conns)
    batch_corr_stacked = np.stack([b_asws, b_pcrs, kbets, g_conns], axis=-1)

    B_ASW.append(np.median(b_asws, axis=-1))
    B_PCR.append(np.median(b_pcrs, axis=-1))
    kBET.append(np.median(kbets, axis=-1))
    G_conn.append(np.median(g_conns, axis=-1))
    batch_corr.append(np.median(np.mean(batch_corr_stacked, axis=-1), axis=-1))

B_ASW = np.array(B_ASW)
B_PCR = np.array(B_PCR)
kBET = np.array(kBET)
G_conn = np.array(G_conn)
batch_corr = np.array(batch_corr)

summary = [B_ASW, B_PCR, kBET, G_conn, batch_corr]
titles = ['Batch ASW', 'Batch PCR', 'kBET', 'Graph connectivity', 'Batch Correction']

for j in range(len(summary)):

    mtx = summary[j]

    fig, ax = plt.subplots(figsize=(7, 3))

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
    plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.77)

    if save:
        save_path = save_dir + f"scenario_1to3_line_chart_{titles[j]}_with_SCALEX.{file_format}"
        plt.savefig(save_path)
plt.show()


models = ['INSTINCT', 'SCALEX']
legend = ['INSTINCT', 'INSTINCT_del', 'SCALEX', 'SCALEX_del']
color_list = ['darkviolet', 'darkviolet', 'sandybrown', 'sandybrown']
metric_groups = ['Batch Correction']


# load results
B_ASW = []
B_PCR = []
kBET = []
G_conn = []
batch_corr = []

for scenario in range(1, 4):

    b_asws = []
    b_pcrs = []
    kbets = []
    g_conns = []

    for model_name in models:

        with open(f'../../results/simulated/scenario_{scenario}/T_{name_concat}/comparison/'
                  f'{model_name}/{model_name}_results_dict.pkl', 'rb') as file:
            results_dict = pickle.load(file)

        b_asws.append(results_dict['Batch_ASWs'])
        b_pcrs.append(results_dict['Batch_PCRs'])
        kbets.append(results_dict['kBETs'])
        g_conns.append(results_dict['Graph_connectivities'])

        with open(f'../../results/simulated/scenario_{scenario}/T_{name_concat}/comparison/'
                  f'{model_name}/{model_name}_results_dict_batch_corr_del.pkl', 'rb') as file:
            results_dict = pickle.load(file)

        b_asws.append(results_dict['Batch_ASWs'])
        b_pcrs.append(results_dict['Batch_PCRs'])
        kbets.append(results_dict['kBETs'])
        g_conns.append(results_dict['Graph_connectivities'])

    b_asws = np.array(b_asws)
    b_pcrs = np.array(b_pcrs)
    kbets = np.array(kbets)
    g_conns = np.array(g_conns)
    batch_corr_stacked = np.stack([b_asws, b_pcrs, kbets, g_conns], axis=-1)

    B_ASW.append(np.median(b_asws, axis=-1))
    B_PCR.append(np.median(b_pcrs, axis=-1))
    kBET.append(np.median(kbets, axis=-1))
    G_conn.append(np.median(g_conns, axis=-1))
    batch_corr.append(np.median(np.mean(batch_corr_stacked, axis=-1), axis=-1))

B_ASW = np.array(B_ASW)
B_PCR = np.array(B_PCR)
kBET = np.array(kBET)
G_conn = np.array(G_conn)
batch_corr = np.array(batch_corr)

summary = [B_ASW, B_PCR, kBET, G_conn, batch_corr]
titles = ['Batch ASW', 'Batch PCR', 'kBET', 'Graph connectivity', 'Batch Correction']

for j in range(len(summary)):

    mtx = summary[j]

    fig, ax = plt.subplots(figsize=(7, 3))

    for i, model in enumerate(models):
        ax.plot(['Scenario 1', 'Scenario 2', 'Scenario 3'], mtx[:, 2*i], label=legend[2*i],
                color=color_list[2*i], marker='o', markersize=6)
        ax.plot(['Scenario 1', 'Scenario 2', 'Scenario 3'], mtx[:, 2*i+1], label=legend[2*i+1],
                color=color_list[2*i+1], marker='o', markersize=6, linestyle='--')

    ax.legend(loc=(1.05, 0.5), fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    ax.set_title(titles[j], fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.77)

    if save:
        save_path = save_dir + f"del_scenario_1to3_line_chart_{titles[j]}_with_SCALEX.{file_format}"
        plt.savefig(save_path)
plt.show()
