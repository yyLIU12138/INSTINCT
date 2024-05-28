import pickle
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

save = False

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

save_dir = f'../../results/simulated/'

slice_index_list = [str(i) for i in range(len(slice_name_list))]
num_iters = 8
num_clusters = 5

models = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']
legend = ['INSTINCT', 'Scanorama', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST']
color_list = ['darkviolet', 'chocolate', 'sandybrown', 'peachpuff', 'darkslategray', 'c', 'cyan']
metric_groups = ['Clustering Performance', 'Representation Quality', 'Batch Correction']


# load results
cluster_mean = []
bio_conserve_mean = []
batch_corr_mean = []
overall_mean = []
cluster_median = []
bio_conserve_median = []
batch_corr_median = []
overall_median = []

for scenario in range(1, 4):

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

        with open(save_dir + f'scenario_{scenario}/T_{name_concat}/comparison/'
                             f'{model_name}/{model_name}_results_dict.pkl', 'rb') as file:
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
    stacked_matrix = np.stack([aris, amis, nmis, fmis, comps, homos, maps,
                               c_asws, b_asws, b_pcrs, kbets, g_conns], axis=-1)

    cluster_mean.append(np.mean(np.mean(cluster_stacked, axis=-1), axis=-1))
    bio_conserve_mean.append(np.mean(np.mean(bio_conserve_stacked, axis=-1), axis=-1))
    batch_corr_mean.append(np.mean(np.mean(batch_corr_stacked, axis=-1), axis=-1))
    overall_mean.append(np.mean((np.mean(cluster_stacked, axis=-1) +
                                 np.mean(bio_conserve_stacked, axis=-1) +
                                 np.mean(batch_corr_stacked, axis=-1)) / 3, axis=-1))

    cluster_median.append(np.median(np.mean(cluster_stacked, axis=-1), axis=-1))
    bio_conserve_median.append(np.median(np.mean(bio_conserve_stacked, axis=-1), axis=-1))
    batch_corr_median.append(np.median(np.mean(batch_corr_stacked, axis=-1), axis=-1))
    overall_median.append(np.median((np.mean(cluster_stacked, axis=-1) +
                                     np.mean(bio_conserve_stacked, axis=-1) +
                                     np.mean(batch_corr_stacked, axis=-1)) / 3, axis=-1))

cluster_mean = np.array(cluster_mean)
bio_conserve_mean = np.array(bio_conserve_mean)
batch_corr_mean = np.array(batch_corr_mean)
overall_mean = np.array(overall_mean)
cluster_median = np.array(cluster_median)
bio_conserve_median = np.array(bio_conserve_median)
batch_corr_median = np.array(batch_corr_median)
overall_median = np.array(overall_median)

summary = [cluster_mean, bio_conserve_mean, batch_corr_mean, overall_mean,
           cluster_median, bio_conserve_median, batch_corr_median, overall_median]
titles = ['Clustering Performance', 'Representation Quality', 'Batch Correction', 'Overall',
          'Clustering Performance', 'Representation Quality', 'Batch Correction', 'Overall']
mode_list = ['mean', 'mean', 'mean', 'mean', 'median', 'median', 'median', 'median']

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
        save_path = save_dir + f"{mode_list[j]}_scenario_1to3_line_chart_{titles[j]}.pdf"
        plt.savefig(save_path)
    plt.show()
