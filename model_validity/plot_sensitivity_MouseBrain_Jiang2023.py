import pickle
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

save = False
file_format = 'png'

# mouse brain
num_iters = 8
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']
cls_list = ['Primary_brain_1', 'Primary_brain_2', 'Midbrain',  'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain',
            'Subpallium_1', 'Subpallium_2', 'Cartilage_1', 'Cartilage_2', 'Cartilage_3', 'Cartilage_4',
            'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv']

save_dir = '../../results/model_validity/MouseBrain_Jiang2023/sensitivity/'

titles = ['clamp_margin', 'training_epoch', 'filter_rate', 'losses_hyper', 'k_neighbors', 'radius']
parameters_dict = {
    'clamp_margin': [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'training_epoch': [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'filter_rate': [0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30],#[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    'losses_hyper': [1, 5, 10, 15, 20, 25, 30],
    'k_neighbors': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'radius': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
}
file_name_dict = {
    'clamp_margin': ['margin'],
    'training_epoch': ['stage1', 'stage2'],
    'filter_rate': ['min_cells_rate'],
    'losses_hyper': ['lambda_cls', 'lambda_la', 'lambda_rec'],
    'k_neighbors': ['k'],
    'radius': ['rad_coef']
}
color_dict = {
    'clamp_margin': ['wheat'],
    'training_epoch': ['violet', 'plum'],
    'filter_rate': ['lightgreen'],
    'losses_hyper': ['dodgerblue', 'deepskyblue', 'cyan'],
    'k_neighbors': ['greenyellow'],
    'radius': ['peachpuff']
}

metric_groups = ['Clustering Performance', 'Representation Quality', 'Batch Correction']

for i, title in enumerate(titles):

    tmp_dir = save_dir + f'{title}/'
    parameter_list = parameters_dict[title]
    file_name_list = file_name_dict[title]
    color_list = color_dict[title]

    for j, file_name in enumerate(file_name_list):

        with open(f'{tmp_dir}/{file_name}_results_dict.pkl', 'rb') as file:
            results_dict = pickle.load(file)

        aris = results_dict['ARIs'][:, 0:num_iters]
        amis = results_dict['AMIs'][:, 0:num_iters]
        nmis = results_dict['NMIs'][:, 0:num_iters]
        fmis = results_dict['FMIs'][:, 0:num_iters]
        comps = results_dict['COMPs'][:, 0:num_iters]
        homos = results_dict['HOMOs'][:, 0:num_iters]
        maps = results_dict['mAPs'][:, 0:num_iters]
        c_asws = results_dict['Cell_type_ASWs'][:, 0:num_iters]
        i_asws = results_dict['Isolated_label_ASWs'][:, 0:num_iters]
        i_f1s = results_dict['Isolated_label_F1s'][:, 0:num_iters]
        b_asws = results_dict['Batch_ASWs'][:, 0:num_iters]
        b_pcrs = results_dict['Batch_PCRs'][:, 0:num_iters]
        kbets = results_dict['kBETs'][:, 0:num_iters]
        g_conns = results_dict['Graph_connectivities'][:, 0:num_iters]

        cluster_stacked = np.stack([aris, amis, nmis, fmis, comps, homos], axis=-1)
        bio_conserve_stacked = np.stack([maps, c_asws, i_asws, i_f1s], axis=-1)
        batch_corr_stacked = np.stack([b_asws, b_pcrs, kbets, g_conns], axis=-1)
        stacked_matrix = np.stack([aris, amis, nmis, fmis, comps, homos, maps,
                                   c_asws, i_asws, i_f1s, b_asws, b_pcrs, kbets, g_conns], axis=-1)

        cluster_scores = np.mean(cluster_stacked, axis=-1)
        bio_conserve_scores = np.mean(bio_conserve_stacked, axis=-1)
        batch_corr_scores = np.mean(batch_corr_stacked, axis=-1)
        stacked_scores = (np.mean(cluster_stacked, axis=-1) +
                          np.mean(bio_conserve_stacked, axis=-1) +
                          np.mean(batch_corr_stacked, axis=-1)) / 3

        cluster_mean = np.mean(cluster_scores, axis=-1)
        bio_conserve_mean = np.mean(bio_conserve_scores, axis=-1)
        batch_corr_mean = np.mean(batch_corr_scores, axis=-1)
        overall_mean = np.mean(stacked_scores, axis=-1)

        cluster_median = np.median(cluster_scores, axis=-1)
        bio_conserve_median = np.median(bio_conserve_scores, axis=-1)
        batch_corr_median = np.median(batch_corr_scores, axis=-1)
        overall_median = np.median(stacked_scores, axis=-1)

        summary = [cluster_mean, bio_conserve_mean, batch_corr_mean, overall_mean,
                   cluster_median, bio_conserve_median, batch_corr_median, overall_median]
        summary_scores = [cluster_scores, bio_conserve_scores, batch_corr_scores, stacked_scores,
                          cluster_scores, bio_conserve_scores, batch_corr_scores, stacked_scores]
        title_list = ['Clustering Performance', 'Representation Quality', 'Batch Correction', 'Overall',
                      'Clustering Performance', 'Representation Quality', 'Batch Correction', 'Overall']
        mode_list = ['mean', 'mean', 'mean', 'mean', 'median', 'median', 'median', 'median']

        if title == 'clamp_margin':
            parameter_list = parameter_list[1:]

        for k in range(len(summary)):

            if title == 'clamp_margin':
                mtx = summary[k][1:]
                scores_mtx = summary_scores[k][1:]
            else:
                mtx = summary[k]
                scores_mtx = summary_scores[k]

            fig, ax = plt.subplots(figsize=(7, 4))

            if mode_list[k] == 'median':

                if title == 'filter_rate' or title == 'radius':
                    positions = [p*100 for p in parameter_list]
                else:
                    positions = parameter_list
                widths = (positions[-1] - positions[-2]) * 0.15
                boxplot = ax.boxplot(scores_mtx.T, positions=positions, patch_artist=True, widths=widths, showfliers=False, zorder=1)

                for patch in boxplot['boxes']:
                    patch.set_facecolor(color_list[j])

                ax.plot(positions, mtx, marker='o', markersize=6, label='median')

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1)
                ax.spines['bottom'].set_linewidth(1)

                ax.set_title(f'{title_list[k]} ({file_name})', fontsize=12)
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.legend()

                ax.set_xticks(positions)
                if title == 'filter_rate':
                    with open(f'{tmp_dir}/n_features.txt', 'r') as file:
                        data_list = [int(line.strip()) for line in file]
                    labels = [f'{str(p)}\n({num})' for p, num in zip(parameter_list, data_list)]
                elif title == 'radius':
                    with open(f'{tmp_dir}/n_neighbors.txt', 'r') as file:
                        data_list = [list(map(float, line.strip().split())) for line in file]
                    data_list = [(np.round(1258 * data_list[n][0]) +
                                  np.round(1777 * data_list[n][1]) +
                                  np.round(1949 * data_list[n][2]) +
                                  np.round(2129 * data_list[n][3])) / (1258+1777+1949+2129) for n in range(len(data_list))]
                    labels = [f'{str(p)}\n({num:.2f})' for p, num in zip(parameter_list, data_list)]
                else:
                    labels = [str(p) for p in parameter_list]
                ax.set_xticklabels(labels, rotation=0)
                # plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.9)

            else:

                if title == 'filter_rate' or title == 'radius':
                    positions = [p*100 for p in parameter_list]
                else:
                    positions = parameter_list

                means = np.mean(scores_mtx, axis=-1)
                std_err = np.std(scores_mtx, axis=-1)

                ax.errorbar(positions, means, yerr=std_err, fmt='-o', markersize=6, label='mean')

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1)
                ax.spines['bottom'].set_linewidth(1)

                ax.set_title(f'{title_list[k]} ({file_name})', fontsize=12)
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.legend()

                ax.set_xticks(positions)
                if title == 'filter_rate':
                    with open(f'{tmp_dir}/n_features.txt', 'r') as file:
                        data_list = [int(line.strip()) for line in file]
                    labels = [f'{str(p)}\n({num})' for p, num in zip(parameter_list, data_list)]
                elif title == 'radius':
                    with open(f'{tmp_dir}/n_neighbors.txt', 'r') as file:
                        data_list = [list(map(float, line.strip().split())) for line in file]
                    data_list = [(np.round(1258*data_list[n][0]) +
                                  np.round(1777*data_list[n][1]) +
                                  np.round(1949*data_list[n][2]) +
                                  np.round(2129*data_list[n][3])) / (1258+1777+1949+2129) for n in range(len(data_list))]
                    labels = [f'{str(p)}\n({num:.2f})' for p, num in zip(parameter_list, data_list)]
                else:
                    labels = [str(p) for p in parameter_list]
                ax.set_xticklabels(labels, rotation=0)
                # plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.9)

            if save:
                save_path = tmp_dir + f"{mode_list[k]}_{file_name}_{title_list[k]}.{file_format}"
                plt.savefig(save_path)
        plt.show()


