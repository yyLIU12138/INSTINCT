import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

save = False
file_format = 'pdf'

save_dir = '../../results/tests/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

models = ['INSTINCT', 'SCALEX', 'PeakVI', 'SEDR', 'STAligner', 'GraphST (PASTE on GPU)', 'GraphST (PASTE on CPU)']
colors_for_labels = ['darkviolet', 'sandybrown', 'peachpuff', 'darkslategray', 'c', 'cyan', 'lightskyblue']
labels = ['250', '500', '750', '1000', '1250', '1500', '1750', '2000']

time_cost = [[0+34/60, 0+47/60, 0+56/60, 1+5/60, 1+14/60, 1+22/60, 1+34/60, 1+46/60],
             [30+49/60, 37+36/60, 33+46/60, 35+58/60, 40+22/60, 45+38/60, 45+53/60, 44+17/60],
             [4+54/60, 9+31/60, 13+16/60, 32+16/60, 37+21/60, 46+45/60, 54+13/60, 63+0/60],
             [0+12/60, 0+19/60, 0+27/60, 0+32/60, 0+39/60, 0+45/60, 0+49/60, 0+58/60],
             [0+18/60, 0+27/60, 0+34/60, 0+41/60, 0+48/60, 0+55/60, 1+0/60, 1+7/60],
             [0+17/60, 0+25/60, 0+32/60, 0+41/60, 0+49/60, 1+0/60, 1+8/60, 1+17/60],
             [0+19/60, 0+26/60, 0+31/60, 0+45/60, 0+54/60, 1+6/60, 1+16/60, 1+22/60]]

memory_usage = [[913, 1029, 1183, 1431, 1771, 2075, 2489, 2971],
                [5853, 5855, 5861, 5863, 5867, 5875, 5853, 5863],
                [6801, 6801, 6798, 6801, 6798, 6801, 6798, 6801],
                [899, 925, 957, 1001, 1047, 1081, 1141, 1173],
                [547, 589, 617, 785, 899, 1017, 1175, 1267],
                [2479, 4105, 5733, 7337, 8973, 10519, 12205, 13841],
                [68, 141, 219, 302, 399, 507, 620, 739]]

time_cost = np.array(time_cost)

fig, ax = plt.subplots(figsize=(9, 4))

for i in range(len(models)):
    plt.plot(labels, time_cost[i], label=models[i], color=colors_for_labels[i], marker='o', markersize=4)

ax.legend(loc=(1.05, 0.5), fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

ax.set_title('Time Cost', fontsize=15)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.set_xlabel('Spots per slice', fontsize=12)
ax.set_ylabel('Time (min)', fontsize=12)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.8)
plt.tight_layout()

if save:
    save_path = save_dir + f"time_cost_7.{file_format}"
    plt.savefig(save_path)


memory_usage = np.array(memory_usage)

fig, ax = plt.subplots(figsize=(9, 4))

for i in range(len(models)):
    plt.plot(labels, memory_usage[i], label=models[i], color=colors_for_labels[i], marker='o', markersize=4)

ax.legend(loc=(1.05, 0.5), fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

ax.set_title('Memory Usage', fontsize=15)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.set_xlabel('Spots per slice', fontsize=12)
ax.set_ylabel('Memory (MiB)', fontsize=12)
plt.gcf().subplots_adjust(left=0.13, top=0.9, bottom=0.15, right=0.8)
plt.tight_layout()

if save:
    save_path = save_dir + f"memory_usage.{file_format}"
    plt.savefig(save_path)


indices = [0, 3, 4, 5, 6]
models = [models[i] for i in indices]
colors_for_labels = [colors_for_labels[i] for i in indices]
time_cost = time_cost[indices]
memory_usage = memory_usage[indices]


fig, ax = plt.subplots(figsize=(9, 4))

for i in range(len(models)):
    plt.plot(labels, time_cost[i], label=models[i], color=colors_for_labels[i], marker='o', markersize=4)

ax.legend(loc=(1.05, 0.5), fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

ax.set_title('Time Cost', fontsize=15)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.set_xlabel('Spots per slice', fontsize=12)
ax.set_ylabel('Time (min)', fontsize=12)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.8)
plt.tight_layout()

if save:
    save_path = save_dir + f"time_cost_5.{file_format}"
    plt.savefig(save_path)
plt.show()
