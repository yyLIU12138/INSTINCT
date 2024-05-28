import os
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

save = False
show = False
mode_index = 3
mode_list = ['E11_0', 'E13_5', 'E15_5', 'E18_5']
mode = mode_list[mode_index]
model = 'INSTINCT'
folder = 'S2_all'

read_path = f'../../results/MouseBrain_Jiang2023/vertical/{mode}/{model}/herit/{folder}/'
save_path = read_path + f'A_figs/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

peak_types = ['Background', 'Midbrain', 'Subpallium_1', 'Subpallium_2', 'Diencephalon_\nand_hindbrain', 'Mesenchyme', 'Cartilage_1', 'Cartilage_2', 'Cartilage_3',
              'Cartilage_4', 'Muscle', 'Thalamus', 'DPallm', 'DPallv']
colors = ['gray', 'chocolate', 'b', 'royalblue', 'orange', 'deepskyblue', 'g', 'limegreen', 'lime', 'springgreen',
          'pink', 'fuchsia', 'yellowgreen', 'olivedrab']

entries = os.listdir(read_path)

for entry in entries:

    entry_path = read_path + entry + '/'
    if entry_path == save_path:
        continue

    name = entry.replace('.sumstats.gz', '')
    if 'PASS' in name:
        name = name.replace('PASS_', '')
    elif 'UKB_460K' in name:
        name = name.replace('UKB_460K.', '')
    print(name)

    file_name = entry_path + '/res.results'

    df = pd.read_csv(file_name, sep='\t')
    df['Category'] = df['Category'].str.replace('L2_0', '', regex=False)
    df = df[df['Category'] != 'base']
    df['Category'] = pd.Categorical(df['Category'], categories=peak_types, ordered=True)
    df = df.sort_values('Category')

    enrich_score = [list(df['Enrichment'])[i] if list(df['Enrichment'])[i] >= 0 else 0 for i in
                    range(len(list(df['Enrichment'])))]
    enrich_std = [list(df['Enrichment_std_error'])[i] if list(df['Enrichment'])[i] >= 0 else 0 for i in
                  range(len(list(df['Enrichment'])))]
    # enrich_std = list(df['Enrichment_std_error'])
    # print(enrich_score)
    # print(enrich_std)

    plt.figure(figsize=(9.5, 4))

    bars = plt.bar(peak_types, enrich_score, alpha=0.7, color=colors)
    for bar, err in zip(bars, enrich_std):
        plt.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=err, lolims=True, uplims=False,
                     fmt='', ecolor='black', capsize=5)

    plt.title(name, fontsize=14)
    plt.ylabel('Enrichment', fontsize=12)
    plt.xticks(rotation=30, ha='center', fontsize=8)

    plt.gcf().subplots_adjust(left=None, top=0.9, bottom=0.2, right=None)

    if show:
        plt.show()
    if save:
        save_root = entry_path + f'{name}.pdf'
        plt.savefig(save_root, dpi=100)
        save_root = save_path + f'{name}.pdf'
        plt.savefig(save_root, dpi=100)

    plt.close()





