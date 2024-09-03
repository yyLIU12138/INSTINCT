import anndata as ad

from ..INSTINCT import peak_sets_alignment

import warnings
warnings.filterwarnings("ignore")

# mouse brain
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']
order_list = [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
              [1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
              [2, 1, 0, 3], [2, 1, 3, 0], [2, 0, 1, 3], [2, 0, 3, 1], [2, 3, 1, 0], [2, 3, 0, 1],
              [3, 1, 2, 0], [3, 1, 0, 2], [3, 2, 1, 0], [3, 2, 0, 1], [3, 0, 1, 2], [3, 0, 2, 1]]

pre = None

for order in order_list:

    # load raw data
    cas_list = []
    for idx in order:
        sample_data = ad.read_h5ad(data_dir + slice_name_list[idx] + '_atac.h5ad')

        if 'insertion' in sample_data.obsm:
            del sample_data.obsm['insertion']

        cas_list.append(sample_data)

    # merge peaks
    cas_list = peak_sets_alignment(cas_list)

    if not pre:
        pre = cas_list[0].var_names.to_list()
        continue
    else:
        if cas_list[0].var_names.to_list() == pre:
            print('Identical !')
            pre = cas_list[0].var_names.to_list()
        else:
            print('Not identical !')

