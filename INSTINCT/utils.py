import scipy
import numpy as np
import episcanpy as epi
import scanpy as sc
import pandas as pd

from typing import Optional
from sklearn.metrics import pairwise_distances


def preprocess_CAS(adata_list, adata_concat, binarize=False, use_fragment_count=False, tfidf=True,
                   min_cells_rate=0.03, min_features=1, tfidf_type=2):

    epi.pp.filter_features(adata_concat, min_cells=int(min_cells_rate * adata_concat.shape[0]))
    epi.pp.filter_cells(adata_concat, min_features=min_features)

    if binarize and use_fragment_count:
        raise ValueError("'binarize' and 'use_fragment_count' cannot be set to True at the same time !")

    elif binarize:
        epi.pp.binarize(adata_concat)

    elif use_fragment_count:
        adata_concat.X = scipy.sparse.csr_matrix(np.ceil((adata_concat.X / 2).toarray()))

    if tfidf:
        adata_concat.X = TFIDF(adata_concat.X.T, type_=tfidf_type).T.copy()
    else:
        epi.pp.normalize_total(adata_concat, target_sum=10000)
        epi.pp.log1p(adata_concat)

    for i in range(len(adata_list)):
        obs_list = [item for item in adata_list[i].obs_names if item in adata_concat.obs_names]
        var_names = adata_concat.var_names
        adata_list[i] = adata_list[i][obs_list, var_names]


def preprocess_SRT(adata_list, adata_concat, n_top_genes=5000, min_cells=1, min_genes=1):

    sc.pp.filter_genes(adata_concat, min_cells=min_cells)
    sc.pp.filter_cells(adata_concat, min_genes=min_genes)

    sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata_concat, target_sum=1e4)
    sc.pp.log1p(adata_concat)
    adata_concat = adata_concat[:, adata_concat.var['highly_variable']]

    for i in range(len(adata_list)):
        obs_list = [item for item in adata_list[i].obs_names if item in adata_concat.obs_names]
        var_names = adata_concat.var_names
        adata_list[i] = adata_list[i][obs_list, var_names]

    return adata_list, adata_concat


def TFIDF(count_mat, type_=2):
    # Perform TF-IDF (count_mat: peak*cell)
    def tfidf1(count_mat):
        if not scipy.sparse.issparse(count_mat):
            count_mat = scipy.sparse.coo_matrix(count_mat)

        nfreqs = count_mat.multiply(1.0 / count_mat.sum(axis=0))
        tfidf_mat = nfreqs.multiply(np.log(1 + 1.0 * count_mat.shape[1] / count_mat.sum(axis=1)).reshape(-1, 1)).tocoo()

        return scipy.sparse.csr_matrix(tfidf_mat)

    # Perform Signac TF-IDF (count_mat: peak*cell) [selected]
    def tfidf2(count_mat):
        if not scipy.sparse.issparse(count_mat):
            count_mat = scipy.sparse.coo_matrix(count_mat)

        tf_mat = count_mat.multiply(1.0 / count_mat.sum(axis=0))
        signac_mat = (1e4 * tf_mat).multiply(1.0 * count_mat.shape[1] / count_mat.sum(axis=1).reshape(-1, 1))
        signac_mat = signac_mat.log1p()

        return scipy.sparse.csr_matrix(signac_mat)

    # Perform TF-IDF (count_mat: ?)
    from sklearn.feature_extraction.text import TfidfTransformer
    def tfidf3(count_mat):
        model = TfidfTransformer(smooth_idf=False, norm="l2")
        model = model.fit(np.transpose(count_mat))
        model.idf_ -= 1
        tf_idf = np.transpose(model.transform(np.transpose(count_mat)))

        return scipy.sparse.csr_matrix(tf_idf)

    if type_ == 1:
        return tfidf1(count_mat)
    elif type_ == 2:
        return tfidf2(count_mat)
    else:
        return tfidf3(count_mat)


def find_peak_overlaps(query, key):
    q_seqname = np.array(query.get_seqnames())
    k_seqname = np.array(key.get_seqnames())
    q_start = np.array(query.get_start())
    k_start = np.array(key.get_start())
    q_width = np.array(query.get_width())
    k_width = np.array(key.get_width())
    q_end = q_start + q_width
    k_end = k_start + k_width

    q_index = 0
    k_index = 0
    overlap_index = [[] for i in range(len(query))]
    overlap_count = [0 for i in range(len(query))]

    while True:
        if q_index == len(query) or k_index == len(key):
            return overlap_index, overlap_count

        if q_seqname[q_index] == k_seqname[k_index]:
            if k_start[k_index] >= q_start[q_index] and k_end[k_index] <= q_end[q_index]:
                overlap_index[q_index].append(k_index)
                overlap_count[q_index] += 1
                k_index += 1
            elif k_start[k_index] < q_start[q_index]:
                k_index += 1
            else:
                q_index += 1
        elif q_seqname[q_index] < k_seqname[k_index]:
            q_index += 1
        else:
            k_index += 1


def peak_sets_alignment(adata_list, sep=(":", "-"), min_width=20, max_width=10000, min_gap_width=1,
                        peak_region: Optional[str] = None):
    from genomicranges import GenomicRanges
    from iranges import IRanges
    from biocutils.combine import combine

    ## Peak merging
    gr_list = []
    for i in range(len(adata_list)):
        seq_names = []
        starts = []
        widths = []
        regions = adata_list[i].var_names if peak_region is None else adata_list[i].obs[peak_region]
        for region in regions:
            seq_names.append(region.split(sep[0])[0])
            if sep[0] == sep[1]:
                start, end = region.split(sep[0])[1:]
            else:
                start, end = region.split(sep[0])[1].split(sep[1])
            width = int(end) - int(start)
            starts.append(int(start))
            widths.append(width)
        gr = GenomicRanges(seqnames=seq_names, ranges=IRanges(starts, widths)).sort()
        peaks = [seqname + sep[0] + str(start) + sep[1] + str(end) for seqname, start, end in
                 zip(gr.get_seqnames(), gr.get_start(), gr.get_end())]
        adata_list[i] = adata_list[i][:, peaks]
        gr_list.append(gr)

    gr_combined = combine(*gr_list)
    gr_merged = gr_combined.reduce(min_gap_width=min_gap_width).sort()
    print("Peak merged")

    ## Peak filtering
    # filter by intesect
    overlap_index_list = []
    index = np.ones(len(gr_merged)).astype(bool)
    for gr in gr_list:
        overlap_index, overlap_count = find_peak_overlaps(gr_merged, gr)
        index = (np.array(overlap_count) > 0) * index
        overlap_index_list.append(overlap_index)
    # filter by width
    index = index * (gr_merged.get_width() > min_width) * (gr_merged.get_width() < max_width)
    gr_merged = gr_merged.get_subset(index)
    common_peak = [seqname + ":" + str(start) + "-" + str(end) for seqname, start, end in
                   zip(gr_merged.get_seqnames(), gr_merged.get_start(), gr_merged.get_end())]
    print("Peak filtered")

    ## Merge count matrix
    adata_merged_list = []
    for adata, overlap_index in zip(adata_list, overlap_index_list):
        overlap_index = [overlap_index[i] for i in range(len(index)) if index[i]]
        X = adata.X.tocsc()
        X_merged = scipy.sparse.hstack([scipy.sparse.csr_matrix(X[:, cur].sum(axis=1)) for cur in overlap_index])
        adata_merged_list.append(
            sc.AnnData(X_merged, obs=adata.obs, var=pd.DataFrame(index=common_peak), obsm=adata.obsm))
    print("Matrix merged")

    return adata_merged_list


def create_neighbor_graph(adata_list, adata_concat, rad_cutoff=None, rad_coef=1.5, coor_key='spatial'):

    if not rad_cutoff:
        rad_cutoff = []

    G_list = []

    # calculate the rad_cutoff based on the min pair distance if it is not provided
    if len(rad_cutoff) == 0:
        for i in range(len(adata_list)):
            adata_copy = adata_list[i].copy()
            loc_copy = np.array(adata_copy.obsm[coor_key])
            pair_dist_ref = pairwise_distances(loc_copy)
            min_dist = np.sort(np.unique(pair_dist_ref), axis=None)[1]

            graph = (pair_dist_ref < rad_coef * min_dist).astype(float)
            G_list.append(graph)
            print(f"Radius for graph connection of slice {i} is {rad_coef * min_dist:.4f}.")
            print('%.4f neighbors per cell on average including itself.' % (np.mean(np.sum(graph, axis=1))))

    elif len(rad_cutoff) != len(adata_list):
        raise ValueError("The length of 'rad_cutoff' should be the number of slices !")

    else:
        for i in range(len(adata_list)):
            adata_copy = adata_list[i].copy()
            loc_copy = np.array(adata_copy.obsm[coor_key])
            pair_dist_ref = pairwise_distances(loc_copy)

            graph = (pair_dist_ref < rad_cutoff[i]).astype(float)
            G_list.append(graph)
            print(f"Radius for graph connection of slice {i} is {rad_cutoff[i]:.4f}.")
            print('%.4f neighbors per cell on average including itself.' % (np.mean(np.sum(graph, axis=1))))

    adata_concat.uns['graph_list'] = G_list
