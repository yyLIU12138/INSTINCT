import numpy as np
import scipy
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import completeness_score
import sklearn
import sklearn.neighbors
import pandas as pd
import networkx as nx

import scib
import scanpy as sc

from sklearn.model_selection import cross_validate, LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score, make_scorer


def match_cluster_labels(true_labels, est_labels):
    true_labels_arr = np.array(list(true_labels))
    est_labels_arr = np.array(list(est_labels))

    org_cat = list(np.sort(list(pd.unique(true_labels))))
    est_cat = list(np.sort(list(pd.unique(est_labels))))

    B = nx.Graph()
    B.add_nodes_from([i + 1 for i in range(len(org_cat))], bipartite=0)
    B.add_nodes_from([-j - 1 for j in range(len(est_cat))], bipartite=1)

    for i in range(len(org_cat)):
        for j in range(len(est_cat)):
            weight = np.sum((true_labels_arr == org_cat[i]) * (est_labels_arr == est_cat[j]))
            B.add_edge(i + 1, -j - 1, weight=-weight)

    match = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)

    if len(org_cat) >= len(est_cat):
        return np.array([match[-est_cat.index(c) - 1] - 1 for c in est_labels_arr])
    else:
        unmatched = [c for c in est_cat if not (-est_cat.index(c) - 1) in match.keys()]
        l = []
        for c in est_labels_arr:
            if (-est_cat.index(c) - 1) in match:
                l.append(match[-est_cat.index(c) - 1] - 1)
            else:
                l.append(len(org_cat) + unmatched.index(c))
        return np.array(l)


def StrLabel2Idx(string_labels):

    label_encoder = LabelEncoder()
    idx_labels = label_encoder.fit_transform(string_labels)

    return np.array(idx_labels)


def knn_label_translation(reference_X, reference_y, target_X, k=20):
    label_encoder = LabelEncoder()
    reference_y_idx = label_encoder.fit_transform(reference_y)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(reference_X, reference_y_idx)
    target_y_idx = neigh.predict(target_X)
    target_y = label_encoder.inverse_transform(target_y_idx)

    return target_y


def knn_cross_validation(mtx, label, Kfold=5, k=20, batch_idx=None):
    if not isinstance(label, np.ndarray): label = np.array(label).astype(str)
    target = StrLabel2Idx(label)
    if batch_idx is not None:
        batch_idx = np.array(batch_idx).astype(str)
        groups = StrLabel2Idx(batch_idx)
        split = LeaveOneGroupOut()
        n_jobs = np.unique(batch_idx).shape[0]
    else:
        groups = None
        split = StratifiedKFold(n_splits=Kfold)
        n_jobs = Kfold
    model = KNeighborsClassifier(n_neighbors=k)
    cv_results = cross_validate(model, mtx, target, groups=groups,
                                scoring=("accuracy", "f1_macro", "f1_weighted"),
                                cv=split, n_jobs=n_jobs)
    model = KNeighborsClassifier(n_neighbors=k)
    kappa_score = make_scorer(cohen_kappa_score)
    kappa = cross_validate(model, mtx, target, groups=groups,
                           scoring=kappa_score,
                           cv=split, n_jobs=n_jobs)["test_score"]
    acc, kappa, mf1, wf1 = cv_results["test_accuracy"].mean(), kappa.mean(), cv_results["test_f1_macro"].mean(), \
                           cv_results["test_f1_weighted"].mean()
    print('Accuracy: %.3f, Kappa: %.3f, mF1: %.3f, wF1: %.3f' % (acc, kappa, mf1, wf1))

    return acc, kappa, mf1, wf1


def cluster_metrics(target, pred):
    target = np.array(target)
    pred = np.array(pred)
    
    ari = adjusted_rand_score(target, pred)
    ami = adjusted_mutual_info_score(target, pred)
    nmi = normalized_mutual_info_score(target, pred)
    fmi = fowlkes_mallows_score(target, pred)
    comp = completeness_score(target, pred)
    homo = homogeneity_score(target, pred)
    print('ARI: %.3f, AMI: %.3f, NMI: %.3f, FMI: %.3f, Comp: %.3f, Homo: %.3f' % (ari, ami, nmi, fmi, comp, homo))
    
    return ari, ami, nmi, fmi, comp, homo


def mean_average_precision(x: np.ndarray, y: np.ndarray, k: int=30, **kwargs) -> float:
    r"""
    Mean average precision
    Parameters
    ----------
    x
        Coordinates
    y
        Cell_type/Layer labels
    k
        k neighbors
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`
    Returns
    -------
    map
        Mean average precision
    """
    
    def _average_precision(match: np.ndarray) -> float:
        if np.any(match):
            cummean = np.cumsum(match) / (np.arange(match.size) + 1)
            return cummean[match].mean().item()
        return 0.0
    
    y = np.array(y)
    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=min(y.shape[0], k + 1), **kwargs).fit(x)
    nni = knn.kneighbors(x, return_distance=False)
    match = np.equal(y[nni[:, 1:]], np.expand_dims(y, 1))
    
    return np.apply_along_axis(_average_precision, 1, match).mean().item()


def rep_metrics(adata, origin_concat, use_rep, label_key, batch_key, k_map=30):
    if label_key not in adata.obs or batch_key not in adata.obs or use_rep not in adata.obsm:
        print("KeyError")
        return None
    
    adata.obs[label_key] = adata.obs[label_key].astype(str).astype("category")
    adata.obs[batch_key] = adata.obs[batch_key].astype(str).astype("category")
    origin_concat.X = origin_concat.X.astype(float)
    sc.pp.neighbors(adata, use_rep=use_rep)

    MAP = mean_average_precision(adata.obsm[use_rep].copy(), adata.obs[label_key], k=k_map)
    cell_type_ASW = scib.me.silhouette(adata, label_key=label_key, embed=use_rep)
    # g_iLISI = scib.me.ilisi_graph(adata, batch_key=batch_key, type_="embed", use_rep=use_rep)
    batch_ASW = scib.me.silhouette_batch(adata, batch_key=batch_key, label_key=label_key, embed=use_rep, verbose=False)
    batch_PCR = scib.me.pcr_comparison(origin_concat, adata, covariate=batch_key, embed=use_rep)
    kBET = scib.me.kBET(adata, batch_key=batch_key, label_key=label_key, type_='embed', embed=use_rep)
    g_conn = scib.me.graph_connectivity(adata, label_key=label_key)
    print('mAP: %.3f, Cell type ASW: %.3f, Batch ASW: %.3f, Batch PCR: %.3f, kBET: %.3f, Graph connectivity: %.3f' %
          (MAP, cell_type_ASW, batch_ASW, batch_PCR, kBET, g_conn))
    
    return MAP, cell_type_ASW, batch_ASW, batch_PCR, kBET, g_conn


def bio_conservation_metrics(adata, use_rep, label_key, batch_key, k_map=30, threshold=1):
    if label_key not in adata.obs or batch_key not in adata.obs or use_rep not in adata.obsm:
        print("KeyError")
        return None

    adata.obs[label_key] = adata.obs[label_key].astype(str).astype("category")
    adata.obs[batch_key] = adata.obs[batch_key].astype(str).astype("category")
    # sc.pp.neighbors(adata, use_rep=use_rep, random_state=1234)

    MAP = mean_average_precision(adata.obsm[use_rep].copy(), adata.obs[label_key], k=k_map)
    cell_type_ASW = scib.me.silhouette(adata, label_key=label_key, embed=use_rep)
    isolated_asw = scib.me.isolated_labels_asw(adata, batch_key=batch_key, label_key=label_key, embed=use_rep,
                                               iso_threshold=threshold)
    isolated_f1 = scib.me.isolated_labels_f1(adata, batch_key=batch_key, label_key=label_key, embed=use_rep,
                                             iso_threshold=threshold)

    print('mAP: %.3f, Cell type ASW: %.3f, Isolated label ASW: %.3f, Isolated label F1: %.3f' %
          (MAP, cell_type_ASW, isolated_asw, isolated_f1))

    return MAP, cell_type_ASW, isolated_asw, isolated_f1


def batch_correction_metrics(adata, origin_concat, use_rep, label_key, batch_key):
    if label_key not in adata.obs or batch_key not in adata.obs or use_rep not in adata.obsm:
        print("KeyError")
        return None

    adata.obs[label_key] = adata.obs[label_key].astype(str).astype("category")
    adata.obs[batch_key] = adata.obs[batch_key].astype(str).astype("category")
    origin_concat.X = origin_concat.X.astype(float)
    sc.pp.neighbors(adata, use_rep=use_rep, random_state=1234)

    # g_iLISI = scib.me.ilisi_graph(adata, batch_key=batch_key, type_="embed", use_rep=use_rep)
    batch_ASW = scib.me.silhouette_batch(adata, batch_key=batch_key, label_key=label_key, embed=use_rep, verbose=False)
    batch_PCR = scib.me.pcr_comparison(origin_concat, adata, covariate=batch_key, embed=use_rep)
    kBET = scib.me.kBET(adata, batch_key=batch_key, label_key=label_key, type_='embed', embed=use_rep)
    g_conn = scib.me.graph_connectivity(adata, label_key=label_key)
    print('Batch ASW: %.3f, Batch PCR: %.3f, kBET: %.3f, Graph connectivity: %.3f' %
          (batch_ASW, batch_PCR, kBET, g_conn))

    return batch_ASW, batch_PCR, kBET, g_conn


def metacell_correlation(raw, imputed, label, metrics="spearman"):
    assert metrics in ("spearman", "pearson"), "metrics should be one of (spearman, pearson)"

    label = np.array(label).astype(str)
    result = np.zeros_like(label)
    if scipy.sparse.issparse(raw): raw = raw.A
    if scipy.sparse.issparse(imputed): imputed = imputed.A
    for domain in np.unique(label):
        idx = (label == domain)
        meta = raw[idx, :].mean(axis=0).reshape(1, -1)
        cur = imputed[idx, :]
        if metrics == "pearson":
            result[idx] = np.corrcoef(meta, cur)[0, 1:]
        elif metrics == "spearman" and cur.shape[0] > 1:
            result[idx] = scipy.stats.spearmanr(meta, cur, axis=1).correlation[0, 1:]
        else:
            result[idx] = np.array([scipy.stats.spearmanr(meta, cur, axis=1).correlation])

    return result.tolist()


def gene_cell_correlation(expr1, expr2, metrics="spearman"):
    assert metrics in ("spearman", "pearson"), "metrics should be one of (spearman, pearson)"
    assert expr1.shape[0] == expr2.shape[0] and expr1.shape[1] == expr2.shape[
        1], "shape of expr1 and expr2 should be the same"

    cell_corr = []
    gene_corr = []
    if metrics == "pearson":
        for i in range(expr1.shape[0]):
            cell_corr.append(np.corrcoef(expr1[i], expr2[i])[1, 0])
        for j in range(expr1.shape[1]):
            gene_corr.append(np.corrcoef(expr1[:, j], expr2[:, j])[1, 0])
    else:
        for i in range(expr1.shape[0]):
            cell_corr.append(scipy.stats.spearmanr(expr1[i], expr2[i]).correlation)
        for j in range(expr1.shape[1]):
            gene_corr.append(scipy.stats.spearmanr(expr1[:, j], expr2[:, j]).correlation)

    return cell_corr, gene_corr


def batch_metrics(adata, use_rep, batch_key, label_key):
    import scib
    if batch_key not in adata.obs or label_key not in adata.obs or use_rep not in adata.obsm:
        print("KeyError")
        return None
    sc.pp.neighbors(adata, use_rep=use_rep, random_state=1234)
    #     GC = scib.me.graph_connectivity(adata, label_key=label_key)
    iLISI = scib.me.ilisi_graph(adata, batch_key=batch_key, type_="embed", use_rep=use_rep)
    kBET = scib.me.kBET(adata, batch_key=batch_key, label_key=label_key, type_="embed", embed=use_rep)
    #     bASW = scib.me.silhouette_batch(adata, batch_key=batch_key, label_key=label_key, embed=use_rep)
    #     print('GC: %.3f, iLISI: %.3f, kBET: %.3f, bASW: %.3f' % (GC, iLISI, kBET, bASW))
    print('iLISI: %.3f, kBET: %.3f' % (iLISI, kBET))

    return iLISI, kBET


def isolated_metrics(adata, use_rep, label_key, batch_key, threshold=1):
    import scib
    if batch_key not in adata.obs or label_key not in adata.obs or use_rep not in adata.obsm:
        print("KeyError")
        return None

    sc.pp.neighbors(adata, use_rep=use_rep, random_state=1234)
    isolated_asw = scib.me.isolated_labels_asw(adata, batch_key=batch_key, label_key=label_key, embed=use_rep,
                                               iso_threshold=threshold)
    isolated_f1 = scib.me.isolated_labels_f1(adata, batch_key=batch_key, label_key=label_key, embed=use_rep,
                                             iso_threshold=threshold)
    print('isolated_asw: %.3f, isolated_f1: %.3f' % (isolated_asw, isolated_f1))

    return isolated_asw, isolated_f1
