import numpy as np
from scipy.stats import multivariate_normal
import os
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy import sparse
import scipy.io as sio
import scanpy as sc
from Bio import Phylo
from io import StringIO
import logging
from scipy.optimize import fsolve
import random
import threading
import scipy.stats as stats
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import logser

from scipy.sparse import csr_matrix, vstack
from scipy.special import rel_entr
from statsmodels.discrete.count_model import (ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson,
                                              ZeroInflatedGeneralizedPoisson)
import statsmodels.api as sm
from scipy.stats import nbinom
from scipy.special import expit


# library size
def cal_lib(adata):
    return np.array(np.sum(adata.X,axis=1)).ravel()

def cal_pm(adata): # peak mean
    return np.array(np.mean(adata.X,axis=0)).ravel()

def cal_pl(adata):# peak length
    start=np.array([int(i.split('_')[1]) for i in adata.var.index])
    end=np.array([int(i.split('_')[2]) for i in adata.var.index])
    return (end-start).ravel()

def cal_spa(adata):# sparsity
    X=adata.X.copy()
    X[X>0]=1
    sparsity=np.sum(X,axis=1)/X.shape[1]
    return np.array(sparsity).ravel()

def cal_nozero(adata):# sparsity
    X=adata.X.copy()
    X[X>0]=1
    sparsity=np.sum(X,axis=1)
    return np.array(sparsity).ravel()

def cal_peak_count(adata):
    return np.array(np.sum(adata.X,axis=0)).ravel()

def Activation(X, method='sigmod'):  # 对peak_effect*cell_embedding 的矩阵进行激活操作，防止其值为0
    if method == 'sigmod':
        return 1 / (1 + np.exp(-1 * X))
    elif method == 'exp':
        return np.exp(X)
    elif method == 'exp_linear':
        exp_num = 4
        k = np.exp(exp_num)
        # k=1
        # X_act = X.copy()
        # X_act[X_act >= exp_num] = k * X_act[X_act >= exp_num] + np.exp(exp_num) - exp_num * np.exp(exp_num)
        # X_act[X_act < exp_num] = np.exp(X_act[X_act < exp_num])
        # X_act = np.where(X_act >= exp_num, k * X_act + np.exp(exp_num) - exp_num * np.exp(exp_num), np.exp(X_act))
        X = np.where(X >= exp_num, k * X + np.exp(exp_num) - exp_num * np.exp(exp_num), np.exp(X))
        return X
    elif method == 'exp_sym':
        X_act = X
        exp_level = 1.5
        X_act[X_act <= 0] = np.power(exp_level, X_act[X_act <= 0])
        X_act[X_act > 0] = 2 - np.power(exp_level, -X_act[X_act > 0])
        return X_act
    elif method == 'sigmod_adj':
        k = 2
        A = 1
        return A / (1 + k ** (-1 * X))

    else:
        raise ValueError('wrong activation method!')

def create_logger(name='', ch=True, fh='', levelname=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(levelname)

    #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if ch:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if fh:
        fh = logging.FileHandler(fh, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    # torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def Get_Effect(n_peak, n_cell_total, len_cell_embed, rand_seed, zero_prob, zero_set, effect_mean, effect_sd):
    # 生成peak effect和library size effect
    # np.random.seed(rand_seed)
    peak_effect = np.random.normal(effect_mean, effect_sd, (n_peak, len_cell_embed))
    lib_size_effect = np.random.normal(effect_mean, effect_sd, (1, len_cell_embed))

    # 対生成的effect vevtor进行置零
    if zero_set == 'by_row':
        # 对于每个peak的effect进行相同概率的置零
        def set_zero(a, zero_prob=0.5):
            a[np.random.choice(len(a), replace=False, size=int(len(a) * zero_prob))] = 0
            return a

        peak_effect = np.apply_along_axis(set_zero, 1, peak_effect, zero_prob=zero_prob)

    if zero_set == 'all':
        # 对于所有index选择进行置零
        indices = np.random.choice(peak_effect.shape[1] * peak_effect.shape[0], replace=False,
                                   size=int(peak_effect.shape[1] * peak_effect.shape[0] * zero_prob))
        peak_effect[np.unravel_index(indices, peak_effect.shape)] = 0

    return peak_effect, lib_size_effect

def Bernoulli_pm_correction(X_peak, param_pm):  # X_peak:等待矫正的矩阵  param_pm:对应的采样得到的peak_mean
    # peak mean correction
    peak_p_list = []
    for i in range(0, X_peak.shape[0]):
        peak_p = X_peak[i, :]  # 单个peak对应的所有cell的值
        peak_mean = np.mean(peak_p)  # 当前矩阵的peak mean
        peak_mean_ex = np.exp(param_pm[i]) - 1  # 期望的peak mean

        # 若期望的peak_mean都是0
        if peak_mean_ex == 0 or peak_mean == 0:
            peak_p_list.append(peak_p * peak_mean_ex)
            continue

        if np.max(peak_p) / peak_mean * peak_mean_ex > 1:
            peak_p_sort = np.sort(peak_p)
            idx = len(peak_p_sort) - 1
            while (1):
                weight = (len(peak_p) * peak_mean_ex + idx - len(peak_p)) / (np.sum(peak_p_sort[0:idx]) + 1e-8)
                if peak_p_sort[idx - 1] * weight <= 1:
                    # print(idx)
                    break

                for idx_2 in range(idx, -1, -1):  # 找到*weight<1 的idx
                    if peak_p_sort[idx_2 - 1] * weight <= 1:
                        # print(idx_2)
                        break
                    # 如果实在没有idx能够使得值*weight<1,此时就会一直循环，需要及时跳出循环
                    if idx_2 <= 1:
                        break
                idx = idx_2
                if idx_2 <= 1:
                    weight = (len(peak_p) * peak_mean_ex + idx - len(peak_p)) / (np.sum(peak_p_sort[0:idx]) + 1e-8)
                    break
            peak_p = peak_p * weight
            peak_p[peak_p > 1] = 1
        else:
            peak_p = peak_p / peak_mean * peak_mean_ex
        peak_p_list.append(peak_p)
    peak_p_matrix = np.vstack(peak_p_list)
    return peak_p_matrix

def Bernoulli_lib_correction(X_peak, param_lib):
    peak_p_list = []
    for i in range(X_peak.shape[1]):
        peak_p = X_peak[:, i]
        lib_size = np.sum(peak_p)
        lib_size_ex = np.exp(param_lib[i]) - 1

        # 若期望的library_size都是0
        if lib_size_ex == 0 or lib_size == 0:
            peak_p_list.append((peak_p * lib_size_ex).reshape(-1, 1))
            continue

        if np.max(peak_p) / lib_size * lib_size_ex > 1:
            peak_p_sort = np.sort(peak_p)
            idx = len(peak_p_sort) - 1
            while (1):
                weight = (lib_size_ex + idx - len(peak_p)) / (np.sum(peak_p_sort[0:idx]) + 1e-8)
                if peak_p_sort[idx - 1] * weight <= 1:
                    break
                for idx_2 in range(idx, -1, -1):
                    if peak_p_sort[idx_2 - 1] * weight <= 1:
                        # print(idx_2)
                        break
                    # 如果实在没有idx能够使得值*weight<1,此时就会一直循环，需要及时跳出循环
                    if idx_2 <= 1:
                        break
                idx = idx_2
                if idx_2 <= 1:
                    raise ValueError('Please correct it !')
                    weight = (len(peak_p) * peak_mean_ex + idx - len(peak_p)) / (np.sum(peak_p_sort[0:idx]) + 1e-8)
                    break
            # 防止出现<1的部分全都是0
            if np.sum(peak_p_sort[0:idx]) == 0:
                peak_p[peak_p > 1] = 1
            else:
                peak_p = peak_p * weight
                peak_p[peak_p > 1] = 1
        else:
            peak_p = peak_p / lib_size * lib_size_ex
        peak_p_list.append(peak_p.reshape(-1, 1))
    peak_p_matrix = np.hstack(peak_p_list)
    return peak_p_matrix

def Get_Single_Embedding(n_cell_total, embed_mean_same, embed_sd_same,
                         n_embed_diff, n_embed_same):
    embed = np.random.normal(embed_mean_same, embed_sd_same, (n_embed_same + n_embed_diff, n_cell_total))
    index = ['embedding_' + str(m + 1) for m in range(n_embed_same + n_embed_diff)]
    columns = ['single cluster' for m in range(n_cell_total)]
    df = pd.DataFrame(embed, columns=columns, index=index)

    return df, columns


def Get_Discrete_Embedding(pops_name, min_popsize, tree_text,
                           n_cell_total, pops_size,
                           embed_mean_same, embed_sd_same,
                           embed_mean_diff, embed_sd_diff,
                           n_embed_diff, n_embed_same, rand_seed):
    # np.random.seed(rand_seed)
    n_pop = len(pops_name)
    if (n_cell_total < min_popsize * n_pop):
        raise ValueError("The size of the smallest population is too big for the total number of cells")

    if not pops_size:
        if min_popsize:  # 若设定了最小pop的size，则其他pop将原来的细胞数目平均分配
            pop_size = np.floor((n_cell_total - min_popsize) / (len(pops_name) - 1))
            left_over = n_cell_total - min_popsize - pop_size * (len(pops_name) - 1)
            pop_name_size = {}  # 每个pop对应的size
            for name in pops_name:
                if name == min_pop:
                    pop_name_size[name] = min_popsize
                else:
                    pop_name_size[name] = pop_size
            pop_name_size[pops_name[pops_name.index(min_pop) - 1]] += left_over
        else:  # 未设置最小pop，直接将每个pop的cell数目均分
            pop_size = np.floor((n_cell_total) / (len(pops_name)))
            left_over = n_cell_total - pop_size * (len(pops_name))
            pop_name_size = {}
            for name in pops_name:
                pop_name_size[name] = pop_size
            pop_name_size[pops_name[0]] += left_over

    else:  # 若直接对每个pop赋予size
        pop_name_size = {}
        for (i, name) in enumerate(pops_name):
            pop_name_size[name] = pops_size[i]
    # 将float转化为int
    for key, value in pop_name_size.items():
        pop_name_size[key] = int(value)

    # --------生成不同pop之间的协方差矩阵，这里需要在你的python环境中使用R包ape
    ape = importr('ape')
    phyla = ape.read_tree(text=tree_text)
    corr_matrix = np.array(ape.vcv_phylo(phyla, cor=True))

    corr_matrix = np.eye(len(pops_name))

    # --------生成embed

    embed_same, embed_diff = [], []
    # 生成差异embedding特征对应的均值，保证不同的pop之间的相关性
    embed_diff_mean_mv = multivariate_normal.rvs(mean=[embed_mean_diff] * n_pop, cov=corr_matrix, size=n_embed_diff)
    for (j, name) in enumerate(pops_name):
        # 生成每个pop对应的非差异embed部分
        embed_same_pop = np.random.normal(embed_mean_same, embed_sd_same, (n_embed_same, pop_name_size[name]))

        # 生成每个pop对应的差异embed部分
        embed_diff_pop = []
        for k in range(n_embed_diff):
            embed = np.random.normal(embed_diff_mean_mv[k, j], embed_sd_diff, (pop_name_size[name],))
            embed_diff_pop.append(embed)
        embed_diff_pop = np.vstack(embed_diff_pop)

        # 对每个pop差异/非差异embed进行汇总
        embed_same.append(embed_same_pop)  # n_embed_same*pop_size
        embed_diff.append(embed_diff_pop)  # n_embed_diff*pop_size

    # embed_param: len_cell_embed*n_cell_total
    embed_same = np.hstack(embed_same)
    embed_diff = np.hstack(embed_diff)
    embed_param = np.vstack([embed_same, embed_diff])

    columns = np.hstack([[name] * pop_name_size[name] for name in pops_name])
    index = ['same_embedding_' + str(m + 1) for m in range(n_embed_same)] + ['diff_embedding_' + str(m + 1) for m in
                                                                             range(n_embed_diff)]
    df = pd.DataFrame(embed_param, columns=columns, index=index)

    return df, columns


def Generate_Tree_Sd(branches, root, depth=0, anchor=0,
                     rand_seed=0):  # depth就是到根节点的深度;一个递归函数,用来获取细胞在每个branch上的位置以及enbedding
    # np.random.seed(rand_seed)

    start_nodes = [i.split('-')[0] for i in branches]

    df = pd.DataFrame({'branches': [], 'cell_places': [], 'embeddings': []})
    for i in range(len(start_nodes)):
        if root == start_nodes[i]:  # 该节点对应的所有branch
            branch = branches[i]
            start, end, branch_len, n_cells = branch.split('-')[0], \
                                              branch.split('-')[1], float(branch.split('-')[2]), int(
                branch.split('-')[3])
            interval = branch_len / (n_cells - 1)  # 获取interval
            cell_places = [depth + interval * i for i in range(n_cells - 1)] + [
                depth + branch_len]  # 以interval为间隔获取cell在branch上的位置

            # 获取单维所有细胞的embedding
            embeddings = np.array([0] + list(np.cumsum(np.random.normal(0, np.sqrt(interval), (n_cells - 1))))) + anchor

            df_ = pd.DataFrame(
                {'branches': [branch] * len(cell_places), 'cell_places': cell_places, 'embeddings': embeddings})
            df = pd.concat([df, df_, Generate_Tree_Sd(branches, end, depth + branch_len, anchor=embeddings[-1])],
                           axis=0)
    return df


def Get_Continuous_Embedding(tree_text, n_cell_total,
                             embed_mean_same, embed_sd_same,
                             embed_mean_diff, embed_sd_diff,
                             n_embed_diff, n_embed_same, rand_seed):
    # np.random.seed(rand_seed)
    # 构建tree
    tree = Phylo.read(StringIO(tree_text), "newick")

    # 获取不同的branch，形式为‘X-X-length’
    clades = [i for i in tree.find_clades()]
    branch_clades = [i for i in clades if i.branch_length]
    branches = [tree.get_path(i)[-2:] for i in branch_clades]
    branches = [branches[i][0].name + '-' + branches[i][1].name + '-' + str(branch_clades[i].branch_length) for i in
                range(len(branches))]

    # 获取所有branch的长度
    total_branch_len = sum([float(i.split('-')[2]) for i in branches])

    # 获取每个branch上的细胞数目（按照branch长度进行均分）
    n_branches_cell = []
    for i in range(len(branches)):
        branch_len = float(branches[i].split('-')[2])
        n_cells = np.floor(n_cell_total * (branch_len / total_branch_len))
        n_branches_cell.append(n_cells)

    # 将偏置加到数目最多的分支上
    n_branches_cell[n_branches_cell.index(max(n_branches_cell))] = n_branches_cell[n_branches_cell.index(
        max(n_branches_cell))] + n_cell_total - sum(n_branches_cell)
    n_branches_cell = [int(i) for i in n_branches_cell]

    # 将细胞数目加入branch，最终branch格式：A-B-1.0-200
    branches = [branches[i] + '-' + str(n_branches_cell[i]) for i in range(len(branches))]

    # 获取root名字
    root = clades[1].name

    # 生成continuous的embedding
    embed_same = np.random.normal(embed_mean_same, embed_sd_same, (n_embed_same, n_cell_total))
    embed_diff = []
    for i in range(n_embed_diff):
        df_continuous = Generate_Tree_Sd(branches, root, depth=0, anchor=embed_mean_diff, rand_seed=rand_seed + i)
        embed_diff.append(np.array(df_continuous['embeddings']))
    embed_diff = np.vstack(embed_diff)
    # print(embed_same.shape,embed_diff.shape)
    # print(branches)
    embed = np.vstack([embed_same, embed_diff])

    # 加上columns和index
    columns = list(df_continuous['branches'])
    index = ['same_embedding_' + str(m + 1) for m in range(n_embed_same)] + ['diff_embedding_' + str(m + 1) for m in
                                                                             range(n_embed_diff)]
    df = pd.DataFrame(embed, columns=columns, index=index)

    return df, columns


def zip_correction(i, simu_param_lib_i, lambdas_i, lambdas_sum_i, simu_param_nozero_i, n_peak):
    global k_dict, pi_dict
    # print(i)
    if i % 200 == 0: print(i)

    def solve_function(unsolved_value):
        k, pi = unsolved_value[0], unsolved_value[1]
        return [
            k * (1 - pi) - simu_param_lib_i / (lambdas_sum_i),
            n_peak * pi + (1 - pi) * np.sum(np.exp(-lambdas_i * k)) - (n_peak - simu_param_nozero_i)
        ]

    solved = fsolve(solve_function, [3, 0.5], maxfev=2000)
    k, pi = solved[0], solved[1]
    simu1 = k * (1 - pi) * (lambdas_sum_i)
    real1 = simu_param_lib_i
    if abs(simu1 - real1) / real1 > 0.1:
        solved = fsolve(solve_function, [20, 0.5], maxfev=2000)
    k, pi = solved[0], solved[1]

    k_dict[i] = solved[0]
    pi_dict[i] = solved[1]


class zip_correction_thread(threading.Thread):
    def __init__(self, i, simu_param_lib_i, lambdas_i, lambdas_sum_i, simu_param_nozero_i, n_peak):
        super(zip_correction_thread, self).__init__()
        self.i = i
        self.simu_param_lib_i = simu_param_lib_i
        self.lambdas_i = lambdas_i
        self.lambdas_sum_i = lambdas_sum_i
        self.simu_param_nozero_i = simu_param_nozero_i
        self.n_peak = n_peak

    def run(self):
        zip_correction(self.i, self.simu_param_lib_i, self.lambdas_i, self.lambdas_sum_i, self.simu_param_nozero_i,
                       self.n_peak)

def Get_Tree_Counts(peak_mean, lib_size, nozero, n_peak, n_cell_total, rand_seed,
                    embeds_peak, embeds_lib, correct_iter, distribution='Bernoulli',
                    activation='exp', bw_pm=1e-4, bw_lib=0.05, bw_nozero=0.05, real_param=True):
    # np.random.seed(rand_seed)
    if distribution == 'Bernoulli' and np.max(np.exp(peak_mean) - 1) > 1:
        raise ValueError('you data may not be Bernoulli distribution!')

    if real_param:  # 如果直接使用真实参数，peak mean直接按照真实参数来，lib size抽样
        param_pm = np.sort(peak_mean, axis=0).ravel()
        param_lib = np.sort(np.random.choice(lib_size, size=n_cell_total), axis=0).ravel()
        param_nozero = np.sort(np.random.choice(nozero, size=n_cell_total), axis=0).ravel()
    else:
        # kde
        kde_pm = KernelDensity(kernel='gaussian', bandwidth=bw_pm).fit(peak_mean.reshape(-1, 1))
        kde_lib = KernelDensity(kernel='gaussian', bandwidth=bw_lib).fit(lib_size.reshape(-1, 1))
        kde_nozero = KernelDensity(kernel='gaussian', bandwidth=bw_nozero).fit(nozero.reshape(-1, 1))

        # 从kde中采样并进行排序（从小到大）
        param_pm = kde_pm.sample(n_peak, random_state=rand_seed)
        param_lib = kde_lib.sample(n_cell_total, random_state=rand_seed)
        param_nozero = kde_nozero.sample(n_cell_total, random_state=rand_seed)

        param_pm = np.sort(param_pm, axis=0).ravel()
        param_lib = np.sort(param_lib, axis=0).ravel()
        param_nozero = np.sort(param_nozero, axis=0).ravel()

    # 从模拟矩阵的参数顺序对应到采样的真实参数
    X_peak = np.dot(peak_effect, embeds_peak)  # peak*cell
    X_peak = Activation(X_peak, method=activation)  # 防止出现负值
    rank = np.arange(len(X_peak))[np.mean(X_peak, axis=1).argsort().argsort()]
    param_pm = param_pm[rank]

    if two_embeds:
        X_lib = np.dot(lib_size_effect, embeds_lib).ravel()
    else:
        X_lib = np.dot(lib_size_effect, embeds_peak).ravel()
    rank = np.arange(len(X_lib))[X_lib.argsort().argsort()]
    param_lib = param_lib[rank]
    param_nozero = param_nozero[rank]

    # 对参数进行修正
    # X_peak维度是peak*cell
    simu_param_peak = X_peak
    if distribution == 'Poisson':
        for i in range(correct_iter):
            print('correct_iter ' + str(i + 1))
            simu_param_peak = simu_param_peak / (np.sum(simu_param_peak, axis=1).reshape(-1, 1) + 1e-8) * (
                (np.exp(param_pm) - 1).reshape(-1, 1)) * simu_param_peak.shape[1]
            simu_param_peak = simu_param_peak / (np.sum(simu_param_peak, axis=0).reshape(1, -1) + 1e-8) * (
                (np.exp(param_lib) - 1).reshape(1, -1))

        simu_param_lib = np.exp(param_lib) - 1
        simu_param_nozero = np.exp(param_nozero) - 1
        # --------使用poisson分布生成ATAC
        lambdas = simu_param_peak
        # lambdas=simu_param_peak*(simu_param_lib.reshape(1,-1))

        # 对sparsity进行修正
        lambdas_sum = np.sum(lambdas, axis=0)

        print("**********start ZIP correction...**********")
        k_list, pi_list = [], []
        # 求解每个cell中lambda扩大的倍数和置零的比例
        for i in range(n_cell_total):
            iter_ = i

            # print(i)
            def solve_function(unsolved_value):
                k, pi = unsolved_value[0], unsolved_value[1]
                return [
                    k * (1 - pi) - simu_param_lib[iter_] / (lambdas_sum[iter_]),
                    n_peak * pi + (1 - pi) * np.sum(np.exp(-lambdas[:, iter_] * k)) - (
                                n_peak - simu_param_nozero[iter_])
                ]

            solved = fsolve(solve_function, [3, 0.5], maxfev=2000)
            k, pi = solved[0], solved[1]
            simu1 = k * (1 - pi) * (lambdas_sum[iter_])
            real1 = simu_param_lib[iter_]
            if abs(simu1 - real1) / real1 > 0.1:
                print('=================================')
                print(i)
                print(simu1, real1)
                # print('=================================')
                solved = fsolve(solve_function, [20, 0.5], maxfev=2000)
            simu1 = solved[0] * (1 - solved[1]) * (lambdas_sum[iter_])
            real1 = simu_param_lib[iter_]
            if abs(simu1 - real1) / real1 > 0.1:
                print(i)
                print(simu1, real1)
                print("=================================")

            k_list.append(solved[0])
            pi_list.append(solved[1])
        # 对每个cell的lambda置零并扩大相应倍数
        for i in range(n_cell_total):
            if k_list[i] == 3 or k_list[i] == 20 or pi_list[i] < 0:
                continue
            a = lambdas[:, i] * k_list[i]
            # print(i)
            # print(k_list[i],pi_list[i])
            # print("=============================")
            # b=atac_counts[:,i]
            a[np.random.choice(n_peak, replace=False, size=int(pi_list[i] * n_peak))] = 0
            lambdas[:, i] = a
        print("**********ZIP correction finished!**********")

        #         print("**********start ZIP correction...**********")
        #         batch_size = 1000 # 并行数目，全局字典
        #         global k_dict,pi_dict
        #         for i in range(0,n_cell_total,batch_size):
        #             if i+batch_size<=n_cell_total:
        #                 my_thread = [zip_correction_thread(j,simu_param_lib[j],lambdas[:,j],lambdas_sum[j],simu_param_nozero[j],n_peak) for j in range(i, i+batch_size)]
        #             else:
        #                 my_thread = [zip_correction_thread(j,simu_param_lib[j],lambdas[:,j],lambdas_sum[j],simu_param_nozero[j],n_peak) for j in range(i, n_cell_total)]
        #             for thread_ in my_thread:
        #                 thread_.start()
        #             for thread_ in my_thread:
        #                 thread_.join()
        #         # 对每个cell的lambda置零并扩大相应倍数
        #         for i in range(n_cell_total):
        #             if k_dict[i]==3 or k_dict[i]==20 or pi_dict[i]<0:
        #                 continue
        #             a=lambdas[:,i]*k_dict[i]
        #             # b=atac_counts[:,i]
        #             a[np.random.choice(n_peak,replace=False,size=int(pi_dict[i]*n_peak))]=0
        #             lambdas[:,i]=a

        #         print("**********ZIP correction finished!**********")

        atac_counts = np.random.poisson(lambdas, lambdas.shape)
    elif distribution == 'Bernoulli':
        for i in range(correct_iter):
            print('correct_iter ' + str(i + 1))
            simu_param_peak = Bernoulli_pm_correction(simu_param_peak, param_pm)
            simu_param_peak = Bernoulli_lib_correction(simu_param_peak, param_lib)
        atac_counts = np.random.binomial(1, p=simu_param_peak, size=simu_param_peak.shape)

    return atac_counts


def kl_div(peak_count, peak_count_simu):
    # -------- K-L散度
    peak_count_combine = np.concatenate((peak_count, peak_count_simu))
    value = np.sort(np.unique(peak_count_combine))
    value_count_ori, value_count_simu = [], []
    for value_ in value:
        value_count_ori.append(len(np.where(peak_count == value_)[0]))
        value_count_simu.append(len(np.where(peak_count_simu == value_)[0]))

    value_count_ori = np.array(value_count_ori)
    value_count_ori = value_count_ori / sum(value_count_ori)
    value_count_simu = np.array(value_count_simu)
    value_count_simu = value_count_simu / sum(value_count_simu)

    epsilon = 0.00001
    value_count_ori += epsilon
    value_count_simu += epsilon

    # print('KL divergence:',sum(rel_entr(value_count_ori, value_count_simu)))
    return sum(rel_entr(value_count_ori, value_count_simu))


def zero_logser(peak_count):
    peak_count_new = np.delete(peak_count, np.where(peak_count == 0))
    zero_prob_ = len(np.where(peak_count == 0)[0]) / len(peak_count)

    def solve_function(unsolved_value):
        p = unsolved_value[0]
        return [
            -1 * p / (np.log(1 - p) * (1 - p)) - np.mean(peak_count_new)
        ]

    solved = fsolve(solve_function, [0.995], maxfev=2000)
    p = solved[0]
    # print(-1*p/(np.log(1-p)*(1-p)),np.mean(peak_count_new))
    peak_count_simu = logser.rvs(p, size=len(peak_count)) * \
                      stats.bernoulli.rvs(p=1 - zero_prob_, size=len(peak_count))

    return peak_count_simu


def one_logser(peak_count):
    zero_prob_ = len(np.where(peak_count == 0)[0]) / len(peak_count)
    one_prob = len(np.where(peak_count == 1)[0]) / len(peak_count)
    peak_count_new = np.delete(peak_count, np.where(peak_count == 0))
    peak_count_new = np.delete(peak_count_new, np.where(peak_count_new == 1)) - 1
    # 固定0、1的概率
    idx_all = range(len(peak_count))
    idx_zero = np.random.choice(idx_all, replace=False, size=int(len(peak_count) * (zero_prob_)))
    idx_one = np.random.choice(np.delete(idx_all, idx_zero), replace=False, size=int(len(peak_count) * (one_prob)))

    def solve_function(unsolved_value):
        p = unsolved_value[0]
        return [
            -1 * p / (np.log(1 - p) * (1 - p)) - np.mean(peak_count_new)
        ]

    solved = fsolve(solve_function, [0.995], maxfev=2000)
    p = solved[0]
    # print(-1*p/(np.log(1-p)*(1-p)),np.mean(peak_count_new))

    peak_count_simu = logser.rvs(p, size=len(peak_count)) + 1
    peak_count_simu[idx_zero] = 0
    peak_count_simu[idx_one] = 1

    return peak_count_simu


def ZINB(peak_count):
    model_zinb = ZeroInflatedNegativeBinomialP(peak_count, np.ones_like(peak_count), p=1)
    res_zinb = model_zinb.fit(method='bfgs', maxiter=5000, maxfun=5000)
    mu = np.exp(res_zinb.params[1])
    alpha = res_zinb.params[2]
    pi = expit(res_zinb.params[0])

    p = 1 / (1 + alpha)
    n = mu * p / (1 - p)

    peak_count_simu = (nbinom.rvs(n, p, size=len(peak_count))) * \
                      stats.bernoulli.rvs(p=1 - pi, size=len(peak_count))

    return peak_count_simu


def zero_NB(peak_count):
    zero_prob = len(np.where(peak_count == 0)[0]) / len(peak_count)
    peak_count_new = np.delete(peak_count, np.where(peak_count == 0))
    res = sm.NegativeBinomial(peak_count_new - 1, np.ones_like(peak_count_new)).fit(start_params=[1, 1])
    mu = np.exp(res.params[0])
    p = 1 / (1 + mu * res.params[1])
    n = mu * p / (1 - p)

    peak_count_simu = (nbinom.rvs(n, p, size=len(peak_count)) + 1) * \
                      stats.bernoulli.rvs(p=1 - zero_prob, size=len(peak_count))

    return peak_count_simu


def NB(peak_count):
    res = sm.NegativeBinomial(peak_count, np.ones_like(peak_count)).fit(start_params=[1, 1])
    mu = np.exp(res.params[0])
    p = 1 / (1 + mu * res.params[1])
    n = mu * p / (1 - p)

    peak_count_simu = nbinom.rvs(n, p, size=len(peak_count))

    return peak_count_simu


def ZIP(peak_count):
    zip_model = ZeroInflatedPoisson(endog=peak_count, exog=np.ones_like(peak_count))
    zip_res = zip_model.fit()
    mu = zip_res.params[1]
    pi = expit(zip_res.params[0])
    peak_count_simu = stats.bernoulli.rvs(p=1 - pi, size=len(peak_count)) * \
                      stats.poisson.rvs(mu=mu, size=len(peak_count))

    return peak_count_simu


def Get_Celltype_Counts(adata_part, two_embeds, embed_mean_same, embed_sd_same,
                        n_embed_diff, n_embed_same, correct_iter=10, lib_simu='real', n_cell_total=None,
                        distribution='Poisson', activation='sigmod'
                        , bw_pm=1e-4, bw_lib=0.05, bw_nozero=0.05,
                        rand_seed=0):  # 如果lib_simu为‘estimate’则需要提供对应的n_cell_total

    # np.random.seed(rand_seed)
    # 计算真实参数
    peak_mean = np.log(cal_pm(adata_part) + 1)
    lib_size = np.log(cal_lib(adata_part) + 1)
    nozero = np.log(cal_nozero(adata_part) + 1)
    peak_count = cal_peak_count(adata_part)

    if distribution == 'Bernoulli' and np.max(np.exp(peak_mean) - 1) > 1:
        raise ValueError('you data may not be Bernoulli distribution!')

    n_peak = len(peak_mean)
    n_cell_total = len(lib_size)  # 总共的细胞数目
    if lib_simu == 'real':
        # param_lib=lib_size
        param_lib = np.sort(np.random.choice(lib_size, size=n_cell_total), axis=0).ravel()
        param_nozero = np.sort(np.random.choice(nozero, size=n_cell_total), axis=0).ravel()
    elif lib_simu == 'estimate':
        # kde_lib = KernelDensity(kernel='gaussian', bandwidth=bw_lib).fit(lib_size.reshape(-1,1))
        # param_lib=kde_lib.sample(n_cell_total,random_state=rand_seed)
        # param_lib=np.sort(param_lib)

        estimation_dis = 'one_logser'  # 'NB'/'one_logser'/'gamma'/'zero_logser'

        print('the estimation method is ', estimation_dis)

        if estimation_dis == 'gamma':
            peak_mean_real = np.exp(peak_mean) - 1
            peak_mean_sqrt = np.sqrt(peak_mean_real)

            fit_alpha, fit_loc, fit_beta = stats.gamma.fit(peak_mean_sqrt, floc=np.min(peak_mean_sqrt) - 0.001)
            peak_mean_sqrt_sample = stats.gamma.rvs(a=fit_alpha, loc=fit_loc, scale=fit_beta, size=n_peak,
                                                    random_state=rand_seed)
            param_pm = np.sort(peak_mean_sqrt_sample)
            param_pm = np.log(param_pm ** 2 + 1)
        elif estimation_dis == 'zero_logser':
            peak_count_simu = zero_logser(peak_count)
            param_pm = np.log(peak_count_simu / n_cell_total + 1)
            param_pm = np.sort(param_pm)
        elif estimation_dis == 'one_logser':
            peak_count_simu = one_logser(peak_count)
            param_pm = np.log(peak_count_simu / n_cell_total + 1)
            param_pm = np.sort(param_pm)
        elif estimation_dis == 'zero_NB':
            peak_count_simu = zero_NB(peak_count)
            param_pm = np.log(peak_count_simu / n_cell_total + 1)
            param_pm = np.sort(param_pm)
        elif estimation_dis == 'NB':
            peak_count_simu = NB(peak_count)
            param_pm = np.log(peak_count_simu / n_cell_total + 1)
            param_pm = np.sort(param_pm)
        elif estimation_dis == 'ZIP':
            peak_count_simu = ZIP(peak_count)
            param_pm = np.log(peak_count_simu / n_cell_total + 1)
            param_pm = np.sort(param_pm)

        elif estimation_dis == 'ZINB':
            peak_count_simu = ZINB(peak_count)
            param_pm = np.log(peak_count_simu / n_cell_total + 1)
            param_pm = np.sort(param_pm)

        else:
            raise ValueError('wrong estimation distribution!')

        lib_size_real = np.exp(lib_size) - 1
        lib_size_log = np.log(lib_size_real)

        # n,random_state = 2,2022
        gmm_lz = GMM(2, random_state=rand_seed)
        gmm_lz.fit(lib_size_log.reshape(-1, 1))
        # [sample[0] for sample in gmm.sample(1000)]
        lib_size_log_sample = gmm_lz.sample(n_cell_total)[0].reshape(-1)
        param_lib = np.sort(lib_size_log_sample)

        non_zero_real = np.exp(nozero) - 1
        non_zero_log = np.log(non_zero_real)
        gmm_nz = GMM(2, random_state=rand_seed)
        gmm_nz.fit(non_zero_log.reshape(-1, 1))
        # [sample[0] for sample in gmm.sample(1000)]
        non_zero_log_sample = gmm_nz.sample(n_cell_total)[0].reshape(-1)
        param_nozero = np.log(np.exp(np.sort(non_zero_log_sample)) + 1)

    param_pm = param_pm[peak_mean.argsort().argsort()]
    # param_pm=np.sort(peak_mean)
    # origin_peak=np.arange(len(peak_mean))[peak_mean.argsort()]#记录实际peak的位置，保证最后输出的与输入peak含义一致

    # 生成effect和embedding
    peak_effect, lib_size_effect = Get_Effect(n_peak, n_cell_total,
                                              len_cell_embed, rand_seed, zero_prob, zero_set, effect_mean, effect_sd)

    # if simu_type=='single':
    embeds_param = {}
    embeds_param['peak'], meta = Get_Single_Embedding(n_cell_total, embed_mean_same, embed_sd_same,
                                                      n_embed_diff, n_embed_same)
    embeds_param['lib_size'], meta = Get_Single_Embedding(n_cell_total, embed_mean_same, embed_sd_same,
                                                          n_embed_diff, n_embed_same)

    # 从模拟矩阵的参数顺序对应到采样的真实参数
    X_peak = np.dot(peak_effect, embeds_param['peak'].values)  # peak*cell
    X_peak = Activation(X_peak, method=activation)
    # rank=np.arange(len(X_peak))[np.mean(X_peak,axis=1).argsort().argsort()]
    # param_pm=param_pm[rank]
    # origin_peak=origin_peak[rank]

    if two_embeds:
        X_lib = np.dot(lib_size_effect, embeds_param['lib_size'].values).ravel()
    else:
        X_lib = np.dot(lib_size_effect, embeds_param['peak'].values).ravel()
    rank = np.arange(len(X_lib))[X_lib.argsort().argsort()]
    param_lib = param_lib[rank]
    param_nozero = param_nozero[rank]

    # 对参数进行修正
    # X_peak维度是peak*cell
    simu_param_peak = X_peak
    if distribution == 'Poisson':
        for i in range(correct_iter):
            print('correct_iter ' + str(i + 1))
            simu_param_peak = simu_param_peak / (np.sum(simu_param_peak, axis=1).reshape(-1, 1) + 1e-8) * (
                (np.exp(param_pm) - 1).reshape(-1, 1)) * simu_param_peak.shape[1]
            simu_param_peak = simu_param_peak / (np.sum(simu_param_peak, axis=0).reshape(1, -1) + 1e-8) * (
                (np.exp(param_lib) - 1).reshape(1, -1))  # 分母加一个很小的数防止nan

        simu_param_lib = np.exp(param_lib) - 1
        simu_param_nozero = np.exp(param_nozero) - 1
        simu_param_pm = np.exp(param_pm) - 1
        # --------使用poisson分布生成ATAC
        lambdas = simu_param_peak
        # lambdas=lambdas[origin_peak.argsort(),:] #保证peak与输入peak一致

        atac_counts = np.random.poisson(lambdas, lambdas.shape)
    elif distribution == 'Bernoulli':
        for i in range(correct_iter):
            print('correct_iter ' + str(i + 1))
            simu_param_peak = Bernoulli_pm_correction(simu_param_peak, param_pm)
            simu_param_peak = Bernoulli_lib_correction(simu_param_peak, param_lib)
        atac_counts = np.random.binomial(1, p=simu_param_peak, size=simu_param_peak.shape)

        lambdas, simu_param_nozero, simu_param_lib, simu_param_pm = None, None, None, None

    return atac_counts, embeds_param['peak'].values, embeds_param[
        'lib_size'].values, lambdas, simu_param_nozero, simu_param_lib, simu_param_pm


prefix_ = 'Buenrostro_2018'
resultdir = '../../data/simulated/' + prefix_ + '/'
peak_mean = pd.read_csv(resultdir+'peak_mean_log.csv',index_col=0)
lib_size = pd.read_csv(resultdir+'library_size_log.csv',index_col=0)
nozero = pd.read_csv(resultdir+'nozero_log.csv',index_col=0)

peak_mean = np.array(peak_mean['peak mean'])
lib_size = np.array(lib_size['library size'])
nozero = np.array(nozero['nozero'])

n_peak = len(peak_mean)  # peak数目
n_cell_total = 10000  # 总共的细胞数目
rand_seed = 2022  # 随机种子
zero_prob = 0.5  # 对于peak_effect的置零个数
zero_set = 'all'  # 'by_row'指的是对于每一个peak的effect vector进行置零；'all'指的是随机在所有的index中选择进行置零
effect_mean = 0  # 生成effect vector的均值
effect_sd = 1  # 生成effect vector的方差

min_popsize = 50  # 离散模式下设定的细胞群的最小数目
min_pop = '0'  # 离散模式下设定最小细胞群的名称，注意需要与下面的tree_text一致
tree_text = ["(((0:0.2,1:0.2):0.2,2:0.4):0.5,(3:0.5,4:0.5):0.4);"]
pops_name = [['0', '1', '2', '3', '4']]  # 输入不同节点的名字，离散模式只需要输入叶子节点的名称就行，注意这里需要与tree_text的前三个顺序保持一致
pops_size = [2000, 2000, 2000, 2000, 2000]  # 设置不同cluster的细胞数目，None则直接取平均

embed_mean_same = 1  # 对embedding非差异特征采样的均值
embed_sd_same = 0.5  # 对embedding的非差异特征采样的方差
embed_mean_diff = 1  # 对embedding差异特征采样的均值
embed_sd_diff = 0.5  # 对embedding的差异特征采样的方差

len_cell_embed = 12  # 仿真细胞的低维特征的特征个数
n_embed_diff = 10  # 使得cell embedding不同的特征维度数目
n_embed_same = len_cell_embed - n_embed_diff

simu_type = 'discrete'  # continuous/discrete/single/cell_type
correct_iter = 2  # 使用参数进行修正的迭代次数
activation = 'exp_linear'  # 对参数矩阵矫正的方式，在连续和离散的条件下使用'exp'，在仿真celltype的时候应该使用'sigmod'

two_embeds = True  # true表明peak mean和library size通过两个不同的矩阵排序对应得到；False 表明通过一个矩阵的值排序对应得到

adata_dir = resultdir + 'adata_forsimulation.h5ad'  # 为了进行cell_type simulation
lib_simu = 'estimate'  # 在仿真cell_type时用的参数，’real‘表示直接使用真实的library_size参数，‘estimate’表示从估计的分布中采样
distribution = 'Poisson'  # 数据的分布，如果二值化就是’Bernoulli‘，count就是‘Poisson’

bw_pm = 1e-4  # 分别为对peak mean、library_size、nozero的核密度估计的窗宽；注：bw_pm若取的过大可能会导致采样的peak mean小于0而报错
bw_lib = 0.05
bw_nozero = 0.05

real_param = False  # 是否使用真实的参数，True则为直接使用真实参数，False

log = None

fix_seed(rand_seed)

Tech_noise = True
Tech_params = [0, 0.1]
Bio_noise = True
Bio_params = [0, 0.5]

# 生成effect和embedding
print("**********start generate effect vector...**********")
peak_effect, lib_size_effect = Get_Effect(n_peak, n_cell_total, len_cell_embed, rand_seed,
                                          zero_prob, zero_set, effect_mean, effect_sd)
print("**********generate effect finished!**********")

print("**********start generate cell embedding...**********")
embeds_peak, meta = Get_Discrete_Embedding(pops_name[0], min_popsize, tree_text[0], n_cell_total, pops_size,
                                           embed_mean_same, embed_sd_same, embed_mean_diff, embed_sd_diff,
                                           n_embed_diff, n_embed_same, rand_seed)
embeds_lib, meta = Get_Discrete_Embedding(pops_name[0], min_popsize, tree_text[0], n_cell_total, pops_size,
                                          embed_mean_same, embed_sd_same, embed_mean_diff, embed_sd_diff,
                                          n_embed_diff, n_embed_same, rand_seed+1)
embeds_peak, embeds_lib = embeds_peak.values, embeds_lib.values
print("**********generate cell embedding finished**********")

# np.random.seed(rand_seed)
if distribution == 'Bernoulli' and np.max(np.exp(peak_mean) - 1) > 1:
    raise ValueError('you data may not be Bernoulli distribution!')

if real_param:  # 如果直接使用真实参数，peak mean直接按照真实参数来，lib size抽样
    param_pm = np.sort(peak_mean, axis=0).ravel()
    param_lib = np.sort(np.random.choice(lib_size, size=n_cell_total), axis=0).ravel()
    param_nozero = np.sort(np.random.choice(nozero, size=n_cell_total), axis=0).ravel()
else:
    # kde
    kde_pm = KernelDensity(kernel='gaussian', bandwidth=bw_pm).fit(peak_mean.reshape(-1, 1))
    kde_lib = KernelDensity(kernel='gaussian', bandwidth=bw_lib).fit(lib_size.reshape(-1, 1))
    kde_nozero = KernelDensity(kernel='gaussian', bandwidth=bw_nozero).fit(nozero.reshape(-1, 1))

    # 从kde中采样并进行排序（从小到大）
    param_pm = kde_pm.sample(n_peak, random_state=rand_seed)
    param_lib = kde_lib.sample(n_cell_total, random_state=rand_seed)
    param_nozero = kde_nozero.sample(n_cell_total, random_state=rand_seed)

    param_pm = np.sort(param_pm, axis=0).ravel()
    param_lib = np.sort(param_lib, axis=0).ravel()
    param_nozero = np.sort(param_nozero, axis=0).ravel()

    # print(kde_pm, param_pm)

# 从模拟矩阵的参数顺序对应到采样的真实参数
if Bio_noise:  # add noise to peak effect (biological batch effect)
    noise = np.random.normal(Bio_params[0], Bio_params[1], peak_effect.shape)
    X_peak = np.dot(peak_effect + noise, embeds_peak)  # peak*cell
else:
    X_peak = np.dot(peak_effect, embeds_peak)  # peak*cell
print(X_peak.shape)
X_peak = Activation(X_peak, method=activation)  # 防止出现负值
rank = np.arange(len(X_peak))[np.mean(X_peak, axis=1).argsort().argsort()]
param_pm = param_pm[rank]

if two_embeds:
    X_lib = np.dot(lib_size_effect, embeds_lib).ravel()
else:
    X_lib = np.dot(lib_size_effect, embeds_peak).ravel()
rank = np.arange(len(X_lib))[X_lib.argsort().argsort()]
param_lib = param_lib[rank]
param_nozero = param_nozero[rank]

# 对参数进行修正
# X_peak维度是peak*cell
simu_param_peak = X_peak.copy()
del X_peak
if distribution == 'Poisson':
    for i in range(correct_iter):
        print('correct_iter ' + str(i + 1))
        simu_param_peak = simu_param_peak / (np.sum(simu_param_peak, axis=1).reshape(-1, 1) + 1e-8) * (
            (np.exp(param_pm) - 1).reshape(-1, 1)) * simu_param_peak.shape[1]
        simu_param_peak = simu_param_peak / (np.sum(simu_param_peak, axis=0).reshape(1, -1) + 1e-8) * (
            (np.exp(param_lib) - 1).reshape(1, -1))

    simu_param_lib = np.exp(param_lib) - 1
    simu_param_nozero = np.exp(param_nozero) - 1
    # --------使用poisson分布生成ATAC
    lambdas = simu_param_peak.copy()
    del simu_param_peak
    # lambdas=simu_param_peak*(simu_param_lib.reshape(1,-1))

    # 对sparsity进行修正
    lambdas_sum = np.sum(lambdas, axis=0)

    lambdas_sum_copy = lambdas_sum.copy()

    print("**********start ZIP correction...**********")
    batch_size = 1000  # 并行数目，全局字典
    k_dict, pi_dict = {}, {}
    for i in range(0, n_cell_total, batch_size):
        if i + batch_size <= n_cell_total:
            my_thread = [
                zip_correction_thread(j, simu_param_lib[j], lambdas[:, j], lambdas_sum[j], simu_param_nozero[j], n_peak)
                for j in range(i, i + batch_size)]
        else:
            my_thread = [
                zip_correction_thread(j, simu_param_lib[j], lambdas[:, j], lambdas_sum[j], simu_param_nozero[j], n_peak)
                for j in range(i, n_cell_total)]
        for thread_ in my_thread:
            thread_.start()
        for thread_ in my_thread:
            thread_.join()
    # 对每个cell的lambda置零并扩大相应倍数
    for i in range(n_cell_total):
        if Bio_noise:
            if k_dict[i] == 3 or k_dict[i] == 20 or pi_dict[i] < 0 or pi_dict[i] > 1:
                continue
        else:
            if k_dict[i] == 3 or k_dict[i] == 20 or pi_dict[i] < 0:
                continue
        a = lambdas[:, i] * k_dict[i]
        # b=atac_counts[:,i]
        a[np.random.choice(n_peak, replace=False, size=int(pi_dict[i] * n_peak))] = 0
        lambdas[:, i] = a

    print("**********ZIP correction finished!**********")

    if not Tech_noise:
        atac_counts = np.random.poisson(lambdas, lambdas.shape)
elif distribution == 'Bernoulli':
    for i in range(correct_iter):
        print('correct_iter ' + str(i + 1))
        simu_param_peak = Bernoulli_pm_correction(simu_param_peak, param_pm)
        simu_param_peak = Bernoulli_lib_correction(simu_param_peak, param_lib)
    atac_counts = np.random.binomial(1, p=simu_param_peak, size=simu_param_peak.shape)

# noise=np.random.normal(8,0.5,lambdas.shape)
# lambdas_noise=lambdas+noise
# lambdas_noise[lambdas_noise<0]=0
# temp=lambdas.copy()
# temp[temp>0]=1
# lambdas_noise=lambdas_noise*temp
# atac_counts=np.random.poisson(lambdas_noise, lambdas_noise.shape)

import anndata

# 生成anndata并按照每个celltype的平均library size按比例添加batch effect
if Tech_noise:

    adata_noise = anndata.AnnData(X=csr_matrix(lambdas.T))
    adata_noise.obs['celltype'] = meta
    del lambdas
    # adata_noise.var=adata.var

    celltype_noise = np.unique(adata_noise.obs.celltype)
    lib_size_list = []
    for celltype_ in celltype_noise:
        # adata_part = adata_noise[adata_noise.obs.celltype.isin([celltype_]), :]
        # lib_size_tmp = np.mean(cal_lib(adata_part))
        lib_size_tmp = np.mean(cal_lib(adata_noise[adata_noise.obs.celltype.isin([celltype_]), :]))
        lib_size_list.append(lib_size_tmp)
    min_lib_size = min(lib_size_list)

    array_list = []
    for i, celltype_ in enumerate(celltype_noise):  # add noise to parameter matrix (technical batch effect)
        # adata_part = adata_noise[adata_noise.obs.celltype.isin([celltype_]), :]
        # temp = adata_part.X.copy()
        temp = adata_noise[adata_noise.obs.celltype.isin([celltype_]), :].X.toarray()
        print('the noise is:', Tech_params[0] * lib_size_list[i] / min_lib_size)
        lambdas_tmp = temp + np.random.normal(Tech_params[0] * lib_size_list[i] / min_lib_size, Tech_params[1], temp.shape)
        lambdas_tmp[lambdas_tmp < 0] = 0
        temp[temp > 0] = 1
        lambdas_tmp = lambdas_tmp * temp
        array_list.append(csr_matrix(lambdas_tmp))

    del adata_noise, temp, lambdas_tmp
    lambdas_noise = vstack(array_list).T
    del array_list
    atac_counts = np.random.poisson(lambdas_noise.toarray(), lambdas_noise.shape)
    del lambdas_noise

# 创建对应的文件夹
mode = f'Tech_{Tech_params[0]}_{Tech_params[1]}_Bio_{Bio_params[0]}_{Bio_params[1]}'
resultdir = "../../data/simulated/" + mode + '/'
if not os.path.exists(resultdir):
    os.makedirs(resultdir)

prefix='_'.join([simu_type,'embed'+str(len_cell_embed),
                'diff'+str(n_embed_diff),'cell'+str(n_cell_total),
                'EmbedSd'+str(embed_sd_diff),'EffectSd'+str(effect_sd),'prob'+str(zero_prob),
                'CorrectIter'+str(correct_iter),'TwoEmbedd'+str(two_embeds),
                'activation_'+activation+f'_{mode}'])
# os.makedirs(os.path.join(resultdir,prefix),exist_ok=True)

with open(resultdir + 'mode.txt', 'w', encoding='utf-8') as file:
    file.write(prefix)

print(prefix)

# save mtx
# sio.mmwrite(os.path.join(resultdir, "matrix.mtx"), csr_matrix(atac_counts.T))

sio.mmwrite(os.path.join(resultdir, "matrix_embed_pm.mtx"),sparse.csr_matrix(embeds_peak.T))
sio.mmwrite(os.path.join(resultdir, "matrix_embed_lib.mtx"),sparse.csr_matrix(embeds_lib.T))

# df=pd.DataFrame({'pop':meta})
# df.to_csv(os.path.join(resultdir, "meta.tsv"),sep=' ')

adata_simulated = anndata.AnnData(csr_matrix(atac_counts.T))
adata_simulated.obs['celltype'] = meta
adata_simulated.write_h5ad(os.path.join(resultdir, f"sc_simulated_{mode}.h5ad"))







