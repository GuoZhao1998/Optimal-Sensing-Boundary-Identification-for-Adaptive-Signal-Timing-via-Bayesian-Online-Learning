# ===================== Linux system compatibility configuration =====================
# ===================== 线程配置（CPU侧单线程，避免GPU调度冲突） =====================
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
# ======================================================================================

import numpy as np
import cupy as cp

# ===================== 1. 全局参数配置 =====================
DATASET_PATH = "dataset_3s0_5000m.npz"
GRID_SIZE = 18.0
CHANNELS_PER_BIN = 4

# --- 可分性驱动贝叶斯参数 ---
PHI_SEP = 1.0
LENGTH_PENALTY = 1.5e-5
EPSILON = 0.02

# --- k-NN基线参数 ---
K_DEFAULT = 2
THETA_DEFAULT = 4.0

# --- 敏感性分析配置 ---
K_CANDIDATES = np.array([1,2, 3,4, 5, 9], dtype=np.int64)
THETA_CANDIDATES = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
N_PHASES = 10

# --- 数据集划分 ---
TRAIN_RATIO = 0.8
RANDOM_SHUFFLE = True
SEED = 42

# ===================== 2. GPU核心函数 =====================
def maximum_a_posteriori_estimation(posterior, hypo_space):
    """使用CuPy原生运算，支持GPU数组"""
    return hypo_space[cp.argmax(posterior)]


def bayesian_update_sep(new_o_n, new_o_v, new_lambda, hist_o_n, hist_o_v, hist_labels,
                        hypo_space, prior_beliefs, Phi, penalty, eps):
    """可分性驱动贝叶斯更新：绝对区分度 + 长度惩罚"""
    n_hypo = len(hypo_space)
    if len(prior_beliefs) == 0:
        prior_beliefs = cp.ones(n_hypo, dtype=cp.float64) / n_hypo

    # 同类/异类掩码
    tra_mask = cp.abs(hist_labels - new_lambda) <= eps
    count_tra = cp.sum(tra_mask)
    count_ter = len(hist_labels) - count_tra

    # 前缀和预计算
    diff_n = hist_o_n - new_o_n
    diff_v = hist_o_v - new_o_v
    sum_sq = diff_n * diff_n + diff_v * diff_v
    prefix_sum = cp.cumsum(sum_sq, axis=1)

    # 批量计算所有候选长度的距离
    ell_idx = hypo_space - 1
    total_sq = prefix_sum[:, ell_idx]
    norm_factor = cp.sqrt(2.0 * hypo_space)
    d_norm = cp.sqrt(total_sq) / norm_factor

    # 同类/异类平均距离
    D_tra = cp.sum(d_norm[tra_mask], axis=0) / count_tra if count_tra > 0 else cp.full(n_hypo, 0.5)
    D_ter = cp.sum(d_norm[~tra_mask], axis=0) / count_ter if count_ter > 0 else cp.full(n_hypo, 0.5)

    # 绝对区分度 + 长度惩罚
    abs_gap = D_ter - D_tra
    score = abs_gap #- penalty * hypo_space

    # 贝叶斯更新
    unnorm_post = cp.exp(Phi * score) * prior_beliefs
    Z = cp.sum(unnorm_post)
    return unnorm_post / Z if Z > 0 else prior_beliefs.copy()


def knn_predict_batch(test_n, test_v, train_n, train_v, train_labels, ell, k_hat, theta, batch_size=256):
    """GPU批量k-NN预测，矩阵乘法加速"""
    n_train = len(train_labels)
    n_test = len(test_n)
    actual_k = min(k_hat, n_train)
    preds = cp.zeros(n_test, dtype=cp.float64)

    train_n_slice = train_n[:, :ell]
    train_v_slice = train_v[:, :ell]
    train_norm_sq = cp.sum(train_n_slice**2 + train_v_slice**2, axis=1)
    inv_norm = 1.0 / cp.sqrt(2.0 * ell)

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        b_size = end - start

        batch_n = test_n[start:end, :ell]
        batch_v = test_v[start:end, :ell]
        batch_norm_sq = cp.sum(batch_n**2 + batch_v**2, axis=1, keepdims=True)

        inner_n = batch_n @ train_n_slice.T
        inner_v = batch_v @ train_v_slice.T
        dist_sq = batch_norm_sq + train_norm_sq[None, :] - 2 * (inner_n + inner_v)
        dist_sq = cp.maximum(dist_sq, 0.0)
        d_norm = cp.sqrt(dist_sq) * inv_norm

        sim = cp.exp(-theta * d_norm)
        top_k_idx = cp.argpartition(sim, -actual_k, axis=1)[:, -actual_k:]
        top_sim = cp.take_along_axis(sim, top_k_idx, axis=1)
        top_lbl = cp.take_along_axis(train_labels[None, :].repeat(b_size, axis=0), top_k_idx, axis=1)

        denom = cp.sum(top_sim, axis=1)
        num = cp.sum(top_sim * top_lbl, axis=1)
        preds[start:end] = cp.where(denom > 0, num / denom, cp.mean(top_lbl, axis=1))

    return preds


# ===================== 3. 数据加载 =====================
def load_data(path):
    data = np.load(path)
    feat_n = data['features_counts']
    feat_v = data['features_avgs']
    labels = data['optimal_lam']
    n_samples = len(labels)
    feat_dim = feat_n.shape[1]

    print("="*70)
    print("可分性驱动贝叶斯分析 GPU版")
    print("="*70)
    print(f"\n【数据集信息】")
    print(f"  总样本数: {n_samples}")
    print(f"  特征维度: {feat_dim} → 对应 {feat_dim//CHANNELS_PER_BIN*GRID_SIZE:.0f}m")
    print(f"  标签范围: [{labels.min():.4f}, {labels.max():.4f}]")

    if RANDOM_SHUFFLE:
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(n_samples)
        feat_n = feat_n[perm]
        feat_v = feat_v[perm]
        labels = labels[perm]
        print(f"  已按种子 {SEED} 打乱")

    # 迁移GPU
    return cp.asarray(feat_n), cp.asarray(feat_v), cp.asarray(labels), feat_dim


def ell2dist(ell):
    return ell / CHANNELS_PER_BIN * GRID_SIZE


# ===================== 4. 全量贝叶斯在线更新 =====================
def run_bayesian_online(feat_n, feat_v, labels, hypo_space):
    M = len(labels)
    n_hypo = len(hypo_space)
    beliefs = cp.zeros((M, n_hypo), dtype=cp.float64)
    opt_ells = cp.zeros(M, dtype=cp.int64)
    current = cp.array([], dtype=cp.float64)

    print("\n【贝叶斯在线更新】")
    for i in range(M):
        current = bayesian_update_sep(
            feat_n[i], feat_v[i], labels[i],
            feat_n[:i], feat_v[:i], labels[:i],
            hypo_space, current, PHI_SEP, LENGTH_PENALTY, EPSILON
        )
        beliefs[i] = current
        opt_ells[i] = maximum_a_posteriori_estimation(current, hypo_space)

        if (i+1) % max(1, M//10) == 0 or i == M-1:
            best = int(opt_ells[i].item())
            print(f"  {i+1:5d}/{M} | ℓ* = {best:4d} ({ell2dist(best):6.1f}m) | 峰值后验 = {cp.max(current).item():.4f}")

    final_idx = int(cp.argmax(beliefs[-1]).item())
    final_ell = int(hypo_space[final_idx].item())
    print(f"\n  最终可分性最优 ℓ*_sep = {final_ell} ({ell2dist(final_ell):.1f}m)")
    return beliefs, opt_ells, final_ell


# ===================== 5. 全局MAE全景扫描 =====================
def run_global_mae_scan(feat_n, feat_v, labels, hypo_space, train_ratio):
    M = len(labels)
    split = int(M * train_ratio)
    train_n, train_v, train_lbl = feat_n[:split], feat_v[:split], labels[:split]
    test_n, test_v, test_lbl = feat_n[split:], feat_v[split:], labels[split:]

    print("\n【全局MAE全景扫描】")
    n_hypo = len(hypo_space)
    mae_all = cp.zeros(n_hypo, dtype=cp.float64)

    for idx, ell in enumerate(hypo_space):
        preds = knn_predict_batch(test_n, test_v, train_n, train_v, train_lbl,
                                  int(ell.item()), K_DEFAULT, THETA_DEFAULT)
        mae_all[idx] = cp.mean(cp.abs(preds - test_lbl))

    best_idx = int(cp.argmin(mae_all).item())
    best_ell = int(hypo_space[best_idx].item())
    best_mae = float(mae_all[best_idx].item())
    print(f"  全局MAE最优 ℓ*_offline = {best_ell} ({ell2dist(best_ell):.1f}m)")
    print(f"  最小MAE = {best_mae:.6f}")
    return mae_all, best_ell, best_mae


# ===================== 6. k值敏感性分析（全局+分阶段） =====================
def run_k_sensitivity(feat_n, feat_v, labels, ell_list, ell_names, k_cand, n_phases):
    M = len(labels)
    n_ell = len(ell_list)
    n_k = len(k_cand)
    phase_edges = np.linspace(0, M, n_phases + 1, dtype=int)

    k_mae_global = np.zeros((n_ell, n_k))
    k_mae_phase = np.zeros((n_ell, n_phases, n_k))

    print("\n【k值敏感性分析】")
    for ei, ell in enumerate(ell_list):
        print(f"  处理 {ell_names[ei]} (ℓ={ell}) ...")
        errors_all = cp.zeros(M - 1, dtype=cp.float64)

        for ki, k in enumerate(k_cand):
            for i in range(M - 1):
                pred = knn_predict_batch(
                    feat_n[i+1:i+2], feat_v[i+1:i+2],
                    feat_n[:i+1], feat_v[:i+1], labels[:i+1],
                    ell, int(k), THETA_DEFAULT
                )
                errors_all[i] = cp.abs(pred[0] - labels[i+1])

            k_mae_global[ei, ki] = float(cp.mean(errors_all).item())

            # 分阶段MAE
            for p in range(n_phases):
                s, e = phase_edges[p], phase_edges[p+1] - 1
                if e <= 0 or s >= M-1:
                    k_mae_phase[ei, p, ki] = np.nan
                else:
                    s = max(s, 0)
                    e = min(e, M-1)
                    k_mae_phase[ei, p, ki] = float(cp.mean(errors_all[s:e]).item())

    return k_mae_global, k_mae_phase


# ===================== 7. theta敏感性分析 =====================
def run_theta_sensitivity(feat_n, feat_v, labels, ell_list, ell_names, theta_cand, best_k):
    M = len(labels)
    split = int(M * TRAIN_RATIO)
    train_n, train_v, train_lbl = feat_n[:split], feat_v[:split], labels[:split]
    test_n, test_v, test_lbl = feat_n[split:], feat_v[split:], labels[split:]

    n_ell = len(ell_list)
    n_theta = len(theta_cand)
    theta_mae = np.zeros((n_ell, n_theta))

    print("\n【theta敏感性分析】")
    for ei, ell in enumerate(ell_list):
        print(f"  处理 {ell_names[ei]} (ℓ={ell}) ...")
        for ti, theta in enumerate(theta_cand):
            preds = knn_predict_batch(test_n, test_v, train_n, train_v, train_lbl,
                                      ell, best_k, float(theta))
            theta_mae[ei, ti] = float(cp.mean(cp.abs(preds - test_lbl)).item())

    return theta_mae


# ===================== 8. 独立测试集泛化对比 =====================
def run_generalization_test(feat_n, feat_v, labels, ell_list, ell_names, k, theta):
    M = len(labels)
    split = int(M * TRAIN_RATIO)
    train_n, train_v, train_lbl = feat_n[:split], feat_v[:split], labels[:split]
    test_n, test_v, test_lbl = feat_n[split:], feat_v[split:], labels[split:]

    n_ell = len(ell_list)
    mae_arr = np.zeros(n_ell)
    rmse_arr = np.zeros(n_ell)

    print("\n【独立测试集泛化对比】")
    for i, ell in enumerate(ell_list):
        preds = knn_predict_batch(test_n, test_v, train_n, train_v, train_lbl, ell, k, theta)
        err = cp.abs(preds - test_lbl)
        mae_arr[i] = float(cp.mean(err).item())
        rmse_arr[i] = float(cp.sqrt(cp.mean(err**2)).item())
        print(f"  {ell_names[i]:<15} MAE={mae_arr[i]:.6f}  RMSE={rmse_arr[i]:.6f}")

    return mae_arr, rmse_arr


# ===================== 9. 结果保存 =====================
def save_results(hypo_space, feat_dim,
                 sep_beliefs, sep_opt_ells, sep_final_ell,
                 global_mae_curve, global_best_ell, global_best_mae,
                 k_mae_global, k_mae_phase,
                 theta_mae,
                 gen_mae, gen_rmse,
                 ell_list, ell_names,
                 k_cand, theta_cand, n_phases):
    # GPU数组转CPU numpy
    save_data = {
        # 基础元数据
        "hypothesis_space": cp.asnumpy(hypo_space),
        "full_ell": feat_dim,
        "grid_size": GRID_SIZE,
        "total_samples": int(len(sep_opt_ells)),
        "train_ratio": TRAIN_RATIO,
        "ell_list": np.array(ell_list),
        "ell_names": np.array(ell_names, dtype=object),
        "k_candidates": k_cand,
        "theta_candidates": theta_cand,
        "n_phases": n_phases,

        # 方法1：可分性驱动贝叶斯
        "sep_beliefs": cp.asnumpy(sep_beliefs),
        "sep_opt_ells": cp.asnumpy(sep_opt_ells),
        "sep_final_ell": sep_final_ell,

        # 全局离线基准
        "global_mae_curve": cp.asnumpy(global_mae_curve),
        "global_best_ell": global_best_ell,
        "global_best_mae": global_best_mae,

        # k值敏感性
        "k_mae_global": k_mae_global,
        "k_mae_phase": k_mae_phase,

        # theta敏感性
        "theta_mae": theta_mae,

        # 泛化性能
        "gen_mae": gen_mae,
        "gen_rmse": gen_rmse,

        # 预留：预测误差驱动方法
        "mae_beliefs": None,
        "mae_opt_ells": None,
        "mae_final_ell": None,
    }

    np.savez_compressed("sep_analysis_results.npz", **save_data)
    print("\n 所有结果已保存至 sep_analysis_results.npz")


# ===================== 主流程 =====================
if __name__ == "__main__":
    # 1. 加载数据
    feat_n, feat_v, labels, feat_dim = load_data(DATASET_PATH)
    hypo_space = cp.arange(20, feat_dim + 4, 4, dtype=cp.int64)
    print(hypo_space)
    print(f"单特征维度: {feat_dim}")
    print(f"对应空间bin数: {feat_dim // CHANNELS_PER_BIN}")
    print(f"对应物理范围: {feat_dim // CHANNELS_PER_BIN * GRID_SIZE:.0f} m")


    # 2. 贝叶斯在线更新
    sep_beliefs, sep_opt_ells, sep_final_ell = run_bayesian_online(
        feat_n, feat_v, labels, hypo_space
    )

    # 3. 全局MAE全景扫描
    global_mae_curve, global_best_ell, global_best_mae = run_global_mae_scan(
        feat_n, feat_v, labels, hypo_space, TRAIN_RATIO
    )

    # 4. 三组对比长度
    ell_list = [sep_final_ell, global_best_ell, feat_dim]
    ell_names = ["可分性最优", "全局MAE最优", "全特征"]

    # 5. k值敏感性分析
    k_mae_global, k_mae_phase = run_k_sensitivity(
        feat_n, feat_v, labels, ell_list, ell_names, K_CANDIDATES, N_PHASES
    )
    best_k_global = int(K_CANDIDATES[np.argmin(k_mae_global[0])])  # 主方法对应的最优k
    print(f"\n  主方法最优k = {best_k_global}")

    # 6. theta敏感性分析
    theta_mae = run_theta_sensitivity(
        feat_n, feat_v, labels, ell_list, ell_names, THETA_CANDIDATES, best_k_global
    )

    # 7. 独立测试集泛化对比
    gen_mae, gen_rmse = run_generalization_test(
        feat_n, feat_v, labels, ell_list, ell_names, best_k_global, THETA_DEFAULT
    )

    # 8. 保存全部结果
    save_results(hypo_space, feat_dim,
                 sep_beliefs, sep_opt_ells, sep_final_ell,
                 global_mae_curve, global_best_ell, global_best_mae,
                 k_mae_global, k_mae_phase,
                 theta_mae,
                 gen_mae, gen_rmse,
                 ell_list, ell_names,
                 K_CANDIDATES, THETA_CANDIDATES, N_PHASES)

    print("\n" + "="*70)
    print("全部分析完成")
    print("="*70)
