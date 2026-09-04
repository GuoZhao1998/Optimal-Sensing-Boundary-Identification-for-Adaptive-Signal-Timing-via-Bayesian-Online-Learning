# ===================== Linux system compatibility configuration =====================
# # ===================== 线程配置 =====================
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
# ==================================================

import numpy as np
import cupy as cp

# ===================== 全局参数 =====================
PHI_SEP = 1.0
LENGTH_PENALTY = 0.0
EPSILON = 0.02
K_DEFAULT = 3
THETA_DEFAULT = 4.0
TRAIN_RATIO = 0.8
CHANNELS_PER_BIN = 4
RANDOM_SHUFFLE = True
SEED = 42

# 三组数据集配置
DATASET_CONFIGS = [
    ("dataset_2s0_5000m.npz", 12.0, "2s₀"),
    ("dataset_3s0_5000m.npz", 18.0, "3s₀"),
    ("dataset_4s0_5000m.npz", 24.0, "4s₀"),
]

# ===================== GPU核心函数 =====================
def bayesian_update_sep(new_o_n, new_o_v, new_lambda, hist_o_n, hist_o_v, hist_labels,
                        hypo_space, prior_beliefs, Phi, penalty, eps):
    n_hypo = len(hypo_space)
    if len(prior_beliefs) == 0:
        prior_beliefs = cp.ones(n_hypo, dtype=cp.float64) / n_hypo

    tra_mask = cp.abs(hist_labels - new_lambda) <= eps
    count_tra = cp.sum(tra_mask)
    count_ter = len(hist_labels) - count_tra

    diff_n = hist_o_n - new_o_n
    diff_v = hist_o_v - new_o_v
    sum_sq = diff_n * diff_n + diff_v * diff_v
    prefix_sum = cp.cumsum(sum_sq, axis=1)

    ell_idx = hypo_space - 1
    total_sq = prefix_sum[:, ell_idx]
    norm_factor = cp.sqrt(2.0 * hypo_space)
    d_norm = cp.sqrt(total_sq) / norm_factor

    D_tra = cp.sum(d_norm[tra_mask], axis=0) / count_tra if count_tra > 0 else cp.full(n_hypo, 0.5)
    D_ter = cp.sum(d_norm[~tra_mask], axis=0) / count_ter if count_ter > 0 else cp.full(n_hypo, 0.5)

    abs_gap = D_ter - D_tra
    score = abs_gap - penalty * hypo_space

    unnorm_post = cp.exp(Phi * score) * prior_beliefs
    Z = cp.sum(unnorm_post)
    return unnorm_post / Z if Z > 0 else prior_beliefs.copy()


def knn_predict_batch(test_n, test_v, train_n, train_v, train_labels, ell, k_hat, theta, batch_size=256):
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


# ===================== 工具函数 =====================
def load_data(path, grid_size):
    data = np.load(path)
    feat_n = data['features_counts']
    feat_v = data['features_avgs']
    labels = data['optimal_lam']
    n_samples = len(labels)
    feat_dim = feat_n.shape[1]
    n_bins = feat_dim // CHANNELS_PER_BIN
    total_range = n_bins * grid_size

    # 与主实验完全一致的随机打乱
    if RANDOM_SHUFFLE:
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(n_samples)
        feat_n = feat_n[perm]
        feat_v = feat_v[perm]
        labels = labels[perm]

    return cp.asarray(feat_n), cp.asarray(feat_v), cp.asarray(labels), feat_dim, n_bins, total_range


def run_bayesian(feat_n, feat_v, labels, hypo_space):
    M = len(labels)
    current = cp.array([], dtype=cp.float64)
    for i in range(M):
        current = bayesian_update_sep(
            feat_n[i], feat_v[i], labels[i],
            feat_n[:i], feat_v[:i], labels[:i],
            hypo_space, current, PHI_SEP, LENGTH_PENALTY, EPSILON
        )
    final_idx = int(cp.argmax(current).item())
    final_ell = int(hypo_space[final_idx].item())
    return final_ell


def calc_test_mae(feat_n, feat_v, labels, ell):
    M = len(labels)
    split = int(M * TRAIN_RATIO)
    train_n, train_v, train_lbl = feat_n[:split], feat_v[:split], labels[:split]
    test_n, test_v, test_lbl = feat_n[split:], feat_v[split:], labels[split:]
    preds = knn_predict_batch(test_n, test_v, train_n, train_v, train_lbl,
                              ell, K_DEFAULT, THETA_DEFAULT)
    mae = float(cp.mean(cp.abs(preds - test_lbl)).item())
    return mae


# ===================== 主流程 =====================
if __name__ == "__main__":
    results = []
    baseline_time = None

    print("=" * 90)
    print("Grid Size Sensitivity Analysis (GPU Accelerated)")
    print("=" * 90)
    print(f"实验条件：PHI={PHI_SEP}，EPSILON={EPSILON}，随机打乱={RANDOM_SHUFFLE}，种子={SEED}")
    print()

    for idx, (path, grid_size, label) in enumerate(DATASET_CONFIGS):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()

        # 1. 加载数据（含打乱）
        feat_n, feat_v, labels, feat_dim, n_bins, total_range = load_data(path, grid_size)
        hypo_space = cp.arange(20, feat_dim + 4, 4, dtype=cp.int64)

        # 2. 贝叶斯求解最优ℓ
        final_ell = run_bayesian(feat_n, feat_v, labels, hypo_space)
        phys_dist = final_ell / CHANNELS_PER_BIN * grid_size

        # 3. 计算最优ℓ下的测试集MAE
        test_mae = calc_test_mae(feat_n, feat_v, labels, final_ell)

        end_event.record()
        end_event.synchronize()
        elapsed = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0

        if label == "3s₀":
            baseline_time = elapsed
        relative_cost = elapsed / baseline_time if baseline_time else 1.0

        results.append({
            "label": label,
            "grid_len": grid_size,
            "feat_dim": feat_dim,
            "n_bins": n_bins,
            "total_range": total_range,
            "opt_ell": final_ell,
            "opt_dist": phys_dist,
            "test_mae": test_mae,
            "time": elapsed,
            "rel_cost": relative_cost,
        })

        print(f"[{label}] 维度={feat_dim:4d} | 最优ℓ={final_ell:4d} ({phys_dist:6.1f}m) | MAE={test_mae:.6f} | 耗时={elapsed:.2f}s")

    # ===================== 打印结果表格 =====================
    print("\n" + "=" * 90)
    print("汇总对比表")
    print("=" * 90)
    print(f"{'Grid':<6} {'Bin/m':<8} {'Total Dim':<10} {'Range/m':<10} {'Opt ℓ':<8} {'Opt Dist/m':<11} {'Test MAE':<10} {'Rel.Cost':<9}")
    print("-" * 90)
    for r in results:
        print(f"{r['label']:<6} {r['grid_len']:<8.1f} {r['feat_dim']:<10d} {r['total_range']:<10.1f} "
              f"{r['opt_ell']:<8d} {r['opt_dist']:<11.1f} {r['test_mae']:<10.6f} {r['rel_cost']:<9.2f}")
    print("=" * 90)
