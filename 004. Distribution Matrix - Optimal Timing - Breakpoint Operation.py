# ===================== Linux system compatibility configuration =====================
# ===================== 必须放在所有 import 之前 =====================
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
# ===================================================================

import time
import multiprocessing
import numpy as np
from numba import njit
from math import ceil

# ===================== 1. 全局物理参数 =====================
s0 = 6
T_D = 1.5
a_max = 2.0
b_max = 2.0
v0 = 15.0
root_2ab = 2 * np.sqrt(a_max * b_max)
T_y = 3.0
α, β, γ, η = 0.5, 5.0, 5.0, -1.5
v_stop = 1.0
T_cycle = 90.0
Δt = 0.1
lam_min = Δt / T_cycle
lam_max = (T_cycle - 2 * T_y - Δt) / T_cycle


# ===================== 网格尺寸敏感性分析配置 =====================
GRID_SIZES = [2 * s0, 3 * s0, 4 * s0]  # 12m / 18m / 24m
GRID_LABELS = ["2s0", "3s0", "4s0"]
OUTPUT_DATASETS = [f"dataset_{label}_5000m.npz" for label in GRID_LABELS]
ENCODE_THRESHOLD_N = 4
ENCODE_THRESHOLD_V = 16.0
ENCODE_X_START = 0.0
ENCODE_X_END = -5000.0 

# ===================== 性能与断点配置 =====================
CHECKPOINT_FILE = "checkpoint_5000m.npz"
SAVE_INTERVAL = 100
NUM_PROCESSES = min(os.cpu_count(), 18)
GRID_SEARCH_POINTS = 100


# ===================== 2. Numba核心动力学函数 =====================
@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, inline='always')
def acc_green_njit(X, V):
    A = np.zeros_like(X)
    free_flow_acc = a_max * (1 - (V / v0)**4)
    A = free_flow_acc
    if len(X) > 1:
        dv = V[1:] - V[:-1]
        sg_e = s0 + V[1:] * T_D + (V[1:] * dv) / root_2ab
        sg_e = np.maximum(sg_e, 0)
        s = np.abs(X[1:] - X[:-1])
        interaction_acc = -a_max * (sg_e / s)**2
        A[1:] += interaction_acc
    return np.clip(A, -b_max, a_max)


@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, inline='always')
def acc_yellow_njit(X, V, t_y):
    A = np.zeros_like(X)
    vy_e = np.maximum(1e-4, v0 * (1 / (α * X) + 1)**β)
    passed_line = X >= 0
    leader_mask = np.zeros_like(X, dtype=np.bool_)
    leader_mask[0] = True
    if len(X) > 1:
        leader_mask[1:] = (X[1:] < 0) & (X[:-1] >= 0)
    sg_e_all = np.empty(len(X) - 1, dtype=np.float64)
    sy_e_all = np.empty(len(X) - 1, dtype=np.float64)
    if len(X) > 1:
        dv = V[1:] - V[:-1]
        sg_e_all = s0 + V[1:] * T_D + (V[1:] * dv) / root_2ab
        sg_e_all = np.maximum(sg_e_all, 0)
        sy_e_all = sg_e_all + γ * (T_y - t_y)**(η)
    for j in range(len(X)):
        if passed_line[j]:
            if j == 0:
                A[j] = a_max * (1 - (V[j] / v0)**4)
            else:
                s = np.abs(X[j] - X[j - 1])
                A[j] = a_max * (1 - (V[j] / v0)**4 - (sg_e_all[j-1] / s)**2)
        else:
            if leader_mask[j]:
                if j == 0:
                    A[j] = a_max * (1 - (V[j] / vy_e[j])**4)
                else:
                    s = np.abs(X[j] - X[j - 1])
                    A[j] = a_max * (1 - (V[j] / vy_e[j])**4 - (sy_e_all[j-1] / s)**2)
            else:
                s = np.abs(X[j] - X[j - 1])
                A[j] = a_max * (1 - (V[j] / vy_e[j])**4 - (sg_e_all[j-1] / s)**2)
    return np.clip(A, -b_max, a_max)


@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, inline='always')
def acc_red_njit(X, V):
    A = np.zeros_like(X)
    vy_e = np.maximum(1e-4, v0 * (1 / (α * X) + 1)**β)
    passed_line = X >= 0
    leader_mask = np.zeros_like(X, dtype=np.bool_)
    leader_mask[0] = True
    if len(X) > 1:
        leader_mask[1:] = (X[1:] < 0) & (X[:-1] >= 0)
    sg_e_all = np.empty(len(X) - 1, dtype=np.float64)
    if len(X) > 1:
        dv = V[1:] - V[:-1]
        sg_e_all = s0 + V[1:] * T_D + (V[1:] * dv) / root_2ab
        sg_e_all = np.maximum(sg_e_all, 0)
    sr_e = s0 + np.maximum(0, V * T_D + (V**2) / root_2ab)
    for j in range(len(X)):
        if passed_line[j]:
            if j == 0:
                A[j] = a_max * (1 - (V[j] / v0)**4)
            else:
                s = np.abs(X[j] - X[j - 1])
                A[j] = a_max * (1 - (V[j] / v0)**4 - (sg_e_all[j-1] / s)**2)
        else:
            if leader_mask[j]:
                s = np.abs(X[j])
                A[j] = a_max * (1 - (V[j] / vy_e[j])**4 - (sr_e[j] / s)**2)
            else:
                s = np.abs(X[j] - X[j - 1])
                A[j] = a_max * (1 - (V[j] / vy_e[j])**4 - (sg_e_all[j-1] / s)**2)
    return np.clip(A, -b_max, a_max)


@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, inline='always')
def idm_vehicle_state_njit(X, V, A):
    v_plus_a_dt = V + A * Δt
    needs_stop = v_plus_a_dt < 0
    safe_A = np.where(A == 0, 1e-9, A)
    t_stop = -V / safe_A
    new_X = np.where(
        needs_stop,
        X + V * t_stop + 0.5 * A * t_stop**2,
        X + V * Δt + 0.5 * A * Δt**2
    )
    new_V = np.where(needs_stop, 0.0, v_plus_a_dt)
    return new_X, new_V


# ===================== 3. 目标函数 =====================
@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, inline='always')
def objective_fun_opt(
    lam: float,
    X1: np.ndarray, V1: np.ndarray,
    X2: np.ndarray, V2: np.ndarray,
    X3: np.ndarray, V3: np.ndarray,
    X4: np.ndarray, V4: np.ndarray
):
    X1_sim = X1.copy()
    V1_sim = V1.copy()
    X2_sim = X2.copy()
    V2_sim = V2.copy()
    X3_sim = X3.copy()
    V3_sim = V3.copy()
    X4_sim = X4.copy()
    V4_sim = V4.copy()

    T_g1 = lam * T_cycle
    T_r1 = T_cycle - T_g1 - T_y
    T_g2 = T_r1 - T_y
    total_steps = int(T_cycle / Δt)
    stop_count_1 = stop_count_2 = stop_count_3 = stop_count_4 = 0

    for k in range(total_steps):
        t_phase = k * Δt
        if t_phase < T_g1:
            A1 = acc_green_njit(X1_sim, V1_sim)
            A3 = acc_green_njit(X3_sim, V3_sim)
            A2 = acc_red_njit(X2_sim, V2_sim)
            A4 = acc_red_njit(X4_sim, V4_sim)
        elif t_phase < T_g1 + T_y:
            A1 = acc_yellow_njit(X1_sim, V1_sim, t_phase - T_g1)
            A3 = acc_yellow_njit(X3_sim, V3_sim, t_phase - T_g1)
            A2 = acc_red_njit(X2_sim, V2_sim)
            A4 = acc_red_njit(X4_sim, V4_sim)
            if t_phase + Δt >= T_g1 + T_y:
                stop_count_2 = np.sum(V2_sim < v_stop)
                stop_count_4 = np.sum(V4_sim < v_stop)
        elif t_phase < T_g1 + T_y + T_g2:
            A1 = acc_red_njit(X1_sim, V1_sim)
            A3 = acc_red_njit(X3_sim, V3_sim)
            A2 = acc_green_njit(X2_sim, V2_sim)
            A4 = acc_green_njit(X4_sim, V4_sim)
        else:
            A1 = acc_red_njit(X1_sim, V1_sim)
            A3 = acc_red_njit(X3_sim, V3_sim)
            already_yellow = t_phase - (T_g1 + T_y + T_g2)
            A2 = acc_yellow_njit(X2_sim, V2_sim, already_yellow)
            A4 = acc_yellow_njit(X4_sim, V4_sim, already_yellow)
            if t_phase + Δt >= T_cycle:
                stop_count_1 = np.sum(V1_sim < v_stop)
                stop_count_3 = np.sum(V3_sim < v_stop)
        
        X1_sim, V1_sim = idm_vehicle_state_njit(X1_sim, V1_sim, A1)
        X2_sim, V2_sim = idm_vehicle_state_njit(X2_sim, V2_sim, A2)
        X3_sim, V3_sim = idm_vehicle_state_njit(X3_sim, V3_sim, A3)
        X4_sim, V4_sim = idm_vehicle_state_njit(X4_sim, V4_sim, A4)
    
    return float(stop_count_1 + stop_count_2 + stop_count_3 + stop_count_4)


# ===================== 4. 轨迹网格化编码 =====================
@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, inline='always')
def segment_and_uniform_njit(
    X1: np.ndarray, V1: np.ndarray,
    X2: np.ndarray, V2: np.ndarray,
    X3: np.ndarray, V3: np.ndarray,
    X4: np.ndarray, V4: np.ndarray,
    x_start: float, x_end: float, h_bins: float,
    threshold_n: int, threshold_v: float
):
    inv_threshold_n = 1.0 / threshold_n
    inv_threshold_v = 1.0 / threshold_v
    
    total_length = x_start - x_end
    n_bins = int(ceil(total_length / h_bins))
    n_rows = 4
    total_size = n_rows * n_bins
    
    list_counts = np.empty(total_size, dtype=np.float64)
    list_avgs = np.empty(total_size, dtype=np.float64)
    
    counts = np.zeros((n_rows, n_bins), dtype=np.int64)
    speed_sums = np.zeros((n_rows, n_bins), dtype=np.float64)
    
    # 进口道1
    x_len = X1.shape[0]
    for i in range(x_len):
        x = X1[i]
        if x <= x_end or x >= x_start:
            continue
        v = V1[i]
        offset = x_start - x
        bin_idx = int(offset // h_bins)
        bin_idx = min(max(bin_idx, 0), n_bins - 1)
        counts[0, bin_idx] += 1
        speed_sums[0, bin_idx] += v
    
    # 进口道2
    x_len = X2.shape[0]
    for i in range(x_len):
        x = X2[i]
        if x <= x_end or x >= x_start:
            continue
        v = V2[i]
        offset = x_start - x
        bin_idx = int(offset // h_bins)
        bin_idx = min(max(bin_idx, 0), n_bins - 1)
        counts[1, bin_idx] += 1
        speed_sums[1, bin_idx] += v
    
    # 进口道3
    x_len = X3.shape[0]
    for i in range(x_len):
        x = X3[i]
        if x <= x_end or x >= x_start:
            continue
        v = V3[i]
        offset = x_start - x
        bin_idx = int(offset // h_bins)
        bin_idx = min(max(bin_idx, 0), n_bins - 1)
        counts[2, bin_idx] += 1
        speed_sums[2, bin_idx] += v
    
    # 进口道4
    x_len = X4.shape[0]
    for i in range(x_len):
        x = X4[i]
        if x <= x_end or x >= x_start:
            continue
        v = V4[i]
        offset = x_start - x
        bin_idx = int(offset // h_bins)
        bin_idx = min(max(bin_idx, 0), n_bins - 1)
        counts[3, bin_idx] += 1
        speed_sums[3, bin_idx] += v
    
    idx = 0
    for j in range(n_bins):
        for i in range(n_rows):
            cnt = counts[i, j]
            avg_v = speed_sums[i, j] / cnt if cnt > 0 else 0.0
            c_norm = cnt * inv_threshold_n
            list_counts[idx] = min(c_norm, 1.0)
            a_norm = avg_v * inv_threshold_v
            list_avgs[idx] = min(a_norm, 1.0)
            idx += 1
    return list_counts, list_avgs


# ===================== 5. 网格搜索+中位数选优 =====================
@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, inline='always')
def grid_search_with_median_numba(x1, v1, x2, v2, x3, v3, x4, v4, x_min:float, x_max:float, Ns:int):
    x = np.linspace(x_min, x_max, Ns)
    y = np.empty(Ns, dtype=np.float64)
    for i in range(Ns):
        y[i] = objective_fun_opt(x[i], x1, v1, x2, v2, x3, v3, x4, v4)
    min_y = np.min(y)
    min_indices = np.where(y == min_y)[0]
    min_x_candidates = x[min_indices]
    unique_sol = np.median(min_x_candidates)
    return unique_sol


# ===================== 6. 子进程初始化 =====================
def _worker_init():
    x_test = np.linspace(-100, 0, 20, dtype=np.float64)
    v_test = np.full_like(x_test, 5.0, dtype=np.float64)
    _ = acc_green_njit(x_test, v_test)
    _ = acc_red_njit(x_test, v_test)
    _ = acc_yellow_njit(x_test, v_test, 1.0)
    _ = idm_vehicle_state_njit(x_test, v_test, np.zeros_like(v_test))
    _ = objective_fun_opt(0.5, x_test,v_test, x_test,v_test, x_test,v_test, x_test,v_test)
    _ = segment_and_uniform_njit(x_test,v_test, x_test,v_test, x_test,v_test, x_test,v_test, 0, -2000, 20, 4, 16)
    _ = grid_search_with_median_numba(x_test,v_test, x_test,v_test, x_test,v_test, x_test,v_test, lam_min, lam_max, 10)


# ===================== 7. 数据加载与合并 =====================
def load_merged_preheating_data(tv_path: str, uni_path: str):
    tv_data = np.load(tv_path, allow_pickle=True)
    uni_data = np.load(uni_path, allow_pickle=True)
    
    X1 = np.concatenate([tv_data['X1'], uni_data['X1']])
    V1 = np.concatenate([tv_data['V1'], uni_data['V1']])
    X2 = np.concatenate([tv_data['X2'], uni_data['X2']])
    V2 = np.concatenate([tv_data['V2'], uni_data['V2']])
    X3 = np.concatenate([tv_data['X3'], uni_data['X3']])
    V3 = np.concatenate([tv_data['V3'], uni_data['V3']])
    X4 = np.concatenate([tv_data['X4'], uni_data['X4']])
    V4 = np.concatenate([tv_data['V4'], uni_data['V4']])
    
    n_tv = len(tv_data['X1'])
    n_uni = len(uni_data['X1'])
    total = n_tv + n_uni
    
    assert len(X1) == len(V1) == len(X2) == len(V2) == len(X3) == len(V3) == len(X4) == len(V4)
    return X1, V1, X2, V2, X3, V3, X4, V4, n_tv, n_uni, total


# ===================== 8. 单样本处理（带索引） =====================
def process_single_sample(task):
    idx, x1, v1, x2, v2, x3, v3, x4, v4 = task
    try:
        # 最优配时仅计算1次
        solution = grid_search_with_median_numba(
            x1, v1, x2, v2, x3, v3, x4, v4,
            lam_min, lam_max, Ns=GRID_SEARCH_POINTS
        )
        
        # 3种网格尺寸编码
        counts_list = []
        avgs_list = []
        for h in GRID_SIZES:
            list_counts, list_avgs = segment_and_uniform_njit(
                x1, v1, x2, v2, x3, v3, x4, v4,
                x_start=ENCODE_X_START,
                x_end=ENCODE_X_END,
                h_bins=h,
                threshold_n=ENCODE_THRESHOLD_N,
                threshold_v=ENCODE_THRESHOLD_V
            )
            counts_list.append(list_counts)
            avgs_list.append(list_avgs)
        
        return idx, counts_list, avgs_list, solution
    except Exception:
        return idx, None, None, None


# ===================== 9. 断点加载与保存 =====================
def load_checkpoint(n_total, feat_dims):
    if not os.path.exists(CHECKPOINT_FILE):
        # 初始化空结果数组
        feat_counts = [np.full((n_total, dim), np.nan, dtype=np.float64) for dim in feat_dims]
        feat_avgs = [np.full((n_total, dim), np.nan, dtype=np.float64) for dim in feat_dims]
        opt_lam = np.full(n_total, np.nan, dtype=np.float64)
        completed = 0
        print("未找到断点文件，从头开始计算")
        return feat_counts, feat_avgs, opt_lam, completed
    
    data = np.load(CHECKPOINT_FILE, allow_pickle=True)
    feat_counts = [data[f'counts_{i}'] for i in range(3)]
    feat_avgs = [data[f'avgs_{i}'] for i in range(3)]
    opt_lam = data['optimal_lam']
    completed = int(data['completed'])
    
    print(f" 加载断点成功 | 已完成 {completed}/{n_total} 个样本")
    return feat_counts, feat_avgs, opt_lam, completed


def save_checkpoint(feat_counts, feat_avgs, opt_lam, completed):
    np.savez_compressed(
        CHECKPOINT_FILE,
        counts_0=feat_counts[0],
        counts_1=feat_counts[1],
        counts_2=feat_counts[2],
        avgs_0=feat_avgs[0],
        avgs_1=feat_avgs[1],
        avgs_2=feat_avgs[2],
        optimal_lam=opt_lam,
        completed=completed
    )


# ===================== 10. 批量并行处理 =====================
def batch_process_with_checkpoint(X1, V1, X2, V2, X3, V3, X4, V4, n_total):
    # 预计算3种网格的特征维度
    feat_dims = []
    for h in GRID_SIZES:
        n_bins = int(ceil((ENCODE_X_START - ENCODE_X_END) / h))
        feat_dims.append(4 * n_bins)
    
    # 加载断点
    feat_counts, feat_avgs, opt_lam, completed = load_checkpoint(n_total, feat_dims)
    
    # 生成待处理任务列表
    pending_indices = np.where(np.isnan(opt_lam))[0]
    tasks = []
    for idx in pending_indices:
        tasks.append((idx, X1[idx], V1[idx], X2[idx], V2[idx], X3[idx], V3[idx], X4[idx], V4[idx]))
    
    n_pending = len(tasks)
    if n_pending == 0:
        print("所有样本已处理完成")
        return feat_counts, feat_avgs, opt_lam
    
    print(f"待处理样本: {n_pending} 个 | 每{SAVE_INTERVAL}个保存一次断点")
    print(f"启动 {NUM_PROCESSES} 个CPU核心并行计算...")
    
    start_time = time.time()
    last_save = completed
    
    with multiprocessing.Pool(
        processes=NUM_PROCESSES,
        initializer=_worker_init
    ) as pool:
        # 流式迭代结果，边算边存
        for result in pool.imap_unordered(process_single_sample, tasks, chunksize=4):
            idx, counts_list, avgs_list, solution = result
            
            if solution is not None:
                # 写入结果数组
                opt_lam[idx] = solution
                for k in range(3):
                    feat_counts[k][idx] = counts_list[k]
                    feat_avgs[k][idx] = avgs_list[k]
                completed += 1
            
            # 达到保存间隔则写盘
            if completed - last_save >= SAVE_INTERVAL:
                save_checkpoint(feat_counts, feat_avgs, opt_lam, completed)
                last_save = completed
                elapsed = time.time() - start_time
                speed = completed / elapsed
                eta = (n_total - completed) / speed / 60
                print(f"进度: {completed}/{n_total} ({completed/n_total*100:.1f}%) | 速度: {speed:.1f} 样本/秒 | 预计剩余: {eta:.1f} 分钟")
    
    # 全部完成后最终保存
    save_checkpoint(feat_counts, feat_avgs, opt_lam, completed)
    total_time = time.time() - start_time
    print(f"\n 全部样本处理完成 | 总耗时: {total_time/60:.1f} 分钟 | 平均速度: {n_total/total_time:.1f} 样本/秒")
    
    return feat_counts, feat_avgs, opt_lam


# ===================== 11. 保存最终数据集 =====================
def save_final_datasets(feat_counts, feat_avgs, opt_lam, n_tv, n_uni):
    for k in range(3):
        h = GRID_SIZES[k]
        label = GRID_LABELS[k]
        output_path = OUTPUT_DATASETS[k]
        
        np.savez_compressed(
            output_path,
            features_counts=feat_counts[k],
            features_avgs=feat_avgs[k],
            optimal_lam=opt_lam,
            n_time_varying=n_tv,
            n_uniform=n_uni,
            n_total=len(opt_lam),
            feature_dim=feat_counts[k].shape[1],
            x_start=ENCODE_X_START,
            x_end=ENCODE_X_END,
            h_bins=h,
            grid_size_label=label,
            threshold_n=ENCODE_THRESHOLD_N,
            threshold_v=ENCODE_THRESHOLD_V,
            lam_min=lam_min,
            lam_max=lam_max,
            grid_search_points=GRID_SEARCH_POINTS,
            s0=s0,
            T_cycle=T_cycle
        )
        file_size = os.path.getsize(output_path) / 1024 / 1024
        print(f" 数据集已保存: {output_path} | 大小: {file_size:.2f} MB")
    
    # 完成后删除断点文件
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(" 临时断点文件已清理")


# ===================== 主流程 =====================
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    # 替换主流程内的文件路径配置
    TV_PREHEAT_FILE = "13_timevarying_preheating_6dist.npz"   # 时变分布预热结果
    UNI_PREHEAT_FILE = "14_uniform_preheating_6dist.npz"      # 均匀分布预热结果

    try:
        print("="*70)
        print("最优配时计算 + 网格尺寸敏感性分析 (断点续跑版)")
        print("="*70)
        
        print("\n【步骤1】加载预热结果")
        X1, V1, X2, V2, X3, V3, X4, V4, n_tv, n_uni, n_total = load_merged_preheating_data(
            TV_PREHEAT_FILE, UNI_PREHEAT_FILE
        )
        print(f"时变分布: {n_tv} 组 | 均匀分布: {n_uni} 组 | 总计: {n_total} 组")
        
        print("\n【步骤2】并行计算")
        feat_counts, feat_avgs, optimal_lam = batch_process_with_checkpoint(
            X1, V1, X2, V2, X3, V3, X4, V4, n_total
        )
        
        print("\n【步骤3】生成最终数据集")
        save_final_datasets(feat_counts, feat_avgs, optimal_lam, n_tv, n_uni)
        
        print("\n" + "="*70)
        print("✅ 全部流程执行完成")
        print("="*70)
        
    except Exception as e:
        print(f"\n 运行错误: {str(e)}")
        import traceback
        traceback.print_exc()
