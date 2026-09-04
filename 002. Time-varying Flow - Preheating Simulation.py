# ===================== Linux system compatibility configuration =====================
import os
import json
import time
import itertools
import multiprocessing
from functools import partial
import numpy as np
from numba import njit


# ===================== 1. 全局参数 =====================
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


# ===================== 2. Numba核心加速函数 =====================
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


@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, inline='always')
def filter_njit(X, V, x_min=-5000.0, x_max=20.0):
    mask = (X >= x_min) & (X <= x_max)
    filtered_X = X[mask]
    filtered_V = V[mask]
    return filtered_X, filtered_V


@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, inline='always')
def objective_fun_opt_numba(
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
        
        X1_sim, V1_sim = idm_vehicle_state_njit(X1_sim, V1_sim, A1)
        X2_sim, V2_sim = idm_vehicle_state_njit(X2_sim, V2_sim, A2)
        X3_sim, V3_sim = idm_vehicle_state_njit(X3_sim, V3_sim, A3)
        X4_sim, V4_sim = idm_vehicle_state_njit(X4_sim, V4_sim, A4)
    
    res_X1, res_V1 = filter_njit(X1_sim, V1_sim)
    res_X2, res_V2 = filter_njit(X2_sim, V2_sim)
    res_X3, res_V3 = filter_njit(X3_sim, V3_sim)
    res_X4, res_V4 = filter_njit(X4_sim, V4_sim)
    return (res_X1, res_V1), (res_X2, res_V2), (res_X3, res_V3), (res_X4, res_V4)


# ===================== 3. 数据读取 =====================
def load_initial_distributions(file_path: str):
    """加载6种初始车辆分布NPZ文件"""
    data = np.load(file_path, allow_pickle=True)
    positions = data['positions']
    speeds = data['speeds']
    n_dist = len(positions)
    
    lane_samples = [ [positions[i], speeds[i]] for i in range(n_dist) ]
    print(f"成功读取 {n_dist} 组初始车辆分布")
    return lane_samples


# ===================== 4. 单任务处理 =====================
def process_single_task(task_idx, lane_samples, lams):
    """单组仿真任务：输入索引，返回仿真结果数组"""
    idx1, idx2, idx3, idx4, lam_idx = task_idx
    lam = lams[lam_idx]
    try:
        X1 = np.array(lane_samples[idx1][0], dtype=np.float64)
        V1 = np.array(lane_samples[idx1][1], dtype=np.float64)
        X2 = np.array(lane_samples[idx2][0], dtype=np.float64)
        V2 = np.array(lane_samples[idx2][1], dtype=np.float64)
        X3 = np.array(lane_samples[idx3][0], dtype=np.float64)
        V3 = np.array(lane_samples[idx3][1], dtype=np.float64)
        X4 = np.array(lane_samples[idx4][0], dtype=np.float64)
        V4 = np.array(lane_samples[idx4][1], dtype=np.float64)
        
        return objective_fun_opt_numba(lam, X1, V1, X2, V2, X3, V3, X4, V4)
    except Exception as e:
        print(f"警告：任务 ({idx1},{idx2},{idx3},{idx4},lam={lam}) 失败: {str(e)}")
        return None


# ===================== 5. 批量并行处理 =====================
def batch_process_initial_conditions(lane_samples: list, lams: list):
    num_processes = min(os.cpu_count(), 18)
    print(f"启动 {num_processes} 个CPU核心并行仿真...")
    
    n_samples = len(lane_samples)
    # 生成索引组合任务，减少进程间数据传输量
    tasks = list(itertools.product(
        range(n_samples), range(n_samples),
        range(n_samples), range(n_samples),
        range(len(lams))
    ))
    total_tasks = len(tasks)
    print(f"总任务量：{total_tasks} 组（{n_samples}^4 × {len(lams)} 种参数组合）")
    
    process_func = partial(process_single_task, lane_samples=lane_samples, lams=lams)
    
    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, tasks)
    
    results = [r for r in results if r is not None]
    print(f"仿真完成：成功 {len(results)} 组，失败 {total_tasks - len(results)} 组")
    print(f"总耗时：{time.time() - start_time:.2f} 秒")
    return results


# ===================== 6. NPZ格式结果保存 =====================
def save_results_to_npz(results, output_path: str, lams: list, n_samples: int):
    """
    以压缩NPZ格式保存全部仿真结果
    包含元数据与4个进口道的位置、速度数组，支持按索引快速检索
    """
    n_results = len(results)
    if n_results == 0:
        raise ValueError("无有效仿真结果可保存")
    
    # 元数据数组
    lam_ids = np.repeat(np.arange(len(lams)), n_samples**4)[:n_results]
    lambdas = np.array(lams)[lam_ids]
    
    # 变长数组用对象存储
    X1_all = np.empty(n_results, dtype=object)
    V1_all = np.empty(n_results, dtype=object)
    X2_all = np.empty(n_results, dtype=object)
    V2_all = np.empty(n_results, dtype=object)
    X3_all = np.empty(n_results, dtype=object)
    V3_all = np.empty(n_results, dtype=object)
    X4_all = np.empty(n_results, dtype=object)
    V4_all = np.empty(n_results, dtype=object)
    
    for i, ((x1,v1), (x2,v2), (x3,v3), (x4,v4)) in enumerate(results):
        X1_all[i] = x1
        V1_all[i] = v1
        X2_all[i] = x2
        V2_all[i] = v2
        X3_all[i] = x3
        V3_all[i] = v3
        X4_all[i] = x4
        V4_all[i] = v4
    
    # 压缩存储
    np.savez_compressed(
        output_path,
        lambdas=lambdas,
        lam_ids=lam_ids,
        n_samples=n_samples,
        X1=X1_all, V1=V1_all,
        X2=X2_all, V2=V2_all,
        X3=X3_all, V3=V3_all,
        X4=X4_all, V4=V4_all
    )
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"结果已保存至: {output_path}")
    print(f"文件大小: {file_size:.2f} MB（压缩格式，无损精度）")


# ===================== 7. 主流程 =====================
if __name__ == "__main__":
    # Linux系统多进程启动方式显式设置
    multiprocessing.set_start_method('fork', force=True)
    
    # 配置参数
    INPUT_FILE = "06_initial_distributions.npz"
    OUTPUT_FILE = "13_timevarying_preheating_6dist.npz"
    LAMS = [0.2, 0.4, 0.6, 0.8]
    
    try:
        print("="*70)
        print("步骤1：读取初始车辆分布数据\n")
        lane_samples = load_initial_distributions(INPUT_FILE)
        
        # Numba函数预热（避免多进程同时编译冲突）
        print("\n步骤2：Numba核心函数编译预热...")
        X_test = np.array([10.0, 0.0], dtype=np.float64)
        V_test = np.array([5.0, 3.0], dtype=np.float64)
        _ = acc_green_njit(X_test, V_test)
        _ = acc_red_njit(X_test, V_test)
        _ = acc_yellow_njit(X_test, V_test, 1.0)
        _ = idm_vehicle_state_njit(X_test, V_test, np.zeros_like(V_test))
        _ = filter_njit(X_test, V_test)
        _ = objective_fun_opt_numba(0.5, X_test,V_test, X_test,V_test, X_test,V_test, X_test,V_test)
        print("预热完成\n")
        
        # 批量仿真
        print("="*70)
        print("步骤3：批量预热仿真\n")
        results = batch_process_initial_conditions(lane_samples, LAMS)
        
        # 保存NPZ结果
        print("\n" + "="*70)
        print("步骤4：保存仿真结果\n")
        save_results_to_npz(results, OUTPUT_FILE, LAMS, len(lane_samples))
        
        print("\n 全部流程执行完成")
        print("="*70)
        
    except Exception as e:
        print(f"\n 运行错误：{str(e)}")
        import traceback
        traceback.print_exc()
