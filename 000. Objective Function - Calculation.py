import numpy as np
import time

# --- 1. 全局参数 ---
s0 = 6.0
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


# --- 2. 纯NumPy向量化加速度模型 ---
def acc_green(X, V):
    A = np.zeros_like(X)
    free_flow_acc = a_max * (1 - (V / v0) ** 4)
    A = free_flow_acc
    if len(X) > 1:
        dv = V[1:] - V[:-1]
        sg_e = s0 + V[1:] * T_D + (V[1:] * dv) / root_2ab
        sg_e = np.maximum(sg_e, 0)
        s = np.abs(X[1:] - X[:-1])
        interaction_acc = -a_max * (sg_e / s) ** 2
        A[1:] += interaction_acc
    return np.clip(A, -b_max, a_max)


def acc_yellow(X, V, t_y):
    A = np.zeros_like(X)
    vy_e = np.maximum(1e-4, v0 * (1 / (α * X) + 1) ** β)
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
        sy_e_all = sg_e_all + γ * (T_y - t_y) ** (η)

    for j in range(len(X)):
        if passed_line[j]:
            if j == 0:
                A[j] = a_max * (1 - (V[j] / v0) ** 4)
            else:
                s = np.abs(X[j] - X[j - 1])
                A[j] = a_max * (1 - (V[j] / v0) ** 4 - (sg_e_all[j - 1] / s) ** 2)
        else:
            if leader_mask[j]:
                if j == 0:
                    A[j] = a_max * (1 - (V[j] / vy_e[j]) ** 4)
                else:
                    s = np.abs(X[j] - X[j - 1])
                    A[j] = a_max * (1 - (V[j] / vy_e[j]) ** 4 - (sy_e_all[j - 1] / s) ** 2)
            else:
                s = np.abs(X[j] - X[j - 1])
                A[j] = a_max * (1 - (V[j] / vy_e[j]) ** 4 - (sg_e_all[j - 1] / s) ** 2)
    return np.clip(A, -b_max, a_max)


def acc_red(X, V):
    A = np.zeros_like(X)
    vy_e = np.maximum(1e-4, v0 * (1 / (α * X) + 1) ** β)
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
    sr_e = s0 + np.maximum(0, V * T_D + (V ** 2) / root_2ab)
    for j in range(len(X)):
        if passed_line[j]:
            if j == 0:
                A[j] = a_max * (1 - (V[j] / v0) ** 4)
            else:
                s = np.abs(X[j] - X[j - 1])
                A[j] = a_max * (1 - (V[j] / v0) ** 4 - (sg_e_all[j - 1] / s) ** 2)
        else:
            if leader_mask[j]:
                s = np.abs(X[j])
                A[j] = a_max * (1 - (V[j] / vy_e[j]) ** 4 - (sr_e[j] / s) ** 2)
            else:
                s = np.abs(X[j] - X[j - 1])
                A[j] = a_max * (1 - (V[j] / vy_e[j]) ** 4 - (sg_e_all[j - 1] / s) ** 2)
    return np.clip(A, -b_max, a_max)


def idm_vehicle_state(X, V, A):
    v_plus_a_dt = V + A * Δt
    needs_stop = v_plus_a_dt < 0
    safe_A = np.where(A == 0, 1e-9, A)
    t_stop = -V / safe_A
    new_X = np.where(
        needs_stop,
        X + V * t_stop + 0.5 * A * t_stop ** 2,
        X + V * Δt + 0.5 * A * Δt ** 2
    )
    new_V = np.where(needs_stop, 0.0, v_plus_a_dt)
    return new_X, new_V


def objective_fun_opt(lam, X_list, V_list):
    X1, X2, X3, X4 = [x.copy() for x in X_list]
    V1, V2, V3, V4 = [v.copy() for v in V_list]
    T_g1 = lam * T_cycle
    T_r1 = T_cycle - T_g1 - T_y
    T_g2 = T_r1 - T_y
    stop_count = [0, 0, 0, 0]
    total_steps = int(T_cycle / Δt)

    for k in range(total_steps):
        t_phase = k * Δt
        if t_phase < T_g1:
            A1, A3 = acc_green(X1, V1), acc_green(X3, V3)
            A2, A4 = acc_red(X2, V2), acc_red(X4, V4)
        elif t_phase < T_g1 + T_y:
            ty_elapsed = t_phase - T_g1
            A1, A3 = acc_yellow(X1, V1, ty_elapsed), acc_yellow(X3, V3, ty_elapsed)
            A2, A4 = acc_red(X2, V2), acc_red(X4, V4)
            if t_phase + Δt >= T_g1 + T_y:
                stop_count[1] = np.sum(V2 < v_stop)
                stop_count[3] = np.sum(V4 < v_stop)
        elif t_phase < T_g1 + T_y + T_g2:
            A1, A3 = acc_red(X1, V1), acc_red(X3, V3)
            A2, A4 = acc_green(X2, V2), acc_green(X4, V4)
        else:
            A1, A3 = acc_red(X1, V1), acc_red(X3, V3)
            ty_elapsed = t_phase - (T_g1 + T_y + T_g2)
            A2, A4 = acc_yellow(X2, V2, ty_elapsed), acc_yellow(X4, V4, ty_elapsed)
            if t_phase + Δt >= T_cycle:
                stop_count[0] = np.sum(V1 < v_stop)
                stop_count[2] = np.sum(V3 < v_stop)

        X1, V1 = idm_vehicle_state(X1, V1, A1)
        X2, V2 = idm_vehicle_state(X2, V2, A2)
        X3, V3 = idm_vehicle_state(X3, V3, A3)
        X4, V4 = idm_vehicle_state(X4, V4, A4)

    return int(sum(stop_count))


# --- 3. 主计算流程 ---
if __name__ == "__main__":
    # 初始化车辆状态
    X_init = [np.arange(20, -1000, -10, dtype=np.float64) for _ in range(4)]
    V_init = [np.full_like(x, 0.0) for x in X_init]

    lam_set = np.linspace(Δt / T_cycle, (T_cycle - 2 * T_y - Δt) / T_cycle, 200)

    print(f"开始计算，共 {len(lam_set)} 个网格点...")
    t_start = time.time()
    obj_set = np.array([objective_fun_opt(lam, X_init, V_init) for lam in lam_set])
    elapsed = time.time() - t_start
    print(f"计算完成！耗时: {elapsed:.2f}s")

    # 求解最优
    min_y = np.min(obj_set)
    min_indices = np.where(obj_set == min_y)[0]
    min_x_candidates = lam_set[min_indices]
    unique_solution = float(np.median(min_x_candidates))

    print(f"最小排队数 y = {min_y}")
    print(f"最优绿灯占比 λ* = {unique_solution:.8f}")

    # 保存到npz文件
    save_path = "idm_opt_result.npz"
    np.savez(
        save_path,
        lam_set=lam_set,
        obj_set=obj_set,
        min_y=min_y,
        unique_solution=unique_solution,
        params=dict(s0=s0, T_D=T_D, a_max=a_max, b_max=b_max, v0=v0,
                    T_y=T_y, α=α, β=β, γ=γ, η=η, v_stop=v_stop,
                    T_cycle=T_cycle, Δt=Δt)
    )
    print(f"结果已保存至: {save_path}")
