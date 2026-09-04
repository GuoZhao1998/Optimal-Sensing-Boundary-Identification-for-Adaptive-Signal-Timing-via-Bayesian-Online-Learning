# ===================== Linux system compatibility configuration =====================
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import json
from typing import Tuple, List

# 无GUI环境后端
plt.switch_backend('Agg')
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Liberation Serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ====================== 全局常量 ======================
TOTAL_LENGTH = 7000.0       # 分布总长度 7000m
SEGMENT_LENGTH = 50.0       # 区段长度 50m
V0 = 15.0                   # 自由流速度
K_MAX = 200.0               # 最大密度
D_MIN = 5.0                 # 最小安全间距
CV = 0.1                    # 速度波动系数
MAX_SAMPLING_ATTEMPTS = 50  # 最大采样次数

# ====================== 6种交通密度分布函数 ======================
def dist1_upstream_congestion(
    x,
    k_congest: float = 100.0,
    k_free: float = 20.0,
    center: float = 2800.0,    # 衰减中心后移，适配7000m
    width: float = 800.0       # 衰减宽度同步放大
):
    """分布1：上游拥堵向右平滑消散（erfc型）"""
    norm_profile = erfc((x - center) / width)
    density = k_free + (k_congest - k_free) * norm_profile
    return np.clip(density, 0.0, 200.0)


def dist2_downstream_congestion(
    x,
    k_free: float = 100.0,
    k_congest: float = 170.0,
    center: float = 4200.0,    # 抬升中心后移，适配7000m
    width: float = 800.0       # 抬升宽度同步放大
):
    """分布2：下游拥堵平滑抬升（反向erfc型）"""
    norm_profile = 1.0 - erfc((x - center) / width)
    density = k_free + (k_congest - k_free) * norm_profile
    return np.clip(density, 0.0, 200.0)


def dist3_valley_concave(
    x,
    k_high: float = 130.0,
    k_low: float = 15.0,
    center1: float = 2100.0,   # 凹坑左起点后移
    center2: float = 4900.0,   # 凹坑右终点后移
    width: float = 500.0       # 过渡宽度放大
):
    """分布3：中间凹坑型（两头高、中间低）"""
    def logistic(t):
        return 1.0 / (1.0 + np.exp(-t))
    log1 = logistic((x - center1) / width)
    log2 = logistic((x - center2) / width)
    density = k_high - (k_high - k_low) * (log1 - log2)
    return np.clip(density, 0.0, 200.0)


def dist4_hump_convex(
    x,
    k_low: float = 30.0,
    k_high: float = 140.0,
    center1: float = 2100.0,   # 驼峰左起点后移
    center2: float = 4900.0,   # 驼峰右终点后移
    width: float = 500.0       # 峰宽系数放大
):
    """分布4：中间驼峰型（两头低、中间高）"""
    def logistic(t):
        return 1.0 / (1.0 + np.exp(-t))
    log1 = logistic((x - center1) / width)
    log2 = logistic((x - center2) / width)
    density = k_low + (k_high - k_low) * (log1 - log2)
    return np.clip(density, 0.0, 200.0)


def dist5_sinusoidal(
    x,
    k_base: float = 55.0,
    amp: float = 35.0,
    period: float = 2100.0     
):
    """分布5：正弦周期波动（多周期波浪形）"""
    density = k_base + amp * np.sin(2 * np.pi * x / period)
    return np.clip(density, 0.0, 200.0)


def dist6_cosine_smooth(
    x,
    k_base: float = 55.0,
    amp: float = 35.0,
    period: float = 2100.0     
):
    """分布6：余弦平缓渐变（全程单峰单谷，长距离缓变）"""
    density = k_base + amp * np.cos(2 * np.pi * x / period)
    return np.clip(density, 0.0, 200.0)


# ====================== 区间平均采样 ======================
def sample_segment_average(dist_func, x_start=0, x_end=TOTAL_LENGTH, segment_len=SEGMENT_LENGTH, n_sample=20):
    """对连续密度函数做区间平均采样，输出离散密度列表"""
    x_edges = np.arange(x_start, x_end + 1e-6, segment_len)
    n_segments = len(x_edges) - 1
    density_list = np.zeros(n_segments, dtype=np.float64)

    for i in range(n_segments):
        a, b = x_edges[i], x_edges[i+1]
        x_samples = np.linspace(a, b, n_sample)
        y_samples = dist_func(x_samples)
        density_list[i] = np.mean(y_samples)

    return np.clip(density_list, 0.0, 200.0), x_edges


# ====================== 6种分布参数配置 ======================
DISTRIBUTION_CONFIGS = [
    {
        "func": dist1_upstream_congestion,
        "name": "1. erfc Decaying Profile",
        "params": {"k_congest": 100.0, "k_free": 20.0, "center": 2800.0, "width": 800.0}
    },
    {
        "func": dist2_downstream_congestion,
        "name": "2. Inverse erfc Rising",
        "params": {"k_free": 100.0, "k_congest": 170.0, "center": 4200.0, "width": 800.0}
    },
    {
        "func": dist3_valley_concave,
        "name": "3. Double Logistic Concave",
        "params": {"k_high": 130.0, "k_low": 15.0, "center1": 2100.0, "center2": 4900.0, "width": 500.0}
    },
    {
        "func": dist4_hump_convex,
        "name": "4. Double Logistic Convex",
        "params": {"k_low": 30.0, "k_high": 140.0, "center1": 2100.0, "center2": 4900.0, "width": 500.0}
    },
    {
        "func": dist5_sinusoidal,
        "name": "5. Sinusoidal Periodic",
        "params": {"k_base": 55.0, "amp": 35.0, "period": 2100.0}
    },
    {
        "func": dist6_cosine_smooth,
        "name": "6. Cosinusoidal Gradient",
        "params": {"k_base": 60.0, "amp": 45.0, "period": 2100.0}
    }
]

DISTRIBUTION_NAMES = [
    "erfc_decaying",
    "inverse_erfc_rising",
    "double_logistic_concave",
    "double_logistic_convex",
    "sinusoidal_periodic",
    "cosinusoidal_gradient"
]


# ====================== 可视化 ======================



def plot_all_distributions(save_path="6_distributions.png"):
    """绘制6种分布：连续曲线 + 50m区间矩形 + 采样点，灰度风格"""
    # ========== 全局样式统一配置 ==========
    plt.rcParams.update({
        # 1. 正文字体：Times New Roman 属于衬线字体，归入 serif 家族
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        # 2. LaTeX 数学公式字体：STIX 与 Times New Roman 视觉高度一致
        'mathtext.fontset': 'stix',
        # 3. 负号兼容：避免坐标轴负号显示为方块
        'axes.unicode_minus': False,
        # 4. 字号统一配置
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
    })

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    x_continuous = np.linspace(0, TOTAL_LENGTH, 1000)
    
    for idx, config in enumerate(DISTRIBUTION_CONFIGS):
        ax = axes[idx]
        y_cont = config["func"](x_continuous, **config["params"])
        
        dens_list, _ = sample_segment_average(lambda x: config["func"](x, **config["params"]))
        x_centers = np.arange(SEGMENT_LENGTH/2, TOTAL_LENGTH, SEGMENT_LENGTH)
        
        # 1. 50m区间灰色矩形
        ax.bar(
            x_centers, 
            dens_list, 
            width=SEGMENT_LENGTH * 0.85, 
            alpha=0.65, 
            color="#7D7D7D",
            zorder=2
        )
        
        # 2. 连续密度曲线
        ax.plot(
            x_continuous, 
            y_cont, 
            '-', 
            color='#000000', 
            linewidth=2.5, 
            label='Continuous Density',
            zorder=3
        )
        
        # 3. 采样点
        ax.scatter(
            x_centers, 
            dens_list, 
            s=15, 
            color='#000000', 
            zorder=5, 
            label='Sampling (accuracy 50m)'
        )
        
        ax.set_title(config["name"], pad=10)
        ax.set_xlabel('Distance to stop line (m)')
        ax.set_ylabel('Density (veh/km)')
        ax.set_ylim(0, 200)
        ax.set_xlim(0, TOTAL_LENGTH)
        ax.grid(alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=10)
        ax.legend(framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"分布图已保存: {save_path}")



# ====================== Numba 车辆生成核心函数 ======================
def generate_segment_positions(
    xu: float, xd: float, ni: int, d_min: float, max_sampling_attempts: int
) -> np.ndarray:
    segment_length = xu - xd
    pos = np.empty(ni, dtype=np.float64)
    if ni == 0: return pos[:0]
    
    n_max_possible = int(np.floor(segment_length / d_min))
    if ni > n_max_possible:
        ni = n_max_possible
        if ni == 0: return pos[:0]
    
    if ni > 0.8 * n_max_possible:
        total_gap = segment_length - ni * d_min
        extra_per_slot = total_gap / (ni + 1)
        for j in range(ni):
            slot_start = xd + (j + 1) * extra_per_slot + j * d_min
            slot_end = slot_start + d_min
            pos[j] = np.random.uniform(slot_start, slot_end)
    else:
        valid = False
        for _ in range(max_sampling_attempts):
            pos = np.random.uniform(xd + 1e-12, xu - 1e-12, ni)
            pos = np.sort(pos)[::-1]
            valid_gaps = True
            if ni > 1:
                for j in range(ni - 1):
                    gap = pos[j] - pos[j + 1]
                    if gap < d_min - 1e-12:
                        valid_gaps = False
                        break
            valid_up = (xu - pos[0]) >= (d_min / 2.0 - 1e-12)
            valid_down = (pos[-1] - xd) >= (d_min / 2.0 - 1e-12)
            if valid_gaps and valid_up and valid_down:
                valid = True
                break
        if not valid:
            total_gap = segment_length - ni * d_min
            extra_per_slot = total_gap / (ni + 1)
            for j in range(ni):
                slot_start = xd + (j + 1) * extra_per_slot + j * d_min
                slot_end = slot_start + d_min
                pos[j] = np.random.uniform(slot_start, slot_end)
    pos = np.sort(pos)[::-1]
    return pos

def generate_segment_speeds(
    ni: int, k: float, v0: float, k_max: float, cv: float
) -> np.ndarray:
    v_vehicles = np.empty(ni, dtype=np.float64)
    if ni == 0: return v_vehicles[:0]
    v_theory = v0 * (1.0 - k / k_max)
    if v_theory < 0: v_theory = 0.0
    if v_theory > v0: v_theory = v0
    if v_theory < 1e-6:
        for i in range(ni): v_vehicles[i] = 0.0
    else:
        noise_std = cv * v_theory
        for i in range(ni):
            noise = np.random.normal(0.0, noise_std)
            v = v_theory + noise
            if v < 0: v = 0.0
            if v > v0: v = v0
            v_vehicles[i] = v
    return v_vehicles

def generate_single_vehicle_distribution(
    densities: np.ndarray, start_x: float = 0.0, segment_length: float = SEGMENT_LENGTH,
    v0: float = V0, k_max: float = K_MAX, d_min: float = D_MIN, cv: float = CV,
    max_sampling_attempts: int = MAX_SAMPLING_ATTEMPTS
) -> Tuple[np.ndarray, np.ndarray]:
    n_segments = len(densities)
    x_up = np.zeros(n_segments, dtype=np.float64)
    x_down = np.zeros(n_segments, dtype=np.float64)
    x_up[0] = start_x
    x_down[0] = start_x - segment_length
    for i in range(1, n_segments):
        x_up[i] = x_down[i-1]
        x_down[i] = x_up[i] - segment_length
    
    all_pos = []
    all_spd = []
    for i in range(n_segments):
        seg_len = x_up[i] - x_down[i]
        n_exp = (densities[i] / 1000.0) * seg_len
        n_int = int(np.floor(n_exp))
        n_dec = n_exp - n_int
        r = np.random.rand()
        ni = n_int + 1 if r <= n_dec else n_int
        ni = max(ni, 0)
        
        pos = generate_segment_positions(x_up[i], x_down[i], ni, d_min, max_sampling_attempts)
        spd = generate_segment_speeds(len(pos), densities[i], v0, k_max, cv)
        
        all_pos.append(pos)
        all_spd.append(spd)
    
    positions_arr = np.concatenate(all_pos) if all_pos else np.array([])
    speeds_arr = np.concatenate(all_spd) if all_spd else np.array([])
    
    if len(positions_arr) > 1:
        sort_idx = np.argsort(positions_arr)[::-1]
        positions_arr = positions_arr[sort_idx]
        speeds_arr = speeds_arr[sort_idx]
    
    return positions_arr, speeds_arr


# ====================== 批量生成 + NPZ保存 ======================
def generate_and_save_all_distributions(output_path: str = "06_initial_distributions.npz"):
    print("生成 6 种密度对应的初始车辆分布...\n")
    
    # 先生成所有密度列表
    density_lists = []
    for config in DISTRIBUTION_CONFIGS:
        dens, _ = sample_segment_average(lambda x: config["func"](x, **config["params"]))
        density_lists.append(dens)
    
    # 生成车辆位置与速度
    pos_all = np.empty(len(density_lists), dtype=object)
    spd_all = np.empty(len(density_lists), dtype=object)
    
    for idx, dens_arr in enumerate(density_lists):
        pos, spd = generate_single_vehicle_distribution(dens_arr)
        pos_all[idx] = pos.round(4)
        spd_all[idx] = spd.round(4)
        print(f"分布 {idx+1} [{DISTRIBUTION_NAMES[idx]}] 生成完成 | 车辆数: {len(pos)}")
    
    # 保存为NPZ
    np.savez_compressed(
        output_path,
        positions=pos_all,
        speeds=spd_all,
        distribution_names=np.array(DISTRIBUTION_NAMES, dtype=object),
        total_length=TOTAL_LENGTH,
        segment_length=SEGMENT_LENGTH,
        n_distributions=len(density_lists)
    )
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n 全部生成完成！文件已保存至: {output_path}")
    print(f"文件大小: {file_size:.2f} MB")
    print(f"总分布数: {len(density_lists)} | 单分布区段数: {len(density_lists[0])}")


# ====================== 主流程 ======================
if __name__ == "__main__":
    print("="*70)
    print("初始车辆分布生成 (6种 | 7000m范围)")
    print("="*70)
    
    # 1. 生成分布图
    print("\n【步骤1】绘制密度分布曲线")
    plot_all_distributions()
    
    # 2. 生成车辆分布并保存
    print("\n【步骤2】生成车辆位置与速度分布")
    generate_and_save_all_distributions()
    
    print("\n" + "="*70)
    print(" 全部完成")
    print("="*70)
