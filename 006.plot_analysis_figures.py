# Note! Recommended to run on Windows system! Avoid missing system font libraries!


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter

# ===================== 全局样式配置=====================
plt.rcParams.update({
    # 正文字体：Times New Roman 属于衬线字体，归入 serif 
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    # LaTeX 数学公式字体：STIX 与 Times New Roman 视觉高度一致
    'mathtext.fontset': 'stix',
    # 负号兼容：避免坐标轴负号显示为方块
    'axes.unicode_minus': False,
    # 分辨率与基础字号
    'figure.dpi': 300,
    'font.size': 14,
})

# 配色统一
COLOR_SEP = '#d62728'       # 可分性最优：红
COLOR_GLOBAL = '#1f77b4'    # 全局最优：蓝
COLOR_FULL = '#2ca02c'      # 全特征：绿
COLOR_CURVE = '#000000'     # MAE曲线：黑

# ===================== 读取数据 =====================
# 主分析结果（图2-图5使用）
main_data = np.load("sep_analysis_results.npz", allow_pickle=True)
main_data = dict(main_data)
# 替换为全英文标签，彻底解决中文字体问题
main_data['ell_names'] = np.array([
    r'Optimal Prefix Length $\ell^*=72$ ',
    r'Prefix Length $\ell =216$',
    'Full Prefix Length'
])
hypo_space = main_data['hypothesis_space']
full_ell = int(main_data['full_ell'])
grid_size = float(main_data['grid_size'])
channels = 4

def ell2dist(ell):
    return ell / channels * grid_size

# 图1专用信念数据
belief_data = np.load("belief_evolution_raw.npz")
beliefs = belief_data['beliefs']
opt_ells = belief_data['opt_ells']
final_ell = int(belief_data['final_ell'])
M = beliefs.shape[0]

# ===================== 图1：后验信念演化3D曲线图 =====================
def plot_belief_3d_lines():
    # ========== 可调参数 ==========
    LINE_STEP = 1               # ℓ抽稀步长：1=显示全部ℓ
    # 背景曲线样式
    BG_COLOR = "#000000"        # 背景曲线统一色（中灰色）
    BG_ALPHA = 0.75             # 背景曲线透明度
    BG_LINEWIDTH = 0.75         # 背景曲线线宽
    # 高亮曲线样式（前5条抬升最明显）
    HIGHLIGHT_COLORS = [
        '#d62728', '#1f77b4', '#2ca02c', 
        '#ff7f0e', '#9467bd'
    ]                         # 5种高区分度配色
    HIGHLIGHT_ALPHA = 0.95     # 高亮曲线透明度
    HIGHLIGHT_LINEWIDTH = 1.6  # 高亮曲线线宽
    # =============================
    # 全量使用所有ℓ
    selected_ells = hypo_space[::LINE_STEP]
    selected_beliefs = beliefs[:, ::LINE_STEP]
    n_ell = len(selected_ells)
    x_axis = np.arange(M)

    # ---------- 筛选前5条“抬升最明显”的曲线, 计算每条曲线的曲线下面积(AUC) ----------
    auc_values = np.trapezoid(selected_beliefs, x=x_axis, axis=0)
    # 按AUC降序取前5
    top5_idx = np.argsort(auc_values)[-5:][::-1]
    top5_ells = selected_ells[top5_idx]
    top5_final_beliefs = selected_beliefs[-1, top5_idx]
    # --------------------------------------------------

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 绘制所有背景曲线：统一灰色
    for i in range(n_ell):
        ell = selected_ells[i]
        z_vals = selected_beliefs[:, i]
        ax.plot3D(
            x_axis,
            np.full(M, ell),
            z_vals,
            color=BG_COLOR,
            linewidth=BG_LINEWIDTH,
            alpha=BG_ALPHA
        )

    # 2. 绘制前5条高亮曲线 + 终点标记
    for i in range(5):
        idx = top5_idx[i]
        ell = selected_ells[idx]
        z_vals = selected_beliefs[:, idx]
        color = HIGHLIGHT_COLORS[i]
        final_belief = z_vals[-1]
        # 高亮曲线
        ax.plot3D(
            x_axis,
            np.full(M, ell),
            z_vals,
            color=color,
            linewidth=HIGHLIGHT_LINEWIDTH,
            alpha=HIGHLIGHT_ALPHA,
            zorder=5
        )
        # 终点标记（带黑边增强辨识度）
        ax.scatter3D(
            M - 1,
            ell,
            final_belief,
            color=color,
            s=55,
            alpha=0.85,
            edgecolors='black',
            linewidth=0.6,
            zorder=10,
            label=f'$\ell$ = {ell:.0f} ({ell2dist(ell):.0f} m)\t Final Posterior = {final_belief:.3f}'
        )

    # 3. 坐标轴与标题
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_xlabel('Training Samples $n$', fontsize=16, labelpad=12)
    ax.set_ylabel('Hypothesis Space', fontsize=16, labelpad=12) 
    ax.set_zlabel('Posterior Probabilities', fontsize=16, labelpad=12)
    ax.set_xlim(0, M - 1)
    ax.set_ylim(selected_ells[0], selected_ells[-1])
    ax.set_zlim(0, np.max(selected_beliefs) * 1.05)
    ax.tick_params(axis='y', pad=6)
    # 4. 图例（仅显示5个终点标注）
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95, bbox_to_anchor=(0.175, 0.82))
    ax.view_init(elev=25, azim=-90)

    plt.tight_layout()
    plt.savefig('belief_evolution_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("finish: belief_evolution_3d.png")
    
    # 控制台输出前5条的信息，方便分析
    print("\n=== Top 5 Hypotheses with Most Significant Lift (Sorted by AUC) ===")
    for i in range(5):
        print(f"Rank {i+1}: ℓ = {top5_ells[i]:.0f} | Physical distance = {ell2dist(top5_ells[i]):.0f}m | Final belief = {top5_final_beliefs[i]:.4f}")

# ===================== 图2：MAE全景曲线 =====================
def plot_mae_curve():
    mae_curve = main_data['global_mae_curve']
    sep_ell = int(main_data['sep_final_ell'])
    global_ell = int(main_data['global_best_ell'])
    global_mae = float(main_data['global_best_mae'])

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(hypo_space, mae_curve, color=COLOR_CURVE, linewidth=1.8, label='MAE on Test Set')

    # 曲线上的3个交点标记
    sep_idx = np.where(hypo_space == sep_ell)[0][0]
    global_idx = np.where(hypo_space == global_ell)[0][0]
    full_idx = np.where(hypo_space == full_ell)[0][0]
    ax.scatter(sep_ell, mae_curve[sep_idx], color=COLOR_SEP, s=80, marker='o',
               edgecolors='black', linewidth=0.8, zorder=6,alpha=0.85,
               label=rf'Optimal Prefix Length $\ell^*$ = {sep_ell} ({ell2dist(sep_ell):.0f} m)')
    ax.scatter(global_ell, mae_curve[global_idx], color=COLOR_GLOBAL, s=80, marker='o',
               edgecolors='black', linewidth=0.8, zorder=6,alpha=0.85,
               label=rf'Prefix Length $\ell$ = {global_ell} ({ell2dist(global_ell):.0f} m)')
    ax.scatter(full_ell, mae_curve[full_idx], color=COLOR_FULL, s=80, marker='o',
               edgecolors='black', linewidth=0.8, zorder=6,alpha=0.85,
               label=f'Full Prefix Length = {full_ell} ({ell2dist(full_ell):.0f} m)')

    ax.set_xlabel(r'Prefix Lengths $\ell$', fontsize=16)
    ax.set_ylabel('MAE on Test Set', fontsize=16)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=14, loc='upper right')

    plt.tight_layout()
    plt.savefig('mae_curve.png', bbox_inches='tight')
    plt.close()
    print("finish: mae_curve.png")


# ===================== 图3：k值全局敏感性 =====================
def plot_k_sensitivity_global():
    k_cand = main_data['k_candidates']
    k_mae = main_data['k_mae_global']
    names = main_data['ell_names']
    colors = [COLOR_SEP, COLOR_GLOBAL, COLOR_FULL]
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    for i in range(len(names)):
        ax.plot(k_cand, k_mae[i], 'o-', color=colors[i], linewidth=1.5, markersize=6, label=names[i])
        best_idx = np.argmin(k_mae[i])
        ax.scatter(k_cand[best_idx], k_mae[i, best_idx], color=colors[i], s=80, zorder=5)
    
    ax.set_xlabel(r'Number of Neighbors $K$', fontsize=16)
    ax.set_ylabel('MAE on Test Set', fontsize=16)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=14)
    ax.set_xticks(k_cand)
    
    plt.tight_layout()
    plt.savefig('k_sensitivity.png', bbox_inches='tight')
    plt.close()
    print("finish: k_sensitivity.png")

# ===================== 图4：k值分阶段演化 =====================
def plot_k_evolution_phase():
    k_cand = main_data['k_candidates']
    k_phase = main_data['k_mae_phase']
    names = main_data['ell_names']
    n_phases = int(main_data['n_phases'])
    phase_x = np.arange(1, n_phases + 1) * 10

    n_k = len(k_cand)
    sample_idx = [0,1,2,3,4]
    plot_k_labels = [rf'$K$ = {k_cand[i]}' for i in sample_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for i, ax in enumerate(axes):
        for ki, label in zip(sample_idx, plot_k_labels):
            ax.plot(phase_x, k_phase[i, :, ki], 'o-', linewidth=1.2, markersize=4, label=label)
        ax.set_title(names[i], fontsize=16)
        ax.set_xlabel('Training Progress (%)', fontsize=14)
        if i == 0:
            ax.set_ylabel('Intra-phase Online MAE', fontsize=16)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, ncol=2)
    
    plt.tight_layout()
    plt.savefig('k_phase_evolution.png', bbox_inches='tight')
    plt.close()
    print("finish: k_phase_evolution.png")

# ===================== 图5：theta敏感性 =====================
def plot_theta_sensitivity():
    theta_cand = main_data['theta_candidates']
    theta_mae = main_data['theta_mae']
    names = main_data['ell_names']
    colors = [COLOR_SEP, COLOR_GLOBAL, COLOR_FULL]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i in range(len(names)):
        ax.plot(theta_cand, theta_mae[i], 's-', color=colors[i], linewidth=1.5, markersize=6, label=names[i])

    ax.set_xlabel(r'Kernel Bandwidth $\vartheta$', fontsize=16)
    ax.set_ylabel('MAE on Test Set', fontsize=16)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=14, loc='upper right', bbox_to_anchor=(1.0, 0.85))

    plt.tight_layout()
    plt.savefig('theta_sensitivity.png', bbox_inches='tight')
    plt.close()
    print("finish: theta_sensitivity.png")

# ===================== 执行绘图 =====================
if __name__ == '__main__':
    print("Starting to plot the analysis chart....\n")
    #plot_belief_heatmap()
    plot_belief_3d_lines()
    plot_mae_curve()
    plot_k_sensitivity_global()
    plot_k_evolution_phase()
    plot_theta_sensitivity()
    print("\n All plotting completed")
