# Recommended to run on Windows system! Avoid missing system font libraries!

import numpy as np
import matplotlib.pyplot as plt

# === 全局字体统一配置 ===
plt.rcParams.update({
    # 衬线字体家族，优先使用 Times New Roman
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'SimSun', 'serif'],
    # 数学公式字体匹配 Times 风格（stix 与 Times 视觉高度一致）
    'mathtext.fontset': 'stix',
    # 字号统一设置
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    # 分辨率
    'figure.dpi': 150,
})

# --- 1. 加载计算结果 ---
data = np.load("idm_opt_result.npz", allow_pickle=True)
lam_set = data["lam_set"]
obj_set = data["obj_set"]
min_y = data["min_y"]
unique_solution = float(data["unique_solution"])
params = data["params"].item()
print(f"已加载数据: λ∈[{lam_set[0]:.4f}, {lam_set[-1]:.4f}], 最优λ*={unique_solution:.6f}, 最小排队数={min_y}")

# --- 2. 绘图 ---
fig, ax = plt.subplots(figsize=(12, 6))

# 标注最优解
# ax.axvline(x=unique_solution, color='r', linestyle='--', alpha=0.7, label=f'$\\lambda^*={unique_solution:.4f}$')
ax.scatter([unique_solution], [min_y], color='r', zorder=5, s=40, alpha=0.7, label=f'Optimal Green Split $\\lambda^*={unique_solution:.4f}$')

# 主曲线
ax.plot(lam_set, obj_set, 'd-', markersize=4, linewidth=1.5, label='Queue Count')



# 坐标轴与网格
ax.set_xlabel(r'Green Split $\lambda$')
ax.set_ylabel('Total number of queued vehicles') 
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1))

# 强制横轴显示两端值
current_xticks = ax.get_xticks()
new_xticks = np.unique(np.concatenate([current_xticks, [lam_set[0], lam_set[-1]]]))
ax.set_xticks(np.sort(new_xticks))
ax.set_xlim(lam_set[0], lam_set[-1])

# 参数信息框
p = params
param_text = (
    rf"$s_0={p['s0']}(m),\;v_s={p['v_stop']}(m/s),\;v_0={p['v0']}(m/s),\;"
    rf"a_{{\max}}=b_{{\max}}={p['a_max']}(m/s^2)$" "\n"
    rf"$T_D={p['T_D']}(s),\;T_c={p['T_cycle']}(s),\;T_y={p['T_y']}(s),\;\Delta t={p['Δt']}(s)$" "\n"
    rf"$\alpha={p['α']},\;\beta={int(p['β'])},\;\gamma={int(p['γ'])},\;\eta={p['η']}$"
)
ax.text(
    0.98, 0.95, param_text,
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=14,
    bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFFFFF', edgecolor='black', alpha=0.7)
)

plt.tight_layout()
plt.savefig('Characteristics_obj_fun.png', dpi=300, bbox_inches='tight')
plt.show()
