# Note! Recommended to run on Windows system! Avoid missing system font libraries!

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.patches import Patch

# ===================== 全局样式配置 =====================
plt.rcParams.update({
    # 正文字体：Times New Roman 属于衬线字体，归入 serif 家族
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    # LaTeX 数学公式字体：STIX 与 Times New Roman 视觉高度一致
    'mathtext.fontset': 'stix',
    # 负号兼容：避免坐标轴负号显示为方块
    'axes.unicode_minus': False,
    # 基础字号基准
    'font.size': 12,
})

# ===================== 1. 读取NPZ文件并提取solution（保留两位小数） =====================
def extract_and_process_solutions(filename: str):
    # 加载NPZ数据集，提取最优绿信比数组
    data = np.load(filename, allow_pickle=True)
    opt_lam = data['optimal_lam']
    
    # 统一保留两位小数，与原逻辑完全一致
    solutions = [round(float(val), 2) for val in opt_lam]
    return solutions

# ===================== 2. 统计概率 + 准备绘图数据 =====================
def prepare_histogram_data(solutions):
    total = len(solutions)
    if total == 0:
        raise ValueError("没有有效的solution数据")
    
    # 统计每个两位小数solution的出现次数
    counter = Counter(solutions)
    
    # 生成完整横轴：0.00 到 1.00，步长0.01（共101个点）
    x = np.arange(0.00, 1.01, 0.01)
    x = np.round(x, 2)  # 确保精度
    
    # 计算概率
    y = np.array([counter.get(val, 0) / total for val in x])
    
    # 准备颜色数组
    threshold = 0.5
    colors = []
    for val in x:
        if val > threshold:
            colors.append('#666666')  # 深灰色（满足条件的部分）
        else:
            colors.append('#CCCCCC')  # 浅灰色（默认部分）
    
    # 计算累计概率
    mask = x > threshold
    cumulative_prob = np.sum(y[mask])
    
    return x, y, colors, cumulative_prob, threshold

# ===================== 3. 绘制直方图 =====================
def plot_histogram(x, y, colors, cumulative_prob, threshold):
    plt.figure(figsize=(14, 5))
    
    # 绘制直方图：精确控制每个柱子的颜色和轮廓
    bar_width = 0.008  # 柱子宽度，略小于0.01的间距，避免重叠
    plt.bar(
        x, y,
        width=bar_width,
        color=colors,
        edgecolor='#000000',  # 纯黑实线轮廓
        linewidth=1
    )
    
    # 坐标轴刻度设置
    # 横轴：0到1，间距0.05
    plt.xticks(np.arange(0.0, 1.05, 0.05), fontsize=12)
    plt.xlim(0, 1.0)
    
    # 纵轴：百分比制
    plt.yticks(fontsize=12)
    ax = plt.gca()
    ax.set_yticklabels([f'{tick*100:.0f}%' for tick in ax.get_yticks()])
    
    # 图例：手动创建两个Patch表示两种颜色
    legend_elements = [
        Patch(facecolor='#CCCCCC', edgecolor='#000000', label=r'$\lambda \leq 0.5$'),
        Patch(facecolor='#666666', edgecolor='#000000', label=r'$\lambda > 0.5$')
    ]
    plt.legend(handles=legend_elements, fontsize=14, loc='upper left')
    
    # 显示累加概率文本框
    text_str = f'Cumulative Probability\n($\lambda$ > {threshold:.1f}): {cumulative_prob:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(
        0.5, 0.95, text_str,
        transform=plt.gca().transAxes,
        fontsize=16,
        verticalalignment='top',
        horizontalalignment='center',
        bbox=props
    )
    
    # 图表美化
    plt.xlabel('Optimal Green Split (2 decimal places)', fontsize=16, labelpad=10)
    plt.ylabel('Probability (%)', fontsize=16, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')  # 只显示y轴网格
    plt.tight_layout()
    
    # 保存并显示
    plt.savefig('P6_histogram.png', dpi=600, bbox_inches='tight')
    plt.show()
    
    # 控制台输出
    print(f"\n=== 累加概率统计 ===")
    print(f"Solution > {threshold:.3f} 的累加概率: {cumulative_prob*100:.4f}%")

# ===================== 4. 主流程 =====================
if __name__ == "__main__":
    INPUT_FILE = "dataset_3s0_5000m.npz"  # 已修正为原文件名
    
    print("正在读取NPZ文件并提取solution...")
    solutions = extract_and_process_solutions(INPUT_FILE)
    print(f"成功提取 {len(solutions)} 个有效solution")
    
    print("正在统计概率并准备绘图数据...")
    x, y, colors, cumulative_prob, threshold = prepare_histogram_data(solutions)
    print("正在绘制直方图...")
    plot_histogram(x, y, colors, cumulative_prob, threshold)
