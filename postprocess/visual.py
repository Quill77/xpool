import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ['uniform', 'random', 'attention']
metrics = ['LR@1', 'LR@3', 'LR@5', 'LP@1', 'LP@3', 'LP@5']
uniform = [70.6122, 98.7755, 100.0000, 70.6122, 68.2993, 65.7143]
random  = [66.6667, 98.3740, 100.0000, 66.6667, 68.2927, 66.1789]
attention=[65.0602, 98.7952, 100.0000, 65.0602, 68.0054, 65.9438]

data = np.array([uniform, random, attention])   # shape: (3, 6)

# 参数
n_method = len(methods)
n_metric = len(metrics)
width = 0.15                                    # 单根柱子宽度
group_gap = 0.3                               # LR与LP之间的空隙
method_gap = 0.5                              # 不同方法之间的总空隙

# 计算 x 轴位置
x = np.zeros((n_method, n_metric))
start = 0
for i in range(n_method):
    # 同一方法内部 6 根柱子
    for j in range(n_metric):
        if j == 0:                       # 第一根
            offset = 0
        elif j == 3:                     # LP@1 前面加空隙
            offset += group_gap
        else:
            offset += width
        x[i, j] = start + offset
    start = x[i, -1] + width + method_gap   # 下一组起始位置

# 绘图
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#6BAED6', '#CAB8D9', '#FDBB84', '#B3D9C6', '#F4A6A6', '#A6D8F0']
bars = []
for j in range(n_metric):
    bar = ax.bar(x[:, j], data[:, j], width,
                 label=metrics[j], color=colors[j], edgecolor='black', linewidth=0.5)
    bars.append(bar)

# 坐标轴与标题
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Impact of Different Key-frame Selection Methods on Retrieval Accuracy', fontsize=14)
ax.set_xticks([(x[i, 2] + x[i, 3])/2 for i in range(n_method)])  # 取每组中间位置
ax.set_xticklabels(methods)
ax.set_ylim(0, 105)
ax.legend(ncol=2, fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 数值标签（可选）
for bar in bars:
    for b in bar:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.5,
                f'{h:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("keyframe_selection.png", dpi=300)