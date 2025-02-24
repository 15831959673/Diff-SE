import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 导入数据
Data = [
    {'X': np.random.multivariate_normal([40], [[60]], 300).flatten()},
    {'X': np.random.multivariate_normal([60], [[60]], 600).flatten()},
    {'X': np.random.multivariate_normal([80], [[60]], 900).flatten()},
    {'X': np.random.multivariate_normal([100], [[60]], 1200).flatten()},
    {'X': np.random.multivariate_normal([120], [[60]], 1200).flatten()},
    {'X': np.random.multivariate_normal([140], [[60]], 1500).flatten()},
    {'X': np.random.multivariate_normal([160], [[60]], 1800).flatten()},
    {'X': np.random.multivariate_normal([180], [[60]], 2000).flatten()},
]

# 一些基础设置
scatterSep = 'on'  # 是否分开绘制竖线散点
totalRatio = 'on'  # 是否各组按比例绘制

# 配色列表
C2 = np.array([[102, 173, 194], [36, 59, 66], [232, 69, 69],
               [194, 148, 102], [54, 43, 33], [120, 120, 180],
               [180, 120, 120], [100, 200, 100]]) / 255
colorList = C2

# 图像绘制
plt.figure(figsize=(10, 6))
N = len(Data)
areaHdl = []
lgdStrs = []

# 计算各类数据量
K = np.array([len(data['X']) for data in Data])

# 循环绘图
for n in range(N):
    # 计算核密度估计
    kernel = gaussian_kde(Data[n]['X'])
    xi = np.linspace(0, 200, 100)  # 更新 x 轴范围以适应新数据
    f = kernel(xi)

    if totalRatio == 'on':
        f *= K[n] / sum(K)

    areaHdl.append(plt.fill_between(xi, f, color=colorList[n], alpha=0.5, label=f'Group {n + 1}'))
    lgdStrs.append(f'Group {n + 1}')

# 绘制图例
plt.legend(lgdStrs, loc='best')

# 调整轴范围
posSep = max(max(f), 0.01)  # 确保 posSep 有效值
if scatterSep == 'on':
    plt.ylim([-posSep / 4, posSep])  # 设置负刻度为正刻度的四分之一长度
else:
    plt.ylim([-posSep / 4, posSep])

# 绘制竖线散点
for n in range(N):
    # 竖线的 y 位置
    y_upper = 0  # 竖线的上端点
    y_lower = -0.005  # 竖线的下端点，设置为 0 以接住 x 轴
    data = list(Data[n].values())[0]
    # 绘制竖线
    plt.vlines(data, y_lower, y_upper, color=colorList[n], linewidth=1, alpha=0.4)

# 绘制 y=0 的水平线
plt.axhline(0, color='black', linewidth=1, linestyle='--')

# 设置 y 轴刻度，只显示正刻度
plt.yticks(np.arange(0, posSep + 0.01, 0.005))

# 坐标区域修饰
plt.box(on=True)
plt.title('Area plot with | scatter', fontsize=14)
plt.xlabel('XXXXX', fontsize=12)
plt.ylabel('YYYYY', fontsize=12)
plt.tick_params(axis='both', which='both', direction='out', length=6)
plt.grid(False)

# 显示图像
plt.show()
