import numpy as np
import matplotlib.pyplot as plt
from Second.DNA_utils import *
import seaborn as sns

def normalize_frequencies(data):
    # 归一化频率，使每行的和为 1
    row_sums = data.sum(axis=1, keepdims=True)  # 计算每行的和
    normalized_data = data / row_sums  # 归一化
    normalized_data[np.isnan(normalized_data)] = 0  # 处理 NaN 值（如果有任何行和为 0）
    return normalized_data

def calculate_entropy(frequencies):
    # 计算熵
    # 忽略概率为 0 的项
    entropy = -np.sum(frequencies * np.log2(frequencies + 1e-10), axis=1)  # 添加一个小常数以避免 log(0)
    return entropy

if __name__=='__main__':
    x1 = read_kmer('mESC', 1234, i=False)
    x2 = read_kmer('myotube', 1234, i=False)
    x3 = read_kmer('macrophage', 1234, i=False)
    x4 = read_kmer('proB-cell', 1234, i=False)
    x5 = read_kmer('Th-cell', 1234, i=False)
    x6 = read_kmer('H2171', 1234, i=False)
    x7 = read_kmer('U87', 1234, i=False)
    x8 = read_kmer('MM1.S', 1234, i=False)

    i = 0

    data_dict = {
        'mESC': x1[i],
        'myotube': x2[i],
        'macrophage': x3[i],
        'proB-cell': x4[i],
        'Th-cell': x5[i],
        'H2171': x6[i],
        'U87': x7[i],
        'MM1.S': x8[i]
    }

    colors = {
        'mESC': '#66adc2',
        'myotube': '#243b42',
        'macrophage': '#e84545',
        'proB-cell': '#c29466',
        'Th-cell': '#362b21',
        'H2171': '#7878b4',
        'U87': '#b47878',
        'MM1.S': '#64c864'
    }

    # 创建画布
    plt.figure(figsize=(12, 8))

    # 绘制每组数据的熵的核密度估计
    for label, data in data_dict.items():
        # 先归一化
        normalized_data = normalize_frequencies(data)
        # 计算每条序列的熵
        entropies = calculate_entropy(normalized_data)
        # 绘制核密度估计
        sns.kdeplot(entropies, fill=True, label=label, alpha=0.5, color=colors[label])

        # 竖线的 y 位置
        y_upper = 0  # 竖线的上端点
        y_lower = -7  # 竖线的下端点，设置为 0 以接住 x 轴

        # 绘制竖线
        plt.vlines(entropies, y_lower, y_upper, color=colors[label], linewidth=1, alpha=0.4)

    plt.xlim(1.7, 2.03)  # x 轴范围从 1.7 到 2.1
    plt.xticks(np.arange(1.7, 2.03, 0.03))

    # 添加标题和标签
    plt.title('Kernel Density Estimation of Sequence Entropies with Vertical Lines')
    plt.xlabel('Entropy')
    plt.ylabel('Density')

    # 添加图例
    plt.legend(title='Cell Types', loc='upper left')

    # 删除网格线
    plt.grid(False)

    # 显示图形
    plt.show()
