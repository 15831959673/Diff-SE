import numpy as np
import matplotlib.pyplot as plt
from Second.DNA_utils import *
import seaborn as sns


def normalize_frequencies(data):
    row_sums = data.sum(axis=1, keepdims=True)
    normalized_data = data / row_sums
    normalized_data[np.isnan(normalized_data)] = 0
    return normalized_data


def calculate_entropy(frequencies):
    entropy = -np.sum(frequencies * np.log2(frequencies + 1e-10), axis=1)
    return entropy


if __name__ == '__main__':
    # 读取数据
    x1 = read_kmer('mESC', 1234, i=False)
    x2 = read_kmer('myotube', 1234, i=False)
    x3 = read_kmer('macrophage', 1234, i=False)
    x4 = read_kmer('proB-cell', 1234, i=False)
    x5 = read_kmer('Th-cell', 1234, i=False)
    x6 = read_kmer('H2171', 1234, i=False)
    x7 = read_kmer('U87', 1234, i=False)
    x8 = read_kmer('MM1.S', 1234, i=False)
    data_dict1 = {'mESC': x1[0][x1[4] == 1], 'myotube': x2[0][x2[4] == 1], 'macrophage': x3[0][x3[4] == 1],
                  'proB-cell': x4[0][x4[4] == 1],
                  'Th-cell': x5[0][x5[4] == 1], 'H2171': x6[0][x6[4] == 1], 'U87': x7[0][x7[4] == 1],
                  'MM1.S': x8[0][x8[4] == 1]}

    data_dict1 = {'mESC': x1[0], 'myotube': x2[0], 'macrophage': x3[0], 'proB-cell': x4[0],
                  'Th-cell': x5[0], 'H2171': x6[0], 'U87': x7[0], 'MM1.S': x8[0]}

    data_dict2 = {'mESC': x1[1], 'myotube': x2[1], 'macrophage': x3[1], 'proB-cell': x4[1],
                  'Th-cell': x5[1], 'H2171': x6[1], 'U87': x7[1], 'MM1.S': x8[1]}

    data_dict3 = {'mESC': x1[2], 'myotube': x2[2], 'macrophage': x3[2], 'proB-cell': x4[2],
                  'Th-cell': x5[2], 'H2171': x6[2], 'U87': x7[2], 'MM1.S': x8[2]}

    data_dict4 = {'mESC': x1[3], 'myotube': x2[3], 'macrophage': x3[3], 'proB-cell': x4[3],
                  'Th-cell': x5[3], 'H2171': x6[3], 'U87': x7[3], 'MM1.S': x8[3]}

    data_dicts = [data_dict1, data_dict2, data_dict3, data_dict4]

    colors = {'mESC': '#66adc2', 'myotube': '#243b42', 'macrophage': '#e84545',
              'proB-cell': '#c29466', 'Th-cell': '#362b21', 'H2171': '#f9ce3d',
              'U87': '#b47878', 'MM1.S': '#d32b2b'}

    # 创建画布，2行2列的子图
    plt.rcParams['font.size'] = 8
    # plt.rc('font', weight='bold')
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    for i, data_dict in enumerate(data_dicts):
        ax = axs[i // 2, i % 2]

        all_entropies = []

        for label, data in data_dict.items():
            normalized_data = normalize_frequencies(data)
            entropies = calculate_entropy(normalized_data)
            all_entropies.append(entropies)

            sns.kdeplot(entropies, fill=True, label=label, alpha=0.1, color=colors[label], ax=ax)


        x_min, x_max = np.concatenate(all_entropies).min(), np.concatenate(all_entropies).max()
        ax.set_xlim(x_min + 0.2, x_max + 0.1)

        # 设置 y 轴的范围
        y_lim = ax.get_ylim()
        pos_length = y_lim[1]
        neg_length = -pos_length / 8
        ax.set_ylim(neg_length, pos_length)
        for label, data in data_dict.items():
            normalized_data = normalize_frequencies(data)
            entropies = calculate_entropy(normalized_data)
            # 计算竖线的上端点和下端点
            y_upper = neg_length * (1 / 10)
            y_lower = neg_length * (9 / 10)  # 竖线的下端点为负轴的9/10

            # 绘制竖线
            ax.vlines(entropies, y_lower, y_upper, color=colors[label], linewidth=1, alpha=0.3)

        # 在 y=0 处画水平线
        ax.axhline(y=0, color='black', linewidth=1)  # 使用虚线或实线

        ax.set_title(f'{i + 1}mer')
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Density')
        ax.legend(loc='upper left')
        ax.grid(False)

    plt.tight_layout()
    plt.savefig('entropy_kde_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
