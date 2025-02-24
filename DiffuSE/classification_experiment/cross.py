import numpy as np
import matplotlib.pyplot as plt
from Second.DNA_utils import *

x1 = data_pre('mESC')  # dna2vec词向量
# generate_data = np.squeeze(np.load(f'diffmodel/mESC/generat_pos{i}.npz')['reslut'])
x2, Y1 = read_data('mESC')
X1 = np.concatenate((x1, x2), axis=1)


# mESC
x1 = data_pre('myotube')  # dna2vec词向量
# generate_data = np.squeeze(np.load(f'diffmodel/myotube/generat_pos{i}.npz')['reslut'])
x2, Y2 = read_data('myotube')
X2 = np.concatenate((x1, x2), axis=1)


# mESC
x1 = data_pre('macrophage')  # dna2vec词向量
# generate_data = np.squeeze(np.load(f'diffmodel/macrophage/generat_pos{i}.npz')['reslut'])
x2, Y3 = read_data('macrophage')
X3 = np.concatenate((x1, x2), axis=1)


# mESC
x1 = data_pre('proB-cell')  # dna2vec词向量
# generate_data = np.squeeze(np.load(f'diffmodel/proB-cell/generat_pos{i}.npz')['reslut'])
x2, Y4 = read_data('proB-cell')
X4 = np.concatenate((x1, x2), axis=1)


# mESC
x1 = data_pre('Th-cell')  # dna2vec词向量
# generate_data = np.squeeze(np.load(f'diffmodel/Th-cell/generat_pos{i}.npz')['reslut'])
x2, Y5 = read_data('Th-cell')
X5 = np.concatenate((x1, x2), axis=1)

mouse_data = np.concatenate((X1, X2, X3, X4, X5), axis=0)

# mESC
x1 = data_pre('H2171')  # dna2vec词向量
# generate_data = np.squeeze(np.load(f'diffmodel/H2171/generat_pos{i}.npz')['reslut'])
x2, Y6 = read_data('H2171')
X6 = np.concatenate((x1, x2), axis=1)


# mESC
x1 = data_pre('U87')  # dna2vec词向量
# generate_data = np.squeeze(np.load(f'diffmodel/U87/generat_pos{i}.npz')['reslut'])
x2, Y7 = read_data('U87')
X7 = np.concatenate((x1, x2), axis=1)


# mESC
x1 = data_pre('MM1.S')  # dna2vec词向量
# generate_data = np.squeeze(np.load(f'diffmodel/MM1.S/generat_pos{i}.npz')['reslut'])
x2, Y8 = read_data('MM1.S')
X8 = np.concatenate((x1, x2), axis=1)

# 定义 Min-Max 归一化函数
def min_max_normalize(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

human_data = np.concatenate((X6, X7, X8), axis=0)

# 随机选择50行数据
human_data = human_data[np.random.choice(human_data.shape[0], 50, replace=False)]
mouse_data = mouse_data[np.random.choice(mouse_data.shape[0], 50, replace=False)]

# 划分数据
human_data_0_100 = human_data[:, :100]
mouse_data_0_100 = mouse_data[:, :100]

human_data_101_184 = human_data[:, 100:184]
mouse_data_101_184 = mouse_data[:, 100:184]

human_data_185_740 = human_data[:, 184:]
mouse_data_185_740 = mouse_data[:, 184:]

# 对各部分数据进行归一化
human_data_0_100_normalized = min_max_normalize(human_data_0_100)
mouse_data_0_100_normalized = min_max_normalize(mouse_data_0_100)

human_data_101_184_normalized = min_max_normalize(human_data_101_184)
mouse_data_101_184_normalized = min_max_normalize(mouse_data_101_184)

human_data_185_740_normalized = min_max_normalize(human_data_185_740)
mouse_data_185_740_normalized = min_max_normalize(mouse_data_185_740)

# 只保留位置为10的倍数的索引
def filter_data(data):
    return data[:, np.arange(9, data.shape[1], 10)]

# 过滤数据
human_data_0_100_filtered = filter_data(human_data_0_100_normalized)
mouse_data_0_100_filtered = filter_data(mouse_data_0_100_normalized)

human_data_101_184_filtered = filter_data(human_data_101_184_normalized)
mouse_data_101_184_filtered = filter_data(mouse_data_101_184_normalized)

human_data_185_740_filtered = filter_data(human_data_185_740_normalized)
mouse_data_185_740_filtered = filter_data(mouse_data_185_740_normalized)

# 绘图
plt.figure(figsize=(15, 18))  # 设置图像大小

# 绘制前100维
plt.subplot(3, 1, 1)  # 第一个子图
plt.boxplot(human_data_0_100_filtered, positions=np.arange(1, 11) - 0.15, widths=0.3,
            boxprops=dict(color="blue"), showfliers=False, medianprops=dict(color="darkblue"))
plt.boxplot(mouse_data_0_100_filtered, positions=np.arange(1, 11) + 0.15, widths=0.3,
            boxprops=dict(color="red"), showfliers=False, medianprops=dict(color="darkred"))

plt.xticks(np.arange(10, 101, 10))  # 设置x轴刻度
plt.xlabel('Dimension Index (10 to 100)')
plt.ylabel('Normalized Values')
plt.title('Boxplot of Human and Mouse Vectors for First 100 Dimensions (10s)')
plt.legend(['Human Data', 'Mouse Data'], loc='upper right')

# 绘制第101到184维
plt.subplot(3, 1, 2)  # 第二个子图
plt.boxplot(human_data_101_184_filtered, positions=np.arange(1, 9) - 0.15, widths=0.3,
            boxprops=dict(color="blue"), showfliers=False, medianprops=dict(color="darkblue"))
plt.boxplot(mouse_data_101_184_filtered, positions=np.arange(1, 9) + 0.15, widths=0.3,
            boxprops=dict(color="red"), showfliers=False, medianprops=dict(color="darkred"))

plt.xticks(np.arange(100, 190, 10))  # 设置x轴刻度
plt.xlabel('Dimension Index (110 to 180)')
plt.ylabel('Normalized Values')
plt.title('Boxplot of Human and Mouse Vectors for Dimensions 101 to 184 (10s)')
plt.legend(['Human Data', 'Mouse Data'], loc='upper right')

# 绘制第185到740维
plt.subplot(3, 1, 3)  # 第三个子图
plt.boxplot(human_data_185_740_filtered, positions=np.arange(1, 56) - 0.15, widths=0.3,
            boxprops=dict(color="blue"), showfliers=False, medianprops=dict(color="darkblue"))
plt.boxplot(mouse_data_185_740_filtered, positions=np.arange(1, 56) + 0.15, widths=0.3,
            boxprops=dict(color="red"), showfliers=False, medianprops=dict(color="darkred"))

plt.xticks(np.arange(185, 741, 10))  # 设置x轴刻度
plt.xlabel('Dimension Index (185 to 740)')
plt.ylabel('Normalized Values')
plt.title('Boxplot of Human and Mouse Vectors for Dimensions 185 to 740 (10s)')
plt.legend(['Human Data', 'Mouse Data'], loc='upper right')

# 显示图表
plt.tight_layout()  # 自动调整子图布局
plt.show()