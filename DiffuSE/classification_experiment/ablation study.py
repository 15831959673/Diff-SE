import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 定义数据
labels = ['ACC', 'PRE', 'MCC', 'F1', 'AUPR']
# 八个数据集的模型评分
datasets = {
    'mESC': ([0.927, 0.244, 0.426, 0.426, 0.407], [0.967, 0.409, 0.455, 0.461, 0.386], [0.969, 0.430, 0.453, 0.465, 0.395]),
    'myotube': ([0.892, 0.466, 0.506, 0.545, 0.491], [0.905, 0.543, 0.480, 0.526, 0.507], [0.906, 0.541, 0.488, 0.537, 0.525]),
    'macrophage': ([0.899, 0.472, 0.496, 0.540, 0.564], [0.922, 0.580, 0.528, 0.569, 0.62], [0.934, 0.706, 0.567, 0.587, 0.628]),
    'proB-cell': ([0.932, 0.264, 0.356, 0.344, 0.319], [0.965, 0.430, 0.430, 0.438, 0.356], [0.966, 0.462, 0.459, 0.468, 0.400]),
    'Th-cell': ([0.978, 0.539, 0.602, 0.602, 0.643], [0.982, 0.630, 0.648, 0.651, 0.664], [0.985, 0.696, 0.688, 0.690, 0.711]),
    'H2171': ([0.957, 0.321, 0.454, 0.429, 0.494], [0.978, 0.503, 0.508, 0.512, 0.485], [0.982, 0.604, 0.579, 0.582, 0.562]),
    'U87': ([0.954, 0.563, 0.610, 0.626, 0.686], [0.964, 0.655, 0.653, 0.668, 0.717], [0.965, 0.667, 0.655, 0.672, 0.733]),
    'MM1.S': ([0.939, 0.456, 0.538, 0.551, 0.609], [0.958, 0.600, 0.589, 0.607, 0.628], [0.959, 0.608, 0.616, 0.627, 0.646]),
}

# 创建子图
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
axs = axs.flatten()  # 将子图转换为一维数组，方便索引

width = 0.2  # 柱子的宽度

# 绘制每个数据集的柱状图
for i, (dataset, scores) in enumerate(datasets.items()):
    model1_scores, model2_scores, model3_scores = scores
    x = np.arange(len(labels))  # 横坐标的位置

    # 绘制柱状图
    axs[i].bar(x - width, model1_scores, width, label='Cnn-smote', color='#FCBB44')
    axs[i].bar(x, model2_scores, width, label='Cnn-Diff', color='#F1766D')
    axs[i].bar(x + width, model3_scores, width, label='Diff-SE', color='#7A70B5')

    # 添加标签和标题
    axs[i].set_ylabel('Value')
    axs[i].set_title(dataset)
    axs[i].set_xticks(x)
    axs[i].set_xticklabels(labels)
    axs[i].legend()
    axs[i].set_ylim(0.2, 1)  # 根据需要调整y轴范围

# 调整布局
plt.tight_layout()
plt.savefig('ablation', dpi=300, bbox_inches='tight')
plt.show()
