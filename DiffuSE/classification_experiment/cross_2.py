import re
import numpy as np
import matplotlib.pyplot as plt

def read_log(file):
    # 定义一个字典，用于存储结果
    results = {}

    # 读取日志文件
    with open('../Second/diffmodel/mESC/cross.log', 'r') as file:
        content = file.readlines()

    # 正则表达式匹配模式
    pattern1 = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - (\w+) - ([-\w]+)")
    pattern2 = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - (\w+) - (\w+):\s+([\d.]+)%")


    for i, line in enumerate(content):
        if i == 0:
            match = pattern1.search(line)
            name = match.group(3).strip("-")
        else:
            match = pattern2.search(line)
            evl = match.group(3)
            result = float(match.group(4)) / 100

        print(results)

if __name__=='__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # 随机种子
    np.random.seed(13)
    matrix = np.array([[97, 90.48, 91.01, 96.28, 96.14, 85.26, 83.14, 73.50],
                    [96.53, 90.5, 91.54, 95.21, 95.87, 95.48, 92.28, 93.48],
                    [97.07, 90.82, 93.42, 96.32, 97.89, 87.83, 96.58, 95.46],
                    [97.30, 90.80, 91.96, 96.7, 97.95, 76.63, 85.32, 94.93],
                    [96.97, 90.23, 92.48, 96.60, 98.6, 97.66, 86.75, 95.70],
                    [85.46, 90.21, 83.33, 86.84, 77.89, 98.2, 96.54, 95.99],
                    [96.26, 90.48, 74.29, 86.38, 87.67, 98.22, 96.5, 96.09],
                    [85.19, 90.10, 83.49, 76.60, 86.77, 97.68, 96.46, 96.1]])

    # 行列标签
    labels = ['mESC', 'myotube', 'macrophage', 'proB-cell', 'Th-cell', 'H2171', 'U87', 'MM1.S']
    custom_cmap = LinearSegmentedColormap.from_list("blue_yellow_red", ["blue", "yellow", "red"])

    # 绘制热图
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap='Blues', aspect='auto')
    plt.colorbar()  # 颜色条并标注为 ACC

    # 设置行列标签
    plt.xticks(ticks=np.arange(8), labels=labels, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(8), labels=labels)

    # 添加标题
    # plt.title("ACC Heatmap")
    plt.savefig('species_cross_2', dpi=300, bbox_inches='tight')
    plt.show()






