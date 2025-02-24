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

    # 随机种子
    np.random.seed(13)
    Datas = np.array([[97, 90.48, 91.01, 96.28, 96.14, 85.26, 83.14, 73.50],
                    [96.53, 90.5, 91.54, 95.21, 95.87, 95.48, 92.28, 93.48],
                    [97.07, 90.82, 93.42, 96.32, 97.89, 87.83, 96.58, 95.46],
                    [97.30, 90.80, 91.96, 96.7, 97.95, 76.63, 85.32, 94.93],
                    [96.97, 90.23, 92.48, 96.60, 98.6, 97.66, 86.75, 95.70],
                    [85.46, 90.21, 83.33, 86.84, 77.89, 98.2, 96.54, 95.99],
                    [96.26, 90.48, 74.29, 86.38, 87.67, 98.22, 96.5, 96.09],
                    [85.19, 90.10, 83.49, 76.60, 86.77, 97.68, 96.46, 96.1]])

    # 数据名称
    Name = ['mESC', 'myotube', 'macrophage', 'proB-cell', 'Th-cell', 'H2171', 'U87', 'MM1.S']
    titles = ['mESC', 'myotube', 'macrophage', 'proB-cell', 'Th-cell', 'H2171', 'U87', 'MM1.S']

    # 配色，8种颜色
    CList = np.array([
        [0.8941, 0.8706, 0.8078],
        [0.9255, 0.7412, 0.5843],
        [0.6078, 0.6941, 0.7333],
        [0.6902, 0.7255, 0.7451],
        [0.4745, 0.6745, 0.7412],
        [0.2039, 0.3961, 0.4588],
        [0.0902, 0.3569, 0.4118],
        [0.0902, 0.2569, 0.4118]  # 新增颜色
    ])

    # plt.rcParams.update({'font.size': 10})

    # 数据展示范围及刻度
    YLim = [0, 100]
    YTick = np.linspace(YLim[0], YLim[1], 5)

    # 创建8个子图的网格
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # 生成并绘制每个子图的随机数据
    for j, ax in enumerate(axes):
        # 为每个子图生成新的数据
        Data = Datas[j]
        ax.set_title(titles[j], fontsize=14)

        # 子图设置
        ax.set_aspect('equal')
        ax.axis('off')
        r = 0.8
        N = len(Data)
        TLim = [np.pi / 2, -np.pi - np.pi / 6]
        t01 = np.linspace(0, 1, 80)
        tT = t01 * np.diff(TLim)[0] + TLim[0]
        tX = np.cos(tT) * (N + N / 2 + 1 + r)
        tY = np.sin(tT) * (N + N / 2 + 1 + r)
        ax.plot(tX, tY, linewidth=0.8, color='k')

        # 绘制刻度线和标签
        tT = (YTick - YLim[0]) / np.diff(YLim) * np.diff(TLim) + TLim[0]
        for i, ytick in enumerate(YTick):
            t_x = np.cos(tT[i]) * (N + N / 2 + 2.6)  # 控制标签与圆圈的距离
            t_y = np.sin(tT[i]) * (N + N / 2 + 2.6)
            rotation_angle = (tT[i] / np.pi * 180 + 90) % 360
            ax.text(t_x, t_y, str(int(ytick)), ha='center', va='center', rotation=rotation_angle, fontsize=10,
                    fontname='Times New Roman')

        # 绘制柱状图，每个数据对应一个颜色
        for i in range(N):
            tR = np.concatenate([(N + N / 2 + 1 - i - 0.4) * np.ones(80), (N + N / 2 + 1 - i + 0.4) * np.ones(80)])
            tT = t01 * (Data[i] - YLim[0]) / np.diff(YLim) * np.diff(TLim) + TLim[0]
            tX = np.cos(np.concatenate([tT, tT[::-1]])) * tR
            tY = np.sin(np.concatenate([tT, tT[::-1]])) * tR
            ax.fill(tX, tY, color=CList[i], edgecolor='k', linewidth=1)

        # 显示数据名称
        for i in range(N):
            ax.text(0, N + N / 2 + 1 - i, f'{Name[i]}', ha='right', va='center', fontsize=10,
                    fontname='Times New Roman')

    plt.tight_layout()
    plt.savefig('species_cross', dpi=300, bbox_inches='tight')
    plt.show()




