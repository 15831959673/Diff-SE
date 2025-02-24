import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
file_path = '../Second/diffmodel/mESC/1-fold_loss.csv'  # 修改为实际文件路径
loss_df = pd.read_csv(file_path)

plt.rcParams.update({'font.size': 14})

# 创建图表和主轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制 Loss1
ax1.plot(loss_df['Epoch'], loss_df['Loss1'], label='Loss1', marker='o', color='#A86A9D')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss1', color='black')
ax1.tick_params(axis='y', labelcolor='black')
# ax1.legend(loc='upper left')

# 创建右边的 y 轴并绘制 Loss2 + Loss3
ax2 = ax1.twinx()
ax2.plot(loss_df['Epoch'], loss_df['Loss2_3'], label='Loss2 + Loss3', marker='o', color='#EE6A33')
ax2.set_ylabel('Loss2 + Loss3', color='black')
ax2.tick_params(axis='y', labelcolor='black')
# ax2.legend(loc='upper right')

# 添加图例并设置位置
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

# 显示图表
# plt.title('Loss1 和 Loss2 + Loss3 随 Epoch 变化')
plt.tight_layout()
plt.savefig('loss.png', dpi=300, bbox_inches='tight')
plt.show()
