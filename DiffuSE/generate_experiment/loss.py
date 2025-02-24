import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv('../diff/H2171_loss.csv')

# 假设 CSV 文件中有一列名为 'loss'
loss_values = df.values

# 绘制损失图
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Loss', color='blue')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 1.5)
plt.legend()
plt.grid()
plt.show()