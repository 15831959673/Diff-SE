import torch
from Unet import *
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
import torch.utils as utils
from tqdm import tqdm
# from pywt import cwt, wavedec2, waverec2, wavedec
from Bio import SeqIO
from Second.DNA_utils import *

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_promoters(self, model, n, file):
        print(f"{file}数据生成开始")
        model.eval()
        m = int(n / 16)
        start_time = time.time()
        with torch.no_grad():
            x1 = torch.randn((m, 1, 740)).to(self.device)
            x2 = torch.randn((m, 1, 740)).to(self.device)
            x3 = torch.randn((m, 1, 740)).to(self.device)
            x4 = torch.randn((m, 1, 740)).to(self.device)
            x5 = torch.randn((m, 1, 740)).to(self.device)
            x6 = torch.randn((m, 1, 740)).to(self.device)
            x7 = torch.randn((m, 1, 740)).to(self.device)
            x8 = torch.randn((m, 1, 740)).to(self.device)
            x9 = torch.randn((m, 1, 740)).to(self.device)
            x10 = torch.randn((m, 1, 740)).to(self.device)
            x11 = torch.randn((m, 1, 740)).to(self.device)
            x12 = torch.randn((m, 1, 740)).to(self.device)
            x13 = torch.randn((m, 1, 740)).to(self.device)
            x14 = torch.randn((m, 1, 740)).to(self.device)
            x15 = torch.randn((m, 1, 740)).to(self.device)
            x16 = torch.randn((int(n-15*m), 1, 740)).to(self.device)
            x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16]
            reslut = np.array([])
            f = 0
            for j in x:
                for i in tqdm(reversed(range(1, self.noise_steps)), desc='Processing'):
                    t = (torch.ones(j.shape[0]) * i).long().to(self.device)
                    predicted_noise = model(j, t)
                    alpha = self.alpha[t][:, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None]
                    beta = self.beta[t][:, None, None]
                    if i > 1:
                        noise = torch.randn_like(j)
                    else:
                        noise = torch.zeros_like(j)
                    x = 1 / torch.sqrt(alpha) * (j - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                x = x.cpu()
                if f == 0:
                    # print()
                    reslut = np.array(x)
                    f = 1
                else:
                    # print()
                    reslut = np.concatenate((reslut, np.array(x)), axis=0)
            np.savez(f'../Second/diffmodel/{file}/generat_pos1800.npz', reslut=reslut)

        # model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        end_time = time.time()
        print("{:.2f}s".format(end_time-start_time))
        return x


def  train_unet(file):
    # 获取数据集
    X1 = data_pre(file)
    # x = x[y == 1]
    # pos_n = torch.sum(y == 1).item
    # neg_n = torch.sum(y == 0).item
    X2, Y = read_data(file)

    X = np.concatenate((X1, X2), axis=1)
    x = X[Y == 1]

    # 归一化
    # scaler = StandardScaler()
    # scaler = scaler.fit(x)
    # train_x = scaler.transform(x)
    x = torch.unsqueeze(torch.from_numpy(x), dim=1)
    x_iter = utils.data.DataLoader(dataset=x, batch_size=8, shuffle=False)

    # 获取模型，优化器
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = UNet(device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)  # 1:1e-4 2:
    mse = torch.nn.MSELoss()
    diffusion = Diffusion(device=device)

    epochs = 1800

    loss_history = []  # 初始化损失记录列表
    for epoch in range(epochs):
        count = 0
        epoch_loss = 0
        max_loss = 100
        pbar = tqdm(x_iter, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')
        for x in pbar:
            x = x.to(device)
            t = diffusion.sample_timesteps(x.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(x, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise.float(), predicted_noise.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            count += 1
            pbar.set_postfix(MSE=loss.item())

        average_loss = epoch_loss / count if count > 0 else float('inf')
        loss_history.append(average_loss)  # 记录每个周期的平均损失

        # 打印每个周期的损失
        print(f'Epoch {epoch + 1}, Average Loss: {average_loss:.4f}')

    # torch.save(model.state_dict(), f'../Second/diffmodel/{file}/unet{epoch + 1}lr5.pth')
    # 保存损失记录
    pd.DataFrame(loss_history, columns=["Average Loss"]).to_csv(f'{file}_loss0.01.csv',
                                                                index=False)
        # if max_loss > (epoch_loss / count):
        #     max_loss = epoch_loss / count
        #     torch.save(model.state_dict(), '../model/unet.pth')
        # f = open('epoch_loss.txt', 'a+')
        # loss_ave = 'epoch_loss %s\n' % (epoch_loss / count)
        # f.write(loss_ave)
        # f.close()

def generate(model, file):
    # x, y = read_data(file)
    # x = x[y == 1]
    # pos_n = torch.sum(torch.from_numpy(y) == 1).item()
    # neg_n = torch.sum(torch.from_numpy(y) == 0).item()
    # bz = neg_n - pos_n
    diffusion = Diffusion(device=device)
    with torch.no_grad():
        data = diffusion.sample_promoters(model, 10000, file).cpu().detach().numpy()

if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:1")
    File = ['H2171', 'mESC_constituent', 'macrophage', 'MM1.S', 'myotube', 'proB-cell', 'Th-cell', 'U87']

    train_unet('H2171')


    # for i in range(20):
    #     load_model = torch.load(f'../Second/diffmodel/{file}/unet1800lr5.pth')
    #     model = UNet(device=device).to(device)
    #     model.load_state_dict(load_model)
    #     generate(model, file)

    # x = torch.randn((32, 1, 740)).to(device)
    # model = UNet(device=device).to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # mse = torch.nn.MSELoss()
    # diffusion = Diffusion(device=device)
    # t = diffusion.sample_timesteps(x.shape[0]).to(device)
    # x_t, noise = diffusion.noise_images(x, t)
    # predicted_noise = model(x_t, t)
    # loss = mse(noise, predicted_noise)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # bz = 1024
    # data = diffusion.sample_promoters(model, bz)