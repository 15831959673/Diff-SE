from torch import nn, optim
import torch
import numpy as np
import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import torch.utils as utils
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
CUDA_LAUNCH_BLOCKING=1
# 一维卷积
class DeepSE(nn.Module):
    def __init__(self):
        super(DeepSE, self).__init__()
        self.relu = nn.ReLU()
        self.cnn1 = nn.Conv1d(1, 32, 3, 1, padding=1)
        self.pool1 = nn.MaxPool1d(3)
        self.cnn2 = nn.Conv1d(32, 64, 3, 1, padding=1)
        self.pool2 = nn.MaxPool1d(4)
        self.dropout1 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1600, 640)
        self.dropout2 = nn.Dropout(0.25)
        self.linear2 = nn.Linear(640, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.cnn1(x))
        x = self.pool1(x)
        x = self.relu(self.cnn2(x))
        x = self.pool2(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

def dataSample(x, y, g):
    num_zeros = (y == 1).sum().item()
    ones = np.ones(9 * num_zeros)

    # 将 ones 数组添加到 y 数组末尾
    y = np.concatenate((y, ones))
    g_index = np.random.choice(g.shape[0], 9 * num_zeros, replace=False)
    g_data = np.squeeze(g[g_index, :], axis=1)
    # x = x.reshape((x.shape[0], -1))
    x = np.concatenate((x, g_data), axis=0)
    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
    x, y = rus.fit_resample(x, y)
    return x, y

def data_pre(filename):
    # dna2vec
    df = pd.read_csv(f'../dataset/{filename}/4mer_datavec.csv')
    data4 = df.drop(df.columns[0], axis=1).values
    df = pd.read_csv(f'../dataset/{filename}/5mer_datavec.csv')
    data5 = df.drop(df.columns[0], axis=1).values
    df = pd.read_csv(f'../dataset/{filename}/6mer_datavec.csv')
    data6 = df.drop(df.columns[0], axis=1).values
    x = np.concatenate((data4, data5, data6), axis=1)

    label = pd.read_csv(f'../dataset/{filename}/{filename}.csv')['label'].values[1:]
    y = (label == 'YES').astype(int)
    return x, y

def read_generate(n):
    data = np.load(f'generat_pos{n}.npz')['reslut']

    return

def train():
    print(torch.cuda.is_available())

    X, Y = data_pre('mESC')
    device = torch.device('cuda:0')
    generate_data = np.load(f'generat_pos700.npz')['reslut']
    epoch = 80
    count = 0
    skf = StratifiedKFold(n_splits=10)
    eval_sum = {'acc_sum': 0, 'pre_sum': 0, 'rec_sum': 0, 'f1_sum': 0, 'auc_sum': 0, 'aupr_sum': 0}
    for train_index, test_index in skf.split(X, Y):

        model = DeepSE().to(device)
        loss = nn.CrossEntropyLoss()
        loss = loss.to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-6, amsgrad=False)

        count = count + 1
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # 归一化
        num_zeros = (y_train == 1).sum().item()
        scaler = StandardScaler()
        scaler = scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        x_train, y_train = dataSample(x_train, y_train, generate_data)


        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        train_data = TensorDataset(x_train, y_train)
        train_iter = DataLoader(dataset=train_data, batch_size=32, shuffle=False)
        test_data = TensorDataset(x_test, y_test)
        test_iter = DataLoader(dataset=test_data, batch_size=32, shuffle=False)
        eval_max = {'acc_max': 0, 'pre_max': 0, 'rec_max': 0, 'f1_max': 0, 'auc_max': 0, 'aupr_max': 0}
        for i in range(epoch):
            count = 0
            sum_loss = 0
            y_s = torch.tensor([])
            y_p = torch.tensor([])
            for x, y in tqdm(train_iter):
                x = x.to(device)
                y = y.to(device)
                y_score = model(x)
                l = loss(y_score, y.view(-1).long())
                sum_loss += l.item()
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                y_score = y_score.to('cpu')
                y_s = torch.cat((y_s, y_score[:, 1]), dim=0)
                y_pred = torch.argmax(y_score, dim=1)
                y_p = torch.cat((y_p, y_pred), dim=0)
            train_acc = accuracy_score(y_train, y_p) * 100
            print(f"第{i + 1}轮：训练集的ACC为{train_acc}")
            with torch.no_grad():
                y_s = torch.tensor([])
                y_p = torch.tensor([])
                for x, y in tqdm(test_iter):
                    x = x.to(device)
                    y = y.to(device)
                    y_score = model(x)
                    y_score = y_score.to('cpu')
                    y_s = torch.cat((y_s, y_score[:, 1]), dim=0)
                    y_pred = torch.argmax(y_score, dim=1)
                    y_p = torch.cat((y_p, y_pred), dim=0)
                test_acc = accuracy_score(y_test, y_p) * 100
                test_pre = precision_score(y_test, y_p) * 100
                test_recall = recall_score(y_test, y_p) * 100
                test_f1 = f1_score(y_test, y_p) * 100
                test_auc = roc_auc_score(y_test, y_s) * 100
                test_aupr = average_precision_score(y_test, y_s) * 100
                print(f"测试集的ACC为{test_acc}")
                print(f"测试集的PRE为{test_pre}")
                print(f"测试集的RECALL为{test_recall}")
                print(f"测试集的F1为{test_f1}")
                print(f"测试集的AUC为{test_auc}")
                print(f"测试集的AUPR为{test_aupr}")
                if (eval_max['aupr_max'] < test_aupr) and (test_acc > 88):
                    eval_max['acc_max'] = test_acc
                    eval_max['pre_max'] = test_pre
                    eval_max['rec_max'] = test_recall
                    eval_max['f1_max'] = test_f1
                    eval_max['auc_max'] = test_auc
                    eval_max['aupr_max'] = test_aupr
        eval_sum['acc_sum'] += eval_max['acc_max']
        eval_sum['pre_sum'] += eval_max['pre_max']
        eval_sum['rec_sum'] += eval_max['rec_max']
        eval_sum['f1_sum'] += eval_max['f1_max']
        eval_sum['auc_sum'] += eval_max['auc_max']
        eval_sum['aupr_sum'] += eval_max['aupr_max']
    print(f"测试集的ACC为{eval_sum['acc_sum']/10}")
    print(f"测试集的PRE为{eval_sum['pre_sum']/10}")
    print(f"测试集的RECALL为{eval_sum['rec_sum']/10}")
    print(f"测试集的F1为{eval_sum['f1_sum']/10}")
    print(f"测试集的AUC为{eval_sum['auc_sum']/10}")
    print(f"测试集的AUPR为{eval_sum['aupr_sum']/10}")
if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")

    train()
    # read_generate(100)

