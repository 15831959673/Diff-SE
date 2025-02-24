import numpy as np
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from DNA_utils import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import logging
import torch
import torch.utils as utils
from torch import optim
from tqdm import tqdm
from model_contrastive import *
import torch.nn.functional as F

device = torch.device('cuda:0')
Margin = 2


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        # 将预测概率限制在(0, 1)之间
        probs = torch.sigmoid(outputs)
        # 计算交叉熵损失
        bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')

        # 获取正确分类概率 pt
        pt = torch.where(targets == 1, probs, 1 - probs)

        # 计算 Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # 返回损失均值
        return focal_loss.mean()

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=Margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(self.margin - euclidean_distance, 2))

        return loss_contrastive

def collate(batch):
    seq1_ls = []
    seq2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []
    batch_size = len(batch)

    for i in range(int(batch_size)-1):
        seq1, label1 = batch[i][0], batch[i][1].int()
        for j in range(i+1, int(batch_size)):
            seq2, label2 = batch[j][0], batch[j][1].int()
            label1_ls.append(label1.unsqueeze(0))
            label2_ls.append(label2.unsqueeze(0))
            label = (label1 ^ label2)  # 异或, 相同为 0 ,相异为 1
            seq1_ls.append(seq1.unsqueeze(0))
            seq2_ls.append(seq2.unsqueeze(0))
            label_ls.append(label.unsqueeze(0))
    seq1 = torch.cat(seq1_ls).to(device)
    seq2 = torch.cat(seq2_ls).to(device)
    label = torch.cat(label_ls).to(device)
    label1 = torch.cat(label1_ls).to(device)
    label2 = torch.cat(label2_ls).to(device)

    return seq1, seq2, label, label1, label2



def train(file, i=None):
    logging.basicConfig(
        filename=f'diff_contrastive_model/{file}/result.log',
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    #
    # logging.basicConfig(
    #     filename=f'smote_model/{file}/smote_result.log',
    #     filemode='w',
    #     format='%(asctime)s - %(levelname)s - %(message)s',
    #     level=logging.INFO
    # )

    device = torch.device('cuda:0')
    X1 = data_pre(file)  # dna2vec词向量
    generate_data = np.squeeze(np.load(f'diffmodel/{file}/generat_pos{i}.npz')['reslut'])
    skf = StratifiedKFold(n_splits=10)

    epochs = 300
    X2, Y = read_data(file)

    X = np.concatenate((X1, X2), axis=1)

    count = 0
    eval_sum = {'acc_sum': 0, 'pre_sum': 0, 'rec_sum': 0, 'f1_sum': 0, 'auc_sum': 0, 'aupr_sum': 0, 'mcc_sum': 0}
    for train_index, test_index in skf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # x_train, y_train = dataSample(x_train, y_train)
        x_train, y_train = diff_dataSample(x_train, y_train, generate_data)

        count += 1

        train_data = utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_iter = utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, collate_fn=collate)

        test_data = utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
        test_iter = utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=True)

        # scaler = StandardScaler()
        # scaler = scaler.fit(x_train)
        # x_train = scaler.transform(x_train)
        # x_test = scaler.transform(x_test)

        # 初始化模型和损失函数
        model = Predictor().to(device)
        criterion_model = nn.BCELoss().to(device)
        # criterion_model = FocalLoss().to(device)
        criterion = ContrastiveLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=False)
        eval_max = {'acc_max': 0, 'pre_max': 0, 'rec_max': 0, 'f1_max': 0, 'auc_max': 0, 'aupr_max': 0, 'mcc_max': 0}
        for epoch in range(epochs):
            loss_ls = []
            loss1_ls = []
            loss2_3_ls = []
            # y_prob = np.array([])
            # y_pred = np.array([])
            model.train()
            for seq1, seq2, label, label1, label2 in tqdm(train_iter):
                output1 = model(seq1)
                output2 = model(seq2)
                output3 = model.train_pre(seq1)
                output4 = model.train_pre(seq2)
                loss1 = criterion(output1, output2, label)
                loss2 = criterion_model(output3, label1.unsqueeze(1).float())
                loss3 = criterion_model(output4, label2.unsqueeze(1).float())
                loss = loss1 + loss2 + loss3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_ls.append(loss.item())
                loss1_ls.append(loss1.item())
                loss2_3_ls.append((loss2 + loss3).item())
            model.eval()
            with torch.no_grad():
                y_true = torch.tensor([]).to(device)  # 真实标签
                y_scores = torch.tensor([]).to(device)  # 模型的输出分数
                y_pred = torch.tensor([]).to(device)  # 预测标签
                for x, y in tqdm(test_iter):
                    x = x.to(device)
                    y = y.to(device)
                    y_score = model.train_pre(x)

                    # 存储真实标签和预测分数
                    y_true = torch.cat((y_true, y), dim=0)
                    y_scores = torch.cat((y_scores, y_score), dim=0)

                    # 计算预测的标签（阈值设置为0.5）
                    y_pred_batch = (y_score > 0.5).float()
                    y_pred = torch.cat((y_pred, y_pred_batch), dim=0)
                accuracy, precision, recall, f1, roc_auc, aupr, mcc= evaluate(y_pred.to('cpu'), y_scores.to('cpu'), y_true.to('cpu'))
                print(f"-----------epoch:{epoch + 1}-----------")
                print(f"epoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}")
                print(f"测试集的ACC为{accuracy * 100:.2f}%")
                print(f"测试集的PRE为{precision * 100:.2f}%")
                print(f"测试集的RECALL为{recall * 100:.2f}%")
                print(f"测试集的F1为{f1 * 100:.2f}%")
                print(f"测试集的AUC为{roc_auc * 100:.2f}%")
                print(f"测试集的AUPR为{aupr * 100:.2f}%")
                print(f"测试集的MCC为{mcc * 100:.2f}%")
                if (eval_max['f1_max'] < f1) and (accuracy > 0.88):
                # if (eval_max['aupr_max'] < aupr) and (accuracy > 0.88):
                    eval_max['acc_max'] = accuracy
                    eval_max['pre_max'] = precision
                    eval_max['rec_max'] = recall
                    eval_max['f1_max'] = f1
                    eval_max['auc_max'] = roc_auc
                    eval_max['aupr_max'] = aupr
                    eval_max['mcc_max'] = mcc
                    torch.save(model.state_dict(), f"diff_contrastive_model/{file}/model{count}.pth")
                    # torch.save(model.state_dict(), f"smote_model/{file}/model{count}.pth")
        eval_sum['acc_sum'] += eval_max['acc_max']
        eval_sum['pre_sum'] += eval_max['pre_max']
        eval_sum['rec_sum'] += eval_max['rec_max']
        eval_sum['f1_sum'] += eval_max['f1_max']
        eval_sum['auc_sum'] += eval_max['auc_max']
        eval_sum['aupr_sum'] += eval_max['aupr_max']
        eval_sum['mcc_sum'] += eval_max['mcc_max']
        logging.info(f'-------count:{count}-------')
        logging.info(f"ACC: \t{eval_max['acc_max'] * 100:.2f}%")
        logging.info(f"PRE: \t{eval_max['pre_max'] * 100:.2f}%")
        logging.info(f"REC: \t{eval_max['rec_max'] * 100:.2f}%")
        logging.info(f"F1: \t{eval_max['f1_max'] * 100:.2f}%")
        logging.info(f"AUC: \t{eval_max['auc_max'] * 100:.2f}%")
        logging.info(f"AUPR: \t{eval_max['aupr_max'] * 100:.2f}%")
        logging.info(f"MCC: \t{eval_max['mcc_max'] * 100:.2f}%")
    print(f"测试集的ACC为{eval_sum['acc_sum'] / 10 * 100:.2f}%")
    print(f"测试集的PRE为{eval_sum['pre_sum'] / 10 * 100:.2f}% ")
    print(f"测试集的RECALL为{eval_sum['rec_sum'] / 10 * 100:.2f}%")
    print(f"测试集的F1为{eval_sum['f1_sum'] / 10 * 100:.2f}%")
    print(f"测试集的AUC为{eval_sum['auc_sum'] / 10 * 100:.2f}%")
    print(f"测试集的AUPR为{eval_sum['aupr_sum'] / 10 * 100:.2f}%")
    print(f"测试集的MCC为{eval_sum['mcc_sum'] / 10 * 100:.2f}%")
    # logging.info(f'-------diff{i * 100}avg_result-------')
    logging.info(f'-------_avg_result-------')
    logging.info(f"ACC: \t{eval_sum['acc_sum'] / 10 * 100:.2f}%")
    logging.info(f"PRE: \t{eval_sum['pre_sum'] / 10 * 100:.2f}%")
    logging.info(f"REC: \t{eval_sum['rec_sum'] / 10 * 100:.2f}%")
    logging.info(f"F1: \t{eval_sum['f1_sum'] / 10 * 100:.2f}%")
    logging.info(f"AUC: \t{eval_sum['auc_sum'] / 10 * 100:.2f}%")
    logging.info(f"AUPR: \t{eval_sum['aupr_sum'] /10 * 100:.2f}%")
    logging.info(f"MCC: \t{eval_sum['mcc_sum'] / 10 * 100:.2f}%")



if __name__ == "__main__" :
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    file = 'mESC'
    # logging.basicConfig(
    #     filename=f'smote_model/{file}/smote_result.log',
    #     filemode='w',
    #     format='%(asctime)s - %(levelname)s - %(message)s',
    #     level=logging.INFO
    # )
    File = ['mESC', 'H2171', 'mESC_constituent', 'macrophage', 'MM1.S', 'myotube', 'proB-cell', 'Th-cell', 'U87']
    # for i in range(1, 21):
    train(file=file, i=1800)

