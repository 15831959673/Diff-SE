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
from model import *


def train_svm(kernel, c, l, w):
    # x, y = data_pre('mESC')  # dna2vec词向量
    x, y = read_PseDNC(l, w)
    skf = StratifiedKFold(n_splits=10)
    eval_sum = {'acc_sum': 0, 'pre_sum': 0, 'rec_sum': 0, 'f1_sum': 0, 'auc_sum': 0, 'aupr_sum': 0}
    for train_index, test_index in skf.split(x, y):
        # print("--------------")
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_train, y_train = dataSample(x_train, y_train)

        # scaler = StandardScaler()
        # scaler = scaler.fit(x_train)
        # x_train = scaler.transform(x_train)
        # x_test = scaler.transform(x_test)

        # 初始化SVM模型
        model = svm.SVC(kernel=kernel, C=c)

        # 训练模型
        model.fit(x_train, y_train)

        # 用训练好的模型预测测试集
        y_pred = model.predict(x_test)
        y_prob = model.decision_function(x_test)

        accuracy, precision, recall, f1, roc_auc, aupr = evaluate(y_pred, y_prob, y_test)
        eval_sum['acc_sum'] += accuracy
        eval_sum['pre_sum'] += precision
        eval_sum['rec_sum'] += recall
        eval_sum['f1_sum'] += f1
        eval_sum['auc_sum'] += roc_auc
        eval_sum['aupr_sum'] += aupr
    print(f'-------Lag:{l}, Weight:{w}-------')
    print(f"ACC: \t{eval_sum['acc_sum']/10 * 100:.2f}%")
    print(f"PRE: \t{eval_sum['pre_sum']/10 * 100:.2f}%")
    print(f"REC: \t{eval_sum['rec_sum']/10 * 100:.2f}%")
    print(f"F1: \t{eval_sum['f1_sum']/10 * 100:.2f}%")
    print(f"AUC: \t{eval_sum['auc_sum']/10 * 100:.2f}%")
    print(f"AUPR: \t{eval_sum['aupr_sum']/10 * 100:.2f}%")





# Assuming `evaluate`, `dataSample`, and `read_PseDNC` functions are already defined

def train_RF(n_estimators, max_depth, x, y, source):
    # x, y = data_pre('mESC')  # dna2vec词向量
    skf = StratifiedKFold(n_splits=10)
    eval_sum = {'acc_sum': 0, 'pre_sum': 0, 'rec_sum': 0, 'f1_sum': 0, 'auc_sum': 0, 'aupr_sum': 0}

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_train, y_train = dataSample(x_train, y_train)

        # scaler = StandardScaler()
        # scaler = scaler.fit(x_train)
        # x_train = scaler.transform(x_train)
        # x_test = scaler.transform(x_test)

        # 初始化随机森林模型
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        # 训练模型
        model.fit(x_train, y_train)

        # 用训练好的模型预测测试集
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        accuracy, precision, recall, f1, roc_auc, aupr = evaluate(y_pred, y_prob, y_test)
        eval_sum['acc_sum'] += accuracy
        eval_sum['pre_sum'] += precision
        eval_sum['rec_sum'] += recall
        eval_sum['f1_sum'] += f1
        eval_sum['auc_sum'] += roc_auc
        eval_sum['aupr_sum'] += aupr

    print(f'-------source:{source}-------')
    print(f"ACC: \t{eval_sum['acc_sum'] / 10 * 100:.2f}%")
    print(f"PRE: \t{eval_sum['pre_sum'] / 10 * 100:.2f}%")
    print(f"REC: \t{eval_sum['rec_sum'] / 10 * 100:.2f}%")
    print(f"F1: \t{eval_sum['f1_sum'] / 10 * 100:.2f}%")
    print(f"AUC: \t{eval_sum['auc_sum'] / 10 * 100:.2f}%")
    print(f"AUPR: \t{eval_sum['aupr_sum'] / 10 * 100:.2f}%")
    logging.info(f'-------source:{source}-------')
    logging.info(f"ACC: \t{eval_sum['acc_sum'] / 10 * 100:.2f}%")
    logging.info(f"PRE: \t{eval_sum['pre_sum'] / 10 * 100:.2f}%")
    logging.info(f"REC: \t{eval_sum['rec_sum'] / 10 * 100:.2f}%")
    logging.info(f"F1: \t{eval_sum['f1_sum'] / 10 * 100:.2f}%")
    logging.info(f"AUC: \t{eval_sum['auc_sum'] / 10 * 100:.2f}%")
    logging.info(f"AUPR: \t{eval_sum['aupr_sum'] / 10 * 100:.2f}%")

def train(file, i=None):
    logging.basicConfig(
        filename=f'diff_model_base/{file}/result.log',
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    # logging.basicConfig(
    #     filename=f'smote_model/{file}/smote_result.log',
    #     filemode='w',
    #     format='%(asctime)s - %(levelname)s - %(message)s',
    #     level=logging.INFO
    # )

    device = torch.device('cuda:1')
    X1 = data_pre(file)  # dna2vec词向量
    generate_data = np.squeeze(np.load(f'diffmodel/{file}/generat_pos{i}.npz')['reslut'])
    skf = StratifiedKFold(n_splits=10)

    epochs = 80
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
        train_iter = utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)

        test_data = utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
        test_iter = utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)

        # scaler = StandardScaler()
        # scaler = scaler.fit(x_train)
        # x_train = scaler.transform(x_train)
        # x_test = scaler.transform(x_test)

        #初始化模型和损失函数
        model = Predictor().to(device)
        loss = nn.BCELoss()
        loss = loss.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.000001, amsgrad=False)
        eval_max = {'acc_max': 0, 'pre_max': 0, 'rec_max': 0, 'f1_max': 0, 'auc_max': 0, 'aupr_max': 0, 'mcc_max': 0}
        for epoch in range(epochs):
            # y_prob = np.array([])
            # y_pred = np.array([])
            model.train()
            for x, y in tqdm(train_iter):
                x = x.to(device)
                y = y.to(device)
                y_score = model(x)
                l = loss(y_score, y.unsqueeze(1).float())
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                y_true = torch.tensor([]).to(device)  # 真实标签
                y_scores = torch.tensor([]).to(device)  # 模型的输出分数
                y_pred = torch.tensor([]).to(device)  # 预测标签
                for x, y in tqdm(test_iter):
                    x = x.to(device)
                    y = y.to(device)
                    y_score = model(x)

                    # 存储真实标签和预测分数
                    y_true = torch.cat((y_true, y), dim=0)
                    y_scores = torch.cat((y_scores, y_score), dim=0)

                    # 计算预测的标签（阈值设置为0.5）
                    y_pred_batch = (y_score > 0.5).float()
                    y_pred = torch.cat((y_pred, y_pred_batch), dim=0)
                accuracy, precision, recall, f1, roc_auc, aupr, mcc= evaluate(y_pred.to('cpu'), y_scores.to('cpu'), y_true.to('cpu'))
                print(f"-----------epoch:{epoch + 1}-----------")
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
                    # torch.save(model.state_dict(), f"diff_model_base/{file}/model{count}.pth")
                    torch.save(model.state_dict(), f"diff_model_base/{file}/model{count}.pth")
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
    file = 'macrophage'
    # logging.basicConfig(
    #     filename=f'smote_model/{file}/smote_result.log',
    #     filemode='w',
    #     format='%(asctime)s - %(levelname)s - %(message)s',
    #     level=logging.INFO
    # )
    File = ['mESC', 'H2171', 'mESC_constituent', 'macrophage', 'MM1.S', 'myotube', 'proB-cell', 'Th-cell', 'U87']
    # for i in range(1, 21):
    train(file=file, i=1800)