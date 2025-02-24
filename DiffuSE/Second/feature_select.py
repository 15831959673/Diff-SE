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

logging.basicConfig(
    filename='combine_features.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

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
        model = RandomForestClassifier(n_estimators=500, max_depth=10)

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

if __name__ == "__main__" :
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    all_combinations, all_sources, y = combine_features()
    for (x, source) in zip(all_combinations, all_sources):
        train_RF(500, 10, x, y, source)  # train_svm('linear', 0.001, l, w)

