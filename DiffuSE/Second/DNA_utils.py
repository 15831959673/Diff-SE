from Bio import SeqIO
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import ast
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import matthews_corrcoef
# 数据过采样与随机欠采样
def dataSample(x, y):
    x = x.reshape((x.shape[0], -1))
    count = dict(Counter(y))
    sm = SMOTE(sampling_strategy={0: int(count[0]), 1: int(count[1]) * 10}, random_state=42)
    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
    x, y = sm.fit_resample(x, y)
    x, y = rus.fit_resample(x, y)
    # x = x.reshape((x.shape[0], 4, -1))
    return x, y

def diff_dataSample(x, y, g):
    num_zeros = (y == 1).sum().item()
    ones = np.ones(9 * num_zeros)

    # 将 ones 数组添加到 y 数组末尾
    y = np.concatenate((y, ones))
    g_index = np.random.choice(g.shape[0], 9 * num_zeros, replace=False)
    g_data = g[g_index, :]
    # x = x.reshape((x.shape[0], -1))
    x = np.concatenate((x, g_data), axis=0)
    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
    x, y = rus.fit_resample(x, y)
    return x, y

# 从fasta文件读出DNA序列，然后将序列与标签一起存储为csv文件。
def read_fasta():
    seqs = []
    for record in SeqIO.parse('../dataset/mESC/mESC.fasta', "fasta"):
        seqs.append(str(record.seq))
    labels = [1] * 230 + [0] * 8561
    data = {'Seqs': seqs, 'Labels': labels}
    df = pd.DataFrame(data)
    df.to_csv('../dataset/mESC/mESC.csv', index=False)

# 读出存储好的csv文件，并返回DNA序列和标签
def read_csv():
    df = pd.read_csv('../dataset/mESC/mESC.csv')
    seqs = df['Seqs'].tolist()
    labels = df['Labels'].tolist()

    return seqs, labels

def evaluate(y_pred, y_prob, y_test):
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"ACC: \t{accuracy * 100:.2f}%")

    # 计算精确率
    precision = precision_score(y_test, y_pred)
    # print(f"PRE: \t{precision * 100:.2f}%")

    # 计算召回率
    recall = recall_score(y_test, y_pred)
    # print(f"REC: \t{recall * 100:.2f}%")

    # 计算F1得分
    f1 = f1_score(y_test, y_pred)
    # print(f"F1: \t{f1 * 100:.2f}%")

    # 计算AUC（ROC曲线下面积）
    roc_auc = roc_auc_score(y_test, y_prob)
    # print(f"AUC: \t{roc_auc * 100:.2f}%")

    # 计算AUPR（PR曲线下面积）
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    aupr = auc(recall_curve, precision_curve)

    mcc = matthews_corrcoef(y_test, y_pred)
    # print(f"AUPR: \t{aupr * 100:.2f}%")
    # print(f"ACC: \t{accuracy * 100:.2f}%")
    # print(f"PRE: \t{precision * 100:.2f}%")
    # print(f"REC: \t{recall * 100:.2f}%")
    # print(f"F1: \t{f1 * 100:.2f}%")
    # print(f"AUC: \t{roc_auc * 100:.2f}%")
    # print(f"AUPR: \t{aupr * 100:.2f}%")

    return accuracy, precision, recall, f1, roc_auc, aupr, mcc

# 读取dna2vec456mer词向量
def data_pre(filename):
    # dna2vec
    df = pd.read_csv(f'../dataset/{filename}/4mer_datavec.csv', header=None)
    data4 = df.drop(df.columns[0], axis=1).values
    df = pd.read_csv(f'../dataset/{filename}/5mer_datavec.csv', header=None)
    data5 = df.drop(df.columns[0], axis=1).values
    df = pd.read_csv(f'../dataset/{filename}/6mer_datavec.csv', header=None)
    data6 = df.drop(df.columns[0], axis=1).values
    x = np.concatenate((data4, data5, data6), axis=1)
    # 要删除的行索引
    # indices_to_delete = [90, 5158]

    # 使用 np.delete 函数删除指定行
    # x = np.delete(x, indices_to_delete, axis=0)
    # 输出新的数组形状
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    # y = np.array([1] * 231 + [0] * 8561)
    return x

# 1234mer
def read_kmer(file, k, i=True):
    if i:
        k1 = int(k / 1000)
        k2 = int(k / 100 % 10)
        k3 = int(k / 10 % 10)
        k4 = k % 10
        df = pd.read_csv(f"../dataset/{file}/{file}_{k1}mer.csv")
        x1 = np.stack(df['Seqs_vec'].apply(ast.literal_eval).apply(np.array).values)
        df = pd.read_csv(f"../dataset/{file}/{file}_{k2}mer.csv")
        x2 = np.stack(df['Seqs_vec'].apply(ast.literal_eval).apply(np.array).values)
        df = pd.read_csv(f"../dataset/{file}/{file}_{k3}mer.csv")
        x3 = np.stack(df['Seqs_vec'].apply(ast.literal_eval).apply(np.array).values)
        df = pd.read_csv(f"../dataset/{file}/{file}_{k4}mer.csv")
        x4 = np.stack(df['Seqs_vec'].apply(ast.literal_eval).apply(np.array).values)
        x = np.concatenate((x1, x2, x3, x4), axis=1)
        y = df['Labels'].values
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        return x, y
    else:
        k1 = int(k / 1000)
        k2 = int(k / 100 % 10)
        k3 = int(k / 10 % 10)
        k4 = k % 10
        df = pd.read_csv(f"../dataset/{file}/{file}_{k1}mer.csv")
        x1 = np.stack(df['Seqs_vec'].apply(ast.literal_eval).apply(np.array).values)
        df = pd.read_csv(f"../dataset/{file}/{file}_{k2}mer.csv")
        x2 = np.stack(df['Seqs_vec'].apply(ast.literal_eval).apply(np.array).values)
        df = pd.read_csv(f"../dataset/{file}/{file}_{k3}mer.csv")
        x3 = np.stack(df['Seqs_vec'].apply(ast.literal_eval).apply(np.array).values)
        df = pd.read_csv(f"../dataset/{file}/{file}_{k4}mer.csv")
        x4 = np.stack(df['Seqs_vec'].apply(ast.literal_eval).apply(np.array).values)
        y = df['Labels'].values

        return [x1, x2, x3, x4, y]

# l=3, w=0.01
def read_PseDNC(l, w):
    df = pd.read_csv(f"../dataset/mESC/mESC_PseDNC{l}_{w}.csv")
    x = np.stack(df['Seqs_vec'].apply(ast.literal_eval).apply(np.array).values)
    y = df['Labels'].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x, y

# K:3, L:5, W:0.05
def read_PseKNC(k, l, w):
    df = pd.read_csv(f"../dataset/mESC/mESC_PseKNC{k}_{l}_{float(w):.2f}.csv")
    x = np.stack(df['Seqs_vec'].apply(ast.literal_eval).apply(np.array).values)
    y = df['Labels'].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x, y

# l=4, w=0.5
def read_PCPseDNC(l, w):
    df = pd.read_csv(f"../dataset/mESC/mESC_PCPseDNC{l}_{float(w):.2f}.csv")
    x = np.stack(df['Seqs_vec'].apply(ast.literal_eval).apply(np.array).values)
    y = df['Labels'].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x, y

# L:4, W:0.1
def read_PCPseTNC(l, w):
    df = pd.read_csv(f"../dataset/mESC/mESC_PCPseTNC{l}_{float(w):.2f}.csv")
    x = np.stack(df['Seqs_vec'].apply(ast.literal_eval).apply(np.array).values)
    y = df['Labels'].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x, y

# L:5, W:0.1
def read_SCPseDNC(file, l, w):
    df = pd.read_csv(f"../dataset/{file}/{file}_SCPseDNC{l}_{float(w):.2f}.csv")
    x = np.stack(df['Seqs_vec'].apply(ast.literal_eval).apply(np.array).values)
    y = df['Labels'].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x, y

# L:5, W:0.05
def read_SCPseTNC(file, l, w):
    df = pd.read_csv(f"../dataset/{file}/{file}_SCPseTNC{l}_{float(w):.2f}.csv")
    x = np.stack(df['Seqs_vec'].apply(ast.literal_eval).apply(np.array).values)
    y = df['Labels'].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x, y

def combine_features():
    # 1234mer
    x1, y = read_kmer(1234)

    # l=3, w=0.01
    x2, y = read_PseDNC(3, 0.01)

    # K:3, L:5, W:0.05
    x3, y = read_PseKNC(3, 5, 0.05)

    # l=4, w=0.5
    x4, y = read_PCPseDNC(4, 0.5)

    # L:4, W:0.1
    x5, y = read_PCPseTNC(4, 0.1)

    # L:5, W:0.1
    x6, y = read_SCPseDNC(5, 0.1)

    # L:5, W:0.05
    x7, y = read_SCPseTNC(5, 0.05)

    features = [
        (x1, "Kmer(1234)"),
        (x2, "PseDNC(L=3, W=0.01)"),
        (x3, "PseKNC(K=3, L=5, W=0.05)"),
        (x4, "PCPseDNC(L=4, W=0.5)"),
        (x5, "PCPseTNC(L=4, W=0.1)"),
        (x6, "SCPseDNC(L=5, W=0.1)"),
        (x7, "SCPseTNC(L=5, W=0.05)")
    ]
    results = []
    sources = []

    for r in range(1, len(features) + 1):
        for combo in combinations(features, r):
            combined = np.hstack([f[0] for f in combo])  # 拼接特征矩阵
            source = [f[1] for f in combo]  # 记录组合的来源标签
            results.append(combined)
            sources.append(source)

    return results, sources, y

def read_data(file):
    # 1234mer
    x1, y = read_kmer(file, 1234)

    # L:5, W:0.1
    x2, y = read_SCPseDNC(file, 5, 0.1)

    # L:5, W:0.05
    x3, y = read_SCPseTNC(file, 5, 0.05)

    # x4, y = read_PCPseDNC(file, 4, 0.5)

    x = np.concatenate((x1, x2, x3), axis=1)

    return x, y

def integrative():
    File = ['mESC', ]


if __name__ == "__main__":
    all_combinations, all_sources, y = combine_features()
    for (combo, source) in zip(all_combinations, all_sources):
        print(f"{combo.shape}, source:{source}")