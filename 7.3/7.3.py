import math
import numpy as np
import pandas as pd

D_keys = {
    '色泽': ['青绿', '乌黑', '浅白'],
    '根蒂': ['蜷缩', '硬挺', '稍蜷'],
    '敲声': ['清脆', '沉闷', '浊响'],
    '纹理': ['稍糊', '模糊', '清晰'],
    '脐部': ['凹陷', '稍凹', '平坦'],
    '触感': ['软粘', '硬滑'],
}
Class, labels = '好瓜', ['是', '否']


# 读取数据
def loadData(filename):
    dataSet = pd.read_csv(filename)
    dataSet.drop(columns=['编号'], inplace=True)
    return dataSet


# 配置测1数据
def load_data_test():
    array = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '']
    dic = {a: b for a, b in zip(dataSet.columns, array)}
    return dic


def calculate_D(dataSet):
    D = []
    for label in labels:
        temp = dataSet.loc[dataSet[Class] == label]
        D.append(temp)

    return D


def calculate_Pcxi(label, i, Dcxi, D):
    D_size = D.shape[0]
    Dcxi_size = Dcxi.shape[0]
    N = len(labels)
    Ni = len(D_keys[i])
    return (Dcxi_size + 1) / (D_size + N * Ni)


def calculate_Pcxi_xj_D(key, value, Dcxi):
    Dcxi_size = Dcxi.shape[0]
    Dcxi_xj_size = Dcxi.loc[Dcxi[key] == value].shape[0]
    Nj = len(D_keys[key])
    return (Dcxi_xj_size + 1) / (Dcxi_size + Nj)


def calculate_Pcxi_xj_C(key, value, Dcxi):
    mean, var = Dcxi[key].mean(), Dcxi[key].var()
    exponent = math.exp(-(math.pow(value - mean, 2) / (2 * var)))
    return (1 / (math.sqrt(2 * math.pi * var)) * exponent)


def calculate_probability(label, i, Dcxi, D, data_test):
    prob = calculate_Pcxi(label, i, Dcxi, D)
    for key in D.columns[:-1]:
        value = data_test[key]
        if key in D_keys:
            prob *= calculate_Pcxi_xj_D(key, value, Dcxi)
        else:
            prob *= calculate_Pcxi_xj_C(key, value, Dcxi)

    return prob


def predict(dataSet, data_test, m=5):
    Dcs = calculate_D(dataSet)
    max_prob = -1
    for label, Dc in zip(labels, Dcs):
        prob = 0
        for key, value in data_test.items():
            # 不考虑xi为连续变量
            if key not in D_keys:
                continue

            Dxi_size = dataSet.loc[dataSet[key] == value].shape[0]
            if Dxi_size < m:
                continue

            Dcxi = Dc.loc[Dc[key] == value]
            prob += calculate_probability(label, key, Dcxi, dataSet, data_test)

        if prob > max_prob:
            best_label = label
            max_prob = prob

        print(label, prob)

    return best_label


if __name__ == '__main__':
    # 读取数据
    filename = 'watermelon3_0a_Ch.txt'
    dataSet = loadData(filename)
    data_test = load_data_test()
    label = predict(dataSet, data_test)
    print('预测结果：', label)
