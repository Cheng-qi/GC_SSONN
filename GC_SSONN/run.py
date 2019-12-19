#!/usr/bin/env python
# encoding: utf-8

"""
@version: 3.7.2
@author: Qi Cheng
@license: Apache Licence 
@site: https://github.com/Cheng-qi
@software: PyCharm
"""
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import SelfOrganizeNN
from unit import plot
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # 数据载入
    data_name = "Wifi" #根据数据选填 Iris, Wifi, Cancer, Engi
    path = "../data/"+data_name+".data"

    if data_name == 'Iris':
        Iris_data = pd.read_csv(path, header=None)
        x_iris = Iris_data[list(range(4))]
        y_iris = pd.Categorical(Iris_data[4]).codes
        x_data = x_iris
        y_data = y_iris
    elif data_name == 'Wifi':
        wifi_data = pd.read_table(path, header=None, sep="\t")
        x_wifi = wifi_data[list(range(7))]
        y_wifi = pd.Categorical(wifi_data[7]).codes
        x_data = x_wifi
        y_data = y_wifi
    elif data_name == "Cancer":
        breast_data = pd.read_table(path, header=None, sep=",")
        x_breast = breast_data[list(range(1, 10))]
        y_breast = pd.Categorical(breast_data[10]).codes
        x_data = x_breast
        y_data = y_breast
    elif data_name == "Engi":
        path = "../data/eng.data"
        eng_data = pd.read_table(path, header=None, sep=",")
        x_eng = eng_data[list(range(8))]
        y_eng = pd.Categorical(eng_data[9]).codes
        x_data = x_eng
        y_data = y_eng
    else:
        print("没有找到此数据集")
        raise IOError

    # 结果保存路径
    result_folder = "../results/results_"+data_name

    # 随机选取合适数据集
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, stratify=y_data, random_state=0)
    # 归一化数据
    stand = StandardScaler()
    stand.fit(x_train)
    x_train_std = stand.transform(x_train)
    x_test_std = stand.transform(x_test)

    # 模型构建
    fix_stru_model = SelfOrganizeNN(x_train_std.shape[1], np.max(y_train)+1,1)
    fix_stru_model.trains(x_train_std, y_train)

    # 测试结果
    test_result, test_pre_loss = fix_stru_model.predict(x_test_std, y_test)
    test_result_pro = fix_stru_model.predict_prob(x_test_std, y_test)

    # 准确率
    test_acc_fixed = accuracy_score(y_test, test_result)
    # f1分数
    test_f1_macro_fixed = f1_score(y_test, test_result, average="macro")
    
    # 打印ACC和F1分数
    print("测试集结果")
    print( "ACC:   ",round(test_acc_fixed, 4), "\nF1_score:  ", round(test_f1_macro_fixed, 4))
    
    # 隐层节点变化图
    plot(np.array(fix_stru_model.record_hidden))