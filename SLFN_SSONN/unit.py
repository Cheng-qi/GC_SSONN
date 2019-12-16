#!/usr/bin/env python
# encoding: utf-8

"""
@version: 3.7.2
@author: Qi Cheng
@license: Apache Licence 
@site: https://github.com/Cheng-qi
@software: PyCharm
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc

rc('mathtext', default='regular')

# 指定默认字体
mpl.rcParams['font.sans-serif'] = ['FangSong']

# 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False
##作图函数
def plot(y,xlabel = "迭代次数",ylabel = "隐层节点个数", title="隐层节点个数迭代图"):
    fig = plt.figure(figsize=(10, 6))
    x = [i for i in range(len(y))]
    plt.plot(x, y, 'r-', linewidth=1.5, markersize=5)
    plt.title(title)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0, )
    plt.grid(True)
    plt.show()