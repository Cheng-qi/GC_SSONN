#!/usr/bin/env python
# encoding: utf-8

"""
@version: 3.7.2
@author: Qi Cheng
@contact: chengqi96@126.com
@site: https://github.com/Cheng-qi
"""
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from pso import pso
import pandas as pd
import copy

class SelfOrganizeNN(torch.nn.Module):

    ## 初始化结构参数
    def __init__(self,n_feature,n_out,begin_n_hidden=1, h_activaty_func = torch.relu, lossfunc=torch.nn.CrossEntropyLoss(),task = "classification"):
        super(SelfOrganizeNN,self).__init__()
        # self.inputs = torch.tensor(trains)

        # self.outs = torch.tensor(exp_outs)
        self.record_hidden = []
        self.task = task
        self.n_feature = n_feature
        self.n_hidden = begin_n_hidden
        self.n_output = n_out
        self.h_actfunc = h_activaty_func
        w_1 = torch.rand(self.n_feature, self.n_hidden)*2-1
        b_1 = torch.rand(1,self.n_hidden)*2-1
        # w_1 = torch.ones(self.n_feature, self.n_hidden)
        # b_1 = torch.ones(1,self.n_hidden)
        self.w_1 = Variable(w_1, requires_grad=True)
        self.b_1 = Variable(b_1, requires_grad=True)
        w_2 = torch.rand(self.n_hidden,self.n_output)*2-1
        b_2 = torch.rand(1,self.n_output)*2-1
        # w_2 = torch.ones(self.n_hidden,self.n_output)
        # b_2 = torch.ones(1,self.n_output)
        self.w_2 = Variable(w_2, requires_grad=True)
        self.b_2 = Variable(b_2, requires_grad=True)
        self.trainable_paras = [self.w_1, self.w_2, self.b_1, self.b_2]
        self.lossfunc = lossfunc
        self.loss_record =[]
        self.growth_num = 0

    ##喂入数据
    def placeholder(self,x, y):
        self.inputs = torch.tensor(x,dtype=torch.float32).reshape(-1,self.n_feature)
        if type(y)==np.ndarray:
            if self.task == "classification":
                if y.ndim>1:
                    _, y = np.where(y == 1)
                self.exp_outs =torch.tensor(y,dtype=torch.long)
            elif self.task == "Regression":
                self.exp_outs = torch.tensor(y, dtype=torch.float32).reshape(-1, self.n_output)
        else:
            self.exp_outs = 0
        return self.inputs,self.exp_outs

    ###前向传播，定义损失函数
    def forward(self, weight_decay=0.0001, train = True):
        self.h_1 = self.h_actfunc(torch.matmul(self.inputs,self.w_1)+self.b_1)
        self.h_2 = torch.matmul(self.h_1, self.w_2)+self.b_2
        self.outs = F.softmax(self.h_2, dim=1)
        if train:
            self.loss = self.lossfunc(self.outs, self.exp_outs)  #无正则项
            for para_i in self.trainable_paras:
                self.loss += weight_decay*torch.norm(para_i)**2   #添加L2正则项
        return self.loss


    ##结构更新方法
    def _update_structure(self, pso_fun=pso,structuring_form = "growing",):
        def object(x):
            w_1 = self.w_1.data.numpy()
            w_2 = self.w_2.data.numpy()
            b_1 = self.b_1.data.numpy()

            dw_1 = x[:,:self.n_feature]
            dw_2 = x[:,self.n_feature:(self.n_feature+self.n_output)]
            db_1 = x[:,-1]

            objects = np.zeros(shape=(x.shape[0]))
            for pos_i in range(x.shape[0]):
                w_1_tem = np.concatenate((w_1, dw_1[pos_i].T.reshape(-1,1)),axis=1)
                w_2_tem = np.concatenate((w_2, dw_2[pos_i].T.reshape(1,-1)),axis=0)
                b_1_tem = np.concatenate((b_1, db_1[pos_i].T.reshape(-1,1)),axis=1)

                self.w_1.data = torch.tensor(w_1_tem, dtype=torch.float32)
                self.w_2.data = torch.tensor(w_2_tem,dtype=torch.float32)
                self.b_1.data = torch.tensor(b_1_tem,dtype=torch.float32)

                self.forward()
                objects[pos_i] = self.loss.data.numpy().reshape(1)
                self.w_1.data = torch.tensor(w_1)
                self.w_2.data = torch.tensor(w_2)
                self.b_1.data = torch.tensor(b_1)
            return objects

        # 存储结构变化前的参数值
        para_store = []
        for para_i in self.trainable_paras:
            para_store.append(para_i.data.numpy())

        # 结构增长阶段
        if structuring_form=="growing" :
            lb = [-1]*(self.n_feature+self.n_output+1) ##粒子群搜索限定的范围
            ub = [1]*(self.n_feature+self.n_output+1)
            # g,fg = pso(object, lb, ub)
            # g,fg = pso_fun(object, lb, ub, swarmsize = 50)
            g,fg = pso_fun(object, lb, ub, swarmsize = 2*(self.n_feature+self.n_output+1))

            # if fg < self.loss_record[-1]: #如果有效

            if self.loss_record[-1]-fg>self.loss_record[-1]*0.001: #如果有效
                self.growth_num = 0

                # 待添加节点的参数
                dw_1 = g[:self.n_feature]
                dw_2 = g[self.n_feature:(self.n_feature + self.n_output)]
                db_1 = g[-1]
                # db_2 = g[-1]

                # 更新节点参数
                w_1_tem = np.concatenate((para_store[0], dw_1.T.reshape(-1,1)), axis=1)
                w_2_tem = np.concatenate((para_store[1], dw_2.T.reshape(1,-1)), axis=0)
                b_1_tem = np.concatenate((para_store[2], db_1.T.reshape(-1,1)), axis=1)

                # 按照新节点重新构造网络
                self.w_1 = Variable(torch.tensor(w_1_tem, dtype=torch.float32), requires_grad=True)
                self.b_1 = Variable(torch.tensor(b_1_tem, dtype=torch.float32), requires_grad=True)
                self.w_2 = Variable(torch.tensor(w_2_tem, dtype=torch.float32), requires_grad=True)
                self.trainable_paras = [self.w_1, self.w_2, self.b_1, self.b_2]
                self.opt = torch.optim.SGD(self.trainable_paras, lr=self.lr)  # 标准梯度优化器

            else:
                self.growth_num +=1

        # 节点合并
        elif structuring_form=="pruning":

            # 取出参数值
            w_1_tem = self.w_1.data.numpy()
            w_2_tem = self.w_2.data.numpy()
            b_1_tem = self.b_1.data.numpy()

            ##合并w1和w_b,计算w_b最小距离的隐层节点，记录在min_index
            wb_1_tem = np.concatenate((w_1_tem, b_1_tem), axis=0)
            # w_1_dist0 = (w_1_tem**2).sum(axis=1)
            wb_1_dist0 = np.linalg.norm(wb_1_tem, axis=0)**2
            wb_1_dist1 = np.repeat(wb_1_dist0, wb_1_tem.shape[1]).reshape(wb_1_tem.shape[1],wb_1_tem.shape[1])
            wb_1_dist2 = wb_1_dist1.T
            wb_1_dist3 = wb_1_tem.T.dot(wb_1_tem)
            wb_1_dist4 = wb_1_dist1+wb_1_dist2-2*wb_1_dist3
            # wb_1_dist4 = np.clip(wb_1_dist4,1e-6,1e7)
            one_trip = np.ones_like(wb_1_dist4)-np.tril(np.ones_like(wb_1_dist4))
            wb_1_dist = (one_trip*wb_1_dist4)**0.5
            wb_1_dist_mean = np.mean(wb_1_dist)*2*wb_1_dist.shape[0]/(wb_1_dist.shape[0]-1)
            wb_1_dist_mean_tril = wb_1_dist+np.tril(np.ones_like(wb_1_dist)*wb_1_dist_mean)
            wb_1_dist_std = wb_1_dist_mean_tril.std()*2*wb_1_dist.shape[0]/(wb_1_dist.shape[0]-1)
            wb_1_dist[np.where(wb_1_dist==0)] = np.inf

            if np.min(wb_1_dist) < wb_1_dist_mean - 1.5 * wb_1_dist_std:
                min_index = np.unravel_index(np.argmin(wb_1_dist),wb_1_dist.shape)

                # 计算更新后的wb_1
                wb_1_dnew = np.mean(wb_1_tem[:,min_index],axis = 1)    #.reshape(w_1_tem.shape[0],-1)
                wb_1_new = copy.deepcopy(wb_1_tem)
                wb_1_new[:,min_index[0]] = wb_1_dnew
                wb_1_new = np.delete(wb_1_new, min_index[1], axis=1)

                # 更新后的参数
                w_1_new = wb_1_new[:-1,:]
                b_1_new = wb_1_new[-1,:].reshape(1,-1)
                w_2_new = copy.deepcopy(w_2_tem)
                w_2_new[min_index[0],:] = w_2_tem[min_index[0],:]+w_2_tem[min_index[1],:]
                # w_2_new[min_index[1],:] = 0
                w_2_new = np.delete(w_2_new, min_index[1], axis=0)

                # 重新构造网络
                self.w_1 = Variable(torch.tensor(w_1_new, dtype=torch.float32), requires_grad=True)
                self.b_1 = Variable(torch.tensor(b_1_new, dtype=torch.float32), requires_grad=True)
                self.w_2 = Variable(torch.tensor(w_2_new, dtype=torch.float32), requires_grad=True)
                self.trainable_paras = [self.w_1, self.w_2, self.b_1, self.b_2]
                self.opt = torch.optim.SGD(self.trainable_paras, lr=self.lr)  # 标准梯度优化器

        # 更改隐含层节点数
        self.n_hidden = self.w_1.data.shape[1]
        # print(fg)
        # print(self.forward())  # 前向传播
        self.forward()  # 前向传播
        return self.n_hidden


    def trains(self,train_data, train_labels, learning_rate = 0.1, maxiter = 300, self_structure = True, pso_fun = pso,batch_size=3000):
        # self.placeholder(train_data, train_labels)
        self.lr = learning_rate
        self.opt = torch.optim.Adam(self.trainable_paras, lr=self.lr)
        # self.opt = torch.optim.Adam([self.w_1, self.b_1, self.w_2, self.b_2], lr=learning_rate)  #Adam优化器
        # _ = self.forward()  # 前向传播
        batchs_num = int(np.ceil(train_data.shape[0]/batch_size))
        for it in range(maxiter):
            for batch_i in range(batchs_num):
                if batch_i != batchs_num-1:
                    self.placeholder(train_data[batch_i:(batch_size+batch_i)], train_labels[batch_i:(batch_size+batch_i)])
                else:
                    self.placeholder(train_data[batch_i:], train_labels[batch_i:])
            # start = time.time()
            # self.forward()#前向传播
                self.forward()  # 前向传播

                self.loss_record.append(self.loss.data.numpy())
                # print("变化前" , self.loss.data)  # 打印损失函数值

                # 是否是自组织结构
                # if self_structure and (it+1)%4==0:
                if self_structure and it>5:

                    #判断是否平滑 dx(k-2)-dx(k-1)<dx(k-1)-dx(k) 其中：dx(n) = x(n-1)-x(n)    即x(k-3)-3(x(k-2)-x(k-1))-x(k)<0
                    if self.loss_record[-1-3]-3*(self.loss_record[-1-2]-self.loss_record[-1-1])-self.loss_record[-1]<0:

                        # if it < maxiter/2 and self.growth_num<6: ##前半阶段，增长，如果超过5次增长无效，则停止增长
                        if self.growth_num<3: ##前半阶段，增长，如果超过3次增长无效，则停止增长
                            # print("增长")
                            self._update_structure(pso_fun=pso_fun,structuring_form="growing")#是否是结构自组织神经网络
                        else: #it> maxiter/2 : ##后半阶段， 合并
                            # print("合并")
                            self._update_structure(structuring_form="pruning")#是否是结构自组织神经网络
                # print(time.time()-start)
                self.loss.backward() #误差反向传播
                self.opt.step()
                self.opt.zero_grad()
            # print("第%d次迭代"%it,self.loss.data) #打印损失函数值
            self.record_hidden.append(self.n_hidden)
        return self.loss.data

    def predict(self,input,exp_out):
        self.placeholder(input, exp_out)
        pre_loss = self.forward(train=False)
        self.pre_one_hot = np.argmax(self.outs.data.numpy(),1)
        return self.pre_one_hot, pre_loss

    def predict_prob(self,input,exp_out):
        self.placeholder(input, exp_out)
        self.forward(train=False)
        return self.outs.data

    def evaluate(self):
        pre_one_hot = np.argmax(self.outs.data.numpy(),1)
        exp_one_hot = self.exp_outs.numpy()
        is_acc_sample = np.array(exp_one_hot==pre_one_hot,dtype=np.int)
        self.accuracy = is_acc_sample.mean()
        return self.accuracy


if __name__=="__main__":
    pass







