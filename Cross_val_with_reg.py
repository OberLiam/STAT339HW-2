# -*- coding: utf-8 -*-
"""
Feb 24, 2020

@author: Peng
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import regressionmain as reg
import math


# (a)
def Mse(target, prediction):
    return (np.square(target - prediction)).mean(axis=None)


# (b)
def cross_val(ordinarydata, K, seed=1, lam=0, train_err=False):
    total = np.shape(ordinarydata)[0]
    if total == 0:
        print("dataset is empty - abort!")
        return
    np.random.seed(seed)
    shuffled_set = np.copy(ordinarydata)
    np.random.shuffle(shuffled_set)
    mse_list = []
    ret = []
    for i in range(K):
        start = math.floor(i * total / K)
        end = math.floor((i + 1) * total / K)
        val_set = shuffled_set[start:end]
        train_set = np.concatenate((shuffled_set[:start], shuffled_set[end:]))
        W = reg.getOLS(train_set, lam)
        ret.append(W)
        pred = np.dot(val_set[:, :-1], np.transpose(W))
        mse = reg.mse(pred, val_set[:, -1])
        mse_list.append(mse)
    return np.mean(mse_list), np.std(mse_list)


# (c)
def best_order(ordinarydata, K, seed, ret_err=False, D=None):
    if D is None:
        D = np.shape(ordinarydata)[0]
    total = np.shape(ordinarydata)[0]
    if total == 0:
        print("dataset is empty - abort!")
        return
    if K < 2:
        print("K too small")
        return
    np.random.seed(seed)
    shuffled_set = np.copy(ordinarydata)
    np.random.shuffle(shuffled_set)
    KD_matrix = np.empty((K, D+1))
    KD_matrix_train = np.empty((K, D+1))
    for i in range(K):
        start = math.floor(i * total / K)
        end = math.floor((i + 1) * total / K)
        val_set = shuffled_set[start:end]
        train_set = np.concatenate((shuffled_set[:start], shuffled_set[end:]))
        for j in range(D+1):
            W = reg.getOLSpoly(train_set, j)
            ########W = reg.getOLS(train_set, j, regparam=1)
            W = np.expand_dims(W, axis=1)
            converted = reg.convertpoly(val_set[:,:-1], j)
            #print(converted)
            #print(j)
            pred = np.dot(converted[:,:-1], W)
            mse = Mse(pred, val_set[:, -1])
            KD_matrix[i,j]=mse
            if ret_err is True:
                pred_train = np.dot(train_set[:, :-1], np.transpose(W))
                KD_matrix_train[i, j]=reg.mse(pred_train, train_set[:, -1])
    means = np.mean(KD_matrix, axis=0)
    sds = np.std(KD_matrix, axis=0)
    means_train = np.mean(KD_matrix_train, axis=0)
    sds_train = np.std(KD_matrix_train, axis=0)
    best_index = np.argmin(means)
    if ret_err is True:
        return means, sds, means_train, sds_train, best_index
    else:
        return means, sds, best_index

#def get_errors(data, ):
    

def main():
    data = reg.getdataset("womens100.csv")
    scale = reg.scaledata(data)
    print(data)
    print(reg.getOLSpoly(data, 16))
    # K=10
    means, sds, best_index = best_order(data, 10, 1)
    
    means *= scale[-1]
    sds *= scale[-1]
    
    x = np.arange(np.shape(means)[0])
    
    
    newdata = reg.getdataset("modernwomens100.csv")
    
    
    
    plt.errorbar(x, means, sds, linestyle='None',fmt='o', ecolor='g', capthick=2)
    #plt.margins(x=0, y=-0.25)
    plt.show()  # This plot looks ugly due to one enormous standard deviation value

    print(means)
    print(sds)
    print(best_index)

    # K=N
    # print(np.shape(data))
    # means, sds, best_index = best_order(data, 10, 1)
    # x = np.arange(np.shape(means)[0])
    #
    # plt.errorbar(x, means, sds, linestyle='None', fmt='o', ecolor='g', capthick=2)
    # plt.margins(x=0, y=-0.25)
    # plt.show()  # This plot looks ugly due to one enormous standard deviation value
    #
    # print(means)
    # print(sds)
    # print(best_index)



if __name__ == "__main__":
    main()
