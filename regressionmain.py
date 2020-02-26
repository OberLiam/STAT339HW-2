# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:18:05 2020

@author: Liam
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# TODO: clean up


# NOTATION: Throughout these comments, the number of data points is N, while
# the number of parameters ('X values') is M


# Takes in a dataset, and returns a Nx(M+1) array of the data, with rows
# of the form (t,x,x,x...)
# also returns the scale, to keep track of mater on.
def getdataset(file):
    rawdata = pd.read_csv(file, header=None).to_numpy()
    N = len(rawdata)
    ones = np.repeat(1, N)
    return np.column_stack((ones, rawdata))


# NOTE: this function actually modifies the data, and it only returns the scales used
# As written, this function doesn't handle an all-zero data-set, if you wanted it to
def scaledata(data):
    mins = data.min(axis=0)
    maxes = data.max(axis=0)
    scale = np.maximum(maxes, -mins)
    data /= scale
    return scale


# (e)
# takes in a dataset (as returned by getdataset), and converts it into a data-set
# corresponding to a degree-D polynomial regression.
# note: will break if you put D=0
def convertpoly(data, D, axis=1):
    ones = data[:, 0]
    col = data[:, axis]
    t = data[:, -1]
    # TODO: make this line prettier
    if D == 0:
        return np.column_stack((ones, t))
    else:
        pdata = np.array([col ** i for i in range(1, D + 1)]).T
        return np.column_stack((ones, pdata, t))


# gets the big X matrix
def getX(data):
    return data[:, :-1]


# (a.)
# This function takes in a data set (as returned by getdataset) and returns the
# OLS parameters (w_0,w_1) such that t_n ~ w_0 + w_1*x_n
# regparam is the regularization parameter (lambda) if you are using it
# TODO: implement regparam here
def getOLS(data, regparam=1):
    X = getX(data)  # the big X matrix
    t = data[:, -1]  # the target data
    M = len(X[0])
    #try:
    regparam=0.05
    operation = np.dot(np.linalg.inv(np.dot(X.T, X) + regparam * np.identity(M)), X.T)
    return np.dot(operation, t)
    # except np.linalg.LinAlgError as e:
    #     print("error")
    #     print(X)
    #     print(M)


# Takes in a data-set (as given by getdataset) and returns the prediction according
# to classifier, which is a 1D array
def getpred(classifier, data):
    return np.dot(getX(data), classifier)


# This takes a (one-dimensional) array and returns the same size array, kind of a
# polynomial version of getpred, but different syntax
def applypoly(poly, x):
    xacc = np.repeat(1.0, len(x))  # the "current value of x^i", which starts as all 1 when i = 0
    ret = np.repeat(0.0, len(x))  # accumulates the answer
    for i in range(len(poly)):
        ret += xacc * poly[i]
        xacc *= x
    return ret


# Takes in a data-set (as given by getdataset) and a pair (w_0, w_1), and plots everything
# Note that this only really works if M = 1.
# Can accomodate a scale, the default is no scale
def plotoutput(data, classifier, title, *, scale=None):
    X_values = data[:, 1]
    Y_true = data[:, -1]  # actual data points
    Y_pred = getpred(classifier, data)  # predicted output

    if scale is not None:
        X_values *= scale[1]
        Y_true *= scale[-1]
        Y_pred *= scale[-1]

    plt.title(title)
    plt.scatter(X_values, Y_true, c='b', label='true data')
    plt.plot(X_values, Y_pred, c='g', label='line of best fit')
    plt.legend()
    plt.show()


# Like plotoutput, but accomodates polynomial functions!
def plotoutputpoly(data, classifier, title, *, scale=None):
    X_values = data[:, 1]
    Y_true = data[:, -1]  # actual data points
    X_fine = np.linspace(min(X_values), max(X_values), num=200)  # for plotting the polynomial
    Y_pred = applypoly(classifier, X_fine)  # predicted output

    if scale is not None:
        X_values *= scale[1]
        X_fine *= scale[1]
        Y_true *= scale[-1]
        Y_pred *= scale[-1]

    plt.title(title)
    plt.scatter(X_values, Y_true, c='y', label='true data')
    plt.plot(X_fine, Y_pred, c='r', label='polynomial of best fit')
    plt.legend()
    plt.show()


# This isn't actually required, but I wrote it, so it's staying.
# returns the total squared error from solver
def getOLSerror(data, classifier, scale=None):
    true = data[:, -1]
    pred = getpred(classifier, data)
    if scale is not None:
        true *= scale[-1]
        pred *= scale[-1]
    ret = (pred - true) * (pred - true)
    return np.sum(ret)


# (f)
def getOLSpoly(ordinarydata, D, axis=1):
    return getOLS(convertpoly(ordinarydata, D, axis), 1)


# (i)
# "Create a version of your solver that, instead of returning the coefficients,
#  returns a function which, when called on a new predictor matrix returns a
#  vector of predictions"
# I'm not sure if this is what it's supposed to do, so if anyone wants to change
# this function to fit part 2 better, be my guest.
def predictorfunc(knowndata):
    classifier = getOLS(knowndata)

    # NOTE: testdata should still have t-values b/c of how the function is written
    # , but the functions here will strip off the t-value and not use it
    def returnfunc(testdata):
        return getpred(classifier, testdata)

    return returnfunc


def main():
    mydata = getdataset("synthdata2016.csv")
    mypolydata = convertpoly(mydata, 4)
    mypolyscale = scaledata(mypolydata)
    myclassifier = getOLS(mypolydata, regparam=0)
    plotoutputpoly(mypolydata, myclassifier, "graph of a quadratic fit to womens 100m times", scale=mypolyscale)


if __name__ == "__main__":
    main()

# myclassifier = getOLS(mydata)
# plotoutput(mydata, myclassifier, "Graph of OLS error for womens top 100 racetimes")
# print("The total error is", getOLSError(mydata, myclassifier))
