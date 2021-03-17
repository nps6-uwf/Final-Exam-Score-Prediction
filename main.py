"""
Title: Multivariate Regression - Predicting student performance on final exam from formative & midterm.
Author: Nick Sebasco
Date: 3/17/2021
Version: Python 3.8
"""
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import scipy
from random import shuffle

frac = 0.9 # fraction used for training
data = []

with open("data.csv","r") as dat:
    """
    format
    student #, id, name, stem, midterm, midterm re-test, formative 1 (f1), 
    f2, f3, f4, f5, formative (total), behavior, final, total, grade
    """
    for i in ([j for j in l.split(",") if j.strip() != ""] for l in dat.readlines()):
        number, id, name, stem, midterm, _, f1, f2, f3, f4, _, _, behavior, final, total, grade = i
        data.append((float(midterm), (float(f1) + float(f2) + float(f3) + float(f4))/4, float(final)))

N = len(data)
train_N = int(N * frac)
test_N = N - train_N
shuffle(data)
train = data[:train_N]
test = data[train_N:]

if True:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = np.array([i[0] for i in data], dtype=float)
    y = np.array([i[1] for i in data], dtype=float)
    z = np.array([i[2] for i in data], dtype=float)
    train_X = np.array([[x[i], y[i]] for i in range(len(x))])
    test_X = np.array([np.array([[i[0],i[1]]]) for i in test], dtype=float)

    # create 3-d scatter plot
    ax.scatter(x,y,z, marker="x")
    reg = LinearRegression().fit(train_X, z)
    print(reg.score(train_X, z), reg.coef_, reg.intercept_)
    z_hat = np.array([j[0] for j in [reg.predict(np.array([i])) for i in train_X]])

    # compare predictive surface vs test data
    pred = [reg.predict(tx)[0] for tx in test_X]
    tx = [i[0] for i in test]
    ty = [i[1] for i in test]
    for i in range(len(pred)):
        print(tx[i],ty[i],"|",z[i],"|",pred[i])

    # plot predictive surface
    ax.plot_trisurf(x,y,z_hat, alpha = 0.2)

    # plot test data relative to predictive surface
    ax.scatter([i[0] for i in test],[i[1] for i in test],[i for i in pred], c="r")

    #labels
    ax.set_xlabel('midterm')
    ax.set_ylabel('average formative')
    ax.set_zlabel('final')
    plt.show()
