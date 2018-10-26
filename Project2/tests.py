import numpy as np
import numpy.linalg as nl
import unittest
import Euler as E
import ObjFunc as OF
import GradCalc as GC
import make_circle_problem as mcp
import Train as T
import Agrad as A
import matplotlib.pyplot as plt

n = 10
M = 1

#self-made function, to get data into useful shape
Y0, C = mcp.YC(n, 50, False)
K = np.full((M,4,4), np.identity(4), dtype=float)
W = np.ones(4)
eps = 0.005
tau = 0.001
h = 1
#tolerance depending on n
TOL = 0.01 * n

def eta(x):
    return np.exp(x)/(np.exp(x) + 1)

def sigma(x):
    return np.tanh(x)

def res_plot(res_list):
    plt.plot(list(range(len(res_list))), res_list)
    plt.xlabel("epoch")
    plt.ylabel("residual")
    plt.title("Residual plot")
    plt.show()

def get_accuracy(YM, W):
    projv = eta(np.dot(YM, W))
    guess = np.around(projv)
    diff = guess - C
    wrong_guesses = np.count_nonzero(diff)
    accuracy = (1 - wrong_guesses / n)
    return accuracy

#arguments
Eargs = (M, h, K, Y0, sigma)
OFargs = (M, h, K, Y0, sigma, eta, C, W)
GCargs = (M, h, K, Y0, sigma, eta, C, W, eps)
Targs = (M, h, K, Y0, sigma, eta, C, W, eps, TOL, tau)

K, W, res_list = T.Train(*Targs)
res_plot(res_list)

#Training data accuracy
YM = E.Euler(M, h, K, Y0, sigma)
accu = get_accuracy(YM, W)
print("Accuracy on training set: " + str(accu))

#Test data accuracy
Y0, C = mcp.YC(n, 50, False)
YM = E.Euler(M, h, K, Y0, sigma)
accu = get_accuracy(YM, W)
print("Accuracy on test set: " + str(accu))
