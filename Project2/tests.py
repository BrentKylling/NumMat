import numpy as np
import numpy.linalg as nl
import unittest
import Euler as E
import ObjFunc as OF
import GradCalc as GC
import make_circle_problem as mcp
import Train as T
import Agrad as A


# Euler(M, h, K, Y0, sigma)
# ObjFunc(M, h, K, Y0, sigma, eta, C, W)
# GradCalc(M, h, K, Y0, sigma, eta, C, W, eps)
# Train(M, h, K, Y0, sigma, eta, C, W, eps, TOL, tau):

n = 10
M = 10

Y0, C = mcp.YC(n, 50, False)
K = np.full((M,4,4), np.identity(4), dtype=float)
W = np.ones(4)
eps = 0.005
tau = 0.1
h = 0.1
TOL = 0.1


def eta(x):
    return np.exp(x)/(np.exp(x) + 1)

def sigma(x):
    return np.tanh(x)

Eargs = (M, h, K, Y0, sigma)
OFargs = (M, h, K, Y0, sigma, eta, C, W)
GCargs = (M, h, K, Y0, sigma, eta, C, W, eps)
Targs = (M, h, K, Y0, sigma, eta, C, W, eps, TOL, tau)

K, W = T.Train(*Targs)
#A.Wgrad(*Targs)

n = 100
Y0, C = mcp.YC(n, 50, False)
YM = E.Euler(M, h, K, Y0, sigma)
projv = eta(np.dot(YM, W))
a = projv - C
a = np.around(a)
a = np.count_nonzero(a)
print(1-a/200)
