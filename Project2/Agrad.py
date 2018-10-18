import numpy as np
import Euler as E

def etader(x):
    return np.exp(x)/(np.exp(x)+1)**2


def Wgrad(M, h, K, Y0, sigma, eta, C, W, eps, TOL, tau):
    YM = E.Euler(M, h, K, Y0, sigma)
    wgrad = np.zeros(4)
    y = np.dot(YM, W)
    first = np.dot(YM.T,(etader(y)*(eta(y)-C)))/eps
    print(first)

