import numpy as np
import EulerBackProp as ebp
import Euler as E

def etader(x):
    return np.exp(x)/(np.exp(x)+1)**2

def sigmader(x):
    return 1/np.cosh(x)**2


def Wgrad(M, h, K, Y0, sigma, eta, C, W):
    YM = E.Euler(M, h, K, Y0, sigma)
    y = np.dot(YM, W)
    return np.dot(YM.T, etader(y)*(eta(y) - C))


def Kgrad(M, h, K, Y0, sigma, eta, C, W):
    Y, sigma_list, Y_list = ebp.EulerBackProp(M, h, K, Y0, sigma)
    y = np.dot(Y, W)
    WT = np.reshape(W, (1,4))
    dJdY = etader(y)*(eta(y)-C)
    dJdY = np.reshape(dJdY, (np.size(dJdY), 1))
    dJdY = np.dot(dJdY, WT)
    dK = np.zeros((M,4,4))
    U = np.array(dJdY)
    for i in range(M):
        U += np.dot(h*(sigmader(np.dot(Y_list[M-(i+1)],K[M-1-i]))*U), K.T)[...,:,0]
        arg = h*(sigmader(np.dot(Y_list[M-i-1], K[M-1-i])))*U
        dK[M-i-1] = np.dot(Y_list[M-i-1].T, arg)
    return dK
