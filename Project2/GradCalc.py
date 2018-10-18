import numpy as np
import ObjFunc as OF

def GradCalc(M, h, K, Y0, sigma, eta, C, W, eps):
    dJ = np.zeros((M, 4, 4))
    j1 = OF.ObjFunc(M, h, K, Y0, sigma, eta, C, W)
    for m in range(M):
        for i in range(4):
            for j in range(4):
                temp = np.zeros((4,4))
                temp[i,j] = eps
                dK = np.array(K)
                dK[m] += temp
                j2 = OF.ObjFunc(M, h, dK, Y0, sigma, eta, C, W)
                dJ[m,i,j] = (j2-j1)/eps

    dW = np.zeros(4)
    w1 = OF.ObjFunc(M, h, K, Y0, sigma, eta, C, W)
    for i in range(4):
        temp = np.zeros(4)
        temp[i] = eps
        Wp = np.array(W) + temp
        w2 = OF.ObjFunc(M, h, K, Y0, sigma, eta, C, Wp)
        dW[i] = (w2-w1)/eps
    return dJ, dW
