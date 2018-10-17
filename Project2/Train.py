import numpy as np
import ObjFunc as OF
import GradCalc as GC

def Train(M, h, K, Y0, sigma, eta, C, W, eps, TOL, tau):
    res = OF.ObjFunc(M, h, K, Y0, sigma, eta, C, W)
    count = 0
    while res > TOL:
        count += 1
        dJ, dW = GC.GradCalc(M, h, K, Y0, sigma, eta, C, W, eps)
        K -= tau * dJ
        W -= tau * dW
        res = OF.ObjFunc(M, h, K, Y0, sigma, eta, C, W)
        if count % 1000 == 0:
            print(res)
            print(W)

    return K, W


