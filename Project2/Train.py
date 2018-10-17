import numpy as np
import ObjFunc as OF
import GradCalc as GC
import ObjFuncBackProp as BP

def Train(M, h, K, Y0, sigma, eta, C, W, eps, TOL, tau, variant=1):
    res = 99999999
    count = 0
    while res > TOL:
        if variant == 0:
            res = OF.ObjFunc(M, h, K, Y0, sigma, eta, C, W)
            dJ, dW = GC.GradCalc(M, h, K, Y0, sigma, eta, C, W, eps)
        if variant == 1:
            res, dJ, dW = BP.ObjFuncAndBackProp(M, h, K, Y0, sigma, eta, C, W)
            #dJ_test, dW_test = GC.GradCalc(M, h, K, Y0, sigma, eta, C, W, eps)

        K -= tau * dJ
        W -= tau * dW

        if count % 500 == 0:
            print(res)
            print(W)
        count += 1

    return K, W


