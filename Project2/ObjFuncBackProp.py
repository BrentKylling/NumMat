import numpy as np
import Euler as E
import numpy.linalg as nl
import EulerBackProp as EBP

def ObjFuncAndBackProp(M, h, K, Y0, sigma, eta, C, W):
    #Forward model
    YM, sigma_list, Y_list = EBP.EulerBackProp(M, h, K, Y0, sigma)
    projv = eta(np.dot(YM, W))
    error = (1/2)*nl.norm(projv - C)**2

    #Backpropagation
    #deriv. w.r.t. W
    dW = np.dot(YM.T, ((projv - C) * projv * (1 - projv)))    # * is elementwise (hadamard) operation

    #deriv. w.r.t. YM
    dYM = np.array(np.matrix((projv - C) * projv * (1 - projv)).T * np.matrix(W.T))  # * is elementwise (hadamard) operation

    #deriv. w.r.t. the different K_m
    #This is done recursively. Starting with dYM as first upstream derivative we pass
    #the gradient through the computational graph
    dJ = np.zeros((M, 4, 4))
    dY_upstream = dYM
    for i in range(len(Y_list)-2, -1, -1):
        #dJ[i] = (Y_list[i] * dY_upstream).T.dot(h * (1 - sigma_list[i]**2))
        dJ[i] = Y_list[i].T.dot(h * (1 - sigma_list[i] ** 2) * dY_upstream)
        dY_upstream = np.array(np.matrix(h * (1 - sigma_list[i]**2) * dY_upstream) * np.matrix(K[i].T)) + dY_upstream
    return error, dJ, dW
