import numpy as np
import Euler as E
import numpy.linalg as nl

def ObjFunc(M, h, K, Y0, sigma, eta, C, W):
    YM = E.Euler(M, h, K, Y0, sigma)
    projv = eta(np.dot(YM, W))
    return (1/2)*nl.norm(projv - C)**2
