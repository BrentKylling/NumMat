import numpy as np

def Euler(M, h, K, Y0, sigma):
    Y = np.array(Y0)
    for i in range(M):
        Y += h * sigma(np.dot(Y, K[i]))
    return Y
