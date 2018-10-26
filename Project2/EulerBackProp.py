import numpy as np

def EulerBackProp(M, h, K, Y0, sigma):
    Y_list = []
    sigma_list = []
    Y = np.array(Y0)
    Y_list.append(np.copy(Y))
    for i in range(M):
        sigma_value = sigma(np.dot(Y, K[i]))
        Y += h * sigma_value
        sigma_list.append(np.copy(sigma_value))
        Y_list.append(np.copy(Y))
    return Y, sigma_list, Y_list
