import numpy as np
import ObjFunc as OF
import GradCalc as GC
import ObjFuncBackProp as BP

max_step = 100000

#Parameter Adam
m_t_K = 0
v_t_K = 0
m_t_W = 0
v_t_W = 0

def adam_update(value, g_t, t, m_t_old, v_t_old):
    t += 1
    alpha = 0.0001
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    m_t = beta_1 * m_t_old + (1 - beta_1) * g_t  # updates the moving averages of the gradient
    v_t = beta_2 * v_t_old + (1 - beta_2) * (g_t * g_t)  # updates the moving averages of the squared gradient
    m_cap = m_t / (1 - (beta_1 ** t))  # calculates the bias-corrected estimates
    v_cap = v_t / (1 - (beta_2 ** t))  # calculates the bias-corrected estimates
    value -= np.divide((alpha * m_cap), (np.sqrt(v_cap) + epsilon))
    return value


def Train(M, h, K, Y0, sigma, eta, C, W, eps, TOL, tau, gradient_variant=1, update_variant=1):
    res = 1000
    res_list = []
    count = 0
    while res > TOL and count < max_step:
        #numerical
        if gradient_variant == 0:
            res = OF.ObjFunc(M, h, K, Y0, sigma, eta, C, W)
            dJ, dW = GC.GradCalc(M, h, K, Y0, sigma, eta, C, W, eps)
        #analytically - backpropagation
        if gradient_variant == 1:
            res, dJ, dW = BP.ObjFuncAndBackProp(M, h, K, Y0, sigma, eta, C, W)
            #dJ_test, dW_test = GC.GradCalc(M, h, K, Y0, sigma, eta, C, W, eps)

        #Updates
        # Stochastic gradient descent
        if update_variant == 0 :
            K -= tau * dJ
            W -= tau * dW

        #Adam updates
        if update_variant == 1:
            W = adam_update(W, dW, count, m_t_W, v_t_W)
            K = adam_update(K, dJ, count, m_t_K, v_t_K)


        if count % 1000 == 0:
            print("Residual at step " + str(count) + ": " + str(res))
        res_list.append(res)
        count += 1
    return K, W, res_list


