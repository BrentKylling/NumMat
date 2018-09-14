import numpy as np, scipy.sparse as ss, matplotlib.pyplot as plt
from matplotlib import rcParams


def simulation(N, M):
    K = np.empty((len(N), len(M)))
    for i, n in enumerate(N):
        A = ss.diags([1, -2, 8, -2, 1], [-2, -1, 0, 1, 2], shape=(n, n)).toarray()
        I = np.eye(n)
        b = np.random.rand(n)
        e = np.ones_like(b)
        for j, m in enumerate(M):
            k = conjugateGradient(A + m * I, b, e)
            #k = conjugateGradient(A + m * I, b, e, I)
            K[i, j] = k
    return K

'''
def conjugateGradient(A, b, x0, P, TOL=1E-15):  # Conditioned, need P as input
    n = len(b)
    r = b - A @ x0
    p = z = np.linalg.solve(P, r)
    for k in range(n):
        rdotz = r.dot(z)
        Ap = A @ p
        alpha = rdotz / (p.dot(Ap))
        x1 = x0 + alpha * p
        if np.linalg.norm(x1 - x0, np.inf) <= TOL:
            return k + 1
        x0 = x1
        r -= alpha * Ap
        z = np.linalg.solve(P, r)
        beta = r.dot(z) / rdotz
        p = z + beta * p
    return k + 1
'''

def conjugateGradient(A, b, x0, TOL=1E-15):   #Non-conditioned
    n = len(b)
    p= r = b - A@x0
    for k in range(n):
        rdotp = r.dot(p)
        Ap = A@p
        pAdotp= p.dot(Ap)
        alpha = rdotp /pAdotp
        x1 = x0 + alpha * p
        if np.linalg.norm(x1 - x0, np.inf) <= TOL:
            return k + 1
        x0 = x1
        r -= alpha * Ap
        beta = r.dot(Ap) / pAdotp
        p = r - beta * p
    return k + 1


def plotIterations(K,N,M):
    #%matplotlib qt
    rcParams.update({'axes.grid': True, 'legend.fontsize': 18, 'legend.handlelength': 2,
                     'axes.labelsize': 18, 'axes.titlesize': 18, 'figure.figsize': (16, 8)})

    colors=rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes=plt.subplots(2,3)
    plt.tight_layout()
    for k in range(6):
        i, j = k % 2, k % 3
        axes[i,j].plot(M, K[k], label=r'$n=%d$'%N[k], color=colors[k])
        axes[i,j].legend(loc='best')

    #plt.savefig('CG_Analysis', format='png')
    plt.show()

if __name__=="__main__":
    N=np.arange(10,70,10)
    M=np.linspace(0,1000,1001)
    #print(N,M,sep='\n')
    plotIterations(simulation(N,M),N,M)
