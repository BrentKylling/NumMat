import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from functools import wraps
from timeit import default_timer as timer

y0 = np.array([1 / 3, 1])

def memoize(func):
    func.cache= {}

    @wraps(func)
    def memoized_func(*args, **kwargs):
        key = args[-1]

        if key in func.cache:
            ret=func.cache[key]
        else:
            ret=func.cache[key] = func(*args, **kwargs)

        if len(func.cache) == len(args[1]): #siste K brukes ikke i noen andre K og vi kan tømme cachen for aktuell y
            func.cache = {}
        return ret

    return memoized_func

def f(y):
    du = y[0] - 2 * y[0] * y[1]
    dv = -y[1] + 3 * y[0] * y[1]
    return np.array([du, dv])


dt = lambda k: tf/2 ** (k)

@memoize
def K(y, A, h, i):
    if len(A)==1:
        return f(y)
    else:
        return f(y + h * sum(0 if not A[i, j] else A[i, j] * K(y, A, h, j) for j in range(i + 1)))  # Assumes explicit A


def F(y, A, b, h):
    ret=sum(b[i] * K(y, A, h, i) for i in range(len(A)))
    return ret


# b[0]*K(0) + b[1]*K[1]+b[2]*K[2]+b[3]*K[3]


def RKStep(y, A, b, h):
    y += h * F(y, A, b, h)
    return y


def simulation():

    b = np.array([np.array(arr) for arr in ([1],
                                           [1/2, 1/2],
                                           [1 / 6, 2 / 3, 1 / 6],
                                           [1 / 6, 1 / 3, 1 / 3, 1 / 6])])

    A = np.array([np.array(arr) for arr in ([0],
                                            [[0, 0], [1, 0]],
                                            [[0, 0, 0], [1 / 2, 0, 0], [-1, 2, 0]],
                                            [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]])])

    def curry(y, A=A[3], b=b[3], h=0.1 * dt(Ne)):
        return RKStep(y, A, b, h)

    yex = y0.copy()  # 1x2
    for i in range(10*2**Ne):
        yex = curry(yex)

    ERR = np.empty((4, Ne))  # 4#Ne

    for k in range(Ne):
        y = np.stack(y0 for _ in range(4))
        h = dt(k + 1)
        #print(h*2**(k+1))
        for j in range(2**(k+1)):
            for i in range(4):
                y[i] = RKStep(y[i], A[i], b[i], h)

        ERR[:, k] = np.linalg.norm(y - yex,2, axis=1)

    return ERR


def visualiation(err):
    plt.loglog(dt(np.arange(1, Ne + 1)), err.T)
    plt.legend(plt.gca().lines, ('ERK%d' % i for i in range(1, 5)))
    #plt.show()


def main():
    start=timer()
    visualiation(simulation())
    end=timer()
    print(end-start)


if __name__ == "__main__":
    y0 = np.array([1 / 3, 1])
    tf = 3
    Ne = 12
    main()
