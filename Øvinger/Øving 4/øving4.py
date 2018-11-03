import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from functools import wraps


def f1(x, y):
    return x ** 2 + y ** 2

def f2(x,y):
    return x**3-y


def G(x):
    u = x[1] ** (1 / 3)
    v = np.sqrt(1 - x[0] ** 2)
    return np.array([u, v])



class _1:

    lower, upper= -2, 2
    N=50
    #plt.gca().set(xlim=[lower, upper], ylim=[lower, upper])
    cords=np.linspace(lower, upper,N)
    X, Y= np.meshgrid(cords, cords)
    TOL=0.5E-3



    @classmethod
    def a(cls):
        plt.contour(cls.X, cls.Y, f1(cls.X, cls.Y), [1], colors='r')
        plt.contour(cls.X, cls.Y, f2(cls.X, cls.Y), [0], colors='b')
        plt.show()

    @classmethod
    def d(cls):

        N=20

        cords=np.linspace(0.01,0.99,N)

        plt.figure()
        r=[]
        c=np.array([], dtype=int)

        i = j= 0
        for u in cords:
            for v in cords:
                x0=np.array([u,v])
                x=G(x0)

                while np.linalg.norm(x-x0,2)>cls.TOL:
                    x0=x
                    x=G(x)
                    #print(x, x0)

                i+=1


                try:
                    mask=np.isclose(r,x, atol=cls.TOL)[:,0]

                except ValueError:
                    mask=0

                if not np.any(mask):
                    r.append(x)
                    c = np.append(c, j)
                    mask = j
                    j += 1


                #print(c.shape, mask)
                #print(c[mask])

                print(c[mask])
                plt.plot(u,v, marker='o', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(c[mask])])

        for x in r:
            plt.plot(x[0],x[1], 'ro')

        plt.gca().set(xlim=[0, 1], ylim=[0, 1])
        plt.show()

def memoize(func):
    func.cache = {}

    @wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in func.cache:
            func.cache[key] = func(*args, **kwargs)

        return func.cache[key]

    return memoized_func

def B04(x):
    #return np.where(x<1, (3/2-x)*x**2, (x-0.5)*(2-x)**2)
    return np.where(x < 1, (3 / 2) * x ** 2 - x ** 3, -2 + 6 * x - (9 / 2) * x ** 2 + x ** 3)



@memoize
def B(i,k,x):
    if k==1:
       if i==0 or i==n-1:
           return 0
       else:
           return np.logical_and(x >= xi[i], x <= xi[i + 1], dtype=int)
    else:
        Bik=B(i, k - 1, x)
        Bi1k=B(i + 1, k - 1, x)

        if not np.any(Bik):
            return (xi[i + k] - x) * Bi1k / (xi[i + k] - xi[i + 1])
        elif not np.any(Bi1k):
            return (x - xi[i]) * Bik / (xi[i + k - 1] - xi[i])
        else:
            return (x - xi[i]) * Bik / (xi[i + k - 1] - xi[i]) + (xi[i + k] - x) * Bi1k / (xi[i + k] - xi[i + 1])


xi=np.array([0,0,1,2,2])
n=len(xi)-1

class _2:
    N=1000
    x=np.linspace(0,2,N)


    @classmethod
    def b(cls):
        plt.plot(cls.x, B04(cls.x))
        plt.plot(cls.x, B(0,4,cls.x))
        plt.show()






def main():
    #_1.a()
    #_1.d()
    _2.b()

if __name__=="__main__":
    main()