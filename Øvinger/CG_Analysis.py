import numpy as np, scipy.sparse as ss, matplotlib.pyplot as plt

def simulation(N,M):
    K=np.empty((len(N), len(M)))
    for i, n in enumerate(N):
        A=ss.diags([1,-2,8,-2,1], [-2,-1,0,1,2], shape=(n,n)).todense()
        #print(A)
        I=np.eye(n)
        b=np.random.rand(n).reshape((n,1))
        e=np.ones_like(b)
        for j,m in enumerate(M):
            k=conjugateGradient(A+m*I,b, e)
            K[i,j]=k
    return K

"""
def conjugateGradient(A,b,x0,P,TOL=1E-15):
    n=len(b)
    #print(b, A.dot(x0))
    r=b-A@x0
    #print(P.shape, r.shape)
    p=z=np.linalg.solve(P,r)
    #print(z.shape)
    for k in range(n):
        rdotz=r.T@z
        Ap=A@p
        alpha=np.asscalar(rdotz/p.T@(Ap))
        x1=x0+alpha*p
        if np.linalg.norm(x1-x0,np.inf)<=TOL:
            return k+1
        x0=x1
        r-=alpha*Ap
        z=np.linalg.solve(P,r)
        beta=np.asscalar(r.T@z/rdotz)
        p=z+beta*p
    return k+1
"""


def conjugateGradient(A,b,x0,TOL=1E-15):
    n=len(b)
    #print(b, A.dot(x0))
    r=b-A@x0
    #print(P.shape, r.shape)
    p=r
    #print(z.shape)
    for k in range(n):
        rdotp=r.T@p
        Ap=A@p
        alpha=np.asscalar(rdotp/p.T@(Ap))
        x1=x0+alpha*p
        temp=np.linalg.norm(x1-x0,np.inf)
        #print(temp)
        if temp<=TOL:
            return k+1
        x0=x1
        r-=alpha*Ap
        beta=np.asscalar(r.T@Ap/rdotp)
        p=r-beta*p
    return k+1



def plotIterations(K,N,M):
    %matplotlib qt
    fig, axes=plt.subplots(2,3, sharey=True)
    plt.rcParams['axes.grid'] = True
    for i in range(2):
        for j in range(3):
            axes[i,j].plot(M,K[i+j])

    #plt.savefig('CG_Analysis', format='png')
    plt.show()

if __name__=="__main__":
    N=np.arange(10,70,10)
    M=np.linspace(0,1000,1001)
    #print(N,M,sep='\n')
    plotIterations(simulation(N,M),N,M)
