import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt


def make_circle_problem(n,nx,PLOT):
    # This python-script uses the following three input parameters:
    #   n       - Number of points.
    #   nx      - Resolution of the plotting.
    #   PLOT    - Boolean variable for plotting.

    # Defining function handles.
    transform_domain = lambda r : 2*r-1
    rad = lambda x1,x2 : np.sqrt(x1**2+x2**2)

    # Initializing essential parameters.
    r = np.linspace(0,1,nx)
    x = transform_domain(r)
    dx = 2/nx
    x1,x2 = np.meshgrid(x,x)

    # Creating the data structure 'problem' in terms of dictionaries.
    problem = {'domain':{'x1':x,'x2':x},'classes':[None,None]}
    group1 = {'mean_rad':0,'sigma':0.1,'prob_unscaled':lambda x1,x2: 0,'prob':lambda x1,x2: 0,'density':0}
    group1['prob_unscaled'] = lambda x,y : np.exp(-(rad(x,y)-group1['mean_rad'])**2/(2*group1['sigma']**2))
    density_group1 = group1['prob_unscaled'](x1,x2)
    int_density_group1 = (dx**2)*sum(sum(density_group1))
    group1['density'] = density_group1/int_density_group1
    group2 = {'mean_rad':0.5,'sigma':0.1,'prob_unscaled':lambda x1,x2: 0,'prob':lambda x1,x2: 0,'density':0}
    group2['prob_unscaled'] = lambda x,y : np.exp(-(rad(x,y)-group2['mean_rad'])**2/(2*group2['sigma']**2))
    density_group2 = group2['prob_unscaled'](x1,x2)
    int_density_group2 = (dx**2)*sum(sum(density_group2))
    group2['density'] = density_group2/int_density_group2
    problem['classes'][0] = group1
    problem['classes'][1] = group2

    # Creating the arrays x1 and x2.
    x1 = np.zeros((n,2))
    x2 = np.zeros((n,2))
    count = 0
    for i in range(0,n):
        count += 1
        N1 = 'x1_'+str(count)+'.png'
        N2 = 'x2_'+str(count)+'.png'
        x1[i,0],x1[i,1] = pinky(problem['domain']['x1'],problem['domain']['x2'],problem['classes'][0]['density'],PLOT,N1)
        x2[i,0],x2[i,1] = pinky(problem['domain']['x1'],problem['domain']['x2'],problem['classes'][1]['density'],PLOT,N2)

    # Creating the data structure 'data' in terms of dictionaries.
    x = np.concatenate((x1[0:n,:],x2[0:n,:]),axis=0)
    y = np.concatenate((np.ones((n,1)),2*np.ones((n,1))),axis=0)
    i = rnd.permutation(2*n)
    data = {'x':x[i,:],'y':y[i]}

    return data, problem


def pinky(Xin,Yin,dist_in,PLOT,NAME):
    # Checking the input.
    if len(np.shape(dist_in)) > 2:
        print("The input must be a N x M matrix.")
        return
    sy,sx = np.shape(dist_in)
    if (len(Xin) != sx) or (len(Yin) != sy):
        print("Dimensions of input vectors and input matrix must match.")
        return
    for i in range(0,sy):
        for j in range(0,sx):
            if dist_in[i,j] < 0:
                print("All input probability values must be positive.")
                return

    # Create column distribution. Pick random number.
    col_dist = np.sum(dist_in,1)
    col_dist /= sum(col_dist)
    Xin2 = Xin
    Yin2 = Yin

    # Generate random value index and saving first value.
    ind1 = gendist(col_dist,1,1,PLOT,NAME)
    ind1 = np.array(ind1,dtype="int")
    x0 = Xin2[ind1]

    # Find corresponding indices and weights in the other dimension.
    A = (x0-Xin)**2
    val_temp = np.sort(A)
    ind_temp = np.array([i[0] for i in sorted(enumerate(A), key=lambda x:x[1])])
    eps = 2**-52
    if val_temp[0] < eps:
        row_dist = dist_in[:,ind_temp[0]]
    else:
        low_val = min(ind_temp[0:2])
        high_val = max(ind_temp[0:2])
        Xlow = Xin[low_val]
        Xhigh = Xin[high_val]
        w1 = 1-(x0-Xlow)/(Xhigh-Xlow)
        w2 = 1-(Xhigh-x0)/(Xhigh-Xlow)
        row_dist = w1*dist_in[:,low_val]+w2*dist_in[:,high_val]
    row_dist = row_dist/sum(row_dist)
    ind2 = gendist(row_dist,1,1,PLOT,NAME)
    y0 = Yin2[ind2]

    return x0,y0


def gendist(P,N,M,PLOT,NAME):
    # Checking input.
    if min(P) < 0:
        print('All elements of first argument, P, must be positive.')
        return
    if (N < 1) or (M < 1):
        print('Output matrix dimensions must be greater than or equal to one.')
        return

    # Normalizing P and creating cumlative distribution.
    Pnorm = np.concatenate([[0],P],axis=0)/sum(P)
    Pcum = np.cumsum(Pnorm)

    # Creating random matrix.
    R = rnd.rand()

    # Calculate output matrix T.
    V = np.linspace(0,len(P)-1,len(P))
    hist,inds = np.histogram(R,Pcum)
    hist = np.argmax(hist)
    T = int(V[hist])

    # Plotting graphs.
    if PLOT == True:
        Pfreq = (N*M*P)/sum(P)
        LP = len(P)
        fig,ax = plt.subplots()
        ax.hist(T,np.linspace(1,LP,LP))
        ax.plot(Pfreq,color='red')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('P-vector Index')
        fig.savefig(NAME)

    return T

def YC(n, nx, bool):
    data, problem = make_circle_problem(int(n/2), nx, bool)
    Y0 = np.zeros((n, 4))
    Y0[..., 0:2] = np.array([data['x']])
    C = np.array([data['y']])
    C = np.reshape(C, n)-1
    return Y0, C

if __name__=='__main__':
    data, problem = make_circle_problem(10,50,False)
