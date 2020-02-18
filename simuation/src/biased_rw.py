# importing basic tools
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import chisquare

def rw_absorbing(j,a,b,r,seed):
    '''
    Generates a 1D random walk with absorbing 
    barrier in a and b and returns in which point the walker ended up.
    
    j = starting position
    a = lower bound
    b = upper bound
    r = probability to move to the right
    seed = seed for the RNG
    
    Note that we must have a < b, a <= j <= b (integrers) and 0<=r<=1 (float)
    '''
    
    j = int(j)
    a = int(a)
    b = int(b)
    
    # checks
    if a >= b:
        raise ValueError('must be a < b')
    if j < a or j > b:
        raise ValueError('must be a <= j <= b')
    if r < 0 or r > 1:
        raise ValueError('must be 0<=r<=1, because it is a probability')
        
    np.random.seed(seed) # initializing a seed
    position = j # starting position
        
    while( position > a and position < b ):
        if (np.random.random() <= r):
            position += 1
        else:
            position -= 1   
        
    return position 
        
def pj(j,a,b,r,n):
    '''
    Return the fraction of n particles performing a biased RW
    to be absorbed in the absorbing barrier at a (note: also b is an absorbing barrier)
    
    j,a,b and r are described in rw_absorbing()
    n = number of walker simulated
    '''
    
    j = int(j)
    a = int(a)
    b = int(b)
    
    # j=a and j=b cases
    if j == a:
        d = {a : n, b : 0}
    elif j == b:
        d = {a : 0, b : n}
    # all other cases
    else:
        # I inizialize each particle with a sequential seed in order to avoid
        # two particle to have the same seed.
        # Note: this method produce the same seeds at each call
        seeds = np.arange(n) # different seeds for the n particles
        final_points = np.array([ rw_absorbing(j,a,b,r,seeds[i]) 
                            for i in range(n) ]) # where the particles are absorbed
    
        unique, counts = np.unique(final_points, return_counts=True) 
        
        # if all RW ends up into one absorbing barrier but j != a and j != b
        if len(unique) == 1 and int(unique[0]) == a:
            d = {a : n, b : 0}
        elif len(unique) == 1 and int(unique[0]) == b:
            d = {a : 0, b : n}
        else:
            d = dict(zip(unique, counts))
    
    return d[a]/n

def distribution(a,b,r,n,visual=False):
    '''
    Test function for pj and rw_absorbing
    '''
    
    sites = b-a+1
    results = np.zeros(sites)
    
    for i,j in zip(range(sites),range(a,b+1)):
        results[i] = pj(j,a,b,r,n)
    
    if visual == True:
        axis = list(range(a,b+1))
        plt.plot(axis,results, '.', c = 'b')
        
    return results

def pj_as(r,N):
    '''
    Returns an array with theoretical values for the asymmetric case when we have an
    absorbing barrier in 1 and N
    '''
    
    s = (1-r)/r
    temp = s**(N-1)
    pp = np.zeros(N)
    
    for i,j in zip(range(N),range(1,N+1)):
        pp[i] = (s**(j-1) - temp)/(1 - temp)
    
    return pp

def pj_s(N):
    '''
    Returns an array with theoretical values for the symmetric case when we have an
    absorbing barrier in 1 and N
    '''
    
    pp = np.zeros(N)
    for i,j in zip(range(N),range(1,N+1)):
        pp[i] = (N - j)/(N - 1)
    
    return pp

def pj_as_thermo(r,N):
    '''
    Returns the theoretical distribution for the r>1/2 case in the Thermodynamic limit
    '''
    # Check to have r > 0.5
    if r <= 0.5:
        raise ValueError('r must be > 0.5')
    
    s = (1-r)/r
    pp = np.zeros(N)
    
    for i,j in zip(range(N),range(1,N+1)):
        pp[i] = s**(j-1)
        
    return pp

def comparison(r,N,n,save=False):
    '''
    Plots comparison between theory and simulations
    Both graphs for the asymmetric case and the symmetric case are displayed
    '''

    axis = list(range(1,N+1))
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
    
    s = 14

    # asymmetric
    sim = distribution(1,N,r,n,visual=False)
    theo_as = abs(pj_as(r,N)) # abs to avoid -0

    ax[0].plot(axis, theo_as, '.', c = 'orange', label='prediction')
    ax[0].plot(axis, sim, '+', c = 'b', label='simulation')
    ax[0].set_title('Asymmetric case with $n = {}$ walkers and $r={}$'.format(n,r), fontsize=s)
    ax[0].set_xlabel('$j$', fontsize=s)
    ax[0].set_ylabel('$p_j^{as}$', fontsize=s)
    #ax[0].text(7.5,0.6,'$r = {}$'.format(r), fontsize=14)
    ax[0].legend(fontsize=s)

    # symmetric
    sim_s = distribution(1,N,0.5,n,visual=False)
    theo_s = pj_s(N)

    ax[1].plot(axis, abs(theo_s), '.', c = 'orange', label='prediction')
    ax[1].plot(axis, sim_s, '+', c = 'b', label='simulation')
    ax[1].set_title('Symmetric case with $n = {}$ walkers'.format(n), fontsize=s)
    ax[1].set_xlabel('$j$', fontsize=s)
    ax[1].set_ylabel('$p_j^{s}$', fontsize=s)
    ax[1].legend(fontsize=s)
    
    if save == True:
        plt.savefig('saved/comparison_r_{}_n_{}_N_{}'.format(r,n,N), format='pdf')
    
def comparison_thermo(r,N,n,save=False):
    '''
    Plots comparison between theory and simulations for the termodynamic case (r>0.5)
    '''

    axis = list(range(1,N+1))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,6))
    
    s = 14

    # asymmetric
    sim = distribution(1,N,r,n,visual=False)
    theo_as = abs(pj_as(r,N)) # abs to avoid -0
    theo_thermo = abs(pj_as_thermo(r,N))

    ax.plot(axis, theo_as, '.', c = 'darkslategray', label='pred for finite $N$')
    ax.plot(axis, theo_thermo, c = 'coral', label='pred for infinite $N$')
    ax.plot(axis, sim, '+', c = 'b', label='simulation')
    ax.set_title('Thermodynamic case with $n = {}$ walkers and $r={}$'.format(n,r), fontsize=s)
    ax.set_xlabel('$j$', fontsize=s)
    ax.set_ylabel('$p_j^{thermo}$', fontsize=s)
    ax.legend(fontsize=s)

    if save == True:
        plt.savefig('saved/comparison_thermo_r_{}_n_{}_N_{}'.format(r,n,N), format='pdf')