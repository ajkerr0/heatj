"""

@author: Alex Kerr
"""

import copy

import numpy as np
import matplotlib.pyplot as plt

def single_modes(modes):
    """Plot a series of 1D modes"""
    
    xs = 1.5*np.arange(modes.shape[1])
    ys = np.arange(modes.shape[0])
    
    for y in ys:
        
        for x,push in zip(xs, modes[y]):
            
            plt.plot([x,x+push], [y,y], lw=5, c='C0')
            plt.scatter(x+push, y, c='C0', s=50)
            
    plt.show()
    
def mode_evolution(lattice, mindex, sindices, g):
    """Plot the evolution of an eigenmode (real and imaginary parts separately)
    as a function of the dissipation from the heat baths.
    
    Arguments
    ---------
    lattice  : Lattice
        Lattice object with modes of interest
    mindex   : int
        Index of the mode in the unperturbed (zero-damping) problem
    sindices : array-like
        Array of indices of the coordinates of interest
    g        : array-like
        Evolution of the gammas
        
    There could be problems if the first gamma is not zero.

    """
    
    s = np.array(sindices)
    g = np.array(g)
    
    
    # get the undamped eigenmode 
    uval, uvec = np.linalg.eig(lattice.k)
    umode = uvec[:,mindex]
    
    # evolve the eigenmode
    pos = np.zeros((g.shape[0], s.shape[0]))
    
    for i,gamma in enumerate(g):
        
        # determine the eigenmode(s)
        lattice.gamma = gamma
        val, vec = lattice._calculate_evec()
        
        # find the eigenmode that corresponds to the previous one
        a = vec[:lattice.n].imag.T
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.true_divide(a, np.linalg.norm(a, axis=1)[:,None])
        a[~np.isfinite(a)] = 0.
        
        print(umode)
        print(a)
        print(np.linalg.norm(a-umode, axis=1))
        mindex = np.argmin(np.linalg.norm(a-umode, 
                                          axis=1))
        pos[i] = a[mindex][s]
        
        umode = a[mindex]
        
    # plot
    x = np.arange(s.shape[0])
    
    for site in x:
        plt.plot(pos[:,site]+site, g, '-x')
        plt.axvline(x=site, ymin=np.min(g), ymax=np.max(g), linestyle='dashed')
        
    # try plotting hlines at the crossover points
    # works when lattice is a Simple1D object, does nothing otherwise
    try:
        plt.axhline(y=2.*lattice.m*lattice.sigma2/lattice.nr, 
                    xmin=np.min(x),
                    xmax=np.max(x),
                    ls='dotted')
        plt.axhline(y=lattice.uk*(1. - (np.sqrt(lattice.m*lattice.ud)/2./lattice.uk)*lattice.bandwidth)/2./lattice.sigma2,
                    xmin=np.min(x),
                    xmax=np.max(x),
                    ls='dotted')
    except AttributeError:
        pass
        
    plt.show()
    
def j_evolution(lattice, g):
    """Plot conductance as a function of the supplied damping"""
    
    g = np.array(g)
    sigma = np.zeros(g.shape[0])
    
    lat = copy.deepcopy(lattice)
    
    for i,gamma in enumerate(g):
        
        lat.gamma = gamma
        lat.set_greensfunc()
        sigma[i] = lat.j()
        
    plt.loglog(g, sigma, lw=4)
    
    # try putting lines at theoretical values
    try:
        plt.axhline(y=lat.sigma2, ls='dotted')
        plt.axvline(x=lat.gamma12, ls='dashed')
        plt.axvline(x=lat.gamma23, ls='dashed')
    except AttributeError:
        pass
    
    plt.show()
        
    