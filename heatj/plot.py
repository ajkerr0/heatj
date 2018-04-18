"""

@author: Alex Kerr
"""

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
        print(a)
        print(np.linalg.norm(a, axis=1))
        a = a/np.linalg.norm(a, axis=1)[:,None]
#        print(a/np.linalg.norm(a, axis=1)[:,None])
#        print(umode)
#        print(a/np.linalg.norm(a, axis=1)[:,None]-umode)
#        print(np.linalg.norm(a/np.linalg.norm(a, axis=1)[:,None]-umode, axis=1))
        mindex = np.argmin(np.linalg.norm(a-umode, 
                                          axis=1))
        pos[i] = a[mindex][s]
        
    # plot
    x = np.arange(s.shape[0])
    
    for site in x:
        plt.plot(pos[:,site]+site, g, '-x')
        plt.axvline(x=site, ymin=np.min(g), ymax=np.max(g), linestyle='dashed')
        
    plt.show()
        
        
    
def norm(vec):
    return vec/np.linalg.norm(vec)
    
    
    