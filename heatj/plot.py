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
    
    # get the undamped eigenmodes
    uval, uvec = np.linalg.eig(lattice.k)
#    print(uval)
#    print(np.sqrt(uval))
#    print(uvec)
    uval = 0+np.sqrt(abs(uval[mindex]))*1j
#    print(uval)
    
    # evolve the eigenmode
    pos_r = np.zeros((g.shape[0], s.shape[0]))
    pos_i = np.zeros((g.shape[0], s.shape[0]))
    
    for i,gamma in enumerate(g):
        
        # determine the damped eigenmodes
        lattice.gamma = gamma
        val, vec = lattice._calculate_evec()
        mindex = np.argmin(np.abs(val-uval))
#        print(val[mindex])
#        print(val)
        vec = vec[:lattice.n,mindex]/np.linalg.norm(vec[:lattice.n,mindex])
#        print(vec.real)
#        print(vec.imag)
        
        if vec[0].imag < 0.:
            vec *= -1.
        
        pos_r[i] = vec.real[s]
        pos_i[i] = vec.imag[s]
        
        uval = val[mindex]
        
    # plot
    x = np.arange(s.shape[0])
    
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax2 = fig2.add_subplot(1,1,1)
    
    for site in x:
        ax1.plot(pos_r[:,site]+site, g, '-x')
        ax1.axvline(x=site, ymin=np.min(g), ymax=np.max(g), linestyle='dashed')
        ax2.plot(pos_i[:,site]+2*site, g, '-x')
        ax2.axvline(x=2*site, ymin=np.min(g), ymax=np.max(g), linestyle='dashed')
        ax2.arrow(2*site, np.min(g), pos_i[0,site], 0,
                  lw=2,
                  zorder=2,
                  facecolor='k',
                  length_includes_head=True,
                  head_width=.1,
                  capstyle='butt')
        
    # try plotting hlines at the crossover points
    # works when lattice is a Simple1D object, does nothing otherwise
    try:
        ax1.axhline(y=lattice.gamma12, 
                    xmin=np.min(x),
                    xmax=np.max(x),
                    ls='dotted',)
        ax1.axhline(y=lattice.gamma23,
                    xmin=np.min(x),
                    xmax=np.max(x),
                    ls='dotted',)
        
        ax2.axhline(y=lattice.gamma12, 
                    xmin=np.min(x),
                    xmax=np.max(x),
                    ls='dotted',)
        ax2.axhline(y=lattice.gamma23,
                    xmin=np.min(x),
                    xmax=np.max(x),
                    ls='dotted',)
    except AttributeError:
        pass
        
    ax1.set_xlabel('Re(Mode)')
    ax2.set_xlabel('Im(Mode)')
    ax1.set_ylabel(r'$\gamma \; \; \sqrt{mK}$')
    ax2.set_ylabel(r'$\gamma \; \; \sqrt{mK}$')
    plt.show()
    
def mode_evolution_all(lattice, sindices, g):
    """Plot the evolution of all the eigenmodes (real and imaginary parts separately)
    as a function of the dissipation from the heat baths.
    
    Arguments
    ---------
    lattice  : Lattice
        Lattice object with modes of interest
    mindices : array-like
        Array of indices of the modes of interest
    sindices : array-like
        Array of indices of the coordinates of interest
    g        : array-like
        Evolution of the gammas

    """
    
    m = np.arange(2*lattice.n)
    s = np.array(sindices)
    g = np.array(g)
    
    # evolve the eigenmode
    pos_r = np.zeros((g.shape[0], s.shape[0], m.shape[0]))
    pos_i = np.zeros((g.shape[0], s.shape[0], m.shape[0]))
    
    for i,gamma in enumerate(g):
        
        # determine the damped eigenmodes
        lattice.gamma = gamma
        val, vec = lattice._calculate_evec()

        vec = vec[:lattice.n]/np.linalg.norm(vec[:lattice.n], axis=0)
        
        neg = np.where(vec[:,0].imag < 0.)[0]
        print(vec)
        print(neg)
        vec[:,neg] *= -1.
        print(vec)
        
        pos_r[i] = vec.real[s]
        pos_i[i] = vec.imag[s]
        
    # plot
    x = np.arange(s.shape[0])
    
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax2 = fig2.add_subplot(1,1,1)
        
    for site in x:
        
        ax1.axvline(x=site, ymin=np.min(g), ymax=np.max(g), linestyle='dashed')
        ax2.axvline(x=2*site, ymin=np.min(g), ymax=np.max(g), linestyle='dashed')
        
        for mode in np.arange(m.shape[0]):
            
            ax1.scatter(pos_r[:,site,mode]+site, g, c='k', s=.5)
            ax2.scatter(pos_i[:,site,mode]+2*site, g, c='k', s=.5)
            
        
    # try plotting hlines at the crossover points
    # works when lattice is a Simple1D object, does nothing otherwise
    try:
        ax1.axhline(y=lattice.gamma12, 
                    xmin=np.min(x),
                    xmax=np.max(x),
                    ls='dotted',)
        ax1.axhline(y=lattice.gamma23,
                    xmin=np.min(x),
                    xmax=np.max(x),
                    ls='dotted',)
        
        ax2.axhline(y=lattice.gamma12, 
                    xmin=np.min(x),
                    xmax=np.max(x),
                    ls='dotted',)
        ax2.axhline(y=lattice.gamma23,
                    xmin=np.min(x),
                    xmax=np.max(x),
                    ls='dotted',)
    except AttributeError:
        pass
        
    ax1.set_xlabel('Re(Mode)')
    ax2.set_xlabel('Im(Mode)')
    ax1.set_ylabel(r'$\gamma \; \; \sqrt{mK}$')
    ax2.set_ylabel(r'$\gamma \; \; \sqrt{mK}$')
    plt.show()
    
def _evolve_j(lattice, g):
    """Evaluate the conductance as a function of the supplied damping."""
    
    g = np.array(g)
    sigma = np.zeros(g.shape[0])
    
    lat = copy.deepcopy(lattice)
    
    for i,gamma in enumerate(g):
        
        lat.gamma = gamma
        lat.set_greensfunc()
        sigma[i] = lat.j()
        
    return g, sigma

def _evolve_j_alt(lattice, g):
    """Evaluate the *alternative* conductance as a function of the supplied damping."""
    
    g = np.array(g)
    sigma = np.zeros(g.shape[0])
    
    lat = copy.deepcopy(lattice)
    
    for i,gamma in enumerate(g):
        
        lat.gamma = gamma
        sigma[i] = lat.j_alt()
        
    return g, sigma
    
    
def j_evolution(lattice, g):
    """Plot conductance as a function of the supplied damping"""
    
    g, sigma = _evolve_j(lattice, g)
        
    plt.loglog(g, sigma, lw=4)
    
    # try putting lines at theoretical values
    try:
        plt.axhline(y=lattice.sigma2, ls='dotted')
        plt.axvline(x=lattice.gamma12, ls='dashed')
        plt.axvline(x=lattice.gamma23, ls='dashed')
    except AttributeError:
        pass
    
    plt.show()
        
    