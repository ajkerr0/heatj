"""
@author: Alex Kerr
"""

import numpy as np

from . import Lattice, general_hessian

class SquareLattice(Lattice):
    """
    
    A 2D square harmonic lattice.
    
    Arguments
    ---------
    
    height (int) : Height of the system
    length (int) : Length (width) of the system
    mass (array-like) : height x length size 1d array of the masses
    k (array-like) : 1d array of the interactions strengths 
        with size ( h x (l-1) + (h-1) x l )
        
    Keywords
    --------
    gamma (float) : Reservoir coupling. Defaults to 1.
    
    """
    
    def __init__(self, height, length, m, k, d, gamma=1.):
        pos = self.square_pos(height, length)
        self.height = height
        self.length = length
        neighbors, channels, drivers = self.interactions(height, length)
        super().__init__(m,
                         general_hessian(pos, k, neighbors, d),
                         gamma,
                         drivers,
                         channels)
    
    @staticmethod
    def interactions(h, l):
        """
        Return list of nearest neighbors in a square lattice 
        of dimensions h x l
        """
        neighbors = []
        channels = []
        drivers = []
        
        a = np.arange(h*l).reshape(h,l)
        
        # getting neighbors
        
        # main body
        for site in np.hstack(a[:-1, :-1]):
            neighbors.append([site,site+1])
            neighbors.append([site,site+l])
        
        # last column
        for site in a[:-1,-1]:
            neighbors.append([site,site+l])
        
        # last row    
        for site in a[-1,:-1]:
            neighbors.append([site,site+1])
            
        # now channels
        
        for site in a[:,0]:
            channels.append([site,site+1])
            
        for site in a[:,-2]:
            channels.append([site,site+1])
            
        # drivers
        drivers.append(a[:,0].tolist())
        drivers.append(a[:,-1].tolist())
            
        neighbors = np.array(neighbors)
        neighbors = neighbors[np.lexsort(neighbors[:,::-1].T, axis=0)]
            
        return neighbors, np.array(channels), drivers
    
    @staticmethod
    def square_pos(h,l):
        pos = np.zeros((h*l,2))
        pos[:,0] = np.tile(np.arange(l),h)
        pos[:,1] = np.repeat(np.arange(h),l)
        return pos