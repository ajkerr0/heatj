"""

@author: Alex Kerr
"""

import numpy as np

from . import Lattice

class Simple1D(Lattice):
    """A 1D harmonic lattice with extend reservoirs at the ends."""
    
    def __init__(self, n, nr=1, k=1., d=1., m=1., gamma=1.):
        super().__init__(m*np.ones(n),
                         get_hessian(get_1dneighbors(n), n, k, d),
                         gamma,
                         [np.arange(0, nr), np.arange(n-nr, n)],
                         [[nr,nr+1]])
        self.ud = d
        self.uk = k
        self.m = m
        self.nr = nr
    
    @property    
    def bandwidth(self):
        return np.sqrt(self.uk/self.m)*(np.sqrt((self.ud/self.uk) + 4.) - np.sqrt(self.ud/self.uk))
    
    @property
    def sigma2(self):
        """Return the Casher-Lebowitz theoretical intrinsic conductance."""
        return self.bandwidth/2./np.pi
    
    @property
    def gamma12(self):
        """Return the crossover points between regimes 1 and 2."""
        return 2.*self.m*self.sigma2/self.nr
    
    @property
    def gamma23(self):
        return self.uk*(1. - (np.sqrt(self.m*self.ud)/2./self.uk)*self.bandwidth)/2./self.sigma2
        
def get_1dneighbors(n):
    return [[x, x+1] for x in np.arange(n-1)]

def get_hessian(neighbors, n, k, d, ends=True):
    
    hessian = np.diag(d*np.ones(n))
    for i,j in neighbors:
        hessian[i,i] += k
        hessian[j,j] += k
        hessian[i,j] = -k
        hessian[j,i] = -k
        
    if ends:
        hessian[0,0] += k
        hessian[-1,-1] += k
        
    return hessian