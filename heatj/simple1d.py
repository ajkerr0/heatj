"""

@author: Alex Kerr
"""

import numpy as np
import scipy.linalg

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
    
    def j_alt(self):
        """Return the heat current as defined by Velizhinan et al. which is a
        seemingly different function than Mullen's GF method."""
        
        a = np.zeros((2*self.n*self.dim, 2*self.n*self.dim))
        a[:self.n*self.dim, self.n*self.dim:] = -self._m_matrix
        a[self.n*self.dim:, :self.n*self.dim] = self.k
        a[self.n*self.dim:, self.n*self.dim:] = np.dot(self._g_matrix, 
                                                       self._m_matrix)
        
        w, vl, vr = scipy.linalg.eig(a, left=True)
        
        val_sigma = np.tile(w, (w.shape[0],1))
        val_tau = np.conjugate(np.transpose(val_sigma))
        
        with np.errstate(divide="ignore", invalid="ignore"):
            valterm = np.true_divide(1.,val_sigma+val_tau)
        valterm[~np.isfinite(valterm)] = 0.
        
        drivers = np.arange(self.nr)
        
        return 2*self.uk/self.m*np.abs(np.einsum('l,k,mk,ml,kl->',
                                       np.conj(vr[self.n+self.nr+1,:]),
                                       vr[self.nr,:],
                                       np.conj(vl[self.n+drivers,:]),
                                       vl[self.n+drivers,:],
                                       valterm))*self.gamma
                                        
    def j_alt2(self):
        """Return the heat current as defined Velizhinan et al. without using
        numpy's einsum function."""
        
        a = np.zeros((2*self.n*self.dim, 2*self.n*self.dim))
        a[:self.n*self.dim, self.n*self.dim:] = -self._m_matrix
        a[self.n*self.dim:, :self.n*self.dim] = self.k
        a[self.n*self.dim:, self.n*self.dim:] = np.dot(self._g_matrix, 
                                                       self._m_matrix)
        
        w, vl, vr = scipy.linalg.eig(a, left=True)
        
        val_sigma = np.tile(w, (w.shape[0],1))
        val_tau = np.conjugate(np.transpose(val_sigma))
        
        with np.errstate(divide="ignore", invalid="ignore"):
            valterm = np.true_divide(1.,val_sigma+val_tau)
        valterm[~np.isfinite(valterm)] = 0.
        
        term1 = np.tile(np.conj(vr[self.n+self.nr+1,:]), (self.val.shape[0],1))
        term2 = np.transpose(np.tile(vr[self.nr,:], (self.val.shape[0],1)))
        
        term3 = np.zeros((self.val.shape[0], self.val.shape[0]))
        term4 = np.copy(term3)
        
        for driver in self.drivers[1]:
    
            term3 += np.transpose(np.tile(np.conj(vl[self.n+driver,:]), (self.val.shape[0],1)))
            term4 += np.tile(vl[self.n+driver,:], (self.val.shape[0],1))
            
        termArr = term1*term2*term3*term4*valterm
        return  2.*self.uk/self.m*self.gamma*np.sum(termArr)
        
        
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