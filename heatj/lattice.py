"""

@author: Alex Kerr
"""

import numpy as np

from .timeit import timeit

class Lattice(object):
    
    def __init__(self, m, k, g, drivers, crossings):
        self.mass = np.asarray(m)
        self.k = np.asarray(k)
        self.gamma = g
        self.drivers = np.asarray(drivers)
        self.crossings = np.asarray(crossings)
        self.dim = self.k.shape[0]//self.mass.shape[0]
        self.n = self.mass.shape[0]
        self.val, self.vec = self._calculate_evec()
        self.coeffs = self._calculate_coeffs()
        
    @property
    def _m_matrix(self):
        """Return the inverse mass matrix."""
        return np.diag(1./np.repeat(self.mass, self.dim))
    
    @property
    def _g_matrix(self):
        """Return the damping (gamma) matrix"""
        drivers = np.zeros(self.mass.shape[0])
        drivers[np.ravel(self.drivers)] = self.gamma
        return np.diag(np.repeat(drivers, self.dim))
    
    def _calculate_evec(self):
        """Return the eigenvalues and eigenvectors of the interacting 
        damped harmonic oscillators:
              | 0   -M^-1 |
          G = |           | 
              | K   YM^-1 |"""
        
        a = np.zeros((2*self.n*self.dim, 2*self.n*self.dim))
        a[:self.n*self.dim, self.n*self.dim:] = -self._m_matrix
        a[self.n*self.dim:, :self.n*self.dim] = self.k
        a[self.n*self.dim:, self.n*self.dim:] = np.dot(self._g_matrix, 
                                                       self._m_matrix)
        
        return np.linalg.eig(a)
    
    def _calculate_coeffs(self):
        """Return the M x N Green's function coefficient matrix where
        N is the number of coordinates (ndim x nmass) and M is the number
        of eigenmodes considered."""
        
        N = self.vec.shape[0]//2
        M = self.vec.shape[1]
        
        # determine coefficients in eigenfunction/vector expansion
        # use linear solver to solve equations from notes
        # AX = B where X is the matrix of expansion coefficients
        
        A = np.zeros((2*N, M), dtype=np.complex128)
        A[:N,:] = self.vec[:N,:]
        
        # adding mass and damping terms to A
        lambda_ = np.tile(self.val, (N,1))
        
        m = np.repeat(self.mass, self.dim)
        g = np.diag(self._g_matrix)
        
        A[N:,:] = np.multiply(A[:N,:], 
                              np.tile(m, (M,1)).T*lambda_ 
                            + np.tile(g, (M,1)).T)
        
        # now prep B
        B = np.concatenate((np.zeros((N,N)), np.eye(N)), axis=0)
    
        return np.linalg.solve(A,B)
    
    def calculate_power_vector(self, i,j):
    
        # assuming same drag constant as other driven atom
        driver1 = self.drivers[1]
        
        n = self.val.shape[0]
        
        kappa = 0.
        
        val_sigma = np.tile(self.val, (n,1))
        val_tau = np.transpose(val_sigma)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            valterm = np.true_divide(val_sigma-val_tau,val_sigma+val_tau)
        valterm[~np.isfinite(valterm)] = 0.
        
        for idim in range(self.dim):
            for jdim in range(self.dim):
                
                term3 = np.tile(self.vec[self.dim*i + idim,:], (n,1))
                term4 = np.transpose(np.tile(self.vec[self.dim*j + jdim,:], (n,1)))
                
                for driver in driver1:
                    
                    dterm = np.zeros((self.coeffs.shape[0],), dtype=np.complex128)
                    for k in range(self.dim):
                        dterm += self.coeffs[:, self.dim*driver + k]
        
                    term1 = np.tile(dterm, (n,1))
                    term2 = np.transpose(term1)
                    termArr = term1*term2*term3*term4*valterm
                    kappa += self.k[self.dim*i + idim, self.dim*j + jdim]* \
                             np.sum(termArr)
                    
        return kappa
    
    def calculate_power_einsum(self, i,j):
        
        # sum over:
        # dimensions of i
        # dimensions of j
        # number of drivers
        # sigma
        # tau
        
        # pick a driven side, we will assume the same uniform damping on both
        driver1 = self.drivers[1]
        
        # include dimensions of the drivers
        driver1 = np.repeat(driver1, self.dim) + np.tile(np.arange(self.dim), driver1.shape[0])

        kappa = 0.
        
        val_sigma = np.tile(self.val, (self.val.shape[0],1))
        val_tau = np.transpose(val_sigma)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            valterm = np.true_divide(val_sigma-val_tau,val_sigma+val_tau)
        valterm[~np.isfinite(valterm)] = 0.
        
        for idim in np.arange(self.dim):
            for jdim in np.arange(self.dim):
                kappa += self.k[self.dim*i + idim, self.dim*j + jdim]* \
                         np.einsum('ik,i,jk,j,ij->', self.coeffs[:,driver1],
                                                     self.vec[self.dim*i + idim,:],
                                                     self.coeffs[:,driver1],
                                                     self.vec[self.dim*j + jdim,:],
                                                     valterm)
        
        return kappa
    
    def calculate_power_einsum2(self, i,j):
        
        # sum over:
        # dimensions of i
        # dimensions of j
        # number of drivers
        # sigma
        # tau
        
        # pick a driven side, we will assume the same uniform damping on both
        driver1 = self.drivers[1]
        
        # include dimensions of the drivers
        driver1 = np.repeat(driver1, self.dim) + np.tile(np.arange(self.dim), driver1.shape[0])
        
        val_sigma = np.tile(self.val, (self.val.shape[0],1))
        val_tau = np.transpose(val_sigma)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            valterm = np.true_divide(val_sigma-val_tau,val_sigma+val_tau)
        valterm[~np.isfinite(valterm)] = 0.
        
        kappa = np.einsum('gh,ik,gi,jk,hj,ij->', self.k[self.dim*i:self.dim*(i+1),self.dim*j:self.dim*(j+1)],
                                                 self.coeffs[:,driver1],
                                                 self.vec[self.dim*i:self.dim*(i+1),:],
                                                 self.coeffs[:,driver1],
                                                 self.vec[self.dim*j:self.dim*(j+1),:],
                                                 valterm)
        
        return kappa
    
    def calculate_power_uncollapsed(self, i,j):
        
        # sum over:
        # dimensions of i
        # dimensions of j
        # number of drivers
        # sigma
        # tau
        
        # pick a driven side, we will assume the same uniform damping on both
        driver1 = self.drivers[1]
        
        # include dimensions of the drivers
        driver1 = np.repeat(driver1, self.dim) + np.tile(np.arange(self.dim), driver1.shape[0])
        
        val_sigma = np.tile(self.val, (self.val.shape[0],1))
        val_tau = np.transpose(val_sigma)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            valterm = np.true_divide(val_sigma-val_tau,val_sigma+val_tau)
        valterm[~np.isfinite(valterm)] = 0.
        
        kappa = np.einsum('gh,ik,gi,jk,hj,ij->ij', self.k[self.dim*i:self.dim*(i+1),self.dim*j:self.dim*(j+1)],
                                                 self.coeffs[:,driver1],
                                                 self.vec[self.dim*i:self.dim*(i+1),:],
                                                 self.coeffs[:,driver1],
                                                 self.vec[self.dim*j:self.dim*(j+1),:],
                                                 valterm)
        
        return kappa
    
    def j(self, choice=0):
        
        if choice == 0:
            power = self.calculate_power_vector
        elif choice == 1:
            power = self.calculate_power_einsum
        elif choice == 2:
            power = self.calculate_power_einsum2
        else:
            return self.calculate_power_uncollapsed(*self.crossings[0])
        
        kappa = 0.
        
        for i,j in self.crossings:
            kappa += power(i,j)
            
        return 2.*self.gamma*kappa
    

    

    
        
        
        