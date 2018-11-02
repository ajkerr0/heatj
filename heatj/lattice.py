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
        self.set_greensfunc()
        
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
    
    @property
    def ppation_ratio(self):
        """Return the participation ratio of the eigenmodes"""
        a = self.vec[:self.dim*self.n]
        a = a/np.linalg.norm(a, axis=0)
        return np.sum(np.abs(a)**2, axis=0)**2/ \
               np.sum(np.abs(a)**4, axis=0)
               
    @property
    def ppation_ratio2(self):
        """Alternative definition of PR"""
        vec = self.vec[:self.dim*self.n]
        vec = vec/np.linalg.norm(vec, axis=0)
        return 1./(self.val.shape[0]*np.sum((vec*vec.conj())**2, axis=0)).real
    
    def set_greensfunc(self):
        self.val, self.vec = self._calculate_evec()
        self.coeffs = self._calculate_coeffs()
    
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
        
        val, vec = np.linalg.eig(a)
        val[np.where(np.abs(val) < 1e-4)[0]] = 0.+0.j
        return val, vec
    
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
    
    def calculate_power_vector_uncollapsed(self, crossings):
    
        # assuming same drag constant as other driven atom
        driver1 = self.drivers[1]
        
        n = self.val.shape[0]
        
        kappa = 0.
        
        val_sigma = np.tile(self.val, (n,1))
        val_tau = np.transpose(val_sigma)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            valterm = np.true_divide(val_sigma-val_tau,val_sigma+val_tau)
        valterm[~np.isfinite(valterm)] = 0.
        
#        kappa = np.array([0.,])
        kappa = []
        
        for i,j in crossings:
        
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
                        termArr = term1*term2*term3*term4*valterm*self.k[self.dim*i + idim, self.dim*j + jdim]
                        
#                        kappa = np.concatenate((kappa, self.k[self.dim*i + idim, self.dim*j + jdim]*np.hstack(termArr)))
                        kappa.extend(np.hstack(termArr).tolist())
        
#        return kappa[1:]
        return np.array(kappa)
    
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
    
    def calculate_power_einsum2(self, i, j):
        
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
    
    def calculate_power_uncollapsed(self, crossings):
        
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
        
        kappa = np.array([0.,])
        
        for i,j in crossings:
        
            kappa = np.concatenate(np.einsum('gh,ik,gi,jk,hj,ij->ij', self.k[self.dim*i:self.dim*(i+1),self.dim*j:self.dim*(j+1)],
                                                 self.coeffs[:,driver1],
                                                 self.vec[self.dim*i:self.dim*(i+1),:],
                                                 self.coeffs[:,driver1],
                                                 self.vec[self.dim*j:self.dim*(j+1),:],
                                                 valterm))
        
        return kappa[1:]
    
    def calculate_power_uncollapsed_brute_force(self, crossings):
        
        # pick a driven side, we will assume the same uniform damping on both
        driver1 = self.drivers[1]
        
        sig_list = []
        
        n = self.val.shape[0]//2
        
        for i,j in crossings:
        
            for idim in range(self.dim):
                for jdim in range(self.dim):
                    for driver in driver1:
                        term = 0.
                        for sigma in range(2*n):
                            cosigma = 0.
                            for k in np.arange(self.dim):
                                cosigma += self.coeffs[sigma, self.dim*driver + k]
                            for tau in range(2*n):
                                cotau = 0.
                                for k in np.arange(self.dim):
                                    cotau += self.coeffs[tau, self.dim*driver + k]
                                    
                                term += self.k[self.dim*i + idim, self.dim*j + jdim]*(cosigma*cotau*(self.vec[:n,:][self.dim*i + idim ,sigma])*(
                                        self.vec[:n,:][self.dim*j + jdim,tau])*((self.val[sigma]-self.val[tau])/(self.val[sigma]+self.val[tau])))
                        sig_list.append(term)
        return np.array(sig_list)
    
    def j(self, choice=0):
        
        if choice == 0:
            power = self.calculate_power_vector
        elif choice == 1:
            power = self.calculate_power_einsum
        elif choice == 2:
            power = self.calculate_power_einsum2
        elif choice == 3:
            return 2.*self.gamma*self.calculate_power_uncollapsed_brute_force(self.crossings)
        elif choice == 4:
            return 2.*self.gamma*self.calculate_power_vector_uncollapsed(self.crossings)
        else:
            return 2.*self.gamma*self.calculate_power_uncollapsed(self.crossings)
        
        kappa = 0.
        
        for i,j in self.crossings:
            kappa += power(i,j)
            
        return 2.*self.gamma*np.abs(kappa.real)
    

    

    
        
        
        