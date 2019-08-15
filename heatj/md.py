

import numpy as np
from scipy.integrate import solve_ivp

class MDBatch(object):
    
    def __init__(self, t, y):
        self.t = t
        self.y = y

def perform_md(lattice, t_span, nt, temp1, temp2, solver_options={}):
    """
    Return the scipy IVP solver batch object for the lattice dynamics
    
    Parameters
    ----------
    lattice : The heatj.Lattice object to be modeled
    t_span : The 2-tuple of time ranges to be simulated
    nt : The number of time evaluations
    temp1 : First reservoir temperature
    temp2 : Other reservoir temperature
    
    Keywords
    --------
    solver_options : Dictionary for additional scipy.solve_ivp arguments.
    
    """
    
    n = lattice.mass.shape[0]
    nd = lattice.drivers.shape[1]  # number of drivers on each side
    dim = lattice.dim
    y0 = np.zeros((2*n*dim))
    inv_mass_mat = lattice._m_matrix
    gamma_mat = lattice._g_matrix
    
    # build the Jacobian which is a constant matrix
    
    jac = np.zeros((2*n*dim, 2*n*dim))
    jac[:n*dim, n*dim:] = np.diag(np.ones(n*dim))
    jac[n*dim:, :n*dim] = -np.matmul(inv_mass_mat, lattice.k)
    jac[n*dim:, n*dim:] = -np.matmul(inv_mass_mat, gamma_mat)
    
    times = np.linspace(*t_span, num=nt)
    force_hot  = np.sqrt(2.*lattice.gamma*temp2)*np.random.randn(nt, nd, dim)
    force_cold = np.sqrt(2.*lattice.gamma*temp1)*np.random.randn(nt, nd, dim)
    
    def rforce(t):
        
        force = np.zeros(dim*n)
        tindex = np.searchsorted(times, t)
        for i in np.arange(dim):
            force[dim*lattice.drivers[0] + i] = force_cold[tindex,:,i]
            force[dim*lattice.drivers[1] + i] = force_hot[tindex,:,i]
        
        return force
    
    # build a function to solve for the velocity; acceleration
    
    def func(t, y):
        
        ydot = np.zeros((2*n*dim))
        ydot[:n*dim] = y[n*dim:]
        
        force = rforce(t)
        
        damping = np.matmul(gamma_mat, y[n*dim:])
        springs = np.matmul(lattice.k, y[:n*dim])
        
        ydot[n*dim:] = np.matmul(inv_mass_mat, (force - damping - springs))
        
        return ydot
    
    return solve_ivp(func, t_span, y0, jac=None, t_eval=times, **solver_options)

def perform_md_gf(lattice, t_span, nt, temp1, temp2):
    """
    Return the Green's function solution to the positions/velocities
    of lattice objects.
    """
    
#    nt -= 1
    n = lattice.mass.shape[0]
    nd = lattice.drivers.shape[1]  # number of drivers on each side
    dim = lattice.dim
    
    times = np.linspace(*t_span, num=nt+1)
    
    force = np.zeros((n*dim, nt))
    for i in np.arange(dim):
        force[dim*lattice.drivers[0] + i, :] = np.sqrt(2.*lattice.gamma*temp1)*np.random.randn(nd, nt)
        force[dim*lattice.drivers[1] + i, :] = np.sqrt(2.*lattice.gamma*temp2)*np.random.randn(nd, nt)
        
    y = np.zeros((2*dim*n, nt+1))
        
    ti, tf = t_span
    dt = (tf-ti)/nt
    
    gf = lattice.val[:,None]*dt*np.ones((2*dim*n, dim*n))
    gf = lattice.coeffs[None,:]*np.exp(gf)
    q = np.matmul(lattice.vec[:n,:], gf)
    qdot = np.matmul(lattice.vec[n:,:], gf)
    
    q = np.matmul(q, force).real
    qdot = np.matmul(qdot, force).real
    

    y[:dim*n,1:] = np.cumsum(q, axis=1)
    y[dim*n:,1:] = np.cumsum(qdot, axis=1)
    
    return MDBatch(times, y)
        