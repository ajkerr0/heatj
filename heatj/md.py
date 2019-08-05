

import numpy as np
from scipy.integrate import solve_ivp

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
    force_hot  = np.sqrt(2.*lattice.gamma*temp2)*np.random.randn(nt, nd)
    force_cold = np.sqrt(2.*lattice.gamma*temp1)*np.random.randn(nt, nd)
    
    def rforce(t):
        
        force = np.zeros(dim*n)
        tindex = np.searchsorted(times, t)
        force[lattice.drivers[0]] = force_cold[tindex]
        force[lattice.drivers[1]] = force_hot[tindex]
        
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
    
    sol = solve_ivp(func, t_span, y0, jac=None, t_eval=times, **solver_options)
    
    return sol
        
        
        