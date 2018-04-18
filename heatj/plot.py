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