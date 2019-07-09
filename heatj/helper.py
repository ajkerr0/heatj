"""

@author: Alex Kerr
"""

import itertools

import numpy as np

def find_heat_channels(r, neighbors, pos, cutoff=9., bond_cutoff=4):
    """
    Given a list of reservoir atoms, bonded atoms, and their positions, 
    return a list of channels that represent the interactions crossing the
    interface between two reservoirs.
    """
    channels = []
    def dfs(index, discoverList, start):
        disList = list(discoverList)
        disList.append(index)
        if len(disList) <= bond_cutoff-1:
            for neighbor in neighbors[index]:
                if neighbor not in r:
                    channels.append([start, neighbor])
                dfs(neighbor, disList, start)
                        
    for res in r:
        disList = []
        dfs(res, disList, res)
        
    # then look for cutoffs
    nonres_pos = np.copy(pos)
    nonres_pos[r] = np.inf
    for res in r:
        rij  = np.linalg.norm(pos[res]-nonres_pos, axis=1)
        under_cutoff = np.where(rij < cutoff)[0]
        for u in under_cutoff:
            channels.append([res,u])
        
    channels.sort()
    return list(k for k,_ in itertools.groupby(channels))

def general_hessian(pos, k, neighbors, d):
    """
    Return a Hessian for kr^2 interactions
    """
    
    n, dim = pos.shape
    i,j = neighbors.T
    
    # create stack of dimxdim blocks that will compose the hessian
    hess = np.zeros((n**2,dim,dim))
    posij = pos[i] - pos[j]
    # create stack of subblocks where each layer is the outerproduct of pos diffs
    block = np.einsum('ki,kj->kij', posij, posij)
    block = k[:,None,None]*block
    # add subblocks to the hessian
    # first add positives at the unraveled diagonal
    np.add.at(hess, (n+1)*i, block)
    np.add.at(hess, (n+1)*j, block)
    # now add off diagonal blocks
    np.add.at(hess, n*i + j, -block)
    np.add.at(hess, n*j + i, -block)
    # add on-site potential
    dblock = d[:,None,None]*np.tile(np.eye(dim), (n,1,1))
    np.add.at(hess, (n+1)*np.arange(n), dblock)
    return np.hstack(np.hstack(hess.reshape(n,n,dim,dim)))