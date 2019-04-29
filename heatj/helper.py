"""

@author: Alex Kerr
"""

import itertools

import numpy as np

def find_heat_channels(r, neighbors, pos, cutoff=6., bond_cutoff=4):
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
        
        
                        
    
    
    

    
        
        
        