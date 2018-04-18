"""

@author: Alex Kerr
"""

import time

def timeit(func):
    
    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(end-start)
        return result
    
    return timed
