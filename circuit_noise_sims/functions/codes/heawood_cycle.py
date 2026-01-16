import numpy as np
from bposd.hgp import hgp
from ldpc.code_util import compute_code_parameters, compute_exact_code_distance

def get_heawood_cycle():
    checks = [[1, 15, 20],[2, 4, 18], [5, 7, 21],[3, 8, 10],[6, 11, 13],[9, 14, 16],[12, 17, 19],[1, 2, 3],[4, 5, 6],
    [7, 8, 9],[10, 11, 12],[13, 14, 15],[16, 17, 18],[19, 20, 21]]
    H = np.zeros([14,21], dtype=int)
    for i in range(14):
        H[i,np.array(checks[i])-1] = [1,1,1]   
    print(f"\t\tHeawood code: [n, k, d] = {compute_code_parameters(H)}")
    
    ### Create HGP code from two of the classical code
    code = hgp(h1=H, h2=H, compute_distance=True)
    code.name = 'Heawood Code HGP'
    print(f"\t\tHGP Code: [[{code.N}, {code.K}, {code.D}]]")
        
    return code, H