import numpy as np
from bposd.hgp import hgp
from ldpc.code_util import compute_code_parameters, compute_exact_code_distance

def get_K33_cycle():
    checks = [[1,2,3],[4,5,6],[7,8,9],[1,4,7],[2,5,8],[3,6,9]]
    H = np.zeros([6,9], dtype=int)
    for i in range(6):
        H[i,np.array(checks[i])-1] = [1,1,1]  
    print(f"\t\tK3,3 code: [n, k, d] = {compute_code_parameters(H)}")
    
    ### Create HGP code from two of the classical code
    code = hgp(h1=H, h2=H, compute_distance=True)
    code.name = 'K33 Code HGP'
    print(f"\t\tHGP Code: [[{code.N}, {code.K}, {code.D}]]")
    return code, H