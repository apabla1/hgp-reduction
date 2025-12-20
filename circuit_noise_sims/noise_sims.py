from reduction_funcs import get_reduced_random_code
from hgp_tannergraphs import XZ_ctype_tannergraph
from tannergraph_to_CNOT_circuit import generate_synd_circuit, generate_full_circuit
from bposd.css import css_code

if __name__ == '__main__':
    
    ### Reduced HGP from random (d_v, d_c), [n, k, d_min] classical code
    ### params: (n, d_v, d_c, d_min, min # of color groups)
    Hxnew, Hznew = get_reduced_random_code(20, 3, 5, 6, 5)
    code = css_code(hx=Hxnew, hz=Hznew, name="Reduced Code")
    
    # params: (code, rounds, p1, p2, p_spam, seed)
    rounds = 2
    c = generate_full_circuit(code, rounds, 1e-3, 1e-2, 1e-3, 1234)
    
    # sample
    shots = 200
    samples = c.compile_sampler().sample(shots)
    
    mz = code.hz.shape[0]
    n = code.hx.shape[1]
    
    # first rounds*mz bits are Z-check outcomes
    z_synd = samples[:, :rounds*mz].reshape(shots, rounds, mz)
    
    final_data = samples[:, rounds*mz: rounds*mz + n]