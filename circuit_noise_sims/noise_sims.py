from reduction_funcs import get_reduced_random_code
from hgp_tannergraphs import XZ_ctype_tannergraph
from tannergraph_to_CNOT_circuit import generate_full_circuit_split, generate_synd_circuit, generate_full_circuit
from bposd.css import css_code

if __name__ == '__main__':
    
    ### Reduced HGP from random (d_v, d_c), [n, k, d_min] classical code
    ### params: (n, d_v, d_c, d_min, min # of color groups)
    Hx1, Hx2, Hz1, Hz2, _ = get_reduced_random_code(20, 3, 5, 6, 5)

    # params: (Hx1, Hx2, Hz1, Hz2, rounds, p1, p2, p_spam, seed, measure_x)
    rounds = 2
    c = generate_full_circuit_split(Hx1, Hx2, Hz1, Hz2, rounds, 1e-3, 1e-2, 1e-3, 1234, measure_x=False)
   
    # sample 
    shots = 200
    samples = c.compile_sampler().sample(shots)

    mz = Hz1.shape[0]
    n = Hx1.shape[1]

    # measurement layout (when measure_x=False):
    # per round:
    #   Z1: mz bits
    #   Z2: mz bits
    #   X stages: no recorded bits
    # after rounds: final data MR gives n bits

    z1_bits = samples[:, :rounds*mz].reshape(shots, rounds, mz)
    z2_bits = samples[:, rounds*mz:2*rounds*mz].reshape(shots, rounds, mz)

    z_eff = z1_bits ^ z2_bits

    final_data = samples[:, 2*rounds*mz : 2*rounds*mz + n]