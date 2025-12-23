from noise_funcs.reduction_funcs import get_reduced_random_code
from noise_funcs.H_to_CNOT_circuit import generate_synd_circuit, generate_full_circuit_split
from bposd.css import css_code

if __name__ == '__main__':
    
    ### Reduced HGP from random (d_v, d_c), [n, k, d_min] classical code
    ### params: (n, d_v, d_c, d_min, min # of color groups)
    Hx1, Hx2, Hz1, Hz2, _ = get_reduced_random_code(20, 3, 5, 6, 5)
    
    # dimensions check
    assert Hx1.shape[0] == Hz1.shape[0]
    assert Hx1.shape[1] == Hx2.shape[1] == Hz2.shape[1] == Hz1.shape[1]
    
    mx = Hx1.shape[0]
    mz = Hz1.shape[0]
    n = Hx1.shape[1]

    # params: (Hx1, Hx2, Hz1, Hz2, rounds, p1, p2, p_spam, seed)
    rounds = 2
    c = generate_full_circuit_split(Hx1, Hx2, Hz1, Hz2, rounds, 1e-3, 1e-2, 1e-3, 1234)
   
    # sample 
    shots = 200
    samples = c.compile_sampler().sample(shots)

    z_syndrome = samples[:, :rounds*mz].reshape(shots, rounds, mz)
    final_data = samples[:, rounds*mz : rounds*mz + n]
