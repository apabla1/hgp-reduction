import argparse
from bposd.css import css_code
from functions.reduction_funcs import get_random_code, get_reduced_code
from functions.H_to_CNOT_circuit import generate_synd_circuit, generate_full_circuit, generate_full_circuit_split
from functions.BP_decoding import num_failures_BP
from functions.matrix_funcs import add

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, required=True,
                        help="Number of circuit samples to decode (e.g., 10000)")
    parser.add_argument("--decode", type=str.upper, choices=["OSD", "LSD"], required=True,
                        help="Decoder type: OSD or LSD")
    parser.add_argument("--max-iter", type=int, required=True,
                        help="Max BP iterations (e.g., 75)")
    parser.add_argument("--order", type=int, required=True,
                        help="OSD order (if OSD) or LSD neighborhood size (if LSD) (e.g., OSD-2 or LSD-6)")
    return parser.parse_args()

def sample_HGP_circuit_noise(code, circ, rounds, p1, p2, p_spam):
    ### Generate CNOT syndrome circuit, enforcing correct ordering for hook errors
   
    ### Sample CNOT circuit and decoding
    # params: (code, dec, circ, par, p2, p_data, p_meas, shots, rounds)
    print(f"Sampling CNOT circuit and decoding via BP-{dec}... (This may take a while)")
    failures = num_failures_BP(code, dec, circ, [max_iter, osd_lsd_order], p2, shots, rounds)

    print(f"Number of failed shots: {failures} out of {shots}")
    print(f"==> Logical error rate is approx. {failures/shots:.4f}")

if __name__ == '__main__':

### Reduced HGP from random (d_v, d_c), [n, k, d_min] classical code
    print("Generating HGP code from random LDPC code...")
    unreduced_code, H = get_random_code(n=20, d_v=3, d_c=4, min_dist=6, max_coloring=3) 
    print("Generating reduced HGP...")
    Hx1, Hx2, Hz1, Hz2, _, _, _, d = get_reduced_code(unreduced_code, H)
    reduced_code = css_code(hx = add(Hx1, Hx2), hz = add(Hz1, Hz2))
    
    # dimensions check
    assert Hx1.shape[1] == Hx2.shape[1] == Hz2.shape[1] == Hz1.shape[1]
    
### Parameters -- adjustable!
    args = parse_args()
    p = 1e-3
    p1 = p/10 # single-qubit error probability
    p2 = p # two-qubit error probability
    p_spam = p # measurement error probability
    rounds = d # rounds of syndrome extraction
    shots = args.shots # number of shots for BP decoding
    dec = args.decode # using OSD or LSD decoding
    max_iter = args.max_iter # maximum number of iterations in BP decoding
    osd_lsd_order = args.order # for OSD, how deep the OSD search goes;
                               # for LSD, how many bits in the neighborhood that post-processing explores
    print(f"Noise parameters: p1={p1}, p2={p2}, p_spam={p_spam}")
    
### Sample unreduced code
    print("Generating **unreduced** CNOT syndrome circuit...")
    unreduced_circ = generate_full_circuit(unreduced_code, rounds, p1, p2, p_spam, 1234)
    sample_HGP_circuit_noise(unreduced_code, unreduced_circ, rounds, p1, p2, p_spam)

### Sample reduced code
    print("Generating **reduced** CNOT syndrome circuit (that avoids hook errors)...")
    reduced_circ = generate_full_circuit_split(Hx1, Hx2, Hz1, Hz2, rounds, p1, p2, p_spam, 1234)
    sample_HGP_circuit_noise(reduced_code, reduced_circ, rounds, p1, p2, p_spam)
