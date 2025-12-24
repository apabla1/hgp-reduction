import argparse
from bposd.css import css_code
from functions.reduction_funcs import get_reduced_random_code
from functions.H_to_CNOT_circuit import generate_synd_circuit, generate_full_circuit_split
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

if __name__ == '__main__':
    
### Parameters -- adjustable!
    args = parse_args()
    p1 = 1e-2 # single-qubit error probability
    p2 = 1e-3 # two-qubit error probability
    p_spam = 1e-3 # measurement error probability
    rounds = 2 # rounds of syndrome extraction
    shots = args.shots # number of shots for BP decoding
    dec = args.decode # using OSD or LSD decoding
    max_iter = args.max_iter # maximum number of iterations in BP decoding
    osd_lsd_order = args.order # for OSD, how deep the OSD search goes;
                    # for LSD, how many bits in the neighborhood that post-processing explores

 
### Reduced HGP from random (d_v, d_c), [n, k, d_min] classical code
    # params: (n, d_v, d_c, d_min, min # of color groups)
    print("Generating reduced HGP from random LDPC code...")
    Hx1, Hx2, Hz1, Hz2, _ = get_reduced_random_code(20, 3, 5, 6, 5)
    
    # dimensions check
    assert Hx1.shape[1] == Hx2.shape[1] == Hz2.shape[1] == Hz1.shape[1]

### Generate CNOT syndrome circuit, enforcing correct ordering for hook errors
    # params: (Hx1, Hx2, Hz1, Hz2, rounds, p1, p2, p_spam, seed)
    print("Generating CNOT syndrome circuit...")
    circ = generate_full_circuit_split(Hx1, Hx2, Hz1, Hz2, rounds, p1, p2, p_spam, 1234)
   
### Sample CNOT circuit and decoding
    # params: (code, dec, circ, par, p2, shots, rounds)
    print("Sampling CNOT circuit and decoding... (This may take a while)")
    code = css_code(hx = add(Hx1, Hx2), hz = add(Hz1, Hz2))
    failures = num_failures_BP(code, dec, circ, [max_iter, osd_lsd_order], p2, shots, rounds)

    print(f"Number of failed shots: {failures} out of {shots}")
    print(f"==> Logical error rate is approx. {failures/shots:.4f}")