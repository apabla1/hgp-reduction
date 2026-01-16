import argparse
import matplotlib.pyplot as plt
from bposd.css import css_code
from functions.codes.random_codes import get_random_code
from functions.codes.heawood_cycle import get_heawood_cycle
from functions.codes.K33_cycle import get_K33_cycle
from functions.reduction_funcs import get_reduced_code
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
    """
    Take a CNOT syndrome extraction circuit, and run samples
    
    :param code: code that we are sampling
    :param circ: circuit corresponding to the code we are sampling
    :param rounds: rounds of syndrome extraction
    :param p1: single-qubit error probability
    :param p2: two-qubit error probability
    :param p_spam: measurement error probability
    """   

### Sample CNOT circuit and decode
    # params: (code, dec, circ, par, p2, p_data, p_meas, shots, rounds)
    print(f"\tSampling CNOT circuit and decoding via BP-{dec}... (This may take a while)")
    failures = num_failures_BP(code, dec, circ, [max_iter, osd_lsd_order], p2, shots, rounds)
    ler = failures/shots
    
    print(f"\tNumber of failed shots: {failures} out of {shots}")
    print(f"\t==> Logical error rate is approx. {ler:.4f}")
    
    return ler
    
def total_sampling(p1, p2, p_spam, rounds):
    """
    Samples the unreduced the reduced HGP codes. Returns the unreduced LER and the reduced LER
    
    :param p1: single-qubit error probability
    :param p2: two-qubit error probability
    :param p_spam: measurement error probability
    :param rounds: rounds of syndrome extraction
    """
    
    ### Parameters -- adjustable!
    print(f"   *******Noise parameters: p1={p1}, p2={p2}, p_spam={p_spam}*******")
  
### Sample unreduced code
    print("\tGenerating *unreduced* CNOT syndrome circuit...")
    unreduced_circ = generate_full_circuit(unreduced_code, rounds, p1, p2, p_spam, 1234)
    unreduced_LER = sample_HGP_circuit_noise(unreduced_code, unreduced_circ, rounds, p1, p2, p_spam)

### Sample reduced code
    print("\tGenerating *reduced* CNOT syndrome circuit (that avoids hook errors)...")
    reduced_circ = generate_full_circuit_split(Hx1, Hx2, Hz1, Hz2, rounds, p1, p2, p_spam, 1234)
    reduced_LER = sample_HGP_circuit_noise(reduced_code, reduced_circ, rounds, p1, p2, p_spam)
    
    return unreduced_LER, reduced_LER

if __name__ == '__main__':
    
### Command-line arguments 
    args = parse_args()
    shots = args.shots # number of shots for BP decoding
    dec = args.decode # using OSD or LSD decoding
    max_iter = args.max_iter # maximum number of iterations in BP decoding
    osd_lsd_order = args.order # for OSD, how deep the OSD search goes;
                               # for LSD, how many bits in the neighborhood that post-processing explores

    samples = ["heawood", "K_33", "random"]
    ps = [1e-3, 2e-3, 3e-3]
    
    reduced_results = {code_type: [] for code_type in samples}

    for code_type in samples:
        match code_type:
            case "random":
                print("------Sampling Random LDPC Code------")
            ### Reduced HGP from random (d_v, d_c), [n, k, d_min] classical code
                print("\tGenerating HGP code from random LDPC code...")
                unreduced_code, H = get_random_code(n=20, d_v=3, d_c=5, min_dist=6, max_coloring=5) 
                print("\tGenerating reduced HGP...")
                Hx1, Hx2, Hz1, Hz2, _, _, _, d = get_reduced_code(unreduced_code, H)
                reduced_code = css_code(hx = add(Hx1, Hx2), hz = add(Hz1, Hz2))
                
            case "heawood":
                print("------Sampling Heawood Cycle Code------")
            ### Reduced HGP from Heawood code
                print("\tGenerating HGP code from Heawood LDPC code...")
                unreduced_code, H = get_heawood_cycle()
                print("\tGenerating reduced HGP...")
                Hx1, Hx2, Hz1, Hz2, _, _, _, d = get_reduced_code(unreduced_code, H)
                reduced_code = css_code(hx = add(Hx1, Hx2), hz = add(Hz1, Hz2))
                
            case "K_33":
                print("------Sampling K_(3, 3) Cycle Code------")
            ### Reduced HGP from K_{3, 3} code
                print("\tGenerating HGP code from K_(3, 3) LDPC code...")
                unreduced_code, H = get_K33_cycle()
                print("\tGenerating reduced HGP...")
                Hx1, Hx2, Hz1, Hz2, _, _, _, d = get_reduced_code(unreduced_code, H)
                reduced_code = css_code(hx = add(Hx1, Hx2), hz = add(Hz1, Hz2))
    
        # dimensions check
        assert Hx1.shape[1] == Hx2.shape[1] == Hz2.shape[1] == Hz1.shape[1]
    
    ### Sample for different error probabilities
        for p in ps:
            unreduced_LER, reduced_LER = total_sampling(p1=p/10, p2=p, p_spam=p, rounds=d)
            reduced_results[code_type].append(reduced_LER)
            
    ### Plot
        for code_type, ys in reduced_results.items():
            plt.plot(ps, ys, marker="o", label=code_type)

        #plt.xscale("log")
        #plt.yscale("log")
        plt.xticks(ps, [f"{p:g}" for p in ps])

        plt.xlabel("p (two-qubit error prob)")
        plt.ylabel("Reduced LER")
        plt.legend()
        plt.tight_layout()
        plt.show()