import argparse
import matplotlib.pyplot as plt
from bposd.css import css_code
from functions.codes.random_codes import get_random_code
from functions.codes.heawood_cycle import get_heawood_cycle
from functions.codes.K33_cycle import get_K33_cycle
from functions.reduction_funcs import get_reduced_code
from functions.H_to_CNOT_circuit import generate_full_circuit, generate_full_circuit_split
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
    Take a CNOT syndrome extraction circuit, and run codes
    
    :param code: code that we are sampling
    :param circ: circuit corresponding to the code we are sampling
    :param rounds: rounds of syndrome extraction
    :param p1: single-qubit error probability
    :param p2: two-qubit error probability
    :param p_spam: measurement error probability
    """   

### Sample CNOT circuit and decode
    # params: (code, dec, circ, decoding params, p2, shots, rounds)
    print(f"\tSampling CNOT circuit and decoding via BP-{dec}... (This may take a while)")
    failures = num_failures_BP(code, dec, circ, [max_iter, osd_lsd_order], p2, shots, rounds)
    ler = failures/shots
    
    print(f"\tNumber of failed shots: {failures} out of {shots}")
    print(f"\t==> Logical error rate is approx. {ler:.4f}")
    
    return ler
    
def total_sampling(p1, p2, p_spam, rounds):
    """
    codes the unreduced the reduced HGP codes. Returns the unreduced LER and the reduced LER
    
    :param p1: single-qubit error probability
    :param p2: two-qubit error probability
    :param p_spam: measurement error probability
    :param rounds: rounds of syndrome extraction
    """
    
    ### Parameters -- adjustable!
    print(f"   *******Noise parameters: p1={p1}, p2={p2}, p_spam={p_spam}*******")
  
### Sample unreduced code with random syndrome extraction
    print("\tGenerating *unreduced* CNOT syndrome circuit with random syndrome extraction...")
    unreduced_random_circ = generate_full_circuit(unreduced_code, rounds, p1, p2, p_spam, 1234)
    unreduced_random_LER = sample_HGP_circuit_noise(unreduced_code, unreduced_random_circ, rounds, p1, p2, p_spam)

### Sample reduced code with random syndrome extraction
    print("\tGenerating *reduced* CNOT syndrome circuit with random syndrome extraction...")
    reduced_random_circ = generate_full_circuit(reduced_code, rounds, p1, p2, p_spam, 1234)
    reduced_random_LER = sample_HGP_circuit_noise(reduced_code, reduced_random_circ, rounds, p1, p2, p_spam)

### Sample reduced code with split syndrome extraction
    print("\tGenerating *reduced* CNOT syndrome circuit with split syndrome extraction...")
    reduced_split_circ = generate_full_circuit_split(Hx1, Hx2, Hz1, Hz2, rounds, p1, p2, p_spam, 1234)
    reduced_split_LER = sample_HGP_circuit_noise(reduced_code, reduced_split_circ, rounds, p1, p2, p_spam)
    
    return unreduced_random_LER, reduced_random_LER, reduced_split_LER

if __name__ == '__main__':
    
### Command-line arguments 
    args = parse_args()
    shots = args.shots # number of shots for BP decoding
    dec = args.decode # using OSD or LSD decoding
    max_iter = args.max_iter # maximum number of iterations in BP decoding
    osd_lsd_order = args.order # for OSD, how deep the OSD search goes;
                               # for LSD, how many bits in the neighborhood that post-processing explores

    codes = ["random"]
    ps = [1e-3, 2e-3, 3e-3]
    
    # (for plotting later)
    results = {
        code: {
            "unreduced_random": [],
            "reduced_random": [],
            "reduced_split": [],
        }
        for code in codes
    }

    for code in codes:
        match code:
            case "random":
                print("------Sampling Random LDPC Code------")
            ### Reduced HGP from random (d_v, d_c), [n, k, d_min] classical code
                print("\tGenerating HGP code from random LDPC code...")
                unreduced_code, H = get_random_code(n=12, d_v=3, d_c=4, min_dist=6, max_coloring=3) 
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
       
    ### Test out the weight changes
        def weight_stats(H):
            rw = H.getnnz(axis=1)
            cw = H.getnnz(axis=0)
            return (rw.min(), rw.max(), rw.mean(), cw.min(), cw.max(), cw.mean())

        print("\t --Format: (rmin, rmax, rmean, cmin, cmax, cmean)--")
        print("\t   unreduced hx:", weight_stats(unreduced_code.hx))
        print("\t   unreduced hz:", weight_stats(unreduced_code.hz))
        print("\t   reduced hx:", weight_stats(reduced_code.hx))
        print("\t   reduced hz:", weight_stats(reduced_code.hz))

    ### Sample for different error probabilities
        for p in ps:
            unreduced_random_LER, reduced_random_LER, reduced_split_LER = total_sampling(p1=p/10, p2=p, p_spam=p, rounds=d)
            results[code]["unreduced_random"].append(unreduced_random_LER)
            results[code]["reduced_random"].append(reduced_random_LER)
            results[code]["reduced_split"].append(reduced_split_LER)
            
### Plot
    fig, axes = plt.subplots(1, len(codes), figsize=(5 * len(codes), 4), sharex=True, sharey=True)

    if len(codes) == 1:
        axes = [axes]

    for ax, code_type in zip(axes, codes):
        ax.plot(ps, results[code_type]["unreduced_random"], marker="o",
                label="unreduced + random synd.")
        ax.plot(ps, results[code_type]["reduced_random"], marker="o",
                label="reduced + random synd.")
        ax.plot(ps, results[code_type]["reduced_split"], marker="o",
                label="reduced + split synd.")

        ax.set_title(code_type)
        ax.set_xticks(ps, [f"{p:g}" for p in ps])
        ax.set_xlabel("p")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Logical error rate")
    axes[0].legend()

    plt.tight_layout()
    plt.show()