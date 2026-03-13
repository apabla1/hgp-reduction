import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import cast
from matplotlib import rcParams
from matplotlib.axes import Axes
from codes.random_codes import get_random_code
from codes.heawood_cycle import get_heawood_cycle
from codes.K33_cycle import get_K33_cycle
from functions.reduction_funcs import get_reduced_code
from functions.H_to_CNOT_circuit import generate_full_circuit, generate_full_circuit_split
from functions.BP_decoding import num_failures_BP

rcParams['font.size'] = 14
try:
    import subprocess
    subprocess.run(['latex', '--version'], check=True, capture_output=True)
    rcParams['text.usetex'] = True
except (FileNotFoundError, subprocess.CalledProcessError):
    pass  # LaTeX not available; fall back to mathtext

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, required=True,
                        help="[REQUIRED] Number of circuit samples to decode (e.g., 10000)")
    parser.add_argument("--decoder", type=str, required=True, choices=["OSD", "LSD", "Relay"],
                        help="[REQUIRED] Decoder to use: OSD, LSD, or Relay")

    # BP-OSD / BP-LSD parameters
    parser.add_argument("--bp-max-iter", type=int, default=80,
                        help="Maximum BP iterations for OSD/LSD (default: 80)")
    parser.add_argument("--bp-max-order", "--bp-order", dest="bp_order", type=int, default=5,
                        help="OSD/LSD order (default: 5)")

    # Relay-BP parameters
    parser.add_argument("--relay-gamma0", type=float, default=0.65,
                        help="Uniform memory weight for the first ensemble (default: 0.65)")
    parser.add_argument("--relay-pre-iter", type=int, default=80,
                        help="Max Relay iterations in the first ensemble (default: 80)")
    parser.add_argument("--relay-num-sets", type=int, default=100,
                        help="Number of Relay ensemble elements (default: 100)")
    parser.add_argument("--relay-max-iter", type=int, default=60,
                        help="Max BP interations per Relay ensemble (default: 60)")
    parser.add_argument("--relay-gamma-dist-interval", type=float, nargs=2,
                        default=(-0.24, 0.66),
                        metavar=("LOW", "HIGH"),
                        help="Uniform distribution range for disordered memory weight (default: -0.24 0.66)")
    parser.add_argument("--relay-stop-nconv", type=int, default=5,
                        help="Number of Relay solutions to find before stopping (default: 5)")

    args = parser.parse_args()

    return args

def sample_HGP_circuit_noise(code, circ, rounds, p2, decoder, dec_params, shots):
    """
    Take a CNOT syndrome extraction circuit, and run codes
    
    :param code: code that we are sampling
    :param circ: circuit corresponding to the code we are sampling
    :param rounds: rounds of syndrome extraction
    :param p2: two-qubit error probability (used as error prior for the decoder)
    """   

### Sample CNOT circuit and decode
    # params: (code, dec, circ, decoding params, p2, shots, rounds)
    print(f"\tSampling CNOT circuit and decoding via {decoder}... (This may take a while)")
    failures = num_failures_BP(code, decoder, circ, dec_params, p2, shots, rounds)
    ler = failures/shots
    
    print(f"\tNumber of failed shots: {failures} out of {shots}")
    print(f"\t==> Logical error rate is approx. {ler:.4f}")
    
    return ler
    
def total_sampling(p1, p2, p_spam, rounds, decoder, dec_params, shots,
                   unreduced_code, reduced_code, Hx1, Hx2, Hz1, Hz2, seed):
    """
    Samples the unreduced and reduced HGP codes. Returns the unreduced LER and the reduced LER.
    
    :param p1: single-qubit error probability
    :param p2: two-qubit error probability
    :param p_spam: measurement error probability
    :param rounds: rounds of syndrome extraction
    :param unreduced_code: original unreduced CSS code
    :param reduced_code: reduced CSS code
    :param Hx1, Hx2, Hz1, Hz2: sub-block PCMs of the reduced code
    :param seed: circuit seed (should vary per noise point for independent samples)
    """
    
    ### Parameters -- adjustable!
    print(f"   *******Noise parameters: p1={p1:.3}, p2={p2:.3}, p_spam={p_spam:.3}*******")
  
### Sample unreduced code with random syndrome extraction
    print("\tGenerating *unreduced* CNOT syndrome circuit with random syndrome extraction...")
    unreduced_random_circ = generate_full_circuit(unreduced_code, rounds, p1, p2, p_spam, seed)
    unreduced_random_LER = sample_HGP_circuit_noise(
        unreduced_code, unreduced_random_circ, rounds, p2, decoder, dec_params, shots
    )

### Sample reduced code with random syndrome extraction
    print("\tGenerating *reduced* CNOT syndrome circuit with random syndrome extraction...")
    reduced_random_circ = generate_full_circuit(reduced_code, rounds, p1, p2, p_spam, seed)
    reduced_random_LER = sample_HGP_circuit_noise(
        reduced_code, reduced_random_circ, rounds, p2, decoder, dec_params, shots
    )

### Sample reduced code with split syndrome extraction
    print("\tGenerating *reduced* CNOT syndrome circuit with split syndrome extraction...")
    reduced_split_circ = generate_full_circuit_split(Hx1, Hx2, Hz1, Hz2, rounds, p1, p2, p_spam, seed)
    reduced_split_LER = sample_HGP_circuit_noise(
        reduced_code, reduced_split_circ, rounds, p2, decoder, dec_params, shots
    )
    
    return unreduced_random_LER, reduced_random_LER, reduced_split_LER

if __name__ == '__main__':
    
### Command-line arguments 
    args = parse_args()
    shots = args.shots # number of shots for BP decoding
    decoder = args.decoder

    if decoder == "Relay":
        dec_params = [
            args.relay_gamma0,
            args.relay_pre_iter,
            args.relay_num_sets,
            args.relay_max_iter,
            tuple(args.relay_gamma_dist_interval),
            args.relay_stop_nconv]
    else:
        dec_params = [args.bp_max_iter, args.bp_order]

    codes = ["heawood", "K_33", "random"]
    ps = [7e-4, 7.5e-4, 8e-4, 8.5e-4, 9e-4, 9.5e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3]
    
    # (for plotting later)
    results = {
        code: {
            "unreduced_random": [],
            "reduced_random": [],
            "reduced_split": [],
        }
        for code in codes
    }

    def weight_stats(H):
        rw = H.getnnz(axis=1)
        cw = H.getnnz(axis=0)
        return (rw.min(), rw.max(), round(rw.mean(), 3), cw.min(), cw.max(), round(cw.mean(), 3))

    for code in codes:
        match code:
            case "random":
                print("------Sampling Random LDPC Code------")
            ### Reduced HGP from random (d_v, d_c), [n, k, d_min] classical code
                print("\tGenerating HGP code from random LDPC code...")
                unreduced_code, H = get_random_code(n=12, d_v=3, d_c=4, min_dist=6, max_coloring=3) 
                print("\tGenerating reduced HGP...")
                Hx1, Hx2, Hz1, Hz2, reduced_code, _, _, d = get_reduced_code(unreduced_code, H)
                
            case "heawood":
                print("------Sampling Heawood Cycle Code------")
            ### Reduced HGP from Heawood code
                print("\tGenerating HGP code from Heawood LDPC code...")
                unreduced_code, H = get_heawood_cycle()
                print("\tGenerating reduced HGP...")
                Hx1, Hx2, Hz1, Hz2, reduced_code, _, _, d = get_reduced_code(unreduced_code, H)
                
            case "K_33":
                print("------Sampling K_(3, 3) Cycle Code------")
            ### Reduced HGP from K_{3, 3} code
                print("\tGenerating HGP code from K_(3, 3) LDPC code...")
                unreduced_code, H = get_K33_cycle()
                print("\tGenerating reduced HGP...")
                Hx1, Hx2, Hz1, Hz2, reduced_code, _, _, d = get_reduced_code(unreduced_code, H)
    
        # dimensions check
        assert Hx1.shape[1] == Hx2.shape[1] == Hz2.shape[1] == Hz1.shape[1]

    ### Test out the weight changes
        print("\t --Format: (rmin, rmax, rmean, cmin, cmax, cmean)--")
        print("\t   unreduced hx:", weight_stats(unreduced_code.hx))
        print("\t   unreduced hz:", weight_stats(unreduced_code.hz))
        print("\t   reduced hx:", weight_stats(reduced_code.hx))
        print("\t   reduced hz:", weight_stats(reduced_code.hz))

    ### Sample for different error probabilities
        for i, p in enumerate(ps):
            unreduced_random_LER, reduced_random_LER, reduced_split_LER = total_sampling(
                p1=p/10, p2=p, p_spam=p, rounds=d, decoder=decoder, dec_params=dec_params, shots=shots,
                unreduced_code=unreduced_code, reduced_code=reduced_code,
                Hx1=Hx1, Hx2=Hx2, Hz1=Hz1, Hz2=Hz2, seed=i+1
            )
            results[code]["unreduced_random"].append(unreduced_random_LER)
            results[code]["reduced_random"].append(reduced_random_LER)
            results[code]["reduced_split"].append(reduced_split_LER)
            
### Plot
    fig, axes = plt.subplots(1, len(codes), figsize=(5 * len(codes), 4), sharex=True, sharey=True)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten().tolist()
    else:
        axes = [axes]

    p_array = np.array(ps, dtype=float)
    sort_idx = np.argsort(p_array)

    for ax, code_type in zip(axes, codes):
        ax = cast(Axes, ax)
        ler_unred = np.array(results[code_type]["unreduced_random"], dtype=float)[sort_idx]
        ler_red_split = np.array(results[code_type]["reduced_split"], dtype=float)[sort_idx]
        ler_red_rand = np.array(results[code_type]["reduced_random"], dtype=float)[sort_idx]
        p_plot = p_array[sort_idx]

        # Binomial standard error for the sampled logical error rate.
        std_unred = np.sqrt(ler_unred * (1 - ler_unred) / shots)
        std_red_split = np.sqrt(ler_red_split * (1 - ler_red_split) / shots)
        std_red_rand = np.sqrt(ler_red_rand * (1 - ler_red_rand) / shots)

        ax.errorbar(p_plot, ler_unred, yerr=std_unred, fmt='.-', capsize=3, alpha=1,
                    label='original, random SE')
        ax.errorbar(p_plot, ler_red_split, yerr=std_red_split, fmt='.-', capsize=3, alpha=1,
                    label='reduced, split SE')
        ax.errorbar(p_plot, ler_red_rand, yerr=std_red_rand, fmt='.-', capsize=3, alpha=1,
                    label='reduced, random SE')

        code_titles = {
            "heawood": r"Heawood Cycle Code",
            "K_33":    r"$K_{3,3}$ Cycle Code",
            "random":  r"Random LDPC Code",
        }
        ax.set_title(code_titles.get(code_type, code_type), fontsize=14)
        ax.set_xlabel(r'$p$', fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(8E-4, 1E-2)
        ax.grid(True, which='both', axis='both')

    axes[0].set_ylabel(r'Logical failure probability', fontsize=16)
    axes[0].legend(fontsize=12, loc='lower right')

    plt.tight_layout()

### Save plot
    os.makedirs('plots', exist_ok=True)
    p_min = min(ps)
    p_max = max(ps)
    codes_str = '-'.join(codes)
    if decoder == 'Relay':
        gamma0, pre_iter, num_sets, max_iter, gamma_dist_interval, stop_nconv = dec_params
        params_str = (f"g0={gamma0}_pre={pre_iter}_sets={num_sets}_iter={max_iter}"
                      f"_gdi={gamma_dist_interval[0]}to{gamma_dist_interval[1]}_nconv={stop_nconv}")
    else:
        params_str = f"bpiter={dec_params[0]}_order={dec_params[1]}"
    plot_filename = (
        f"plots/{codes_str}_{decoder}_shots={shots}"
        f"_p={p_min:.0e}to{p_max:.0e}_{params_str}.pdf"
    )
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    plt.show()