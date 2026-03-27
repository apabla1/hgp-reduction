import argparse
import os
import sys
import importlib
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import cast, Dict, List, Tuple, Any, Optional
from matplotlib import rcParams
from matplotlib.axes import Axes
from pathlib import Path
from multiprocessing import Pool
from functions.reduction_funcs import get_reduced_code
from functions.H_to_CNOT_circuit import generate_full_circuit, generate_full_circuit_split
from functions.decoding import num_failures_BP

rcParams['font.size'] = 14
rcParams['text.usetex'] = True

# Get the codes directory path
CODES_DIR = Path(__file__).parent / "codes"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def get_available_codes() -> Dict[str, str]:
    """
    Dynamically discover all available code getter functions.
    Returns dict mapping code name to module name.
    Excludes helper functions and only includes actual code generators.
    """
    codes = {}
    exclude_prefixes = ('get_check_', 'get_coloring_', 'get_adj_')  # Helper function patterns
    
    for module_file in CODES_DIR.glob("*.py"):
        if module_file.name.startswith("__"):
            continue
        module_name = module_file.stem
        try:
            module = importlib.import_module(f"codes.{module_name}")
            # Find all functions starting with 'get_' but exclude helpers
            for attr_name in dir(module):
                if not attr_name.startswith('get_'):
                    continue
                # Skip known helper function patterns
                if any(attr_name.startswith(prefix) for prefix in exclude_prefixes):
                    continue
                code_name = attr_name[4:]  # Remove 'get_' prefix
                codes[code_name] = (module_name, attr_name)
        except Exception as e:
            print(f"Warning: Could not load code module {module_name}: {e}", file=sys.stderr)
    return codes

def load_code(code_name: str) -> Tuple[Any, np.ndarray]:
    """
    Load a code by name. Uses dynamic import to avoid circular dependencies.
    Returns (HGP_code, H_matrix)
    """
    available = get_available_codes()
    if code_name not in available:
        raise ValueError(f"Unknown code: {code_name}. Available: {list(available.keys())}")
    
    module_name, func_name = available[code_name]
    module = importlib.import_module(f"codes.{module_name}")
    getter_func = getattr(module, func_name)
    return getter_func()

def parse_args():
    parser = argparse.ArgumentParser(description="Run circuit noise simulations for reduced HGP codes")
    parser.add_argument("--shots", type=int, default=None,
                        help="Number of circuit samples to decode (e.g., 10000).")
    parser.add_argument("--decoder", type=str, default=None, choices=["OSD", "LSD", "Relay"],
                        help="Decoder to use: OSD, LSD, or Relay.")
    parser.add_argument("--codes", type=str, nargs='+', default=None,
                        help="Codes to simulate. If not specified, all available codes are used. "
                             "Use --list-codes to see available options.")
    parser.add_argument("--list-codes", action='store_true',
                        help="List all available codes and exit")
    parser.add_argument("--print-plots", action='store_true',
                        help="Plot existing data without running experiments")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of parallel threads for sampling (default: 4)")
    parser.add_argument("--resume-data", action='store_true',
                        help="Resume from existing data files if available")

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
    
    if args.list_codes:
        available = get_available_codes()
        print("Available codes:")
        for code_name in sorted(available.keys()):
            print(f"  - {code_name}")
        sys.exit(0)

    return args

def sample_HGP_circuit_noise(code, circ, rounds, p2, decoder, dec_params, shots):
    """
    Take a CNOT syndrome extraction circuit, and run codes
    
    :param code: code that we are sampling
    :param circ: circuit corresponding to the code we are sampling
    :param rounds: rounds of syndrome extraction
    :param p2: two-qubit error probability (used as error prior for the decoder)
    :return: logical error rate
    """   
    # Sample CNOT circuit and decode
    # params: (code, dec, circ, decoding params, p2, shots, rounds)
    print(f"\t\tSampling CNOT circuit and decoding via {decoder}...")
    failures = num_failures_BP(code, decoder, circ, dec_params, p2, shots, rounds)
    ler = failures/shots
    
    print(f"\t\tNumber of failed shots: {failures} out of {shots}")
    print(f"\t\t==> Logical error rate is approx. {ler:.4f}")
    
    return ler

def parallel_sample_wrapper(params_tuple):
    """Wrapper for parallel sampling that calls num_failures_BP"""
    code, dec, circ, dec_params, p2, shots, rounds = params_tuple
    return num_failures_BP(code, dec, circ, dec_params, p2, shots, rounds)
    
def total_sampling(p1, p2, p_spam, rounds, decoder, dec_params, shots,
                   unreduced_code, reduced_code, Hx1, Hx2, Hz1, Hz2, seed,
                   code_name=None, error_idx=None, results_for_code=None, metadata=None):
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
    :return: tuple of (unreduced_LER, reduced_random_LER, reduced_split_LER)
    """
    print(f"   *******Noise parameters: p1={p1:.3}, p2={p2:.3}, p_spam={p_spam:.3}*******")
  
    # Sample unreduced code with random syndrome extraction
    print("\tGenerating *unreduced* CNOT syndrome circuit with random syndrome extraction...")
    unreduced_random_circ = generate_full_circuit(unreduced_code, rounds, p1, p2, p_spam, seed)
    unreduced_random_LER = sample_HGP_circuit_noise(
        unreduced_code, unreduced_random_circ, rounds, p2, decoder, dec_params, shots
    )
    if results_for_code is not None and error_idx is not None and code_name is not None:
        unreduced_failures = int(unreduced_random_LER * shots)
        results_for_code["unreduced_random"][error_idx, 0] += unreduced_failures
        results_for_code["unreduced_random"][error_idx, 1] += shots
        save_data(results_for_code["unreduced_random"], code_name, "unreduced_random", metadata=metadata)

    # Sample reduced code with random syndrome extraction
    print("\tGenerating *reduced* CNOT syndrome circuit with random syndrome extraction...")
    reduced_random_circ = generate_full_circuit(reduced_code, rounds, p1, p2, p_spam, seed)
    reduced_random_LER = sample_HGP_circuit_noise(
        reduced_code, reduced_random_circ, rounds, p2, decoder, dec_params, shots
    )
    if results_for_code is not None and error_idx is not None and code_name is not None:
        reduced_random_failures = int(reduced_random_LER * shots)
        results_for_code["reduced_random"][error_idx, 0] += reduced_random_failures
        results_for_code["reduced_random"][error_idx, 1] += shots
        save_data(results_for_code["reduced_random"], code_name, "reduced_random", metadata=metadata)

    # Sample reduced code with split syndrome extraction
    print("\tGenerating *reduced* CNOT syndrome circuit with split syndrome extraction...")
    reduced_split_circ = generate_full_circuit_split(Hx1, Hx2, Hz1, Hz2, rounds, p1, p2, p_spam, seed)
    reduced_split_LER = sample_HGP_circuit_noise(
        reduced_code, reduced_split_circ, rounds, p2, decoder, dec_params, shots
    )
    if results_for_code is not None and error_idx is not None and code_name is not None:
        reduced_split_failures = int(reduced_split_LER * shots)
        results_for_code["reduced_split"][error_idx, 0] += reduced_split_failures
        results_for_code["reduced_split"][error_idx, 1] += shots
        save_data(results_for_code["reduced_split"], code_name, "reduced_split", metadata=metadata)
    
    return unreduced_random_LER, reduced_random_LER, reduced_split_LER

def weight_stats(H):
    """Compute weight statistics for a parity check matrix."""
    if hasattr(H, 'getnnz'):  # scipy sparse matrix
        rw = H.getnnz(axis=1)
        cw = H.getnnz(axis=0)
    else:  # numpy array
        rw = H.sum(axis=1)
        cw = H.sum(axis=0)
    return (rw.min(), rw.max(), round(float(rw.mean()), 3), cw.min(), cw.max(), round(float(cw.mean()), 3))

def get_data_filename(code_name: str, variant: str) -> Path:
    """Generate standard data filename for a code and variant."""
    return DATA_DIR / f"{code_name}_{variant}.npy"

def get_metadata_filename(code_name: str, variant: str) -> Path:
    """Generate standard metadata filename for a code and variant."""
    return DATA_DIR / f"{code_name}_{variant}_metadata.json"

def load_or_create_data(code_name: str, variant: str, num_error_points: int, decoder: Optional[str] = None, 
                        dec_params: Optional[List] = None, shots: Optional[int] = None, resume: bool = False):
    """
    Load existing data or create new array for storing results.
    Data format: [[failures, total_shots], ...]
    
    If existing data has different size or incompatible metadata, creates new array.
    """
    filename = get_data_filename(code_name, variant)
    
    if resume and filename.exists():
        try:
            prior_data = np.load(filename)
            if prior_data.shape[0] == num_error_points:
                # Check metadata compatibility
                existing_meta = load_metadata(code_name, variant)
                if decoder is not None and dec_params is not None:
                    if not metadata_compatible(existing_meta, decoder, dec_params):
                        print(f"\t\t{color_note('⚠ Metadata mismatch')} - starting fresh")
                        return np.zeros([num_error_points, 2], dtype=int)
                
                print(f"\t\tLoading prior data from {filename}")
                return prior_data
            else:
                print(f"\t\tPrior data size mismatch ({prior_data.shape[0]} vs {num_error_points}), starting fresh")
                return np.zeros([num_error_points, 2], dtype=int)
        except Exception as e:
            print(f"\t\tCould not load prior data: {e}, starting fresh")
            return np.zeros([num_error_points, 2], dtype=int)
    else:
        return np.zeros([num_error_points, 2], dtype=int)

def color_note(text: str) -> str:
    """Simple text helper (for formatting messages)."""
    return text

def save_data(data: np.ndarray, code_name: str, variant: str, metadata: Optional[Dict] = None):
    """Save simulation data and metadata to file."""
    filename = get_data_filename(code_name, variant)
    np.save(filename, data)
    print(f"\t\tData saved to {filename}")
    
    if metadata is not None:
        metadata_filename = get_metadata_filename(code_name, variant)
        # Convert tuples to lists for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, tuple):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            else:
                return obj
        
        serializable_metadata = make_serializable(metadata)
        with open(metadata_filename, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        print(f"\t\tMetadata saved to {metadata_filename}")

def load_metadata(code_name: str, variant: str) -> Optional[Dict]:
    """Load metadata for a data file if it exists."""
    metadata_filename = get_metadata_filename(code_name, variant)
    if metadata_filename.exists():
        try:
            with open(metadata_filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"\t\tWarning: Could not load metadata: {e}")
            return None
    return None

def metadata_compatible(existing_meta: Optional[Dict], decoder: str, dec_params: List) -> bool:
    """Check if existing metadata is compatible with current decoder/parameters."""
    if existing_meta is None:
        return False
    
    if existing_meta.get("decoder") != decoder:
        print(f"\t\tDecoder mismatch: existing={existing_meta.get('decoder')}, current={decoder}")
        return False
    
    if existing_meta.get("dec_params") != dec_params:
        print(f"\t\tDecoder parameters mismatch")
        return False
    
    return True

def plot_results(results, selected_codes, ps, decoder=None, dec_params=None, shots=None):
    """
    Generate and display/save plots of results.
    
    :param results: dict of {code_name: {variant: data_array}}
    :param selected_codes: list of code names to plot
    :param ps: list of error probabilities
    :param decoder: decoder name (optional, for filename generation)
    :param dec_params: decoder parameters (optional, for filename generation)
    :param shots: number of shots (optional, for filename generation)
    """
    # Create 3-column grid layout
    import math
    num_codes = len(selected_codes)
    num_cols = 3
    num_rows = math.ceil(num_codes / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), 
                              sharex=True, sharey=True)
    
    # Flatten axes array for easier indexing
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])
    
    p_array = np.array(ps, dtype=float)
    sort_idx = np.argsort(p_array)
    
    for idx, code_name in enumerate(selected_codes):
        ax = axes[idx]
        ax = cast(Axes, ax)
        
        # Calculate LER from accumulated failures and shots
        unred_data = results[code_name]["unreduced_random"]
        red_rand_data = results[code_name]["reduced_random"]
        red_split_data = results[code_name]["reduced_split"]
        
        ler_unred = np.divide(unred_data[:, 0], unred_data[:, 1], 
                              where=unred_data[:, 1] != 0, out=np.zeros_like(unred_data[:, 0], dtype=float))
        ler_red_rand = np.divide(red_rand_data[:, 0], red_rand_data[:, 1], 
                                 where=red_rand_data[:, 1] != 0, out=np.zeros_like(red_rand_data[:, 0], dtype=float))
        ler_red_split = np.divide(red_split_data[:, 0], red_split_data[:, 1], 
                                  where=red_split_data[:, 1] != 0, out=np.zeros_like(red_split_data[:, 0], dtype=float))
        
        p_plot = p_array[sort_idx]
        
        # Sort by error probability
        ler_unred_sorted = ler_unred[sort_idx]
        ler_red_rand_sorted = ler_red_rand[sort_idx]
        ler_red_split_sorted = ler_red_split[sort_idx]
        
        # Use total accumulated shots for standard error calculation
        std_unred = np.sqrt(ler_unred_sorted * (1 - ler_unred_sorted) / unred_data[sort_idx, 1])
        std_red_split = np.sqrt(ler_red_split_sorted * (1 - ler_red_split_sorted) / red_split_data[sort_idx, 1])
        std_red_rand = np.sqrt(ler_red_rand_sorted * (1 - ler_red_rand_sorted) / red_rand_data[sort_idx, 1])
        
        ax.errorbar(p_plot, ler_unred_sorted, yerr=std_unred, fmt='.-', capsize=3, alpha=1,
                    label='original, random SE')
        ax.errorbar(p_plot, ler_red_split_sorted, yerr=std_red_split, fmt='.-', capsize=3, alpha=1,
                    label='reduced, split SE')
        ax.errorbar(p_plot, ler_red_rand_sorted, yerr=std_red_rand, fmt='.-', capsize=3, alpha=1,
                    label='reduced, random SE')
        
        ax.set_title(code_name.replace('_', ' ').title(), fontsize=14)
        ax.set_xlabel(r'$p$', fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(8E-4, 1E-2)
        ax.grid(True, which='both', axis='both')
    
    # Hide extra subplots if number of codes is not a multiple of 3
    for idx in range(num_codes, len(axes)):
        axes[idx].axis('off')
    
    # Add legend and labels
    axes[0].set_ylabel(r'Logical failure probability', fontsize=16)
    axes[0].legend(fontsize=12, loc='lower right')
    
    plt.tight_layout()
    
    # Only save if decoder and shots info provided
    if decoder is not None and dec_params is not None and shots is not None:
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        p_min = min(ps)
        p_max = max(ps)
        codes_str = '-'.join(selected_codes)
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


def main():
    args = parse_args()
    
    # Handle --print-plots mode
    if args.print_plots:
        print(f"\n{'='*60}")
        print("Loading and plotting existing data...")
        print(f"{'='*60}")
        
        available_codes = get_available_codes()
        if args.codes:
            selected_codes = args.codes
            for code_name in selected_codes:
                if code_name not in available_codes:
                    print(f"Error: Unknown code '{code_name}'", file=sys.stderr)
                    print(f"Available codes: {list(available_codes.keys())}", file=sys.stderr)
                    sys.exit(1)
        else:
            selected_codes = sorted(available_codes.keys())
        
        ps = [7e-4, 7.5e-4, 8e-4, 8.5e-4, 9e-4, 9.5e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3]
        num_error_points = len(ps)
        
        results = {}
        for code in selected_codes:
            results[code] = {
                "unreduced_random": load_or_create_data(code, "unreduced_random", num_error_points, resume=True),
                "reduced_random": load_or_create_data(code, "reduced_random", num_error_points, resume=True),
                "reduced_split": load_or_create_data(code, "reduced_split", num_error_points, resume=True),
            }
        
        plot_results(results, selected_codes, ps)
        return
    
    # Validate required arguments for simulation mode
    if args.shots is None or args.decoder is None:
        print("Error: --shots and --decoder are required for simulation.", file=sys.stderr)
        print("Use --list-codes to see available codes.", file=sys.stderr)
        sys.exit(1)
    
    # Setup decoder parameters
    if args.decoder == "Relay":
        dec_params = [
            args.relay_gamma0,
            args.relay_pre_iter,
            args.relay_num_sets,
            args.relay_max_iter,
            tuple(args.relay_gamma_dist_interval),
            args.relay_stop_nconv]
    else:
        dec_params = [args.bp_max_iter, args.bp_order]
    
    # Determine which codes to run
    available_codes = get_available_codes()
    if args.codes:
        selected_codes = args.codes
        for code_name in selected_codes:
            if code_name not in available_codes:
                print(f"Error: Unknown code '{code_name}'", file=sys.stderr)
                print(f"Available codes: {list(available_codes.keys())}", file=sys.stderr)
                sys.exit(1)
    else:
        selected_codes = sorted(available_codes.keys())
    
    print(f"Selected codes: {selected_codes}")
    
    # Define error probability sweep
    ps = [7e-4, 7.5e-4, 8e-4, 8.5e-4, 9e-4, 9.5e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3]
    num_error_points = len(ps)
    
    # Results storage - format: {code: {variant: [[failures, total_shots], ...]}}
    results = {}
    for code in selected_codes:
        results[code] = {
            "unreduced_random": load_or_create_data(code, "unreduced_random", num_error_points, decoder=args.decoder, dec_params=dec_params, shots=args.shots, resume=args.resume_data),
            "reduced_random": load_or_create_data(code, "reduced_random", num_error_points, decoder=args.decoder, dec_params=dec_params, shots=args.shots, resume=args.resume_data),
            "reduced_split": load_or_create_data(code, "reduced_split", num_error_points, decoder=args.decoder, dec_params=dec_params, shots=args.shots, resume=args.resume_data),
        }
    
    # Process each code
    for code_name in selected_codes:
        print(f"\n{'='*60}")
        print(f"Sampling {code_name.upper()} Code")
        print(f"{'='*60}")
        
        try:
            print(f"\tGenerating HGP code from {code_name} LDPC code...")
            unreduced_code, H = load_code(code_name)
            print(f"\tGenerating reduced HGP...")
            Hx1, Hx2, Hz1, Hz2, reduced_code, _, _, d = get_reduced_code(unreduced_code, H)
            
            # Dimensions check
            assert Hx1.shape[1] == Hx2.shape[1] == Hz2.shape[1] == Hz1.shape[1]
            
            # Weight statistics
            print("\t--Format: (rmin, rmax, rmean, cmin, cmax, cmean)--")
            print("\t  unreduced hx:", weight_stats(unreduced_code.hx))
            print("\t  unreduced hz:", weight_stats(unreduced_code.hz))
            print("\t  reduced hx:  ", weight_stats(reduced_code.hx))
            print("\t  reduced hz:  ", weight_stats(reduced_code.hz))
            
        except Exception as e:
            print(f"Error generating code {code_name}: {e}", file=sys.stderr)
            continue
        
        # Sample for different error probabilities
        for i, p in enumerate(ps):
            print(f"\n\tError point {i+1}/{num_error_points}: p={p:.3e}")
            try:
                metadata = {
                    "decoder": args.decoder,
                    "dec_params": dec_params,
                    "shots": args.shots,
                    "error_probs": ps
                }
                unreduced_random_LER, reduced_random_LER, reduced_split_LER = total_sampling(
                    p1=p/10, p2=p, p_spam=p, rounds=d, decoder=args.decoder, dec_params=dec_params, 
                    shots=args.shots,
                    unreduced_code=unreduced_code, reduced_code=reduced_code,
                    Hx1=Hx1, Hx2=Hx2, Hz1=Hz1, Hz2=Hz2, seed=i+1,
                    code_name=code_name, error_idx=i, results_for_code=results[code_name], metadata=metadata
                )
                
            except Exception as e:
                print(f"Error at error point {i}: {e}", file=sys.stderr)
                continue
    
    # Plot results
    print(f"\n{'='*60}")
    print("Generating plots...")
    print(f"{'='*60}")
    
    plot_results(results, selected_codes, ps, decoder=args.decoder, dec_params=dec_params, shots=args.shots)



if __name__ == '__main__':
    main()
