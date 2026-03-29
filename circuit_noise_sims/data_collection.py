import argparse
import os
import sys
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np

if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

if "MPLBACKEND" not in os.environ and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")

from functions.H_to_CNOT_circuit import generate_full_circuit, generate_full_circuit_split
from functions.decoding import num_failures_BP
from functions.reduction_funcs import get_reduced_code
from functions.sim_common import (
    VARIANTS,
    append_result_row,
    get_available_codes,
    get_data_path,
    load_code,
    load_data_table,
    parse_decoder_params,
    parse_p_values,
    save_data_table,
    validate_selected_codes,
    weight_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run noisy simulations and append data tables.")
    parser.add_argument("--shots", type=int, default=None,
                        help="Number of circuit samples to decode.")
    parser.add_argument("--decoder", type=str, default=None, choices=["OSD", "LSD", "Relay"],
                        help="Decoder to use: OSD, LSD, or Relay.")
    parser.add_argument("--codes", type=str, nargs="+", default=None,
                        help="Codes to simulate. (DEFAULT: all available codes)")
    parser.add_argument("--list-codes", action="store_true",
                        help="List available codes and exit.")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of parallel worker processes for sampling. (DEFAULT: 4)")
    parser.add_argument("--p-values", nargs="+", default=None,
                        help=(
                            "Specific p values to simulate. Accepts space-separated and/or comma-separated values, "
                            "for example: --p-values 5e-4 1e-3 2e-3 or --p-values 5e-4,1e-3,2e-3 "
                            "(DEFAULT: 0.0005 to 0.01 in steps of 0.0005)"
                        ))

    parser.add_argument("--bp-max-iter", type=int, default=80,
                        help="Maximum BP iterations for OSD/LSD (DEFAULT: 80)")
    parser.add_argument("--bp-max-order", "--bp-order", dest="bp_order", type=int, default=5,
                        help="OSD/LSD order (DEFAULT: 5)")

    parser.add_argument("--relay-gamma0", type=float, default=0.65,
                        help="Uniform memory weight for first Relay ensemble. (DEFAULT: 0.65)")
    parser.add_argument("--relay-pre-iter", type=int, default=80,
                        help="Max Relay iterations in first ensemble. (DEFAULT: 80)")
    parser.add_argument("--relay-num-sets", type=int, default=100,
                        help="Number of Relay ensemble elements. (DEFAULT: 100)")
    parser.add_argument("--relay-max-iter", type=int, default=60,
                        help="Max BP iterations per Relay ensemble. (DEFAULT: 60)")
    parser.add_argument("--relay-gamma-dist-interval", type=float, nargs=2,
                        default=(-0.24, 0.66), metavar=("LOW", "HIGH"),
                        help="Uniform range for disordered memory weight. (DEFAULT: -0.24 0.66)")
    parser.add_argument("--relay-stop-nconv", type=int, default=5,
                        help="Number of Relay solutions to find before stopping. (DEFAULT: 5)")

    args = parser.parse_args()

    if args.list_codes:
        available = get_available_codes()
        print("Available codes:")
        for code_name in sorted(available.keys()):
            print(f"  - {code_name}")
        sys.exit(0)

    if args.shots is None or args.decoder is None:
        parser.error("the following arguments are required: --shots, --decoder")

    if args.shots <= 0:
        raise ValueError("--shots must be positive")

    return args


def sample_hgp_circuit_noise(code: Any, circ: Any, rounds: int, p2: float, decoder: str,
                             dec_params: List[Any], shots: int, threads: int = 1) -> int:
    """Sample circuit outcomes and return number of logical failures."""
    print(f"\t\tSampling CNOT circuit and decoding via {decoder}...")

    workers = max(1, min(int(threads), int(shots)))
    if workers == 1:
        failures = num_failures_BP(code, decoder, circ, dec_params, p2, shots, rounds)
    else:
        base, rem = divmod(shots, workers)
        shot_chunks = [base + (1 if i < rem else 0) for i in range(workers)]
        params = [
            (code, decoder, circ, dec_params, p2, s, rounds, True, None, i + 1)
            for i, s in enumerate(shot_chunks)
            if s > 0
        ]
        print(f"\t\tRunning in parallel with {len(params)} worker processes...")
        with Pool(processes=len(params)) as pool:
            failures = int(np.sum(pool.starmap(num_failures_BP, params)))

    ler = failures / shots
    print(f"\t\tNumber of failed shots: {failures} out of {shots}")
    print(f"\t\t==> Logical error rate is approx. {ler:.4f}")
    return int(failures)


def run_one_probability(
    p: float,
    shots: int,
    threads: int,
    decoder: str,
    dec_params: List[Any],
    rounds: int,
    seed: int,
    unreduced_code: Any,
    reduced_code: Any,
    hx1: Any,
    hx2: Any,
    hz1: Any,
    hz2: Any,
) -> Dict[str, int]:
    print(f"\n\t*******Noise parameters: p1={p/10:.3e}, p2={p:.3e}, p_spam={p:.3e}*******")

    print("\tGenerating *unreduced* CNOT syndrome circuit with random syndrome extraction...")
    unreduced_random_circ = generate_full_circuit(unreduced_code, rounds, p / 10, p, p, seed)
    unreduced_failures = sample_hgp_circuit_noise(
        unreduced_code,
        unreduced_random_circ,
        rounds,
        p,
        decoder,
        dec_params,
        shots,
        threads=threads,
    )

    print("\tGenerating *reduced* CNOT syndrome circuit with random syndrome extraction...")
    reduced_random_circ = generate_full_circuit(reduced_code, rounds, p / 10, p, p, seed)
    reduced_random_failures = sample_hgp_circuit_noise(
        reduced_code,
        reduced_random_circ,
        rounds,
        p,
        decoder,
        dec_params,
        shots,
        threads=threads,
    )

    print("\tGenerating *reduced* CNOT syndrome circuit with split syndrome extraction...")
    reduced_split_circ = generate_full_circuit_split(hx1, hx2, hz1, hz2, rounds, p / 10, p, p, seed)
    reduced_split_failures = sample_hgp_circuit_noise(
        reduced_code,
        reduced_split_circ,
        rounds,
        p,
        decoder,
        dec_params,
        shots,
        threads=threads,
    )

    return {
        "unreduced_random": unreduced_failures,
        "reduced_random": reduced_random_failures,
        "reduced_split": reduced_split_failures,
    }


def main() -> None:
    args = parse_args()
    dec_params = parse_decoder_params(args)
    p_values = parse_p_values(args.p_values)

    available_codes = get_available_codes()
    selected_codes = sorted(available_codes.keys()) if args.codes is None else list(args.codes)
    if args.codes is not None:
        validate_selected_codes(selected_codes)

    print(f"{'=' * 60}")
    print(f"Selected codes: {selected_codes}")
    print(f"Using p values: {p_values}")
    print(f"{'=' * 60}\n\n")

    for code_name in selected_codes:
        print(f"\n{'-' * 60}")
        print(f"Sampling {code_name.upper()} Code")
        print(f"{'-' * 60}")

        try:
            print("\tGenerating HGP code...")
            unreduced_code, h = load_code(code_name)
            print("\tGenerating reduced HGP...")
            hx1, hx2, hz1, hz2, reduced_code, _, _, d = get_reduced_code(unreduced_code, h)

            assert hx1.shape[1] == hx2.shape[1] == hz2.shape[1] == hz1.shape[1]

            print("\t--Format: (rmin, rmax, rmean, cmin, cmax, cmean)--")
            print("\t  unreduced hx:", weight_stats(unreduced_code.hx))
            print("\t  unreduced hz:", weight_stats(unreduced_code.hz))
            print("\t  reduced hx:  ", weight_stats(reduced_code.hx))
            print("\t  reduced hz:  ", weight_stats(reduced_code.hz))
        except Exception as exc:
            print(f"Error generating code {code_name}: {exc}", file=sys.stderr)
            continue

        results: Dict[str, np.ndarray] = {}
        for variant in VARIANTS:
            path = get_data_path(code_name, variant, args.decoder, dec_params)
            try:
                results[variant] = load_data_table(path)
            except Exception as exc:
                print(f"\tWarning: Could not read existing data at {path}: {exc}", file=sys.stderr)
                results[variant] = np.zeros((0, 3), dtype=float)

        for idx, p in enumerate(p_values):
            print(f"\n\tError point {idx + 1}/{len(p_values)}: p={p:.3e}")
            try:
                failure_counts = run_one_probability(
                    p=p,
                    shots=args.shots,
                    threads=args.threads,
                    decoder=args.decoder,
                    dec_params=dec_params,
                    rounds=d,
                    seed=idx + 1,
                    unreduced_code=unreduced_code,
                    reduced_code=reduced_code,
                    hx1=hx1,
                    hx2=hx2,
                    hz1=hz1,
                    hz2=hz2,
                )

                for variant in VARIANTS:
                    results[variant] = append_result_row(
                        results[variant],
                        p=p,
                        failures=failure_counts[variant],
                        shots=args.shots,
                    )
                    out_path = get_data_path(code_name, variant, args.decoder, dec_params)
                    save_data_table(out_path, results[variant])
                    print(f"\tSaved {variant} data to {out_path}")

            except Exception as exc:
                print(f"Error at p={p:.3e}: {exc}", file=sys.stderr)
                continue


if __name__ == "__main__":
    main()
