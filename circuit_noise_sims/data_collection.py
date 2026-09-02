import argparse
import hashlib
import os
import queue
import sys
import time
from multiprocessing import Manager, Pool
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
import numpy as np

if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

if "MPLBACKEND" not in os.environ and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")

from functions.H_to_CNOT_circuit import (
    generate_full_circuit,
    generate_full_circuit_cardinal,
    generate_full_circuit_split,
)
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


def _fmt_secs(sec: float) -> str:
    if not np.isfinite(sec):
        return "inf"
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def derive_sampler_seed(base_seed: int, *labels: Any) -> int:
    """Derive a stable unsigned 64-bit Stim seed without Python's hash()."""

    payload = "|".join([str(int(base_seed)), *(str(label) for label in labels)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run noisy simulations and append data tables.")
    parser.add_argument("--shots", type=int, default=None,
                        help="Number of circuit samples to decode.")
    parser.add_argument("--decoder", type=str, default=None, choices=["OSD", "LSD", "Relay"],
                        help="Decoder to use: OSD, LSD, or Relay.")
    parser.add_argument("--codes", type=str, nargs="+", default=None,
                        help="Codes to simulate. (DEFAULT: all available codes)")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help=(
            "Circuit variants to simulate. Choices: "
            f"{', '.join(VARIANTS)}. (DEFAULT: all)"
        ),
    )
    parser.add_argument("--list-codes", action="store_true",
                        help="List available codes and exit.")
    parser.add_argument("--processes", type=int, default=4,
                        help="Number of parallel worker processes for sampling. (DEFAULT: 4)")
    parser.add_argument("--sampler-seed", type=int, default=0,
                        help="Base seed for deterministic Stim sampling. (DEFAULT: 0)")
    parser.add_argument("--schedule-seed", type=int, default=1,
                        help=("Fixed seed for reduced-circuit layer shuffles at every p value. "
                              "The cardinal baseline uses seed 0. (DEFAULT: 1)"))
    parser.add_argument("--p-values", nargs="+", default=None,
                        help=(
                            "P-value selection mode. Use comma-separated explicit values "
                            "(for example: --p-values 5e-4,1e-3,2e-3) or a space-separated range "
                            "triplet <low> <high> <step> "
                            "(for example: --p-values 5e-4 1e-2 5e-4) "
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
    if args.sampler_seed < 0:
        raise ValueError("--sampler-seed must be nonnegative")
    if args.schedule_seed < 0:
        raise ValueError("--schedule-seed must be nonnegative")

    return args


def sample_hgp_circuit_noise(code: Any, circ: Any, rounds: int, p2: float, decoder: str,
                             dec_params: List[Any], shots: int, processes: int = 1,
                             sampler_seed: int = 0) -> int:
    """Sample circuit outcomes and return number of logical failures."""
    print(f"\t\tSampling CNOT circuit and decoding via {decoder}...")

    workers = max(1, min(int(processes), int(shots)))
    if workers == 1:
        failures = num_failures_BP(
            code, decoder, circ, dec_params, p2, shots, rounds,
            sampler_seed=sampler_seed,
        )
    else:
        base, rem = divmod(shots, workers)
        shot_chunks = [base + (1 if i < rem else 0) for i in range(workers)]
        worker_specs = [(i + 1, s) for i, s in enumerate(shot_chunks) if s > 0]
        print(f"\t\tRunning in parallel with {len(worker_specs)} worker processes...")

        if not sys.stdout.isatty(): # if stdout is not a terminal
            params = [
                (
                    code, decoder, circ, dec_params, p2, s, rounds, True,
                    derive_sampler_seed(sampler_seed, worker_id), worker_id,
                )
                for worker_id, s in worker_specs
            ]
            with Pool(processes=len(params)) as pool:
                failures = int(np.sum(pool.starmap(num_failures_BP, params)))
        else:
            progress_t0 = time.perf_counter()
            worker_state: Dict[int, Dict[str, float]] = {
                worker_id: {"shot_num": 0.0, "shots": float(s), "num_failures": 0.0, "done_elapsed": float("nan")}
                for worker_id, s in worker_specs
            }

            def render_worker_lines() -> List[str]:
                lines = []
                now = time.perf_counter()
                elapsed = max(0.0, now - progress_t0)
                for worker_id, _ in worker_specs:
                    state = worker_state[worker_id]
                    shot_num = int(state["shot_num"])
                    total = int(state["shots"])
                    num_failures = int(state["num_failures"])
                    worker_elapsed = state["done_elapsed"] if np.isfinite(state["done_elapsed"]) else elapsed
                    rate = (shot_num / worker_elapsed) if worker_elapsed > 0 else 0.0
                    eta = ((total - shot_num) / rate) if rate > 0 else float("inf")
                    lines.append(
                        f"\t\tWorker {worker_id}: Shot {shot_num} out of {total} "
                        f"; {num_failures} failed so far "
                        f"(elapsed {_fmt_secs(worker_elapsed)}, eta {_fmt_secs(eta)})"
                    )
                return lines

            def redraw_worker_lines(initial: bool = False) -> None:
                lines = render_worker_lines()
                if initial:
                    print("\n".join(lines), flush=True)
                    return

                if lines:
                    sys.stdout.write(f"\x1b[{len(lines)}F")
                    for line in lines:
                        sys.stdout.write("\x1b[2K")
                        sys.stdout.write(line)
                        sys.stdout.write("\n")
                    sys.stdout.flush()

            def apply_progress_msg(msg: Dict[str, float]) -> None:
                worker_id = int(msg["worker_id"])
                worker_state[worker_id] = {
                    "shot_num": float(msg["shot_num"]),
                    "shots": float(msg["shots"]),
                    "num_failures": float(msg.get("num_failures", worker_state[worker_id]["num_failures"])),
                    "done_elapsed": worker_state[worker_id]["done_elapsed"],
                }

                if (
                    worker_state[worker_id]["shot_num"] >= worker_state[worker_id]["shots"]
                    and not np.isfinite(worker_state[worker_id]["done_elapsed"])
                ):
                    worker_state[worker_id]["done_elapsed"] = max(0.0, time.perf_counter() - progress_t0)

            redraw_worker_lines(initial=True)

            with Manager() as manager:
                progress_queue = manager.Queue()
                params = [
                    (
                        code, decoder, circ, dec_params, p2, s, rounds, False,
                        derive_sampler_seed(sampler_seed, worker_id), worker_id,
                        progress_queue,
                    )
                    for worker_id, s in worker_specs
                ]

                with Pool(processes=len(params)) as pool:
                    result = pool.starmap_async(num_failures_BP, params)

                    last_redraw = 0.0
                    max_msgs_per_cycle = 64
                    while not result.ready():
                        updated = False
                        try:
                            msg = progress_queue.get(timeout=0.1)
                        except queue.Empty:
                            pass
                        else:
                            apply_progress_msg(msg)
                            updated = True

                            drained = 0
                            while drained < max_msgs_per_cycle:
                                try:
                                    msg = progress_queue.get_nowait()
                                except queue.Empty:
                                    break
                                apply_progress_msg(msg)
                                drained += 1

                        now = time.perf_counter()
                        if updated or (now - last_redraw) >= 1.0:
                            redraw_worker_lines()
                            last_redraw = now

                    while True:
                        try:
                            msg = progress_queue.get_nowait()
                        except queue.Empty:
                            break

                        apply_progress_msg(msg)

                    redraw_worker_lines()
                    failures = int(np.sum(result.get()))

    ler = failures / shots
    print(f"\t\tNumber of failed shots: {failures} out of {shots}")
    print(f"\t\t==> Logical error rate is approx. {ler:.4f}")
    return int(failures)


def run_one_probability(
    p: float,
    shots: int,
    processes: int,
    decoder: str,
    dec_params: List[Any],
    rounds: int,
    seed: int,
    sampler_seed: int,
    variants: Sequence[str],
    unreduced_code: Any,
    h: Any,
    reduced_code: Any,
    hx1: Any,
    hx2: Any,
    hx3: Any,
    hz1: Any,
    hz2: Any,
    hz3: Any,
) -> Dict[str, int]:
    print(f"\n\t*******Noise parameters: p1={p/10:.3e}, p2={p:.3e}, p_spam={p:.3e}*******")
    failures: Dict[str, int] = {}

    if "unreduced_cardinal" in variants:
        print("\tGenerating *unreduced* cardinal syndrome circuit (fixed schedule seed 0)...")
        cardinal_circ = generate_full_circuit_cardinal(
            unreduced_code, h, h, rounds, (p / 10, p, p), seed=0,
        )
        failures["unreduced_cardinal"] = sample_hgp_circuit_noise(
            unreduced_code,
            cardinal_circ,
            rounds,
            p,
            decoder,
            dec_params,
            shots,
            processes=processes,
            sampler_seed=derive_sampler_seed(sampler_seed, "unreduced_cardinal"),
        )

    if "unreduced_random" in variants:
        print("\tGenerating *unreduced* syndrome circuit with shuffled edge-color layers...")
        unreduced_random_circ = generate_full_circuit(
            unreduced_code, rounds, (p / 10, p, p), seed,
        )
        failures["unreduced_random"] = sample_hgp_circuit_noise(
            unreduced_code,
            unreduced_random_circ,
            rounds,
            p,
            decoder,
            dec_params,
            shots,
            processes=processes,
            sampler_seed=derive_sampler_seed(sampler_seed, "unreduced_random"),
        )

    if "reduced_random" in variants:
        print("\tGenerating *reduced* syndrome circuit with shuffled edge-color layers...")
        reduced_random_circ = generate_full_circuit(
            reduced_code, rounds, (p / 10, p, p), seed,
        )
        failures["reduced_random"] = sample_hgp_circuit_noise(
            reduced_code,
            reduced_random_circ,
            rounds,
            p,
            decoder,
            dec_params,
            shots,
            processes=processes,
            sampler_seed=derive_sampler_seed(sampler_seed, "reduced_random"),
        )

    if "reduced_split" in variants:
        print("\tGenerating *reduced* syndrome circuit with split extraction...")
        reduced_split_circ = generate_full_circuit_split(
            hx1, hx2, hx3, hz1, hz2, hz3, rounds, (p / 10, p, p), seed,
            code=reduced_code,
        )
        failures["reduced_split"] = sample_hgp_circuit_noise(
            reduced_code,
            reduced_split_circ,
            rounds,
            p,
            decoder,
            dec_params,
            shots,
            processes=processes,
            sampler_seed=derive_sampler_seed(sampler_seed, "reduced_split"),
        )

    return failures


def main() -> None:
    args = parse_args()
    dec_params = parse_decoder_params(args)
    p_values = parse_p_values(args.p_values)

    available_codes = get_available_codes()
    selected_codes = sorted(available_codes.keys()) if args.codes is None else list(args.codes)
    if args.codes is not None:
        validate_selected_codes(selected_codes)
    selected_variants = list(dict.fromkeys(args.variants))

    print(f"{'=' * 60}")
    print(f"Selected codes: {selected_codes}")
    print(f"Selected variants: {selected_variants}")
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
            hx1, hx2, hx3, hz1, hz2, hz3, reduced_code, _, _, d = get_reduced_code(unreduced_code, h)

            assert hx1.shape[1] == hx2.shape[1] == hx3.shape[1] == hz1.shape[1] == hz2.shape[1] == hz3.shape[1]

            print("\t--Column and row weights ; format: (rmin, rmax, rmean, cmin, cmax, cmean)--")
            print("\t  unreduced hx:", weight_stats(unreduced_code.hx))
            print("\t  unreduced hz:", weight_stats(unreduced_code.hz))
            print("\t  reduced hx:  ", weight_stats(reduced_code.hx))
            print("\t  reduced hz:  ", weight_stats(reduced_code.hz))
        except Exception as exc:
            print(f"Error generating code {code_name}: {exc}", file=sys.stderr)
            continue

        results: Dict[str, np.ndarray] = {}
        for variant in selected_variants:
            path = get_data_path(code_name, variant, args.decoder, dec_params)
            try:
                results[variant] = load_data_table(path)
            except Exception as exc:
                print(f"\tWarning: Could not read existing data at {path}: {exc}", file=sys.stderr)
                results[variant] = np.zeros((0, 3), dtype=float)

        for idx, p in enumerate(p_values):
            print(f"\n\tError point {idx + 1}/{len(p_values)}: p={p:.3e}")
            point_sampler_seed = derive_sampler_seed(
                args.sampler_seed,
                code_name,
                format(float(p), ".17g"),
            )
            try:
                failure_counts = run_one_probability(
                    p=p,
                    shots=args.shots,
                    processes=args.processes,
                    decoder=args.decoder,
                    dec_params=dec_params,
                    rounds=d,
                    seed=args.schedule_seed,
                    sampler_seed=point_sampler_seed,
                    variants=selected_variants,
                    unreduced_code=unreduced_code,
                    h=h,
                    reduced_code=reduced_code,
                    hx1=hx1,
                    hx2=hx2,
                    hx3=hx3,
                    hz1=hz1,
                    hz2=hz2,
                    hz3=hz3,
                )

                for variant in selected_variants:
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
