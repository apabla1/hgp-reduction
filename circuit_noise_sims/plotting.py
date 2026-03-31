import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes

if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

if "MPLBACKEND" not in os.environ and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")

from functions.sim_common import (
    VARIANTS,
    decoder_config_tag,
    get_available_codes,
    get_data_path,
    load_data_table,
    parse_decoder_params,
    select_rows_in_range,
    validate_selected_codes,
)

rcParams["font.size"] = 14
rcParams["text.usetex"] = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot existing noisy-simulation data.")
    parser.add_argument("--decoder", type=str, default=None, choices=["OSD", "LSD", "Relay"],
                        help="Decoder used to generate data: Choose \"OSD\", \"LSD\", or \"Relay\".")
    parser.add_argument("--codes", type=str, nargs="+", default=None,
                        help="Codes to plot. (DEFAULT: all available codes).")
    parser.add_argument("--list-codes", action="store_true",
                        help="List available codes and exit.")
    parser.add_argument("--p-min", type=float, default=None,
                        help="Minimum p to include on plots (inclusive). (DEFAULT: minimum value)")
    parser.add_argument("--p-max", type=float, default=None,
                        help="Maximum p to include on plots (inclusive). (DEFAULT: maximum value)")

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

    if args.decoder is None:
        parser.error("the following arguments are required: --decoder")

    if args.p_min is not None and args.p_max is not None and args.p_min > args.p_max:
        raise ValueError("--p-min must be <= --p-max")

    return args


def _safe_binom_std(ler_values: np.ndarray, shot_counts: np.ndarray) -> np.ndarray:
    var = ler_values * (1 - ler_values)
    return np.sqrt(np.divide(var, shot_counts, where=shot_counts > 0, out=np.zeros_like(var, dtype=float)))


def _format_p_for_path(value: float) -> str:
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    if text.startswith("."):
        return f"0{text}"
    if text == "":
        return "0"
    return text


def _load_variant_data(code_name: str, decoder: str, dec_params: List[Any],
                       p_min: Optional[float], p_max: Optional[float]) -> Dict[str, np.ndarray]:
    loaded: Dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        path = get_data_path(code_name, variant, decoder, dec_params)
        if not path.exists():
            loaded[variant] = np.zeros((0, 3), dtype=float)
            continue
        data = load_data_table(path)
        loaded[variant] = select_rows_in_range(data, p_min, p_max)
    return loaded


def _collect_available_p_values(selected_codes: List[str], decoder: str, dec_params: List[Any]) -> np.ndarray:
    p_values: List[float] = []
    for code_name in selected_codes:
        for variant in VARIANTS:
            path = get_data_path(code_name, variant, decoder, dec_params)
            if not path.exists():
                continue
            data = load_data_table(path)
            if data.size > 0:
                p_values.extend(data[:, 0].tolist())

    if not p_values:
        return np.array([], dtype=float)
    return np.unique(np.array(p_values, dtype=float))


def _plot_one_variant(ax: Axes, data: np.ndarray, label: str) -> None:
    if data.size == 0:
        return

    ordered = data[np.argsort(data[:, 0])]
    p_vals = ordered[:, 0]
    failures = ordered[:, 1]
    shots = ordered[:, 2]
    lers = np.divide(failures, shots, where=shots > 0, out=np.zeros_like(failures, dtype=float))
    stds = _safe_binom_std(lers, shots)

    ax.errorbar(p_vals, lers, yerr=stds, fmt=".-", capsize=3, alpha=1, label=label)


def plot_results(results: Dict[str, Dict[str, np.ndarray]], selected_codes: List[str],
                 decoder: str, dec_params: List[Any], p_min: Optional[float], p_max: Optional[float]) -> None:
    import math

    if len(selected_codes) == 0:
        print("No codes selected for plotting.")
        return

    num_codes = len(selected_codes)
    num_cols = 3
    num_rows = math.ceil(num_codes / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), sharex=True, sharey=True)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    global_p_values_list: List[np.ndarray] = []
    for code_name in selected_codes:
        for variant in VARIANTS:
            variant_data = results[code_name][variant]
            if variant_data.size > 0:
                global_p_values_list.append(variant_data[:, 0])

    global_p_values = (
        np.unique(np.sort(np.concatenate(global_p_values_list)))
        if global_p_values_list
        else np.array([], dtype=float)
    )

    for idx, code_name in enumerate(selected_codes):
        ax = axes[idx]

        _plot_one_variant(ax, results[code_name]["unreduced_random"], "original, random SE")
        _plot_one_variant(ax, results[code_name]["reduced_split"], "reduced, split SE")
        _plot_one_variant(ax, results[code_name]["reduced_random"], "reduced, random SE")

        ax.set_title(code_name.replace("_", " ").title(), fontsize=14)
        ax.set_xlabel(r"$p$", fontsize=16)
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.grid(True, which="both", axis="both")

    if global_p_values.size > 0:
        if p_min is not None and p_max is not None:
            x_left, x_right = p_min, p_max
        elif p_min is not None:
            x_left, x_right = p_min, float(np.max(global_p_values))
        elif p_max is not None:
            x_left, x_right = float(np.min(global_p_values)), p_max
        else:
            x_left, x_right = float(np.min(global_p_values)), float(np.max(global_p_values))

        for ax in axes[:num_codes]:
            ax.set_xlim(x_left, x_right)
            ax.set_xticks(global_p_values)
            ax.set_xticklabels([f"{p:.4f}" for p in global_p_values], rotation=45, ha="right")
            ax.tick_params(axis="x", which="both", labelbottom=True)

    for idx in range(num_codes, len(axes)):
        axes[idx].axis("off")

    axes[0].set_ylabel(r"Logical failure probability", fontsize=16)
    axes[0].legend(fontsize=12, loc="lower right")

    plt.tight_layout()

    plotted_p_values: List[float] = []
    for code_name in selected_codes:
        for variant in VARIANTS:
            data = results[code_name][variant]
            if data.size > 0:
                plotted_p_values.extend(data[:, 0].tolist())

    if not plotted_p_values:
        print("No p values found for saved plot path.", file=sys.stderr)
        return

    tag = decoder_config_tag(decoder, dec_params)
    p_low = p_min if p_min is not None else min(plotted_p_values)
    p_high = p_max if p_max is not None else max(plotted_p_values)
    range_label = f"{_format_p_for_path(float(p_low))}to{_format_p_for_path(float(p_high))}"

    plots_dir = Path(__file__).parent / "plots" / tag / range_label
    plots_dir.mkdir(parents=True, exist_ok=True)

    code_label = "-".join(selected_codes)
    out_path = plots_dir / f"{code_label}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    plt.show()


def main() -> None:
    args = parse_args()
    dec_params = parse_decoder_params(args)
    
    if args.decoder == "Relay":
        params_str = f"relay_gamma0={dec_params[0]}, relay_pre_iter={dec_params[1]}, relay_num_sets={dec_params[2]}, relay_max_iter={dec_params[3]}, relay_gamma_dist_interval={dec_params[4]}, relay_stop_nconv={dec_params[5]}"
    else:  # OSD or LSD
        params_str = f"bp_max_iter={dec_params[0]}, bp_order={dec_params[1]}"
    
    print(f"Plotting data with {args.decoder} decoder with parameters {params_str}")

    available_codes = get_available_codes()
    if args.codes:
        selected_codes = list(args.codes)
        validate_selected_codes(selected_codes)
    else:
        selected_codes = sorted(available_codes.keys())
    
    print(f"Plotting codes {', '.join(selected_codes)}")

    available_p_values = _collect_available_p_values(selected_codes, args.decoder, dec_params)
    if (args.p_min is None or args.p_max is None) and available_p_values.size == 0:
        raise ValueError("Could not infer --p-min/--p-max defaults because no saved data was found.")

    resolved_p_min = float(np.min(available_p_values)) if args.p_min is None else float(args.p_min)
    resolved_p_max = float(np.max(available_p_values)) if args.p_max is None else float(args.p_max)

    if resolved_p_min > resolved_p_max:
        raise ValueError("--p-min must be <= --p-max")
    if np.isclose(resolved_p_min, resolved_p_max, atol=1e-15, rtol=0.0):
        raise ValueError("Resolved p range is degenerate: --p-min must be different from --p-max")

    print(f"Using p range: [{resolved_p_min:.6g}, {resolved_p_max:.6g}]")

    results: Dict[str, Dict[str, np.ndarray]] = {}
    plotted_codes: List[str] = []

    for code_name in selected_codes:
        code_data = _load_variant_data(code_name, args.decoder, dec_params, resolved_p_min, resolved_p_max)
        has_any = any(code_data[variant].size > 0 for variant in VARIANTS)
        if not has_any:
            print(
                f"Skipping {code_name}: no matching data found for decoder/config and p range.",
                file=sys.stderr,
            )
            continue
        results[code_name] = code_data
        plotted_codes.append(code_name)

    if not plotted_codes:
        print("No data found to plot for the requested selection.", file=sys.stderr)
        sys.exit(1)

    plot_results(results, plotted_codes, args.decoder, dec_params, resolved_p_min, resolved_p_max)


if __name__ == "__main__":
    main()
