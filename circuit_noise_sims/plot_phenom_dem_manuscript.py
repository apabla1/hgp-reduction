#!/usr/bin/env python3
"""Plot the corrected phenomenological-DEM manuscript data in the April style."""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from functions.phenom_dem_config import DECODER_CONFIG


@dataclass(frozen=True)
class CurveStyle:
    variant: str
    label: str
    color: str


MAIN_CURVES: Tuple[CurveStyle, ...] = (
    CurveStyle(
        variant="unreduced_cardinal",
        label="original, balanced-cardinal SE",
        color="#1f77b4",
    ),
    CurveStyle(
        variant="reduced_split",
        label="reduced, split SE",
        color="#ff7f0e",
    ),
    CurveStyle(
        variant="reduced_random",
        label="reduced, unsplit SE",
        color="#2ca02c",
    ),
)

QC24_CURVE = CurveStyle(
    variant="reduced_split",
    label=r"$[[684,36,10]]$ reduced, split SE",
    color="#d62728",
)

DEFAULT_QC_FILENAME = "_20,5,9__LERs.pdf"
DEFAULT_HEAWOOD_FILENAME = "Heawood_LERs.pdf"

# Match the April manuscript panels: the QC grid begins at 1e-3 and the
# Heawood grid begins at 5e-4, with a small logarithmic margin on each left
# endpoint so that markers and error-bar caps are not clipped.
QC_X_LIMITS = (8e-4, 1e-2)
HEAWOOD_X_LIMITS = (4e-4, 1e-2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the audited flattened-DEM-prior phenomenological results in "
            "the style of the April manuscript figures."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help=(
            "The isolated collection root or its single decoder-configuration "
            "directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "figures"
            / "corrected_schedule"
        ),
        help="Directory for the two PDF files.",
    )
    parser.add_argument(
        "--use-tex",
        action="store_true",
        help="Use the system LaTeX installation, as in the April figures.",
    )
    parser.add_argument(
        "--allow-unverified-data",
        action="store_true",
        help="Skip the immutable-ledger audit. Never use for final figures.",
    )
    return parser.parse_args(argv)


def resolve_decoder_root(data_root: Path) -> Path:
    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root is not a directory: {root}")
    if root.name == DECODER_CONFIG:
        return root
    candidate = root / DECODER_CONFIG
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Could not find decoder configuration {DECODER_CONFIG!r} below {root}"
    )


def load_result_table(path: Path) -> np.ndarray:
    raw = np.load(path, allow_pickle=False)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"{path} has shape {raw.shape}; expected (N, 3)")
    data = raw.astype(float, copy=False)
    if data.size == 0:
        raise ValueError(f"{path} is empty")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path} contains non-finite values")

    p_values, failures, shots = data.T
    if np.any(p_values <= 0) or np.any(p_values > 1):
        raise ValueError(f"{path} contains p outside (0, 1]")
    if np.any(shots <= 0):
        raise ValueError(f"{path} contains non-positive shot counts")
    if np.any(failures < 0) or np.any(failures > shots):
        raise ValueError(f"{path} contains invalid failure counts")
    if not np.allclose(failures, np.rint(failures), rtol=0, atol=1e-9):
        raise ValueError(f"{path} contains non-integer failure counts")
    if not np.allclose(shots, np.rint(shots), rtol=0, atol=1e-9):
        raise ValueError(f"{path} contains non-integer shot counts")
    if len(np.unique(p_values)) != len(p_values):
        raise ValueError(f"{path} contains duplicate p values")
    return data[np.argsort(p_values)]


def binomial_standard_error(failures: np.ndarray, shots: np.ndarray) -> np.ndarray:
    rates = failures / shots
    return np.sqrt(rates * (1.0 - rates) / shots)


def _plot_curve(ax: Axes, data: np.ndarray, style: CurveStyle, source: Path) -> None:
    failures = data[:, 1]
    shots = data[:, 2]
    rates = failures / shots
    positive = rates > 0
    if not np.all(positive):
        warnings.warn(
            f"{source}: omitting {np.count_nonzero(~positive)} zero-failure point(s)",
            stacklevel=2,
        )
    if not np.any(positive):
        raise ValueError(f"{source} has no positive BLER values")

    ax.errorbar(
        data[positive, 0],
        rates[positive],
        yerr=binomial_standard_error(failures[positive], shots[positive]),
        fmt=".-",
        color=style.color,
        label=style.label,
        linewidth=1.5,
        markersize=7,
        markerfacecolor=style.color,
        markeredgecolor=style.color,
        capsize=3,
        elinewidth=1.0,
        alpha=1.0,
    )


def _load_curve(decoder_root: Path, code: str, style: CurveStyle) -> tuple[np.ndarray, Path]:
    source = decoder_root / style.variant / f"{code}.npy"
    if not source.is_file():
        raise FileNotFoundError(f"Required result table was not found: {source}")
    return load_result_table(source), source


def _plot_panel(
    decoder_root: Path,
    code: str,
    output_path: Path,
    *,
    include_qc24: bool,
    use_tex: bool,
    x_limits: tuple[float, float],
) -> None:
    rc = {
        "font.size": 14,
        "text.usetex": use_tex,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        for style in MAIN_CURVES:
            data, source = _load_curve(decoder_root, code, style)
            _plot_curve(ax, data, style, source)
        if include_qc24:
            data, source = _load_curve(decoder_root, "qc_24_6_10", QC24_CURVE)
            _plot_curve(ax, data, QC24_CURVE, source)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*x_limits)
        ax.set_xlabel(r"$p$")
        ax.set_ylabel("BLER")
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.65)
        ax.legend(
            loc="lower right",
            frameon=True,
            facecolor="white",
            framealpha=0.8,
        )
        fig.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            format="pdf",
            transparent=True,
        )
        plt.close(fig)


def audit_dataset(collection_root: Path) -> None:
    try:
        from audit_phenom_dem_results import PhenomDemAuditError, audit_results
    except ImportError as exc:
        raise RuntimeError(
            "The phenomenological-DEM auditor is unavailable; refusing to plot "
            "unverified production data"
        ) from exc
    try:
        summary = audit_results(collection_root)
    except PhenomDemAuditError as exc:
        raise ValueError(f"immutable-ledger audit failed: {exc}") from exc
    if summary.get("status") != "ok":
        raise ValueError("immutable-ledger audit did not return status 'ok'")


def generate_plots(
    decoder_root: Path,
    output_dir: Path,
    *,
    use_tex: bool = False,
) -> Tuple[Path, Path]:
    qc_path = output_dir / DEFAULT_QC_FILENAME
    heawood_path = output_dir / DEFAULT_HEAWOOD_FILENAME
    _plot_panel(
        decoder_root,
        "qc_20_5_9",
        qc_path,
        include_qc24=True,
        use_tex=use_tex,
        x_limits=QC_X_LIMITS,
    )
    _plot_panel(
        decoder_root,
        "heawood_cycle",
        heawood_path,
        include_qc24=False,
        use_tex=use_tex,
        x_limits=HEAWOOD_X_LIMITS,
    )
    return qc_path, heawood_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        decoder_root = resolve_decoder_root(args.data_root)
        if not args.allow_unverified_data:
            audit_dataset(decoder_root.parent)
        qc_path, heawood_path = generate_plots(
            decoder_root,
            args.output_dir.expanduser().resolve(),
            use_tex=args.use_tex,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"QC figure: {qc_path}")
    print(f"Heawood figure: {heawood_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
