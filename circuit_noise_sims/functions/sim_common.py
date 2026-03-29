import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODES_DIR = PROJECT_ROOT / "codes"
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

VARIANTS: Tuple[str, str, str] = (
    "unreduced_random",
    "reduced_random",
    "reduced_split",
)

DEFAULT_PS_SWEEP: List[float] = [
    0.5e-3,
    1.0e-3,
    1.5e-3,
    2.0e-3,
    2.5e-3,
    3.0e-3,
    3.5e-3,
    4.0e-3,
    4.5e-3,
    5.0e-3,
    5.5e-3,
    6.0e-3,
    6.5e-3,
    7.0e-3,
    7.5e-3,
    8.0e-3,
    8.5e-3,
    9.0e-3,
    9.5e-3,
    1.0e-2,
]


def get_available_codes() -> Dict[str, Tuple[str, str]]:
    """Dynamically discover all available code getter functions."""
    codes: Dict[str, Tuple[str, str]] = {}
    exclude_prefixes = ("get_check_", "get_coloring_", "get_adj_")

    for module_file in CODES_DIR.glob("*.py"):
        if module_file.name.startswith("__"):
            continue
        module_name = module_file.stem
        try:
            module = importlib.import_module(f"codes.{module_name}")
            for attr_name in dir(module):
                if not attr_name.startswith("get_"):
                    continue
                if any(attr_name.startswith(prefix) for prefix in exclude_prefixes):
                    continue
                code_name = attr_name[4:]
                codes[code_name] = (module_name, attr_name)
        except Exception as exc:
            print(f"Warning: Could not load code module {module_name}: {exc}", file=sys.stderr)

    return codes


def load_code(code_name: str) -> Tuple[Any, np.ndarray]:
    """Load a code by name. Returns (HGP_code, H_matrix)."""
    available = get_available_codes()
    if code_name not in available:
        raise ValueError(f"Unknown code: {code_name}. Available: {list(available.keys())}")

    module_name, func_name = available[code_name]
    module = importlib.import_module(f"codes.{module_name}")
    getter_func = getattr(module, func_name)
    return getter_func()


def validate_selected_codes(selected_codes: Sequence[str]) -> None:
    available = get_available_codes()
    for code_name in selected_codes:
        if code_name not in available:
            raise ValueError(f"Unknown code '{code_name}'. Available: {sorted(available.keys())}")


def weight_stats(H: Any) -> Tuple[Any, Any, float, Any, Any, float]:
    """Compute weight statistics for a parity check matrix."""
    if hasattr(H, "getnnz"):
        rw = H.getnnz(axis=1)
        cw = H.getnnz(axis=0)
    else:
        rw = H.sum(axis=1)
        cw = H.sum(axis=0)
    return (rw.min(), rw.max(), round(float(rw.mean()), 3), cw.min(), cw.max(), round(float(cw.mean()), 3))


def parse_decoder_params(args: Any) -> List[Any]:
    if args.decoder == "Relay":
        return [
            args.relay_gamma0,
            args.relay_pre_iter,
            args.relay_num_sets,
            args.relay_max_iter,
            tuple(args.relay_gamma_dist_interval),
            args.relay_stop_nconv,
        ]
    if args.decoder in ("OSD", "LSD"):
        return [args.bp_max_iter, args.bp_order]
    raise ValueError("Decoder must be one of OSD, LSD, Relay")


def _fmt_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m")


def decoder_config_tag(decoder: str, dec_params: Sequence[Any]) -> str:
    """Build folder name for decoder and decoder parameters."""
    if decoder == "Relay":
        gamma0, pre_iter, num_sets, max_iter, gamma_dist_interval, stop_nconv = dec_params
        low, high = gamma_dist_interval
        return (
            f"relay_g0{_fmt_float(float(gamma0))}"
            f"_pre{int(pre_iter)}"
            f"_sets{int(num_sets)}"
            f"_iter{int(max_iter)}"
            f"_gdi{_fmt_float(float(low))}to{_fmt_float(float(high))}"
            f"_nconv{int(stop_nconv)}"
        )

    return f"{decoder.lower()}_bpiter{int(dec_params[0])}_order{int(dec_params[1])}"


def get_decoder_root(decoder: str, dec_params: Sequence[Any]) -> Path:
    return DATA_DIR / decoder_config_tag(decoder, dec_params)


def get_data_path(code_name: str, variant: str, decoder: str, dec_params: Sequence[Any]) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Expected one of {VARIANTS}")

    path = get_decoder_root(decoder, dec_params) / variant / f"{code_name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_data_table(path: Path) -> np.ndarray:
    """Load per-code results table with rows [p, failures, shots]."""
    if not path.exists():
        return np.zeros((0, 3), dtype=float)

    data = np.load(path)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Unexpected data shape at {path}: {data.shape}. Expected (N, 3).")
    return data.astype(float, copy=False)


def save_data_table(path: Path, data: np.ndarray) -> None:
    np.save(path, data)


def append_result_row(data: np.ndarray, p: float, failures: int, shots: int) -> np.ndarray:
    """Add shot counts to existing p row or create a new row."""
    if data.size == 0:
        return np.array([[float(p), float(failures), float(shots)]], dtype=float)

    matches = np.where(np.isclose(data[:, 0], float(p), atol=1e-15, rtol=0.0))[0]
    if len(matches) > 0:
        idx = int(matches[0])
        data[idx, 1] += float(failures)
        data[idx, 2] += float(shots)
    else:
        data = np.vstack((data, np.array([[float(p), float(failures), float(shots)]], dtype=float)))

    order = np.argsort(data[:, 0])
    return data[order]


def parse_p_values(raw_values: Optional[Iterable[str]]) -> List[float]:
    if raw_values is None:
        return list(DEFAULT_PS_SWEEP)

    parsed: List[float] = []
    for token in raw_values:
        for piece in str(token).split(","):
            piece = piece.strip()
            if not piece:
                continue
            parsed.append(float(piece))

    if not parsed:
        raise ValueError("No valid p values were provided.")

    unique_sorted = sorted(set(parsed))
    return unique_sorted


def select_rows_in_range(data: np.ndarray, p_min: Optional[float], p_max: Optional[float]) -> np.ndarray:
    if data.size == 0:
        return data

    selected = data
    if p_min is not None:
        selected = selected[selected[:, 0] >= p_min]
    if p_max is not None:
        selected = selected[selected[:, 0] <= p_max]
    return selected
