"""Fixed configuration for the April-grid phenomenological-DEM simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


MODEL_NAME = "stim_flattened_dem_priors_phenomenological_graph"
DECODER = "Relay"

RELAY_GAMMA0 = 0.65
RELAY_PRE_ITER = 80
RELAY_NUM_SETS = 100
RELAY_MAX_ITER = 60
RELAY_GAMMA_INTERVAL = (-0.24, 0.66)
RELAY_STOP_NCONV = 5
RELAY_PARAMETERS = (
    RELAY_GAMMA0,
    RELAY_PRE_ITER,
    RELAY_NUM_SETS,
    RELAY_MAX_ITER,
    RELAY_GAMMA_INTERVAL,
    RELAY_STOP_NCONV,
)

DECODER_CONFIG = (
    "phenom_stim_dem_priors_"
    "relay_g00.65_pre80_sets100_iter60_gdim0.24to0.66_nconv5"
)

RANDOM_SCHEDULE_SEED = 1
CARDINAL_SCHEDULE_SEED = 0
SCHEDULE_SEEDS_BY_VARIANT: Dict[str, int] = {
    "unreduced_cardinal": CARDINAL_SCHEDULE_SEED,
    "unreduced_random": RANDOM_SCHEDULE_SEED,
    "reduced_random": RANDOM_SCHEDULE_SEED,
    "reduced_split": RANDOM_SCHEDULE_SEED,
}
QC_P_VALUES_ASCENDING: Tuple[float, ...] = tuple(
    float(value) for value in np.geomspace(1e-3, 7e-3, 10)
)
HEAWOOD_P_VALUES_ASCENDING: Tuple[float, ...] = tuple(
    float(value) for value in np.geomspace(5e-4, 7e-3, 10)
)
P_VALUES_BY_CODE_ASCENDING: Dict[str, Tuple[float, ...]] = {
    "qc_20_5_9": QC_P_VALUES_ASCENDING,
    "qc_24_6_10": QC_P_VALUES_ASCENDING,
    "heawood_cycle": HEAWOOD_P_VALUES_ASCENDING,
}
P_VALUES_BY_CODE_DESCENDING: Dict[str, Tuple[float, ...]] = {
    code: tuple(reversed(values))
    for code, values in P_VALUES_BY_CODE_ASCENDING.items()
}

TARGET_RELATIVE_SE = 0.15
INITIAL_SHOTS = 1000
BATCH_QUANTUM = 1000

LOCAL_MAX_WORKERS = 15
CPU200_MAX_WORKERS = 210


@dataclass(frozen=True, order=True)
class Curve:
    code: str
    variant: str

    @property
    def curve_id(self) -> str:
        return f"{self.code}|{self.variant}"


MANUSCRIPT_CURVES: Tuple[Curve, ...] = (
    Curve("qc_20_5_9", "unreduced_cardinal"),
    Curve("qc_20_5_9", "reduced_random"),
    Curve("qc_20_5_9", "reduced_split"),
    Curve("qc_24_6_10", "reduced_split"),
    Curve("heawood_cycle", "unreduced_cardinal"),
    Curve("heawood_cycle", "reduced_random"),
    Curve("heawood_cycle", "reduced_split"),
)


def schedule_seed(variant: str) -> int:
    """Return the fixed April schedule seed for one circuit variant."""

    try:
        return SCHEDULE_SEEDS_BY_VARIANT[variant]
    except KeyError as exc:
        raise ValueError(f"no schedule seed is configured for {variant!r}") from exc


def p_values_for_code(code: str, *, descending: bool = False) -> Tuple[float, ...]:
    """Return the exact ten-point grid used by the corresponding April panel."""

    grids = (
        P_VALUES_BY_CODE_DESCENDING
        if descending
        else P_VALUES_BY_CODE_ASCENDING
    )
    try:
        return grids[code]
    except KeyError as exc:
        raise ValueError(f"no April p grid is configured for {code!r}") from exc


def relative_se_target(_p: float) -> float:
    """Return the constant adaptive-sampling target."""

    return TARGET_RELATIVE_SE


def worker_limit(execution_site: str) -> int:
    if execution_site == "local":
        return LOCAL_MAX_WORKERS
    if execution_site == "cpu200":
        return CPU200_MAX_WORKERS
    raise ValueError("execution_site must be 'local' or 'cpu200'")
