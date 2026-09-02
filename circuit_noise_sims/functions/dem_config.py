"""Fixed configuration for the full-DEM manuscript simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


MODEL_NAME = "stim_z_detector_dem_canonical"
DECODER = "Relay"

RELAY_GAMMA0 = 0.125
RELAY_PRE_ITER = 80
RELAY_NUM_SETS = 300
RELAY_MAX_ITER = 60
RELAY_GAMMA_INTERVAL = (-0.16, 0.66)
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
    "z_dem_canonical_"
    "relay_g00.125_pre80_sets300_iter60_gdim0.16to0.66_nconv5"
)

SCHEDULE_SEED = 1
P_HIGH = 0.006
P_LOW = 0.001
P_STEP = 0.0005
P_VALUES_DESCENDING: Tuple[float, ...] = tuple(
    round(P_HIGH - index * P_STEP, 15)
    for index in range(int(round((P_HIGH - P_LOW) / P_STEP)) + 1)
)

HIGH_P_RELATIVE_SE = 0.05
LOW_P_RELATIVE_SE = 0.20
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


def relative_se_target(p: float) -> float:
    """Linearly interpolate 5% at p=0.006 to 20% at p=0.001."""

    p = float(p)
    if p >= P_HIGH:
        return HIGH_P_RELATIVE_SE
    if p <= P_LOW:
        return LOW_P_RELATIVE_SE
    fraction_toward_low = (P_HIGH - p) / (P_HIGH - P_LOW)
    return HIGH_P_RELATIVE_SE + fraction_toward_low * (
        LOW_P_RELATIVE_SE - HIGH_P_RELATIVE_SE
    )


def worker_limit(execution_site: str) -> int:
    if execution_site == "local":
        return LOCAL_MAX_WORKERS
    if execution_site == "cpu200":
        return CPU200_MAX_WORKERS
    raise ValueError("execution_site must be 'local' or 'cpu200'")
