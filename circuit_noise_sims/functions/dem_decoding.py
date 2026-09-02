"""Decode annotated Z-memory circuits using Stim's correlated DEM.

Each ``error`` instruction in the detector error model is represented by one
binary Relay-BP variable.  In particular, detector hyperedges and components
separated by ``^`` are not decomposed into graph edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple
import warnings

import numpy as np
from scipy.sparse import csc_matrix
import relay_bp
import stim


@dataclass(frozen=True)
class DemMatrices:
    """Sparse detector/observable incidence matrices for DEM mechanisms."""

    check_matrix: csc_matrix
    observables_matrix: csc_matrix
    error_priors: np.ndarray

    @property
    def num_errors(self) -> int:
        return int(self.error_priors.size)


@dataclass(frozen=True)
class DemBuildResult:
    """A Stim DEM together with whether disjoint channels were approximated."""

    model: stim.DetectorErrorModel
    used_approximate_disjoint_errors: bool
    exact_rejection: Optional[str] = None


def _xor_target(targets: set[int], index: int) -> None:
    """Toggle an index, implementing parity when a DEM target is repeated."""

    if index in targets:
        targets.remove(index)
    else:
        targets.add(index)


def dem_to_matrices(dem: stim.DetectorErrorModel) -> DemMatrices:
    """Convert a full Stim DEM without decomposing correlated mechanisms.

    The returned check matrix has shape ``(detectors, error instructions)``.
    The observable matrix has shape ``(observables, error instructions)``.
    A separator target only partitions a mechanism for graph decoders; it does
    not create another Relay-BP variable here.
    """

    detector_rows = []
    detector_columns = []
    observable_rows = []
    observable_columns = []
    priors = []

    for instruction in dem.flattened():
        if instruction.type != "error":
            if instruction.type not in ("detector", "logical_observable"):
                raise NotImplementedError(
                    f"Unsupported flattened DEM instruction: {instruction.type}"
                )
            continue

        arguments = instruction.args_copy()
        if len(arguments) != 1:
            raise ValueError(
                "A DEM error instruction must have exactly one probability."
            )
        probability = float(arguments[0])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid DEM error probability: {probability}")

        detectors: set[int] = set()
        observables: set[int] = set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                _xor_target(detectors, int(target.val))
            elif target.is_logical_observable_id():
                _xor_target(observables, int(target.val))
            elif target.is_separator():
                # All separator components belong to this same error event.
                continue
            else:
                raise ValueError(f"Unsupported DEM target: {target}")

        column = len(priors)
        priors.append(probability)
        for detector in sorted(detectors):
            detector_rows.append(detector)
            detector_columns.append(column)
        for observable in sorted(observables):
            observable_rows.append(observable)
            observable_columns.append(column)

    num_errors = len(priors)
    check_matrix = csc_matrix(
        (
            np.ones(len(detector_rows), dtype=np.uint8),
            (detector_rows, detector_columns),
        ),
        shape=(int(dem.num_detectors), num_errors),
        dtype=np.uint8,
    )
    observables_matrix = csc_matrix(
        (
            np.ones(len(observable_rows), dtype=np.uint8),
            (observable_rows, observable_columns),
        ),
        shape=(int(dem.num_observables), num_errors),
        dtype=np.uint8,
    )
    return DemMatrices(
        check_matrix=check_matrix,
        observables_matrix=observables_matrix,
        error_priors=np.asarray(priors, dtype=np.float64),
    )


def _is_disjoint_channel_rejection(error: ValueError) -> bool:
    message = str(error).lower()
    return "disjoint" in message and "approximate_disjoint_errors" in message


def build_detector_error_model(
    circuit: stim.Circuit,
    *,
    allow_disjoint_channel_approximation: bool = True,
) -> DemBuildResult:
    """Build an undecomposed DEM, trying the exact Stim conversion first.

    Stim DEMs contain independent binary mechanisms and therefore cannot
    exactly express mutually exclusive branches of instructions such as
    ``DEPOLARIZE2``.  Only when Stim specifically rejects such a channel do we
    optionally retry its standard small-probability approximation.  Detector
    hyperedges remain undecomposed in either case.
    """

    try:
        model = circuit.detector_error_model(decompose_errors=False)
        return DemBuildResult(
            model=model,
            used_approximate_disjoint_errors=False,
        )
    except ValueError as exact_error:
        if (
            not allow_disjoint_channel_approximation
            or not _is_disjoint_channel_rejection(exact_error)
        ):
            raise

        model = circuit.detector_error_model(
            decompose_errors=False,
            approximate_disjoint_errors=True,
        )
        warnings.warn(
            "Stim rejected an exact DEM because the circuit contains a "
            "disjoint error channel; using Stim's disjoint-channel "
            "approximation while retaining each correlated detector "
            "hyperedge as one Relay-BP mechanism.",
            RuntimeWarning,
            stacklevel=2,
        )
        return DemBuildResult(
            model=model,
            used_approximate_disjoint_errors=True,
            exact_rejection=str(exact_error),
        )


def _validate_relay_params(params: Sequence[Any]) -> Tuple[Any, ...]:
    if len(params) != 6:
        raise ValueError(
            "Relay parameters must be [gamma0, pre_iter, num_sets, "
            "set_max_iter, gamma_dist_interval, stop_nconv]."
        )
    gamma0, pre_iter, num_sets, set_max_iter, gamma_interval, stop_nconv = params
    if len(gamma_interval) != 2:
        raise ValueError("gamma_dist_interval must contain two endpoints.")
    return (
        float(gamma0),
        int(pre_iter),
        int(num_sets),
        int(set_max_iter),
        (float(gamma_interval[0]), float(gamma_interval[1])),
        int(stop_nconv),
    )


def num_failures_dem(
    circ: stim.Circuit,
    params: Sequence[Any],
    shots: int,
    sampler_seed: Optional[int] = None,
    worker_id: Optional[int] = None,
    progress_queue: Optional[Any] = None,
) -> int:
    """Sample and decode an annotated circuit, returning block failures.

    A block fails when any predicted canonical observable differs from Stim's
    sampled observable.  Relay batch parallelism is intentionally avoided so
    an outer Monte Carlo worker occupies one CPU worker.
    """

    shots = int(shots)
    if shots <= 0:
        raise ValueError("shots must be positive")

    relay_params = _validate_relay_params(params)
    build = build_detector_error_model(circ)
    matrices = dem_to_matrices(build.model)
    if matrices.check_matrix.shape[0] == 0:
        raise ValueError("The circuit contains no detectors.")
    if matrices.observables_matrix.shape[0] == 0:
        raise ValueError("The circuit contains no logical observables.")
    if matrices.num_errors == 0:
        raise ValueError("The detector error model contains no error mechanisms.")

    gamma0, pre_iter, num_sets, set_max_iter, gamma_interval, stop_nconv = relay_params
    decoder = relay_bp.RelayDecoderF32(
        matrices.check_matrix,
        error_priors=matrices.error_priors,
        gamma0=gamma0,
        pre_iter=pre_iter,
        num_sets=num_sets,
        set_max_iter=set_max_iter,
        gamma_dist_interval=gamma_interval,
        stop_nconv=stop_nconv,
    )

    sampler = (
        circ.compile_detector_sampler(seed=int(sampler_seed))
        if sampler_seed is not None
        else circ.compile_detector_sampler()
    )
    failures = 0
    completed = 0
    sample_batch_size = 256

    while completed < shots:
        batch_size = min(sample_batch_size, shots - completed)
        detectors, actual_observables = sampler.sample(
            shots=batch_size,
            separate_observables=True,
        )
        detectors = np.asarray(detectors, dtype=np.uint8)
        actual_observables = np.asarray(actual_observables, dtype=np.uint8)

        for index in range(batch_size):
            estimated_errors = np.asarray(
                decoder.decode(detectors[index]),
                dtype=np.uint8,
            )
            predicted_observables = np.asarray(
                matrices.observables_matrix @ estimated_errors,
                dtype=np.uint8,
            ).reshape(-1) % 2
            failures += int(
                np.any(predicted_observables != actual_observables[index])
            )
            completed += 1

            if progress_queue is not None and worker_id is not None:
                progress_queue.put(
                    {
                        "worker_id": int(worker_id),
                        "shot_num": int(completed),
                        "shots": int(shots),
                        "num_failures": int(failures),
                    }
                )

    return int(failures)
