"""Phenomenological decoding with priors derived from a Stim DEM.

The decoder graph is the same spacetime phenomenological graph used by the
legacy simulations.  Circuit faults are first converted to Stim detector-error
mechanisms.  Each mechanism is then represented on the phenomenological graph
while preserving its detector syndrome and canonical logical effect.  Finally,
the correlated projected mechanisms are replaced by independent graph
variables with the same one-variable marginals.

Only the priors are flattened in this last step.  The detector graph itself is
not changed, and this module does not claim to retain correlations between its
variables.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import relay_bp
import scipy.sparse as sp
from ldpc.bposd_decoder import BpOsdDecoder

from functions.dem_decoding import build_detector_error_model, dem_to_matrices
from functions.logical_basis import canonical_z_logicals


PROJECTION_VERSION = "phenom-dem-independent-marginals-v2"
SPATIAL_SOLVER = "exact-weight-0-to-4-then-bp-osd-cs-order-0"
GRAPH_PROJECTION_SOLVER = (
    "exact-global-weight-0-to-4-with-certified-weight-5-upper-bound"
)
SPATIAL_SOLVER_ERROR_PRIOR = 0.01
SPATIAL_SOLVER_MAX_ITER = 100


def _sha256_bytes(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _binary_sparse(matrix: sp.spmatrix, *, format: str) -> sp.spmatrix:
    """Return a canonical GF(2) sparse matrix in the requested format."""

    value = sp.coo_matrix(matrix, copy=True)
    data = np.asarray(value.data)
    if np.any(~np.isfinite(data)) or np.any(data != np.rint(data)):
        raise ValueError("A binary matrix contains a nonintegral entry.")
    value.data = np.asarray(np.rint(data), dtype=np.int64)
    value = value.asformat(format)
    value.sum_duplicates()
    value.data = np.remainder(value.data, 2).astype(np.uint8, copy=False)
    value.eliminate_zeros()
    value.sort_indices()
    return value


def _sparse_sha256(matrix: sp.spmatrix) -> str:
    value = _binary_sparse(matrix, format="csc")
    return _sha256_bytes(
        np.asarray(value.shape, dtype="<i8").tobytes(),
        np.asarray(value.indptr, dtype="<i8").tobytes(),
        np.asarray(value.indices, dtype="<i8").tobytes(),
        np.asarray(value.data, dtype=np.uint8).tobytes(),
    )


def _indices_signature(indices) -> int:
    signature = 0
    for index in indices:
        signature ^= 1 << int(index)
    return signature


def _vector_signature(vector: np.ndarray) -> int:
    return _indices_signature(np.flatnonzero(np.asarray(vector, dtype=np.uint8)))


def _column_signatures(matrix: sp.spmatrix) -> Tuple[int, ...]:
    value = _binary_sparse(matrix, format="csc")
    return tuple(
        _indices_signature(
            value.indices[value.indptr[column] : value.indptr[column + 1]]
        )
        for column in range(value.shape[1])
    )


@dataclass(frozen=True)
class PhenomenologicalDemModel:
    """A phenomenological graph and its DEM-derived independent priors."""

    check_matrix: sp.csc_matrix
    observables_matrix: sp.csc_matrix
    error_priors: np.ndarray
    graph_sha256: str
    observables_sha256: str
    priors_sha256: str
    projection_sha256: str
    used_approximate_disjoint_errors: bool
    exact_dem_rejection: Optional[str]
    num_dem_mechanisms: int
    num_unique_spatial_rhs: int
    num_zero_projection_mechanisms: int
    num_osd_spatial_solutions: int
    max_spatial_solution_weight: int
    num_unique_projection_rhs: int
    num_improved_initial_projections: int
    max_initial_projection_weight: int
    max_projection_weight: int
    all_spatial_solutions_minimum_weight_certified: bool
    all_projections_minimum_weight_certified: bool

    @property
    def num_errors(self) -> int:
        return int(self.error_priors.size)

    def metadata(self) -> Dict[str, Any]:
        return {
            "projection_version": PROJECTION_VERSION,
            "projection_method": (
                "preserve each DEM mechanism's detectors and canonical "
                "observables, then retain independent one-variable marginals"
            ),
            "spatial_solver": SPATIAL_SOLVER,
            "graph_projection_solver": GRAPH_PROJECTION_SOLVER,
            "spatial_solver_error_prior": SPATIAL_SOLVER_ERROR_PRIOR,
            "spatial_solver_max_iter": SPATIAL_SOLVER_MAX_ITER,
            "num_detectors": int(self.check_matrix.shape[0]),
            "num_observables": int(self.observables_matrix.shape[0]),
            "num_phenomenological_variables": self.num_errors,
            "num_dem_mechanisms": int(self.num_dem_mechanisms),
            "num_unique_spatial_rhs": int(self.num_unique_spatial_rhs),
            "num_zero_projection_mechanisms": int(
                self.num_zero_projection_mechanisms
            ),
            "num_osd_spatial_solutions": int(self.num_osd_spatial_solutions),
            "max_spatial_solution_weight": int(
                self.max_spatial_solution_weight
            ),
            "num_unique_projection_rhs": int(self.num_unique_projection_rhs),
            "num_improved_initial_projections": int(
                self.num_improved_initial_projections
            ),
            "max_initial_projection_weight": int(
                self.max_initial_projection_weight
            ),
            "max_projection_weight": int(self.max_projection_weight),
            "all_spatial_solutions_minimum_weight_certified": bool(
                self.all_spatial_solutions_minimum_weight_certified
            ),
            "all_projections_minimum_weight_certified": bool(
                self.all_projections_minimum_weight_certified
            ),
            "used_approximate_disjoint_errors": bool(
                self.used_approximate_disjoint_errors
            ),
            "exact_dem_rejection": self.exact_dem_rejection,
            "graph_sha256": self.graph_sha256,
            "observables_sha256": self.observables_sha256,
            "priors_sha256": self.priors_sha256,
            "projection_sha256": self.projection_sha256,
        }


def build_phenomenological_graph(code, rounds: int):
    """Return the legacy spacetime graph and its canonical observable map."""

    rounds = int(rounds)
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    hz = _binary_sparse(code.hz, format="csr")
    num_checks, num_qubits = hz.shape
    logicals = _binary_sparse(canonical_z_logicals(code), format="csr")

    data_part = sp.block_diag([hz] * (rounds + 1), format="csc")
    num_measurement_variables = num_checks * rounds
    measurement_columns = np.arange(num_measurement_variables, dtype=np.int64)
    measurement_rows = np.concatenate(
        (measurement_columns, measurement_columns + num_checks)
    )
    measurement_part = sp.csc_matrix(
        (
            np.ones(2 * num_measurement_variables, dtype=np.uint8),
            (
                measurement_rows,
                np.concatenate((measurement_columns, measurement_columns)),
            ),
        ),
        shape=(num_checks * (rounds + 1), num_measurement_variables),
        dtype=np.uint8,
    )
    check_matrix = sp.hstack((data_part, measurement_part), format="csc")

    logical_data_part = sp.hstack([logicals] * (rounds + 1), format="csc")
    logical_measurement_part = sp.csc_matrix(
        (logicals.shape[0], num_measurement_variables), dtype=np.uint8
    )
    observables_matrix = sp.hstack(
        (logical_data_part, logical_measurement_part), format="csc"
    )
    return check_matrix, observables_matrix


class _SparseSpatialSolver:
    """Find deterministic sparse representatives of ``[Hz; Lz] e = b``."""

    def __init__(self, matrix: sp.spmatrix):
        self.matrix = _binary_sparse(matrix, format="csr")
        self.column_signatures = _column_signatures(self.matrix)
        self.direct: Dict[int, int] = {}
        for index, signature in enumerate(self.column_signatures):
            self.direct.setdefault(signature, index)

        self.pairs: Dict[int, list[Tuple[int, int]]] = {}
        for left in range(len(self.column_signatures)):
            left_signature = self.column_signatures[left]
            for right in range(left + 1, len(self.column_signatures)):
                signature = left_signature ^ self.column_signatures[right]
                self.pairs.setdefault(signature, []).append((left, right))

        self.decoder = BpOsdDecoder(
            self.matrix,
            error_channel=np.full(
                self.matrix.shape[1], SPATIAL_SOLVER_ERROR_PRIOR, dtype=float
            ).tolist(),
            max_iter=SPATIAL_SOLVER_MAX_ITER,
            bp_method="ms",
            osd_method="osd_cs",
            osd_order=0,
            schedule="parallel",
        )
        self.cache: Dict[int, Tuple[Tuple[int, ...], bool, bool]] = {}

    def solve(self, syndrome: np.ndarray) -> Tuple[Tuple[int, ...], bool, bool]:
        """Return indices, minimum-weight certification, and OSD-use flag."""

        syndrome = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
        signature = _vector_signature(syndrome)
        cached = self.cache.get(signature)
        if cached is not None:
            return cached

        if signature == 0:
            result = ((), True, False)
            self.cache[signature] = result
            return result

        direct = self.direct.get(signature)
        if direct is not None:
            result = ((direct,), True, False)
            self.cache[signature] = result
            return result

        pairs = self.pairs.get(signature)
        if pairs:
            result = (pairs[0], True, False)
            self.cache[signature] = result
            return result

        best_three: Optional[Tuple[int, int, int]] = None
        for index, column_signature in enumerate(self.column_signatures):
            for left, right in self.pairs.get(signature ^ column_signature, ()):
                if index == left or index == right:
                    continue
                candidate = tuple(sorted((index, left, right)))
                if best_three is None or candidate < best_three:
                    best_three = candidate
        if best_three is not None:
            result = (best_three, True, False)
            self.cache[signature] = result
            return result

        for left_signature, left_pairs in self.pairs.items():
            right_pairs = self.pairs.get(signature ^ left_signature)
            if not right_pairs:
                continue
            for left_pair in left_pairs:
                for right_pair in right_pairs:
                    if set(left_pair).isdisjoint(right_pair):
                        result = (
                            tuple(sorted((*left_pair, *right_pair))),
                            True,
                            False,
                        )
                        self.cache[signature] = result
                        return result

        decoded = np.asarray(self.decoder.decode(syndrome), dtype=np.uint8)
        if decoded.shape != (self.matrix.shape[1],):
            raise ValueError("The spatial projection decoder returned the wrong shape.")
        if np.any(np.asarray(self.matrix @ decoded).reshape(-1) % 2 != syndrome):
            raise ValueError("A DEM mechanism is outside the phenomenological model.")
        indices = tuple(int(index) for index in np.flatnonzero(decoded))
        if len(indices) < 5:
            raise AssertionError(
                "The fallback found a weight below five after exhaustive search."
            )
        result = (indices, len(indices) == 5, True)
        self.cache[signature] = result
        return result


class _SparseGraphProjectionSolver:
    """Improve a graph representative and certify minima through weight five.

    The temporal construction supplies a valid upper bound. Direct and pair
    lookups are exact. Higher-weight searches anchor on a nonzero target row,
    which every realization must touch an odd number of times. This makes the
    weight-three and weight-four searches exhaustive without storing all
    quadratic pairs of phenomenological graph columns.
    """

    def __init__(self, matrix: sp.spmatrix):
        self.matrix = _binary_sparse(matrix, format="csc")
        self.column_signatures = _column_signatures(self.matrix)

        direct_lists: Dict[int, list[int]] = {}
        for index, signature in enumerate(self.column_signatures):
            direct_lists.setdefault(signature, []).append(index)
        self.direct = {
            signature: tuple(indices)
            for signature, indices in direct_lists.items()
        }

        rows = self.matrix.tocsr()
        self.row_columns = tuple(
            tuple(
                int(index)
                for index in rows.indices[rows.indptr[row] : rows.indptr[row + 1]]
            )
            for row in range(rows.shape[0])
        )
        self.cache: Dict[int, Tuple[Tuple[int, ...], bool]] = {}

    @staticmethod
    def _signature_rows(signature: int) -> Tuple[int, ...]:
        rows = []
        remaining = int(signature)
        while remaining:
            least_bit = remaining & -remaining
            rows.append(least_bit.bit_length() - 1)
            remaining ^= least_bit
        return tuple(rows)

    def _find_pair(
        self, signature: int, *, excluded: Tuple[int, ...] = ()
    ) -> Optional[Tuple[int, int]]:
        """Return the lexicographically first distinct pair with this XOR."""

        for left, left_signature in enumerate(self.column_signatures):
            if left in excluded:
                continue
            for right in self.direct.get(signature ^ left_signature, ()):
                if right > left and right not in excluded:
                    return left, right
        return None

    def _find_three(
        self, signature: int, *, excluded: Tuple[int, ...] = ()
    ) -> Optional[Tuple[int, int, int]]:
        target_rows = self._signature_rows(signature)
        if not target_rows:
            return None
        pivot = min(
            target_rows,
            key=lambda row: (len(self.row_columns[row]), row),
        )

        best: Optional[Tuple[int, int, int]] = None
        for anchor in self.row_columns[pivot]:
            if anchor in excluded:
                continue
            other_pair = self._find_pair(
                signature ^ self.column_signatures[anchor],
                excluded=(*excluded, anchor),
            )
            if other_pair is None:
                continue
            candidate = tuple(sorted((anchor, *other_pair)))
            if best is None or candidate < best:
                best = candidate
        return best

    def _find_four(
        self,
        signature: int,
        target_rows: Sequence[int],
    ) -> Optional[Tuple[int, int, int, int]]:
        unique_target_rows = tuple(sorted(set(int(row) for row in target_rows)))
        if not unique_target_rows:
            return None
        pivot = min(
            unique_target_rows,
            key=lambda row: (len(self.row_columns[row]), row),
        )

        best: Optional[Tuple[int, int, int, int]] = None
        for anchor in self.row_columns[pivot]:
            other_three = self._find_three(
                signature ^ self.column_signatures[anchor],
                excluded=(anchor,),
            )
            if other_three is None:
                continue
            candidate = tuple(sorted((anchor, *other_three)))
            if best is None or candidate < best:
                best = candidate
        return best

    def solve(
        self,
        target_signature: int,
        target_rows: Sequence[int],
        initial_projection: Tuple[int, ...],
    ) -> Tuple[Tuple[int, ...], bool]:
        """Return a no-heavier projection and a global-minimum certificate."""

        cached = self.cache.get(target_signature)
        if cached is not None:
            return cached

        initial = tuple(sorted(int(index) for index in initial_projection))
        if len(set(initial)) != len(initial) or any(
            index < 0 or index >= self.matrix.shape[1] for index in initial
        ):
            raise ValueError("The initial graph projection is not a binary vector.")
        initial_signature = 0
        for index in initial:
            initial_signature ^= self.column_signatures[index]
        if initial_signature != target_signature:
            raise ValueError("The initial graph projection has the wrong target.")

        if target_signature == 0:
            result = ((), True)
            self.cache[target_signature] = result
            return result

        direct = self.direct.get(target_signature)
        if direct:
            result = ((direct[0],), True)
            self.cache[target_signature] = result
            return result

        if len(initial) <= 2:
            result = (initial, True)
            self.cache[target_signature] = result
            return result

        pair = self._find_pair(target_signature)
        if pair is not None:
            result = (pair, True)
            self.cache[target_signature] = result
            return result

        if len(initial) <= 3:
            result = (initial, True)
            self.cache[target_signature] = result
            return result

        three = self._find_three(target_signature)
        if three is not None:
            result = (three, True)
            self.cache[target_signature] = result
            return result

        if len(initial) <= 4:
            result = (initial, True)
            self.cache[target_signature] = result
            return result

        four = self._find_four(target_signature, target_rows)
        if four is not None:
            result = (four, True)
            self.cache[target_signature] = result
            return result

        # We have exhaustively excluded weights zero through four. Therefore
        # an existing weight-five representative certifies a global minimum.
        result = (initial, len(initial) == 5)
        self.cache[target_signature] = result
        return result


def _best_temporal_projection(
    detector_slices: np.ndarray,
    spatial_syndrome: np.ndarray,
    data_indices: Tuple[int, ...],
    *,
    num_qubits: int,
    num_checks: int,
    rounds: int,
) -> Tuple[int, ...]:
    """Place a spatial representative at the least-cost time boundary."""

    best_key = None
    best_projection: Optional[Tuple[int, ...]] = None
    data_variable_count = num_qubits * (rounds + 1)

    for time_slice in range(rounds + 1):
        previous = np.zeros(num_checks, dtype=np.uint8)
        measurement_indices = []
        for detector_time in range(rounds + 1):
            current = detector_slices[detector_time] ^ previous
            if detector_time == time_slice:
                current = current ^ spatial_syndrome
            if detector_time < rounds:
                active = np.flatnonzero(current)
                measurement_indices.extend(
                    data_variable_count + detector_time * num_checks + int(index)
                    for index in active
                )
                previous = current
            elif np.any(current):
                raise ValueError(
                    "The temporal projection does not close at the final boundary."
                )

        projected = tuple(
            sorted(
                [time_slice * num_qubits + index for index in data_indices]
                + measurement_indices
            )
        )
        key = (len(projected), time_slice, projected)
        if best_key is None or key < best_key:
            best_key = key
            best_projection = projected

    if best_projection is None:
        raise AssertionError("No temporal projection was considered.")
    return best_projection


def build_phenomenological_dem_model(
    circuit,
    code,
    rounds: int,
) -> PhenomenologicalDemModel:
    """Build DEM-derived independent priors on the legacy decoder graph."""

    rounds = int(rounds)
    check_matrix, observables_matrix = build_phenomenological_graph(code, rounds)
    if check_matrix.shape[0] != int(circuit.num_detectors):
        raise ValueError(
            "The circuit detector ordering does not match the phenomenological graph."
        )
    if observables_matrix.shape[0] != int(circuit.num_observables):
        raise ValueError(
            "The circuit observables do not match the canonical graph observables."
        )

    dem_build = build_detector_error_model(circuit)
    dem = dem_to_matrices(dem_build.model)
    if dem.check_matrix.shape[0] != check_matrix.shape[0]:
        raise ValueError("Stim and the phenomenological graph disagree on detectors.")
    if dem.observables_matrix.shape[0] != observables_matrix.shape[0]:
        raise ValueError("Stim and the phenomenological graph disagree on observables.")

    hz = _binary_sparse(code.hz, format="csr")
    logicals = _binary_sparse(canonical_z_logicals(code), format="csr")
    num_checks, num_qubits = hz.shape
    spatial_matrix = sp.vstack((hz, logicals), format="csr")
    spatial_solver = _SparseSpatialSolver(spatial_matrix)

    combined_graph = sp.vstack(
        (check_matrix, observables_matrix), format="csc"
    )
    graph_signatures = _column_signatures(combined_graph)
    graph_projection_solver = _SparseGraphProjectionSolver(combined_graph)
    detector_count = check_matrix.shape[0]
    bias = np.ones(check_matrix.shape[1], dtype=np.float64)
    projection_digest = hashlib.sha256()
    projection_digest.update(PROJECTION_VERSION.encode("utf-8"))
    projection_digest.update(b"\0")

    zero_projection_mechanisms = 0
    num_improved_initial_projections = 0
    max_initial_projection_weight = 0
    max_projection_weight = 0
    all_minimum_certified = True
    all_projection_minimum_certified = True

    for mechanism in range(dem.num_errors):
        detector_indices = dem.check_matrix.indices[
            dem.check_matrix.indptr[mechanism] : dem.check_matrix.indptr[
                mechanism + 1
            ]
        ]
        observable_indices = dem.observables_matrix.indices[
            dem.observables_matrix.indptr[mechanism] : dem.observables_matrix.indptr[
                mechanism + 1
            ]
        ]

        detector_slices = np.zeros((rounds + 1, num_checks), dtype=np.uint8)
        for detector in detector_indices:
            detector_slices[int(detector) // num_checks, int(detector) % num_checks] ^= 1
        spatial_syndrome = np.bitwise_xor.reduce(detector_slices, axis=0)
        rhs = np.zeros(num_checks + logicals.shape[0], dtype=np.uint8)
        rhs[:num_checks] = spatial_syndrome
        for observable in observable_indices:
            rhs[num_checks + int(observable)] ^= 1

        data_indices, minimum_certified, _ = spatial_solver.solve(rhs)
        all_minimum_certified &= minimum_certified
        initial_projection = _best_temporal_projection(
            detector_slices,
            spatial_syndrome,
            data_indices,
            num_qubits=num_qubits,
            num_checks=num_checks,
            rounds=rounds,
        )

        target_signature = _indices_signature(detector_indices)
        target_signature ^= _indices_signature(
            detector_count + int(index) for index in observable_indices
        )
        target_rows = tuple(int(index) for index in detector_indices) + tuple(
            detector_count + int(index) for index in observable_indices
        )
        projection, projection_minimum_certified = graph_projection_solver.solve(
            target_signature,
            target_rows,
            initial_projection,
        )
        all_projection_minimum_certified &= projection_minimum_certified
        num_improved_initial_projections += int(
            len(projection) < len(initial_projection)
        )
        max_initial_projection_weight = max(
            max_initial_projection_weight, len(initial_projection)
        )
        projected_signature = 0
        for variable in projection:
            projected_signature ^= graph_signatures[variable]
        if projected_signature != target_signature:
            raise AssertionError(
                "A projected DEM mechanism changed its detectors or observables."
            )

        probability = float(dem.error_priors[mechanism])
        if projection:
            bias[np.asarray(projection, dtype=np.int64)] *= 1.0 - 2.0 * probability
        elif probability > 0:
            zero_projection_mechanisms += 1
        max_projection_weight = max(max_projection_weight, len(projection))
        projection_digest.update(
            np.asarray([len(target_rows)], dtype="<i8").tobytes()
        )
        projection_digest.update(np.asarray(target_rows, dtype="<i8").tobytes())
        projection_digest.update(
            np.asarray([len(projection)], dtype="<i8").tobytes()
        )
        projection_digest.update(np.asarray(projection, dtype="<i8").tobytes())

    raw_priors = (1.0 - bias) / 2.0
    if (
        np.any(~np.isfinite(raw_priors))
        or np.any(raw_priors < 0)
        or np.any(raw_priors > 1)
    ):
        raise ValueError("DEM-derived phenomenological priors are invalid.")
    error_priors = np.clip(raw_priors, 1e-15, 1.0 - 1e-15)

    spatial_solutions = list(spatial_solver.cache.values())
    max_spatial_weight = max(
        (len(value[0]) for value in spatial_solutions), default=0
    )
    num_osd = sum(bool(value[2]) for value in spatial_solutions)

    return PhenomenologicalDemModel(
        check_matrix=check_matrix,
        observables_matrix=observables_matrix,
        error_priors=error_priors,
        graph_sha256=_sparse_sha256(check_matrix),
        observables_sha256=_sparse_sha256(observables_matrix),
        priors_sha256=_sha256_bytes(
            np.asarray(error_priors, dtype="<f8").tobytes()
        ),
        projection_sha256=projection_digest.hexdigest(),
        used_approximate_disjoint_errors=dem_build.used_approximate_disjoint_errors,
        exact_dem_rejection=dem_build.exact_rejection,
        num_dem_mechanisms=dem.num_errors,
        num_unique_spatial_rhs=len(spatial_solver.cache),
        num_zero_projection_mechanisms=zero_projection_mechanisms,
        num_osd_spatial_solutions=num_osd,
        max_spatial_solution_weight=max_spatial_weight,
        num_unique_projection_rhs=len(graph_projection_solver.cache),
        num_improved_initial_projections=num_improved_initial_projections,
        max_initial_projection_weight=max_initial_projection_weight,
        max_projection_weight=max_projection_weight,
        all_spatial_solutions_minimum_weight_certified=all_minimum_certified,
        all_projections_minimum_weight_certified=(
            all_projection_minimum_certified
        ),
    )


def _validate_relay_params(params: Sequence[Any]):
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


def num_failures_phenom_dem(
    circuit,
    model: PhenomenologicalDemModel,
    params: Sequence[Any],
    shots: int,
    sampler_seed: Optional[int] = None,
    worker_id: Optional[int] = None,
    progress_queue: Optional[Any] = None,
) -> int:
    """Decode detector samples on the phenomenological graph."""

    shots = int(shots)
    if shots <= 0:
        raise ValueError("shots must be positive")
    relay_params = _validate_relay_params(params)
    gamma0, pre_iter, num_sets, set_max_iter, gamma_interval, stop_nconv = relay_params
    decoder = relay_bp.RelayDecoderF32(
        model.check_matrix,
        error_priors=model.error_priors,
        gamma0=gamma0,
        pre_iter=pre_iter,
        num_sets=num_sets,
        set_max_iter=set_max_iter,
        gamma_dist_interval=gamma_interval,
        stop_nconv=stop_nconv,
    )

    sampler = (
        circuit.compile_detector_sampler(seed=int(sampler_seed))
        if sampler_seed is not None
        else circuit.compile_detector_sampler()
    )
    failures = 0
    completed = 0
    while completed < shots:
        batch_size = min(256, shots - completed)
        detectors, actual_observables = sampler.sample(
            shots=batch_size, separate_observables=True
        )
        detectors = np.asarray(detectors, dtype=np.uint8)
        actual_observables = np.asarray(actual_observables, dtype=np.uint8)
        for index in range(batch_size):
            estimated = np.asarray(decoder.decode(detectors[index]), dtype=np.uint8)
            if estimated.shape != (model.num_errors,):
                raise ValueError("Relay-BP returned a vector with the wrong shape.")
            estimated %= 2
            predicted = np.asarray(
                model.observables_matrix @ estimated, dtype=np.uint8
            ).reshape(-1) % 2
            failures += int(np.any(predicted != actual_observables[index]))
            completed += 1
            if progress_queue is not None and worker_id is not None:
                progress_queue.put(
                    {
                        "worker_id": int(worker_id),
                        "shot_num": int(completed),
                        "shots": shots,
                        "num_failures": int(failures),
                    }
                )
    return int(failures)

