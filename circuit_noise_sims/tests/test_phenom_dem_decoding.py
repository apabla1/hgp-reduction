"""Tests for DEM-derived priors on the phenomenological decoder graph."""

from __future__ import annotations

import queue
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import stim


SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from functions.phenom_dem_decoding import (
    _SparseSpatialSolver,
    _sparse_sha256,
    build_phenomenological_dem_model,
    build_phenomenological_graph,
    num_failures_phenom_dem,
)


class _OneQubitCode:
    N = 1
    K = 1
    hz = sp.csr_matrix([[1]], dtype=np.uint8)
    canonical_lz = sp.csr_matrix([[1]], dtype=np.uint8)


class _TwoQubitProjectionCode:
    N = 2
    K = 1
    hz = sp.csr_matrix([[1, 1], [1, 0]], dtype=np.uint8)
    canonical_lz = sp.csr_matrix([[0, 1]], dtype=np.uint8)


def _one_qubit_phenom_circuit(p0=0.1, p1=0.2, pm=0.3):
    # q0 flips D0 and L0, q1 flips D1 and L0, and q2 flips D0 D1.
    return stim.Circuit(
        f"""
        R 0 1 2
        X_ERROR({p0}) 0
        X_ERROR({p1}) 1
        X_ERROR({pm}) 2
        M 0 1 2
        DETECTOR rec[-3] rec[-1]
        DETECTOR rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-3] rec[-2]
        """
    )


class PhenomenologicalGraphTests(unittest.TestCase):
    def test_binary_sparse_hash_is_representation_independent(self):
        canonical = sp.eye(2, dtype=np.uint8, format="csc")
        with_duplicates = sp.coo_matrix(
            (
                np.array([3, 1, 2, 0], dtype=np.int64),
                (
                    np.array([0, 1, 1, 0]),
                    np.array([0, 1, 1, 1]),
                ),
            ),
            shape=(2, 2),
        )
        self.assertEqual(_sparse_sha256(canonical), _sparse_sha256(with_duplicates))

    def test_spatial_weight_four_is_exactly_certified(self):
        solver = _SparseSpatialSolver(sp.eye(4, dtype=np.uint8, format="csr"))
        indices, minimum_certified, used_osd = solver.solve(
            np.ones(4, dtype=np.uint8)
        )
        self.assertEqual(indices, (0, 1, 2, 3))
        self.assertTrue(minimum_certified)
        self.assertFalse(used_osd)

    def test_global_projection_repairs_single_time_heuristic(self):
        # This target is graph columns 0 XOR 3. The single-time construction
        # has weight three, while the global graph minimum has weight two.
        circuit = stim.Circuit(
            """
            R 0
            X_ERROR(0.125) 0
            M 0
            DETECTOR rec[-1]
            DETECTOR rec[-1]
            DETECTOR rec[-1]
            DETECTOR
            DETECTOR
            DETECTOR
            OBSERVABLE_INCLUDE(0) rec[-1]
            """
        )
        model = build_phenomenological_dem_model(
            circuit, _TwoQubitProjectionCode(), rounds=2
        )
        expected = np.full(10, 1e-15)
        expected[[0, 3]] = 0.125
        np.testing.assert_allclose(model.error_priors, expected)
        self.assertEqual(model.num_improved_initial_projections, 1)
        self.assertEqual(model.max_initial_projection_weight, 3)
        self.assertEqual(model.max_projection_weight, 2)
        self.assertTrue(model.all_projections_minimum_weight_certified)

        repeated = build_phenomenological_dem_model(
            circuit, _TwoQubitProjectionCode(), rounds=2
        )
        self.assertEqual(model.graph_sha256, repeated.graph_sha256)
        self.assertEqual(model.observables_sha256, repeated.observables_sha256)
        self.assertEqual(model.priors_sha256, repeated.priors_sha256)
        self.assertEqual(model.projection_sha256, repeated.projection_sha256)

    def test_legacy_graph_and_observable_columns(self):
        check, observable = build_phenomenological_graph(_OneQubitCode(), rounds=1)
        np.testing.assert_array_equal(
            check.toarray(),
            np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            observable.toarray(),
            np.array([[1, 1, 0]], dtype=np.uint8),
        )

    def test_dem_mechanisms_map_to_expected_phenom_variables(self):
        model = build_phenomenological_dem_model(
            _one_qubit_phenom_circuit(), _OneQubitCode(), rounds=1
        )
        np.testing.assert_allclose(model.error_priors, [0.1, 0.2, 0.3])
        self.assertEqual(model.num_dem_mechanisms, 3)
        self.assertEqual(model.max_spatial_solution_weight, 1)
        self.assertEqual(model.max_projection_weight, 1)
        self.assertTrue(model.all_spatial_solutions_minimum_weight_certified)
        self.assertEqual(model.num_zero_projection_mechanisms, 0)

    def test_repeated_mechanisms_collapse_to_parity_marginal(self):
        circuit = stim.Circuit(
            """
            R 0 1 2
            X_ERROR(0.1) 0
            TICK
            X_ERROR(0.2) 0
            M 0 1 2
            DETECTOR rec[-3] rec[-1]
            DETECTOR rec[-2] rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-3] rec[-2]
            """
        )
        model = build_phenomenological_dem_model(circuit, _OneQubitCode(), rounds=1)
        expected = 0.1 * (1 - 0.2) + (1 - 0.1) * 0.2
        self.assertAlmostEqual(model.error_priors[0], expected)
        self.assertAlmostEqual(model.error_priors[1], 1e-15)
        self.assertAlmostEqual(model.error_priors[2], 1e-15)


class PhenomenologicalSamplingTests(unittest.TestCase):
    def test_sampled_observables_are_compared_with_graph_prediction(self):
        circuit = _one_qubit_phenom_circuit(p0=0.25, p1=0.125, pm=0.2)
        model = build_phenomenological_dem_model(circuit, _OneQubitCode(), rounds=1)
        shots = 31
        seed = 24680
        _, observables = circuit.compile_detector_sampler(seed=seed).sample(
            shots=shots, separate_observables=True
        )
        expected_failures = int(np.count_nonzero(observables[:, 0]))

        class ZeroDecoder:
            calls = []

            def __init__(self, check_matrix, **kwargs):
                self.num_errors = check_matrix.shape[1]
                self.kwargs = kwargs
                type(self).calls.append((check_matrix.copy(), kwargs))

            def decode(self, _detectors):
                return np.zeros(self.num_errors, dtype=np.uint8)

        progress = queue.SimpleQueue()
        relay = [0.65, 80, 100, 60, (-0.24, 0.66), 5]
        with patch(
            "functions.phenom_dem_decoding.relay_bp.RelayDecoderF32",
            ZeroDecoder,
        ):
            failures = num_failures_phenom_dem(
                circuit,
                model,
                relay,
                shots,
                sampler_seed=seed,
                worker_id=9,
                progress_queue=progress,
            )

        self.assertEqual(failures, expected_failures)
        self.assertEqual(len(ZeroDecoder.calls), 1)
        relay_matrix, relay_kwargs = ZeroDecoder.calls[0]
        np.testing.assert_array_equal(
            relay_matrix.toarray(), model.check_matrix.toarray()
        )
        np.testing.assert_allclose(relay_kwargs.pop("error_priors"), model.error_priors)
        self.assertEqual(
            relay_kwargs,
            {
                "gamma0": 0.65,
                "pre_iter": 80,
                "num_sets": 100,
                "set_max_iter": 60,
                "gamma_dist_interval": (-0.24, 0.66),
                "stop_nconv": 5,
            },
        )
        updates = [progress.get_nowait() for _ in range(shots)]
        self.assertEqual(updates[-1]["worker_id"], 9)
        self.assertEqual(updates[-1]["shot_num"], shots)
        self.assertEqual(updates[-1]["num_failures"], expected_failures)


if __name__ == "__main__":
    unittest.main()
