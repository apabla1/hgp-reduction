"""Focused tests for correlated Stim-DEM Relay decoding."""

import queue
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import stim


SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from functions.dem_decoding import (
    build_detector_error_model,
    dem_to_matrices,
    num_failures_dem,
)


def _three_detector_hyperedge_circuit(probability=0.125):
    return stim.Circuit(
        f"""
        R 0 1 2
        CORRELATED_ERROR({probability}) X0 X1 X2
        M 0 1 2
        DETECTOR rec[-3]
        DETECTOR rec[-2]
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-3]
        """
    )


class DemMatrixTests(unittest.TestCase):
    def test_actual_correlated_fault_remains_one_three_detector_variable(self):
        circuit = _three_detector_hyperedge_circuit()
        build = build_detector_error_model(circuit)
        matrices = dem_to_matrices(build.model)

        self.assertFalse(build.used_approximate_disjoint_errors)
        self.assertEqual(matrices.check_matrix.shape, (3, 1))
        self.assertEqual(matrices.observables_matrix.shape, (1, 1))
        np.testing.assert_array_equal(
            matrices.check_matrix.toarray(),
            np.ones((3, 1), dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            matrices.observables_matrix.toarray(),
            np.ones((1, 1), dtype=np.uint8),
        )
        np.testing.assert_allclose(matrices.error_priors, [0.125])

    def test_separator_components_stay_one_mechanism_and_combine_by_parity(self):
        dem = stim.DetectorErrorModel(
            "error(0.2) D0 D1 ^ D1 D2 L0 ^ D3 L0 L1"
        )
        matrices = dem_to_matrices(dem)

        self.assertEqual(matrices.num_errors, 1)
        np.testing.assert_array_equal(
            matrices.check_matrix.toarray(),
            np.array([[1], [0], [1], [1]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            matrices.observables_matrix.toarray(),
            np.array([[0], [1]], dtype=np.uint8),
        )

    def test_repeat_blocks_flatten_without_merging_error_instructions(self):
        dem = stim.DetectorErrorModel(
            """
            repeat 2 {
                error(0.1) D0 D1 L0
                shift_detectors 2
            }
            """
        )
        matrices = dem_to_matrices(dem)

        self.assertEqual(matrices.check_matrix.shape, (4, 2))
        np.testing.assert_array_equal(
            matrices.check_matrix.toarray(),
            np.array(
                [
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                ],
                dtype=np.uint8,
            ),
        )
        np.testing.assert_array_equal(
            matrices.observables_matrix.toarray(),
            np.array([[1, 1]], dtype=np.uint8),
        )


class DemConstructionTests(unittest.TestCase):
    def test_exact_conversion_is_attempted_without_approximation(self):
        build = build_detector_error_model(_three_detector_hyperedge_circuit())
        self.assertFalse(build.used_approximate_disjoint_errors)
        self.assertIsNone(build.exact_rejection)

    def test_disjoint_channel_fallback_occurs_only_after_exact_rejection(self):
        circuit = stim.Circuit(
            """
            R 0 1
            H 0
            CX 0 1
            PAULI_CHANNEL_1(0.1, 0.2, 0) 0
            CX 0 1
            H 0
            M 0 1
            DETECTOR rec[-2]
            DETECTOR rec[-1]
            """
        )
        with self.assertRaisesRegex(ValueError, "approximate_disjoint_errors"):
            build_detector_error_model(
                circuit,
                allow_disjoint_channel_approximation=False,
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build = build_detector_error_model(circuit)
        self.assertTrue(build.used_approximate_disjoint_errors)
        self.assertIn("approximate_disjoint_errors", build.exact_rejection)
        self.assertEqual(len(caught), 1)
        self.assertIn("disjoint-channel approximation", str(caught[0].message))


class DemSamplingTests(unittest.TestCase):
    def test_num_failures_compares_predicted_and_sampled_observables(self):
        circuit = _three_detector_hyperedge_circuit(probability=0.25)
        shots = 17
        seed = 918273
        _, expected_observables = circuit.compile_detector_sampler(seed=seed).sample(
            shots=shots,
            separate_observables=True,
        )
        expected_failures = int(np.count_nonzero(expected_observables[:, 0]))

        class ZeroDecoder:
            def __init__(self, check_matrix, **kwargs):
                self.num_errors = check_matrix.shape[1]
                self.kwargs = kwargs

            def decode(self, _detectors):
                return np.zeros(self.num_errors, dtype=np.uint8)

        progress = queue.SimpleQueue()
        params = [0.125, 80, 300, 60, (-0.16, 0.66), 5]
        with patch(
            "functions.dem_decoding.relay_bp.RelayDecoderF32",
            ZeroDecoder,
        ):
            failures = num_failures_dem(
                circuit,
                params,
                shots,
                sampler_seed=seed,
                worker_id=7,
                progress_queue=progress,
            )

        self.assertEqual(failures, expected_failures)
        updates = [progress.get_nowait() for _ in range(shots)]
        self.assertEqual(updates[-1]["worker_id"], 7)
        self.assertEqual(updates[-1]["shot_num"], shots)
        self.assertEqual(updates[-1]["shots"], shots)
        self.assertEqual(updates[-1]["num_failures"], expected_failures)

    def test_invalid_relay_parameter_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Relay parameters"):
            num_failures_dem(
                _three_detector_hyperedge_circuit(),
                [0.125],
                shots=1,
                sampler_seed=1,
            )


if __name__ == "__main__":
    unittest.main()
