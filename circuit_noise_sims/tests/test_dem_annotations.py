"""Tests for canonical HGP observables and Stim Z-detector annotations."""

import sys
import unittest
from pathlib import Path

import numpy as np


SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from codes.heawood_cycle import get_heawood_cycle
from codes.quasi_cyclic_codes import get_qc_20_5_9
from functions.H_to_CNOT_circuit import (
    generate_full_circuit,
    generate_full_circuit_cardinal,
    generate_full_circuit_split,
)
from functions.logical_basis import canonical_hgp_left_logicals, canonical_z_logicals
from functions.reduction_funcs import get_reduced_code


def _is_zero(matrix):
    matrix = matrix.tocsr(copy=True)
    matrix.data %= 2
    matrix.eliminate_zeros()
    return matrix.nnz == 0


class CanonicalLogicalBasisTests(unittest.TestCase):
    def _check_family(self, getter):
        code, h = getter()
        lx, lz = canonical_hgp_left_logicals(code.h1, code.h2)

        self.assertEqual(lx.shape, (int(code.K), int(code.N)))
        self.assertEqual(lz.shape, (int(code.K), int(code.N)))
        np.testing.assert_array_equal(
            (lx @ lz.T).toarray() % 2,
            np.eye(int(code.K), dtype=np.uint8),
        )
        self.assertTrue(_is_zero(code.hz @ lx.T))
        self.assertTrue(_is_zero(code.hx @ lz.T))

        reduction = get_reduced_code(code, h)
        reduced_code = reduction[6]
        kept = reduced_code.original_qubit_indices
        np.testing.assert_array_equal(
            reduced_code.canonical_lx.toarray(),
            lx[:, kept].toarray(),
        )
        np.testing.assert_array_equal(
            reduced_code.canonical_lz.toarray(),
            lz[:, kept].toarray(),
        )
        np.testing.assert_array_equal(
            (reduced_code.canonical_lx @ reduced_code.canonical_lz.T).toarray()
            % 2,
            np.eye(int(reduced_code.K), dtype=np.uint8),
        )
        self.assertTrue(_is_zero(reduced_code.hz @ reduced_code.canonical_lx.T))
        self.assertTrue(_is_zero(reduced_code.hx @ reduced_code.canonical_lz.T))

    def test_qc_canonical_basis_survives_reduction(self):
        self._check_family(get_qc_20_5_9)

    def test_heawood_canonical_basis_survives_reduction(self):
        self._check_family(get_heawood_cycle)


class ZMemoryAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code, cls.h = get_heawood_cycle()
        cls.reduction = get_reduced_code(cls.code, cls.h)
        cls.reduced_code = cls.reduction[6]

    def _assert_annotations(self, circuit, code, rounds, records_x_checks):
        mz = int(code.hz.shape[0])
        mx = int(code.hx.shape[0])
        n = int(code.N)

        self.assertEqual(circuit.num_detectors, (rounds + 1) * mz)
        self.assertEqual(circuit.num_observables, int(code.K))
        expected_measurements = rounds * (mz + (mx if records_x_checks else 0)) + n
        self.assertEqual(circuit.num_measurements, expected_measurements)

        noiseless = circuit.without_noise().compile_detector_sampler(seed=7)
        detectors, observables = noiseless.sample(8, separate_observables=True)
        self.assertFalse(detectors.any())
        self.assertFalse(observables.any())

        # Stim also verifies that every declared detector and observable is
        # deterministic when it constructs the model without gauge options.
        dem = circuit.detector_error_model(decompose_errors=False)
        self.assertEqual(dem.num_detectors, circuit.num_detectors)
        self.assertEqual(dem.num_observables, circuit.num_observables)

        observable_instructions = [
            instruction
            for instruction in circuit
            if instruction.name == "OBSERVABLE_INCLUDE"
        ]
        logicals = canonical_z_logicals(code).tocsr()
        self.assertEqual(len(observable_instructions), logicals.shape[0])
        for index, instruction in enumerate(observable_instructions):
            actual = sorted(int(target.value) for target in instruction.targets_copy())
            support = logicals.indices[logicals.indptr[index] : logicals.indptr[index + 1]]
            expected = sorted(-n + int(qubit) for qubit in support)
            self.assertEqual(actual, expected)

    def test_unsplit_reduced_annotations(self):
        rounds = 2
        circuit = generate_full_circuit(
            self.reduced_code,
            rounds=rounds,
            noise_pars=(6e-4, 6e-3, 6e-3),
            seed=1,
        )
        self._assert_annotations(circuit, self.reduced_code, rounds, False)

    def test_cardinal_annotations(self):
        rounds = 2
        circuit = generate_full_circuit_cardinal(
            self.code,
            self.h,
            self.h,
            rounds=rounds,
            noise_pars=(6e-4, 6e-3, 6e-3),
            seed=0,
        )
        self._assert_annotations(circuit, self.code, rounds, False)

    def test_split_annotations_and_correlated_hyperedges(self):
        rounds = 2
        circuit = generate_full_circuit_split(
            *self.reduction[:6],
            rounds=rounds,
            noise_pars=(6e-4, 6e-3, 6e-3),
            seed=1,
            code=self.reduced_code,
        )
        self._assert_annotations(circuit, self.reduced_code, rounds, True)

        dem = circuit.detector_error_model(decompose_errors=False)
        max_detector_weight = max(
            sum(target.is_relative_detector_id() for target in instruction.targets_copy())
            for instruction in dem.flattened()
            if instruction.type == "error"
        )
        self.assertGreater(max_detector_weight, 2)


if __name__ == "__main__":
    unittest.main()
