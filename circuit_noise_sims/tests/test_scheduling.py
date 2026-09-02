"""Focused regression tests for manuscript syndrome-extraction schedules."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import stim


SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from data_collection import parse_args
from codes.heawood_cycle import get_heawood_cycle
from codes.quasi_cyclic_codes import get_qc_20_5_9, get_qc_24_6_10
from functions.H_to_CNOT_circuit import (
    generate_cardinal_synd_circuit,
    generate_full_circuit,
    generate_full_circuit_cardinal,
    generate_full_circuit_split,
    generate_synd_circuit,
)
from functions.cardinal_schedule import get_cardinal_schedule
from functions.reduction_funcs import get_reduced_code


def _matrix_depth(matrix):
    row_weights = np.asarray(matrix.sum(axis=1)).ravel()
    column_weights = np.asarray(matrix.sum(axis=0)).ravel()
    return int(max(row_weights.max(initial=0), column_weights.max(initial=0)))


def _expected_hgp_cnot_edges(code):
    n = int(code.N)
    mx = int(code.hx.shape[0])
    expected = set()

    x_rows, x_columns = code.hx.nonzero()
    expected.update(
        (n + int(row), int(column))
        for row, column in zip(x_rows, x_columns)
    )

    z_rows, z_columns = code.hz.nonzero()
    expected.update(
        (int(column), n + mx + int(row))
        for row, column in zip(z_rows, z_columns)
    )
    return expected


def _assert_matching(test_case, layer):
    qubits = [qubit for gate in layer for qubit in gate]
    test_case.assertEqual(len(qubits), len(set(qubits)))


def _instruction_targets(instruction):
    return [int(target.value) for target in instruction.targets_copy()]


def _assert_cnot_layer_noise(test_case, circuit, all_qubits):
    """Check CX, two-qubit noise, idle noise, TICK structure."""

    instructions = list(circuit)
    for index, instruction in enumerate(instructions):
        if instruction.name != "CX":
            continue

        active = set(_instruction_targets(instruction))
        test_case.assertEqual(instructions[index + 1].name, "DEPOLARIZE2")
        test_case.assertEqual(
            _instruction_targets(instructions[index + 1]),
            _instruction_targets(instruction),
        )
        test_case.assertEqual(instructions[index + 2].name, "DEPOLARIZE1")
        idle = set(_instruction_targets(instructions[index + 2]))
        test_case.assertEqual(idle, set(all_qubits) - active)
        test_case.assertEqual(instructions[index + 3].name, "TICK")


class CardinalScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qc_code, cls.qc_h = get_qc_20_5_9()
        cls.heawood_code, cls.heawood_h = get_heawood_cycle()

    def _check_schedule(self, code, h, expected_gate_count):
        schedule, component_depths = get_cardinal_schedule(code, h, h, seed=0)
        repeat_schedule, repeat_depths = get_cardinal_schedule(code, h, h, seed=0)

        self.assertEqual(component_depths, {"E": 2, "N": 2, "S": 2, "W": 2})
        self.assertEqual(sum(component_depths.values()), 8)
        self.assertEqual(schedule, repeat_schedule)
        self.assertEqual(component_depths, repeat_depths)

        gates = [gate for layer in schedule for gate in layer]
        self.assertEqual(len(gates), expected_gate_count)
        self.assertEqual(len(gates), len(set(gates)))
        self.assertEqual(set(gates), _expected_hgp_cnot_edges(code))
        for layer in schedule:
            _assert_matching(self, layer)

    def _check_noiseless_action(self, code, h):
        n = int(code.N)
        mx = int(code.hx.shape[0])
        mz = int(code.hz.shape[0])
        all_qubits = range(n + mx + mz)
        x_checks = range(n, n + mx)
        z_checks = range(n + mx, n + mx + mz)

        cardinal = generate_cardinal_synd_circuit(
            code, h, h, p1=0.0, p2=0.0, seed=0
        )[0].without_noise()
        separate = (
            generate_synd_circuit(
                code.hz,
                z_checks,
                stab_type=0,
                p1=0.0,
                p2=0.0,
                seed=0,
                all_qubits=all_qubits,
            )
            + generate_synd_circuit(
                code.hx,
                x_checks,
                stab_type=1,
                p1=0.0,
                p2=0.0,
                seed=0,
                all_qubits=all_qubits,
            )
        ).without_noise()

        self.assertEqual(
            stim.Tableau.from_circuit(cardinal),
            stim.Tableau.from_circuit(separate),
        )

    def test_qc_seed_zero_schedule(self):
        self._check_schedule(self.qc_code, self.qc_h, expected_gate_count=3150)
        self._check_noiseless_action(self.qc_code, self.qc_h)

        _, seed_one_depths = get_cardinal_schedule(
            self.qc_code, self.qc_h, self.qc_h, seed=1
        )
        self.assertEqual(seed_one_depths, {"E": 3, "N": 3, "S": 3, "W": 3})

    def test_heawood_seed_zero_schedule(self):
        self._check_schedule(
            self.heawood_code, self.heawood_h, expected_gate_count=2652
        )
        self._check_noiseless_action(self.heawood_code, self.heawood_h)


class CodeAndSplitScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qc_code, cls.qc_h = get_qc_20_5_9()
        cls.qc24_code, cls.qc24_h = get_qc_24_6_10()
        cls.heawood_code, cls.heawood_h = get_heawood_cycle()
        cls.qc_reduction = get_reduced_code(cls.qc_code, cls.qc_h)
        cls.qc24_reduction = get_reduced_code(cls.qc24_code, cls.qc24_h)
        cls.heawood_reduction = get_reduced_code(cls.heawood_code, cls.heawood_h)

    def test_heawood_dimensions_match_manuscript(self):
        self.assertEqual(self.heawood_h.shape, (13, 21))
        self.assertEqual(
            (self.heawood_code.N, self.heawood_code.K, self.heawood_code.D),
            (610, 64, 6),
        )

        reduced_code = self.heawood_reduction[6]
        self.assertEqual((reduced_code.N, reduced_code.K), (441, 64))

    def test_qc_dimensions(self):
        self.assertEqual(
            (self.qc_code.N, self.qc_code.K, self.qc_code.D),
            (625, 25, 9),
        )
        reduced_code = self.qc_reduction[6]
        self.assertEqual((reduced_code.N, reduced_code.K), (475, 25))

    def test_qc24_fixed_footprint_dimensions(self):
        self.assertEqual(
            (self.qc24_code.N, self.qc24_code.K, self.qc24_code.D),
            (900, 36, 10),
        )
        reduced_code = self.qc24_reduction[6]
        self.assertEqual((reduced_code.N, reduced_code.K), (684, 36))
        self.assertEqual(self.qc24_reduction[-1], 10)

    def test_split_component_depths(self):
        qc_parts = self.qc_reduction[:6]
        heawood_parts = self.heawood_reduction[:6]

        self.assertEqual(
            tuple(_matrix_depth(part) for part in qc_parts),
            (5, 3, 3, 5, 3, 3),
        )
        self.assertEqual(sum(_matrix_depth(part) for part in qc_parts), 22)

        self.assertEqual(
            tuple(_matrix_depth(part) for part in heawood_parts),
            (3, 0, 3, 3, 0, 3),
        )
        self.assertEqual(sum(_matrix_depth(part) for part in heawood_parts), 12)

    def test_split_heawood_uses_two_hadamard_layers_with_empty_middle(self):
        hx1, hx2, hx3, hz1, hz2, hz3 = self.heawood_reduction[:6]
        self.assertEqual(hx2.nnz, 0)
        circuit = generate_full_circuit_split(
            hx1,
            hx2,
            hx3,
            hz1,
            hz2,
            hz3,
            rounds=1,
            noise_pars=(1e-4, 1e-3, 1e-3),
            seed=1,
            code=self.heawood_reduction[6],
        )

        h_instructions = [instruction for instruction in circuit if instruction.name == "H"]
        self.assertEqual(len(h_instructions), 2)
        self.assertTrue(
            all(
                len(_instruction_targets(instruction)) == hx1.shape[0]
                for instruction in h_instructions
            )
        )


class CircuitNoiseStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code, cls.h = get_heawood_cycle()
        cls.reduction = get_reduced_code(cls.code, cls.h)

    def test_data_preparation_spam_in_all_builders(self):
        hx1, hx2, hx3, hz1, hz2, hz3 = self.reduction[:6]
        circuits = (
            generate_full_circuit(
                self.code, rounds=1, noise_pars=(1e-4, 1e-3, 2e-3), seed=1
            ),
            generate_full_circuit_cardinal(
                self.code,
                self.h,
                self.h,
                rounds=1,
                noise_pars=(1e-4, 1e-3, 2e-3),
                seed=0,
            ),
            generate_full_circuit_split(
                hx1,
                hx2,
                hx3,
                hz1,
                hz2,
                hz3,
                rounds=1,
                noise_pars=(1e-4, 1e-3, 2e-3),
                seed=1,
                code=self.reduction[6],
            ),
        )

        data_qubits = set(range(int(self.code.N)))
        reduced_data_qubits = set(range(int(self.reduction[6].N)))
        expected_data_sets = (data_qubits, data_qubits, reduced_data_qubits)
        for circuit, expected_data in zip(circuits, expected_data_sets):
            first_error = next(
                instruction for instruction in circuit if instruction.name == "X_ERROR"
            )
            self.assertTrue(expected_data.issubset(set(_instruction_targets(first_error))))

    def test_original_cnot_layers_have_full_idle_noise(self):
        circuit = generate_full_circuit(
            self.code, rounds=1, noise_pars=(1e-4, 1e-3, 2e-3), seed=1
        )
        all_qubits = range(
            int(self.code.N + self.code.hx.shape[0] + self.code.hz.shape[0])
        )
        _assert_cnot_layer_noise(self, circuit, all_qubits)

    def test_cardinal_cnot_layers_have_full_idle_noise(self):
        circuit, component_depths = generate_cardinal_synd_circuit(
            self.code, self.h, self.h, p1=1e-4, p2=1e-3, seed=0
        )
        self.assertEqual(sum(component_depths.values()), 8)
        all_qubits = range(
            int(self.code.N + self.code.hx.shape[0] + self.code.hz.shape[0])
        )
        _assert_cnot_layer_noise(self, circuit, all_qubits)

    def test_split_cnot_layers_have_full_idle_noise(self):
        hx1, hx2, hx3, hz1, hz2, hz3 = self.reduction[:6]
        circuit = generate_full_circuit_split(
            hx1,
            hx2,
            hx3,
            hz1,
            hz2,
            hz3,
            rounds=1,
            noise_pars=(1e-4, 1e-3, 2e-3),
            seed=1,
            code=self.reduction[6],
        )
        all_qubits = range(
            int(hx1.shape[1] + hx1.shape[0] + hz1.shape[0])
        )
        _assert_cnot_layer_noise(self, circuit, all_qubits)


class ScheduleSeedArgumentTests(unittest.TestCase):
    def test_schedule_seed_is_fixed_by_default(self):
        argv = ["data_collection.py", "--shots", "1", "--decoder", "Relay"]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.schedule_seed, 1)

    def test_schedule_seed_can_be_selected(self):
        argv = [
            "data_collection.py",
            "--shots", "1",
            "--decoder", "Relay",
            "--schedule-seed", "7",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.schedule_seed, 7)


if __name__ == "__main__":
    unittest.main()
