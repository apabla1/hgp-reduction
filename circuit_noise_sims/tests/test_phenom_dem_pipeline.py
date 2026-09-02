"""Focused tests for the phenomenological-DEM batch pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import stim


SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

import phenom_dem_data_collection as collector
from functions import phenom_dem_config as config


class ConfigurationTests(unittest.TestCase):
    def test_april_relay_parameters_and_worker_caps(self):
        self.assertEqual(
            config.RELAY_PARAMETERS,
            (0.65, 80, 100, 60, (-0.24, 0.66), 5),
        )
        self.assertEqual(config.worker_limit("local"), 15)
        self.assertEqual(config.worker_limit("cpu200"), 210)

    def test_curve_ids_and_probability_order_are_unique(self):
        curve_ids = [curve.curve_id for curve in config.MANUSCRIPT_CURVES]
        self.assertEqual(len(curve_ids), len(set(curve_ids)))
        for code, descending in config.P_VALUES_BY_CODE_DESCENDING.items():
            self.assertEqual(tuple(sorted(descending, reverse=True)), descending)
            self.assertEqual(
                tuple(reversed(config.P_VALUES_BY_CODE_ASCENDING[code])),
                descending,
            )

    def test_schedule_seeds_are_fixed_by_variant(self):
        self.assertEqual(config.schedule_seed("unreduced_cardinal"), 0)
        self.assertEqual(config.schedule_seed("unreduced_random"), 1)
        self.assertEqual(config.schedule_seed("reduced_random"), 1)
        self.assertEqual(config.schedule_seed("reduced_split"), 1)


class BatchCollectorTests(unittest.TestCase):
    def _args(self, result: Path) -> argparse.Namespace:
        return argparse.Namespace(
            batch_id="batch-0001",
            result=result,
            code="qc_20_5_9",
            variant="reduced_split",
            p=0.002,
            shots=1000,
            sampler_seed=1234,
            schedule_seed=1,
            processes=15,
            execution_site="local",
        )

    def test_sampler_seed_derivation_is_repeatable_and_worker_specific(self):
        seeds = [collector.derive_sampler_seed(1234, index) for index in range(1, 8)]
        self.assertEqual(
            seeds,
            [collector.derive_sampler_seed(1234, index) for index in range(1, 8)],
        )
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertNotIn(0, seeds)

    def test_request_records_fixed_noise_and_relay_configuration(self):
        spec = collector.request_spec(self._args(Path("unused.json")))
        self.assertEqual(spec["model"], config.MODEL_NAME)
        self.assertEqual(spec["decoder_config"], config.DECODER_CONFIG)
        self.assertEqual((spec["p1"], spec["p2"], spec["p_spam"]), (0.0002, 0.002, 0.002))
        self.assertEqual(spec["schedule_seed"], 1)
        self.assertEqual(
            spec["relay_parameters"],
            {
                "gamma0": 0.65,
                "pre_iter": 80,
                "num_sets": 100,
                "set_max_iter": 60,
                "gamma_dist_interval": [-0.24, 0.66],
                "stop_nconv": 5,
            },
        )

    def test_cnot_fingerprint_retains_pair_order_and_layers(self):
        circuit = stim.Circuit(
            """
            CX 0 1 2 3
            TICK
            CX 2 1
            """
        )
        metadata = collector.cnot_order_metadata(circuit)
        self.assertEqual(metadata["cnot_pair_count"], 3)
        self.assertEqual(metadata["cnot_layer_count"], 2)
        self.assertEqual(metadata["cnot_layer_pair_counts"], [[0, 2], [1, 1]])
        self.assertEqual(len(metadata["cnot_order_sha256"]), 64)

    def test_collect_writes_once_and_reuses_an_identical_result(self):
        class FakeModel:
            def metadata(self):
                return {"graph_sha256": "a" * 64, "num_phenomenological_variables": 3}

        circuit = stim.Circuit("R 0\nM 0")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.json"
            args = self._args(path)
            built = (
                circuit,
                FakeModel(),
                {"code_n": 1, "code_k": 0, "rounds": 1, "schedule_seed": 1},
                {"circuit_sha256": "b" * 64},
                {"graph_sha256": "a" * 64, "metadata_sha256": "c" * 64},
            )
            with (
                patch.object(collector, "build_circuit_and_model", return_value=built) as build,
                patch.object(collector, "sample_batch", return_value=7) as sample,
            ):
                first = collector.collect(args)
                second = collector.collect(args)

            self.assertEqual(first, second)
            self.assertEqual((first["failures"], first["shots"]), (7, 1000))
            self.assertEqual(build.call_count, 1)
            self.assertEqual(sample.call_count, 1)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), first)

    def test_cardinal_cli_infers_zero_and_rejects_seed_one(self):
        argv = [
            "--batch-id", "b",
            "--result", "result.json",
            "--code", "qc_20_5_9",
            "--variant", "unreduced_cardinal",
            "--p", "0.002",
            "--shots", "10",
            "--sampler-seed", "1",
            "--processes", "1",
        ]
        self.assertEqual(collector.parse_args(argv).schedule_seed, 0)
        with self.assertRaises(SystemExit):
            collector.parse_args(argv + ["--schedule-seed", "1"])

    def test_runtime_version_keys_are_complete(self):
        self.assertEqual(
            set(collector.runtime_versions()),
            {"python", "numpy", "scipy", "stim", "ldpc", "relay_bp"},
        )

    def test_argument_parser_enforces_site_caps_and_shot_count(self):
        common = [
            "--batch-id", "b",
            "--result", "result.json",
            "--code", "qc_20_5_9",
            "--variant", "reduced_split",
            "--p", "0.002",
            "--shots", "1000",
            "--sampler-seed", "1",
            "--execution-site", "cpu200",
        ]
        self.assertEqual(collector.parse_args(common + ["--processes", "210"]).processes, 210)
        with self.assertRaises(SystemExit):
            collector.parse_args(common + ["--processes", "211"])


if __name__ == "__main__":
    unittest.main()
