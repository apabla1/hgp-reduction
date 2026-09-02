"""Focused tests for the adaptive phenomenological-DEM sweep."""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

import phenom_dem_adaptive_collection as adaptive
import phenom_dem_data_collection as collector
from functions.phenom_dem_config import (
    CPU200_MAX_WORKERS,
    DECODER_CONFIG,
    HEAWOOD_P_VALUES_ASCENDING,
    LOCAL_MAX_WORKERS,
    MANUSCRIPT_CURVES,
    P_VALUES_BY_CODE_DESCENDING,
    QC_P_VALUES_ASCENDING,
    RELAY_GAMMA0,
    RELAY_GAMMA_INTERVAL,
    RELAY_NUM_SETS,
    SCHEDULE_SEEDS_BY_VARIANT,
    TARGET_RELATIVE_SE,
    p_values_for_code,
    relative_se_target,
)


class FixedConfigurationTests(unittest.TestCase):
    def test_exact_per_code_april_grids_and_task_count(self):
        expected_qc = tuple(float(value) for value in np.geomspace(1e-3, 7e-3, 10))
        expected_heawood = tuple(
            float(value) for value in np.geomspace(5e-4, 7e-3, 10)
        )
        self.assertEqual(QC_P_VALUES_ASCENDING, expected_qc)
        self.assertEqual(HEAWOOD_P_VALUES_ASCENDING, expected_heawood)
        self.assertEqual(
            p_values_for_code("qc_24_6_10"), QC_P_VALUES_ASCENDING
        )
        self.assertEqual(len(adaptive.TASK_LEVELS), 10)
        self.assertTrue(
            all(len(level) == len(MANUSCRIPT_CURVES) for level in adaptive.TASK_LEVELS)
        )
        self.assertEqual(len(adaptive.TASKS), 70)

    def test_relative_error_target_is_constant(self):
        for values in P_VALUES_BY_CODE_DESCENDING.values():
            self.assertTrue(
                all(relative_se_target(p) == TARGET_RELATIVE_SE for p in values)
            )

    def test_relay_parameters_model_and_grids_are_encoded_in_header(self):
        header = adaptive.header_value(adaptive.DEFAULT_SEED_NAMESPACE)
        relay = header["relay_parameters"]
        self.assertEqual(relay["gamma0"], RELAY_GAMMA0)
        self.assertEqual(relay["num_sets"], RELAY_NUM_SETS)
        self.assertEqual(relay["gamma_dist_interval"], list(RELAY_GAMMA_INTERVAL))
        self.assertEqual(
            header["p_values_by_code_descending"],
            {code: list(values) for code, values in P_VALUES_BY_CODE_DESCENDING.items()},
        )
        self.assertEqual(
            header["schedule_seeds_by_variant"],
            SCHEDULE_SEEDS_BY_VARIANT,
        )
        self.assertIn("phenom", DECODER_CONFIG)


class AdaptiveRuleTests(unittest.TestCase):
    def test_relative_standard_error(self):
        self.assertAlmostEqual(
            adaptive.relative_standard_error(20, 1000),
            math.sqrt((1000 - 20) / (1000 * 20)),
        )
        self.assertTrue(math.isinf(adaptive.relative_standard_error(0, 1000)))

    def test_batch_starts_at_one_thousand_and_doubles_after_zero_failures(self):
        self.assertEqual(
            adaptive.choose_batch_shots(adaptive.Observation(), 0.05), 1000
        )
        self.assertEqual(
            adaptive.choose_batch_shots(adaptive.Observation(0, 1000), 0.05),
            1000,
        )
        self.assertEqual(
            adaptive.choose_batch_shots(adaptive.Observation(0, 2000), 0.20),
            2000,
        )

    def test_batch_stops_only_after_target(self):
        done = adaptive.Observation(400, 10000)
        self.assertLessEqual(done.relative_se, 0.05)
        self.assertEqual(adaptive.choose_batch_shots(done, 0.05), 0)
        self.assertGreater(adaptive.choose_batch_shots(adaptive.Observation(100, 10000), 0.05), 0)

    def test_global_worker_budget_is_never_exceeded(self):
        allocation = adaptive.allocate_workers(210, [1000] * 7)
        self.assertEqual(allocation, [30] * 7)
        self.assertEqual(sum(allocation), 210)
        allocation = adaptive.allocate_workers(210, [1000] * 3)
        self.assertEqual(sum(allocation), 210)
        self.assertTrue(all(value <= 1000 for value in allocation))


class WorkerLimitTests(unittest.TestCase):
    def test_local_limit_is_fifteen(self):
        args = adaptive.parse_args(["--data-root", "/tmp/x", "--processes", "15"])
        self.assertEqual(args.processes, LOCAL_MAX_WORKERS)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                adaptive.parse_args(
                    ["--data-root", "/tmp/x", "--processes", "16"]
                )

    def test_cpu200_allows_exactly_two_hundred_ten(self):
        args = adaptive.parse_args(
            [
                "--data-root",
                "/tmp/x",
                "--execution-site",
                "cpu200",
                "--processes",
                "210",
            ]
        )
        self.assertEqual(args.processes, CPU200_MAX_WORKERS)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                adaptive.parse_args(
                    [
                        "--data-root",
                        "/tmp/x",
                        "--execution-site",
                        "cpu200",
                        "--processes",
                        "211",
                    ]
                )

    def test_batch_collector_enforces_the_same_site_limits(self):
        common = [
            "--batch-id",
            "b0",
            "--result",
            "/tmp/b0.json",
            "--code",
            "heawood_cycle",
            "--variant",
            "reduced_split",
            "--p",
            "0.006",
            "--shots",
            "1000",
            "--sampler-seed",
            "1",
        ]
        local = collector.parse_args([*common, "--processes", "15"])
        self.assertEqual(local.processes, 15)
        remote = collector.parse_args(
            [
                *common,
                "--execution-site",
                "cpu200",
                "--processes",
                "210",
            ]
        )
        self.assertEqual(remote.processes, 210)


class LedgerTests(unittest.TestCase):
    def test_batch_result_requires_the_complete_fixed_request(self):
        header = adaptive.header_value(adaptive.DEFAULT_SEED_NAMESPACE)
        state = adaptive.ManifestState(header, [header], {}, {}, {})
        task = adaptive.TASKS[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = adaptive.make_plan(
                state, task, adaptive.Observation(), 1000, 1, "local", root,
                adaptive.DEFAULT_SEED_NAMESPACE,
            )
            path = Path(plan["result_path"])
            path.parent.mkdir(parents=True)
            request = adaptive.expected_batch_request(plan)
            request["relay_parameters"]["gamma0"] = 0.5
            path.write_text(
                json.dumps({"request": request, "failures": 50, "shots": 1000}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                adaptive.AdaptivePhenomDemError,
                "does not exactly match",
            ):
                adaptive.read_batch_result(plan)

    def test_restart_rebuilds_tables_before_completion_check(self):
        header = adaptive.header_value(adaptive.DEFAULT_SEED_NAMESPACE)
        task = adaptive.TASKS[0]
        committed = {
            "event": "batch_committed",
            "batch_id": "committed",
            "task_id": task.task_id,
            "code": task.code,
            "variant": task.variant,
            "p": task.p,
            "failures": 50,
            "shots": 1000,
        }
        state = adaptive.ManifestState(
            header, [header], {}, {"committed": committed}, {}
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            args = adaptive.parse_args(
                [
                    "--data-root", str(root),
                    "--processes", "7",
                    "--max-levels", "1",
                ]
            )
            with (
                patch.object(adaptive, "initialize_state", return_value=state),
                patch.object(adaptive, "TASK_LEVELS", ((task,),)),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(adaptive.run_controller(args), 0)
            table = np.load(
                adaptive.table_path(root, task.code, task.variant),
                allow_pickle=False,
            )
        np.testing.assert_array_equal(table, [[task.p, 50, 1000]])

    def test_nonzero_exit_abandons_even_if_stale_result_is_valid(self):
        header = adaptive.header_value(adaptive.DEFAULT_SEED_NAMESPACE)
        state = adaptive.ManifestState(header, [header], {}, {}, {})
        task = adaptive.TASKS[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = adaptive.make_plan(
                state, task, adaptive.Observation(), 1000, 1, "local", root,
                adaptive.DEFAULT_SEED_NAMESPACE,
            )
            adaptive.register_event(state, plan)
            manifest = root / adaptive.DEFAULT_MANIFEST_NAME
            with (
                patch.object(adaptive, "_run_subprocess", return_value=9),
                patch.object(adaptive, "read_batch_result") as read_result,
            ):
                with self.assertRaises(adaptive.AdaptivePhenomDemError):
                    adaptive.run_plans([plan], state, manifest, root)
            read_result.assert_not_called()
            terminal = state.terminals[plan["batch_id"]]
            self.assertEqual(terminal["event"], "batch_abandoned")
            self.assertEqual(terminal["returncode"], 9)

    def test_run_plans_rechecks_global_worker_cap(self):
        header = adaptive.header_value(adaptive.DEFAULT_SEED_NAMESPACE)
        state = adaptive.ManifestState(header, [header], {}, {}, {})
        plans = [
            {"execution_site": "local", "processes": 8},
            {"execution_site": "local", "processes": 8},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                adaptive.AdaptivePhenomDemError, "exceeding the local cap"
            ):
                adaptive.run_plans(
                    plans, state, root / "manifest.jsonl", root
                )

    def test_first_plan_is_json_safe_and_uses_thirty_of_210_workers(self):
        header = adaptive.header_value(adaptive.DEFAULT_SEED_NAMESPACE)
        state = adaptive.ManifestState(header, [header], {}, {}, {})
        task = adaptive.TASKS[0]
        with tempfile.TemporaryDirectory() as temporary:
            plan = adaptive.make_plan(
                state,
                task,
                adaptive.Observation(),
                1000,
                30,
                "cpu200",
                Path(temporary),
                adaptive.DEFAULT_SEED_NAMESPACE,
            )
        self.assertIsNone(plan["before_relative_se"])
        self.assertEqual(plan["processes"], 30)
        self.assertEqual(plan["command"][-1], "30")
        seed_index = plan["command"].index("--schedule-seed") + 1
        self.assertEqual(plan["command"][seed_index], "0")
        json.dumps(plan, allow_nan=False)

    def test_tables_are_rebuilt_only_from_committed_batches(self):
        header = adaptive.header_value(adaptive.DEFAULT_SEED_NAMESPACE)
        task = adaptive.TASKS[0]
        committed = {
            "event": "batch_committed",
            "batch_id": "committed",
            "task_id": task.task_id,
            "code": task.code,
            "variant": task.variant,
            "p": task.p,
            "failures": 17,
            "shots": 1000,
        }
        abandoned = {
            "event": "batch_abandoned",
            "batch_id": "abandoned",
            "task_id": task.task_id,
            "code": task.code,
            "variant": task.variant,
            "p": task.p,
            "failures": 999,
            "shots": 1000,
        }
        state = adaptive.ManifestState(
            header,
            [header],
            {},
            {"committed": committed, "abandoned": abandoned},
            {},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adaptive.rebuild_tables(state, root)
            table = np.load(
                adaptive.table_path(root, task.code, task.variant),
                allow_pickle=False,
            )
        np.testing.assert_array_equal(table, [[task.p, 17, 1000]])


if __name__ == "__main__":
    unittest.main()
