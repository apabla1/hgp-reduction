"""Focused tests for selected phenomenological-DEM precision refinement."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

import phenom_dem_adaptive_collection as adaptive
import phenom_dem_refinement_collection as refinement


class SelectionTests(unittest.TestCase):
    def test_p_max_and_variant_filter_retain_fixed_level_order(self):
        levels = refinement.selected_task_levels(
            "heawood_cycle",
            ["reduced_split"],
            p_max=0.0013,
            p_values=None,
        )
        tasks = [task for level in levels for task in level]
        self.assertTrue(tasks)
        self.assertTrue(all(task.code == "heawood_cycle" for task in tasks))
        self.assertTrue(all(task.variant == "reduced_split" for task in tasks))
        self.assertTrue(all(task.p <= 0.0013 for task in tasks))
        self.assertEqual(
            [task.p for task in tasks],
            sorted((task.p for task in tasks), reverse=True),
        )

    def test_explicit_p_values_are_canonicalized_to_the_fixed_grid(self):
        requested = [0.0005, 0.0008988042625421107]
        levels = refinement.selected_task_levels(
            "heawood_cycle",
            ["unreduced_cardinal", "reduced_random"],
            p_max=None,
            p_values=requested,
        )
        tasks = [task for level in levels for task in level]
        self.assertEqual({task.p for task in tasks}, set(requested))
        self.assertEqual(len(tasks), 4)


class PlanTests(unittest.TestCase):
    def test_refinement_plan_keeps_canonical_id_seed_and_adds_metadata(self):
        header = adaptive.header_value(adaptive.DEFAULT_SEED_NAMESPACE)
        state = adaptive.ManifestState(header, [header], {}, {}, {})
        task = next(
            task
            for task in adaptive.TASKS
            if task.code == "heawood_cycle"
            and task.variant == "reduced_split"
            and task.p == 0.0005
        )
        before = adaptive.Observation(45, 48000)
        shots = adaptive.choose_batch_shots(before, 0.10)
        with tempfile.TemporaryDirectory() as temporary:
            plan = refinement.make_refinement_plan(
                state,
                task,
                before,
                shots,
                70,
                "cpu200",
                Path(temporary),
                adaptive.DEFAULT_SEED_NAMESPACE,
                0.10,
            )

        self.assertEqual(plan["batch_id"], adaptive.make_batch_id(task, 0))
        self.assertEqual(
            plan["sampler_seed"],
            adaptive.derive_batch_seed(adaptive.DEFAULT_SEED_NAMESPACE, task, 0),
        )
        self.assertEqual(plan["sampling_mode"], refinement.SAMPLING_MODE)
        self.assertEqual(plan["refinement_target_relative_se"], 0.10)
        self.assertEqual(
            plan["refinement_controller_sha256"],
            refinement.refinement_controller_sha256(),
        )
        self.assertEqual(len(plan["refinement_controller_sha256"]), 64)
        json.dumps(plan, allow_nan=False)


class ControllerTests(unittest.TestCase):
    @staticmethod
    def _manifest(root: Path) -> Path:
        path = root / adaptive.DEFAULT_MANIFEST_NAME
        path.write_text(
            json.dumps(
                adaptive.header_value(adaptive.DEFAULT_SEED_NAMESPACE),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_completed_selected_point_launches_no_job(self):
        task = next(
            task
            for task in adaptive.TASKS
            if task.code == "heawood_cycle"
            and task.variant == "reduced_split"
            and task.p == 0.0005
        )
        header = adaptive.header_value(adaptive.DEFAULT_SEED_NAMESPACE)
        terminal = {
            "event": "batch_committed",
            "batch_id": "done",
            "task_id": task.task_id,
            "code": task.code,
            "variant": task.variant,
            "p": task.p,
            "failures": 400,
            "shots": 10000,
        }
        state = adaptive.ManifestState(
            header, [header, terminal], {}, {"done": terminal}, {}
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._manifest(root)
            args = refinement.parse_args(
                [
                    "--data-root", str(root),
                    "--variants", "reduced_split",
                    "--p-values", "0.0005",
                    "--target-relative-se", "0.10",
                ]
            )
            with (
                patch.object(
                    refinement, "load_existing_state",
                    return_value=(state, adaptive.DEFAULT_SEED_NAMESPACE),
                ),
                patch.object(adaptive, "rebuild_tables"),
                patch.object(adaptive, "run_plans") as run_plans,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(refinement.run_controller(args), 0)
        run_plans.assert_not_called()

    def test_cpu200_wave_uses_at_most_global_worker_cap(self):
        tasks = tuple(
            task
            for task in adaptive.TASK_LEVELS[-1]
            if task.code == "heawood_cycle"
        )
        header = adaptive.header_value(adaptive.DEFAULT_SEED_NAMESPACE)
        state = adaptive.ManifestState(header, [header], {}, {}, {})
        captured = []

        def capture(plans, *_args, **_kwargs):
            captured.extend(plans)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._manifest(root)
            args = refinement.parse_args(
                [
                    "--data-root", str(root),
                    "--p-values", "0.0005",
                    "--target-relative-se", "0.10",
                    "--execution-site", "cpu200",
                    "--processes", "210",
                    "--max-waves", "1",
                ]
            )
            with (
                patch.object(
                    refinement, "load_existing_state",
                    return_value=(state, adaptive.DEFAULT_SEED_NAMESPACE),
                ),
                patch.object(
                    refinement, "selected_task_levels", return_value=(tasks,)
                ),
                patch.object(adaptive, "rebuild_tables"),
                patch.object(adaptive, "run_plans", side_effect=capture),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(refinement.run_controller(args), 0)

        self.assertEqual(len(captured), 3)
        self.assertEqual(sum(plan["processes"] for plan in captured), 210)
        self.assertTrue(all(plan["processes"] <= 210 for plan in captured))
        self.assertTrue(
            all(plan["sampling_mode"] == refinement.SAMPLING_MODE for plan in captured)
        )

    def test_site_defaults_and_worker_cap_are_enforced(self):
        args = refinement.parse_args(
            [
                "--data-root", "/tmp/refinement",
                "--p-max", "0.001",
                "--execution-site", "cpu200",
            ]
        )
        self.assertEqual(args.processes, 210)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                refinement.parse_args(
                    [
                        "--data-root", "/tmp/refinement",
                        "--p-max", "0.001",
                        "--execution-site", "cpu200",
                        "--processes", "211",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
