"""End-to-end tests for strict phenomenological-DEM result auditing."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path


SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

import audit_phenom_dem_results as audit
import phenom_dem_adaptive_collection as adaptive
import phenom_dem_data_collection as collector
import phenom_dem_refinement_collection as refinement
from functions.phenom_dem_config import schedule_seed


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _CompletedSweep:
    def __init__(self, root: Path, *, workers_per_task: int = 2):
        self.root = root
        header = adaptive.header_value(adaptive.DEFAULT_SEED_NAMESPACE)
        self.state = adaptive.ManifestState(header, [header], {}, {}, {})
        self.result_paths = {}

        for level in adaptive.TASK_LEVELS:
            plans = []
            for task in level:
                plan = adaptive.make_plan(
                    self.state,
                    task,
                    adaptive.Observation(),
                    1000,
                    workers_per_task,
                    "local",
                    root,
                    adaptive.DEFAULT_SEED_NAMESPACE,
                )
                adaptive.register_event(self.state, plan)
                plans.append(plan)

            for plan in plans:
                result = self._result(plan)
                path = Path(plan["result_path"])
                collector.atomic_write_json(path, result)
                path.with_suffix(path.suffix + ".lock").touch()
                terminal = adaptive.make_terminal(
                    plan,
                    "batch_committed",
                    result=result,
                    returncode=0,
                )
                adaptive.register_event(self.state, terminal)
                self.result_paths[plan["task_id"]] = path

        self.manifest = root / adaptive.DEFAULT_MANIFEST_NAME
        for event in self.state.events:
            adaptive.append_jsonl(self.manifest, event)
        adaptive.rebuild_tables(self.state, root)

    def _result(self, plan):
        task_id = str(plan["task_id"])
        code_name = str(plan["code"])
        variant = str(plan["variant"])
        reduced = variant.startswith("reduced_")
        code_k = 5 if reduced else 7
        code_n = 40 if reduced else 50
        detectors = 27
        circuit_metadata = {
            "circuit_sha256": _digest(f"circuit|{task_id}"),
            "num_qubits": 80,
            "num_measurements": 120,
            "num_detectors": detectors,
            "num_observables": code_k,
            "cnot_order_sha256": _digest(f"cnot|{code_name}|{variant}"),
            "cnot_pair_count": 100,
            "cnot_layer_count": 20,
            "cnot_layer_pair_counts": [[0, 5]],
        }
        model_metadata = {
            "projection_version": audit.PROJECTION_VERSION,
            "projection_method": (
                "preserve each DEM mechanism's detectors and canonical "
                "observables, then retain independent one-variable marginals"
            ),
            "spatial_solver": audit.SPATIAL_SOLVER,
            "graph_projection_solver": audit.GRAPH_PROJECTION_SOLVER,
            "spatial_solver_error_prior": 0.01,
            "spatial_solver_max_iter": 100,
            "num_detectors": detectors,
            "num_observables": code_k,
            "num_phenomenological_variables": 90,
            "num_dem_mechanisms": 200,
            "num_unique_spatial_rhs": 20,
            "num_zero_projection_mechanisms": 0,
            "num_osd_spatial_solutions": 0,
            "max_spatial_solution_weight": 3,
            "num_unique_projection_rhs": 20,
            "num_improved_initial_projections": 4,
            "max_initial_projection_weight": 10,
            "max_projection_weight": 8,
            "all_spatial_solutions_minimum_weight_certified": True,
            "all_projections_minimum_weight_certified": True,
            "used_approximate_disjoint_errors": False,
            "exact_dem_rejection": None,
            "graph_sha256": _digest(f"graph|{code_name}|{variant}"),
            "observables_sha256": _digest(
                f"observables|{code_name}|{variant}"
            ),
            "priors_sha256": _digest(f"priors|{task_id}"),
            "projection_sha256": _digest(f"projection|{code_name}|{variant}"),
        }
        model_metadata["metadata_sha256"] = collector.sha256_json(model_metadata)
        return {
            "request": audit._expected_request(plan),
            "completed_utc": "2026-08-31T00:00:00Z",
            "failures": 50,
            "shots": int(plan["batch_shots"]),
            "code_metadata": {
                "code_n": code_n,
                "code_k": code_k,
                "rounds": 9,
                "schedule_seed": schedule_seed(variant),
            },
            "circuit_metadata": circuit_metadata,
            "phenomenological_model": model_metadata,
            "runtime_versions": {
                "python": "3.11.13",
                "numpy": "2.3.0",
                "scipy": "1.16.0",
                "stim": "1.15.0",
                "ldpc": "2.4.0",
                "relay_bp": "0.2.0",
            },
        }

    def add_refinement_batch(self, *, plan_updates=None):
        task = adaptive.TASKS[0]
        before = adaptive.observation(self.state, task)
        target = refinement.DEFAULT_TARGET_RELATIVE_SE
        shots = adaptive.choose_batch_shots(before, target)
        plan = refinement.make_refinement_plan(
            self.state,
            task,
            before,
            shots,
            2,
            "local",
            self.root,
            adaptive.DEFAULT_SEED_NAMESPACE,
            target,
        )
        if plan_updates is not None:
            plan.update(plan_updates)
        adaptive.register_event(self.state, plan)
        adaptive.append_jsonl(self.manifest, plan)

        result = self._result(plan)
        path = Path(plan["result_path"])
        collector.atomic_write_json(path, result)
        path.with_suffix(path.suffix + ".lock").touch()
        terminal = adaptive.make_terminal(
            plan,
            "batch_committed",
            result=result,
            returncode=0,
        )
        adaptive.register_event(self.state, terminal)
        adaptive.append_jsonl(self.manifest, terminal)
        adaptive.rebuild_tables(self.state, self.root)
        return plan


class StrictAuditTests(unittest.TestCase):
    def _rewrite_manifest(self, sweep, events):
        sweep.manifest.write_text(
            "".join(
                json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                for event in events
            ),
            encoding="utf-8",
        )

    def test_complete_synthetic_sweep_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep = _CompletedSweep(Path(directory))
            summary = audit.audit_results(sweep.root)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["tasks"], 70)
        self.assertEqual(summary["global"]["committed_batches"], 70)
        self.assertEqual(summary["global"]["batch_result_files"], 70)
        self.assertEqual(summary["global"]["batch_lock_files"], 70)
        self.assertLessEqual(summary["global"]["max_achieved_target_ratio"], 1)

    def test_canonical_precision_refinement_batch_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep = _CompletedSweep(Path(directory))
            plan = sweep.add_refinement_batch()
            summary = audit.audit_results(sweep.root)
            task = adaptive.TASKS[0]
            table = __import__("numpy").load(
                adaptive.table_path(sweep.root, task.code, task.variant),
                allow_pickle=False,
            )
            row = table[table[:, 0] == task.p][0]

        self.assertEqual(plan["sampling_mode"], "precision_refinement_v1")
        self.assertEqual(plan["refinement_target_relative_se"], 0.10)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["global"]["committed_batches"], 71)
        self.assertEqual(summary["global"]["batch_result_files"], 71)
        self.assertEqual(row[1:].tolist(), [100.0, 2000.0])

    def test_tampered_precision_refinement_metadata_is_rejected(self):
        cases = (
            (
                "target",
                {"refinement_target_relative_se": adaptive.TASKS[0].target},
                "does not tighten the uncertainty target",
            ),
            (
                "controller hash",
                {"refinement_controller_sha256": "0" * 64},
                "has an unknown controller hash",
            ),
        )
        for label, updates, error in cases:
            with self.subTest(field=label):
                with tempfile.TemporaryDirectory() as directory:
                    sweep = _CompletedSweep(Path(directory))
                    sweep.add_refinement_batch(plan_updates=updates)
                    with self.assertRaisesRegex(
                        audit.PhenomDemAuditError, error,
                    ):
                        audit.audit_results(sweep.root)

    def test_wrong_precision_refinement_batch_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep = _CompletedSweep(Path(directory))
            sweep.add_refinement_batch(plan_updates={"batch_shots": 2000})
            with self.assertRaisesRegex(
                audit.PhenomDemAuditError,
                "does not follow the precision-refinement batch-size rule",
            ):
                audit.audit_results(sweep.root)

    def test_unmet_precision_refinement_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep = _CompletedSweep(Path(directory))
            sweep.add_refinement_batch(
                plan_updates={"refinement_target_relative_se": 0.05}
            )
            with self.assertRaisesRegex(
                audit.PhenomDemAuditError,
                "misses its recorded relative-error target",
            ):
                audit.audit_results(sweep.root)

    def test_retrieved_remote_collector_paths_are_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep = _CompletedSweep(Path(directory))
            events = adaptive.read_jsonl(sweep.manifest)
            remote_source = Path(
                "/home/ubuntu/"
                "hgp-phenom-src-"
                "c788ec2af870ea0b3310fbbb2bd3338c338e1000d455c3c718bdff49ba1df67a"
            )
            for event in events:
                if event.get("event") == "batch_planned":
                    event["command"][1] = str(remote_source / adaptive.COLLECTOR.name)
            self._rewrite_manifest(sweep, events)
            summary = audit.audit_results(sweep.root)
        self.assertEqual(summary["status"], "ok")

    def test_wrong_collector_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep = _CompletedSweep(Path(directory))
            events = adaptive.read_jsonl(sweep.manifest)
            plan = next(
                event for event in events
                if event.get("event") == "batch_planned"
            )
            plan["command"][1] = str(
                Path(plan["command"][1]).with_name("other.py")
            )
            self._rewrite_manifest(sweep, events)
            with self.assertRaisesRegex(
                audit.PhenomDemAuditError,
                "does not name the fixed DEM collector",
            ):
                audit.audit_results(sweep.root)

    def test_missing_result_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep = _CompletedSweep(Path(directory))
            path = next(iter(sweep.result_paths.values()))
            path.with_suffix(path.suffix + ".lock").unlink()
            with self.assertRaisesRegex(
                audit.PhenomDemAuditError, "missing_locks",
            ):
                audit.audit_results(sweep.root)

    def test_unexpected_batch_files_are_rejected(self):
        for filename in ("orphan.json.lock", ".batch.json.123.tmp", "notes.txt"):
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as directory:
                    sweep = _CompletedSweep(Path(directory))
                    unexpected = sweep.root / "batches" / filename
                    unexpected.write_text("unexpected\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        audit.PhenomDemAuditError, "unexpected",
                    ):
                        audit.audit_results(sweep.root)

    def test_global_concurrent_worker_cap_is_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep = _CompletedSweep(Path(directory), workers_per_task=3)
            with self.assertRaisesRegex(
                audit.PhenomDemAuditError, "exceeds the local worker cap"
            ):
                audit.audit_results(sweep.root)

    def test_tampered_model_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep = _CompletedSweep(Path(directory))
            path = next(iter(sweep.result_paths.values()))
            value = json.loads(path.read_text(encoding="utf-8"))
            value["phenomenological_model"]["priors_sha256"] = "f" * 64
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(
                audit.PhenomDemAuditError, "differs from its result in priors_sha256"
            ):
                audit.audit_results(sweep.root)

    def test_table_not_equal_to_committed_batches_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep = _CompletedSweep(Path(directory))
            first = adaptive.TASKS[0]
            path = adaptive.table_path(sweep.root, first.code, first.variant)
            table = __import__("numpy").load(path, allow_pickle=False)
            table[0, 1] += 1
            with path.open("wb") as handle:
                __import__("numpy").save(handle, table, allow_pickle=False)
            with self.assertRaisesRegex(
                audit.PhenomDemAuditError,
                "not exactly the sum of committed immutable batches",
            ):
                audit.audit_results(sweep.root)

    def test_request_spec_exactly_matches_auditor_schema(self):
        header = adaptive.header_value(adaptive.DEFAULT_SEED_NAMESPACE)
        for task in adaptive.TASK_LEVELS[0][:2]:
            state = adaptive.ManifestState(header, [header], {}, {}, {})
            plan = adaptive.make_plan(
                state, task, adaptive.Observation(), 1000, 2, "local",
                Path("/tmp/unused"), adaptive.DEFAULT_SEED_NAMESPACE,
            )
            args = SimpleNamespace(
                batch_id=plan["batch_id"],
                code=plan["code"],
                variant=plan["variant"],
                p=plan["p"],
                shots=plan["batch_shots"],
                sampler_seed=plan["sampler_seed"],
                schedule_seed=schedule_seed(plan["variant"]),
                processes=plan["processes"],
                execution_site=plan["execution_site"],
            )
            self.assertEqual(
                collector.request_spec(args), audit._expected_request(plan)
            )

    def test_abandoned_batch_prevents_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep = _CompletedSweep(Path(directory))
            events = adaptive.read_jsonl(sweep.manifest)
            terminal = next(
                event
                for event in reversed(events)
                if event.get("event") == "batch_committed"
            )
            terminal["event"] = "batch_abandoned"
            self._rewrite_manifest(sweep, events)
            with self.assertRaisesRegex(
                audit.PhenomDemAuditError, "manifest contains 1 abandoned batch"
            ):
                audit.audit_results(sweep.root)

    def test_across_p_cnot_topology_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep = _CompletedSweep(Path(directory))
            path = next(iter(sweep.result_paths.values()))
            value = json.loads(path.read_text(encoding="utf-8"))
            changed_hash = _digest("changed-cnot-topology")
            value["circuit_metadata"]["cnot_order_sha256"] = changed_hash
            path.write_text(json.dumps(value), encoding="utf-8")
            events = adaptive.read_jsonl(sweep.manifest)
            batch_id = value["request"]["batch_id"]
            terminal = next(
                event for event in events
                if event.get("event") == "batch_committed"
                and event.get("batch_id") == batch_id
            )
            terminal["cnot_order_sha256"] = changed_hash
            self._rewrite_manifest(sweep, events)
            with self.assertRaisesRegex(
                audit.PhenomDemAuditError, "changes cnot_order_sha256 across p"
            ):
                audit.audit_results(sweep.root)

    def test_uncertified_projection_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep = _CompletedSweep(Path(directory))
            path = next(iter(sweep.result_paths.values()))
            value = json.loads(path.read_text(encoding="utf-8"))
            model = value["phenomenological_model"]
            model["all_projections_minimum_weight_certified"] = False
            metadata_without_hash = dict(model)
            metadata_without_hash.pop("metadata_sha256")
            model["metadata_sha256"] = collector.sha256_json(metadata_without_hash)
            path.write_text(json.dumps(value), encoding="utf-8")
            events = adaptive.read_jsonl(sweep.manifest)
            batch_id = value["request"]["batch_id"]
            terminal = next(
                event for event in events
                if event.get("event") == "batch_committed"
                and event.get("batch_id") == batch_id
            )
            terminal["metadata_sha256"] = model["metadata_sha256"]
            self._rewrite_manifest(sweep, events)
            with self.assertRaisesRegex(
                audit.PhenomDemAuditError, "uncertified graph projections"
            ):
                audit.audit_results(sweep.root)

    def test_runtime_version_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep = _CompletedSweep(Path(directory))
            path = next(iter(sweep.result_paths.values()))
            value = json.loads(path.read_text(encoding="utf-8"))
            value["runtime_versions"]["python"] = "3.12.0"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(
                audit.PhenomDemAuditError, "inconsistent runtime versions"
            ):
                audit.audit_results(sweep.root)


if __name__ == "__main__":
    unittest.main()
