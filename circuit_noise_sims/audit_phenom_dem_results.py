#!/usr/bin/env python3
"""Read-only completion audit for the adaptive DEM-informed phenomenological simulations."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from phenom_dem_adaptive_collection import (
    COLLECTOR,
    DEFAULT_MANIFEST_NAME,
    SCHEMA_VERSION,
    TASKS,
    TASK_LEVELS,
    Observation,
    Task,
    batch_result_path,
    build_command,
    choose_batch_shots,
    derive_batch_seed,
    expected_batch_request,
    make_batch_id,
    parse_state,
    read_jsonl,
    relevant_source_hash,
    relative_standard_error,
    table_path,
    validate_header,
)
from functions.phenom_dem_config import (
    BATCH_QUANTUM,
    DECODER_CONFIG,
    MANUSCRIPT_CURVES,
    MODEL_NAME,
    P_VALUES_BY_CODE_DESCENDING,
    RELAY_GAMMA0,
    RELAY_GAMMA_INTERVAL,
    RELAY_MAX_ITER,
    RELAY_NUM_SETS,
    RELAY_PRE_ITER,
    RELAY_STOP_NCONV,
    schedule_seed,
    worker_limit,
)
from functions.phenom_dem_decoding import (
    GRAPH_PROJECTION_SOLVER,
    PROJECTION_VERSION,
    SPATIAL_SOLVER,
    SPATIAL_SOLVER_ERROR_PRIOR,
    SPATIAL_SOLVER_MAX_ITER,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_TASK_COUNT = 70
REFINEMENT_CONTROLLER = Path(__file__).with_name(
    "phenom_dem_refinement_collection.py"
)
REFINEMENT_SAMPLING_MODE = "precision_refinement_v1"
RUNTIME_VERSION_KEYS = (
    "python",
    "numpy",
    "scipy",
    "stim",
    "ldpc",
    "relay_bp",
)


class PhenomDemAuditError(RuntimeError):
    """Raised when a result directory cannot prove simulation completion."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhenomDemAuditError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PhenomDemAuditError(f"cannot read immutable batch result {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PhenomDemAuditError(f"immutable batch result is not an object: {path}")
    return value


def _same_number(actual: Any, expected: float) -> bool:
    try:
        return float(actual) == float(expected)
    except (TypeError, ValueError):
        return False


def _has_path_suffix(path: Path, suffix: Path) -> bool:
    """Return whether an absolute provenance path lexically has a fixed suffix."""

    suffix_parts = suffix.parts
    return (
        path.is_absolute()
        and ".." not in path.parts
        and len(path.parts) >= len(suffix_parts)
        and path.parts[-len(suffix_parts):] == suffix_parts
    )


def _validate_plan(
    plan: Mapping[str, Any],
    task: Task,
    data_root: Path,
    seed_namespace: int,
    expected_batch_index: int,
    before: Observation,
) -> Path:
    batch_id = str(plan.get("batch_id", ""))
    _require(int(plan.get("schema_version", -1)) == SCHEMA_VERSION,
             f"plan {batch_id} has the wrong schema version")
    _require(plan.get("task_id") == task.task_id,
             f"plan {batch_id} has inconsistent task_id")
    _require(plan.get("code") == task.code and plan.get("variant") == task.variant,
             f"plan {batch_id} has inconsistent curve fields")
    _require(_same_number(plan.get("p"), task.p),
             f"plan {batch_id} has inconsistent p")
    _require(_same_number(plan.get("target_relative_se"), task.target),
             f"plan {batch_id} has inconsistent uncertainty target")
    _require(int(plan.get("batch_index", -1)) == expected_batch_index,
             f"task {task.task_id} has a nonsequential batch index")
    _require(batch_id == make_batch_id(task, expected_batch_index),
             f"plan {batch_id} has a noncanonical batch id")

    seed = int(plan.get("sampler_seed", -1))
    _require(seed == derive_batch_seed(seed_namespace, task, expected_batch_index),
             f"plan {batch_id} has a noncanonical sampler seed")
    shots = int(plan.get("batch_shots", -1))
    _require(shots > 0 and shots % BATCH_QUANTUM == 0,
             f"plan {batch_id} has an invalid batch size")
    sampling_mode = str(plan.get("sampling_mode", "adaptive"))
    if sampling_mode == "adaptive":
        _require(
            "refinement_target_relative_se" not in plan
            and "refinement_controller_sha256" not in plan,
            f"adaptive plan {batch_id} contains refinement metadata",
        )
        batch_target = task.target
        batch_rule = "adaptive"
    elif sampling_mode == REFINEMENT_SAMPLING_MODE:
        try:
            batch_target = float(plan["refinement_target_relative_se"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PhenomDemAuditError(
                f"refinement plan {batch_id} has an invalid target"
            ) from exc
        _require(
            0 < batch_target < task.target,
            f"refinement plan {batch_id} does not tighten the uncertainty target",
        )
        _require(
            REFINEMENT_CONTROLLER.is_file(),
            "refinement controller is unavailable to the auditor",
        )
        _require(
            plan.get("refinement_controller_sha256")
            == _sha256_file(REFINEMENT_CONTROLLER),
            f"refinement plan {batch_id} has an unknown controller hash",
        )
        batch_rule = "precision-refinement"
    else:
        raise PhenomDemAuditError(
            f"plan {batch_id} has an unknown sampling mode {sampling_mode!r}"
        )
    expected_shots = choose_batch_shots(before, batch_target)
    _require(
        shots == expected_shots,
        f"plan {batch_id} does not follow the {batch_rule} batch-size rule",
    )
    site = str(plan.get("execution_site", ""))
    try:
        limit = worker_limit(site)
    except ValueError as exc:
        raise PhenomDemAuditError(f"plan {batch_id} has an invalid execution site") from exc
    processes = int(plan.get("processes", -1))
    _require(1 <= processes <= limit and processes <= shots,
             f"plan {batch_id} exceeds the {site} worker cap")

    _require(int(plan.get("before_failures", -1)) == before.failures,
             f"plan {batch_id} has inconsistent before_failures")
    _require(int(plan.get("before_shots", -1)) == before.shots,
             f"plan {batch_id} has inconsistent before_shots")
    expected_before_se = before.relative_se
    recorded_before_se = plan.get("before_relative_se")
    if math.isfinite(expected_before_se):
        _require(_same_number(recorded_before_se, expected_before_se),
                 f"plan {batch_id} has inconsistent before_relative_se")
    else:
        _require(recorded_before_se is None,
                 f"plan {batch_id} must record null before_relative_se")

    expected_path = batch_result_path(data_root, task, batch_id).resolve()
    expected_relative_path = expected_path.relative_to(data_root)
    # Provenance paths can name the frozen source and data roots on another
    # machine. Check them lexically rather than resolving them through the
    # auditor host's filesystem, where an unrelated path or symlink could
    # otherwise change the result.
    recorded_path = Path(str(plan.get("result_path", ""))).expanduser()
    _require(_has_path_suffix(recorded_path, expected_relative_path),
             f"plan {batch_id} points outside the isolated batch layout")

    command = plan.get("command")
    _require(
        isinstance(command, list)
        and len(command) >= 2
        and all(isinstance(value, str) for value in command),
        f"plan {batch_id} has no collector command",
    )
    recorded_collector = Path(command[1]).expanduser()
    collector_suffix = Path(COLLECTOR.name)
    _require(_has_path_suffix(recorded_collector, collector_suffix),
             f"plan {batch_id} does not name the fixed DEM collector")
    expected_command = build_command(
        task,
        batch_id,
        recorded_path,
        shots,
        seed,
        processes,
        site,
        python_executable=str(command[0]),
    )
    expected_command[1] = command[1]
    _require(command == expected_command,
             f"plan {batch_id} collector command does not match fixed parameters")
    return expected_path


def _expected_request(plan: Mapping[str, Any]) -> Dict[str, Any]:
    return expected_batch_request(plan)


def _validate_committed_batch(
    plan: Mapping[str, Any],
    terminal: Mapping[str, Any],
    result_path: Path,
) -> Tuple[Observation, Dict[str, Any], str, Dict[str, str]]:
    batch_id = str(plan["batch_id"])
    _require(
        int(terminal.get("schema_version", -1)) == SCHEMA_VERSION,
        f"commit {batch_id} has the wrong schema version",
    )
    for key in (
        "batch_id",
        "batch_index",
        "task_id",
        "code",
        "variant",
        "p",
        "target_relative_se",
        "sampler_seed",
        "processes",
    ):
        _require(
            terminal.get(key) == plan.get(key),
            f"commit {batch_id} differs from its plan in {key}",
        )
    _require(
        int(terminal.get("returncode", -1)) == 0,
        f"committed batch {batch_id} has a nonzero return code",
    )
    _require(
        terminal.get("result_path") == plan.get("result_path"),
        f"commit {batch_id} has an inconsistent result path",
    )

    result = _read_json_object(result_path)
    _require(
        result.get("request") == _expected_request(plan),
        f"immutable batch {batch_id} request differs from fixed parameters",
    )
    shots = int(result.get("shots", -1))
    failures = int(result.get("failures", -1))
    _require(
        shots == int(plan["batch_shots"]) and 0 <= failures <= shots,
        f"immutable batch {batch_id} has invalid counts",
    )
    _require(
        int(terminal.get("shots", -1)) == shots
        and int(terminal.get("failures", -1)) == failures,
        f"commit {batch_id} counts differ from its immutable result",
    )

    runtime_versions = result.get("runtime_versions")
    _require(
        isinstance(runtime_versions, dict)
        and set(runtime_versions) == set(RUNTIME_VERSION_KEYS)
        and all(
            isinstance(runtime_versions[key], str) and runtime_versions[key]
            for key in RUNTIME_VERSION_KEYS
        ),
        f"immutable batch {batch_id} has invalid runtime versions",
    )

    circuit_metadata = result.get("circuit_metadata")
    model_metadata = result.get("phenomenological_model")
    code_metadata = result.get("code_metadata")
    _require(
        isinstance(circuit_metadata, dict)
        and isinstance(model_metadata, dict)
        and isinstance(code_metadata, dict),
        f"immutable batch {batch_id} lacks model metadata",
    )

    circuit_hash_keys = ("circuit_sha256", "cnot_order_sha256")
    model_hash_keys = (
        "graph_sha256",
        "observables_sha256",
        "priors_sha256",
        "projection_sha256",
        "metadata_sha256",
    )
    for metadata, keys in (
        (circuit_metadata, circuit_hash_keys),
        (model_metadata, model_hash_keys),
    ):
        for key in keys:
            value = str(metadata.get(key, ""))
            _require(
                bool(SHA256_RE.fullmatch(value)),
                f"immutable batch {batch_id} has an invalid {key}",
            )
            _require(
                terminal.get(key) == value,
                f"commit {batch_id} differs from its result in {key}",
            )

    model_metadata_for_hash = dict(model_metadata)
    recorded_metadata_hash = model_metadata_for_hash.pop("metadata_sha256")
    _require(
        _sha256_json(model_metadata_for_hash) == recorded_metadata_hash,
        f"immutable batch {batch_id} has inconsistent model metadata hashing",
    )

    circuit_count_keys = (
        "num_qubits",
        "num_measurements",
        "num_detectors",
        "num_observables",
        "cnot_pair_count",
        "cnot_layer_count",
    )
    model_count_keys = (
        "num_detectors",
        "num_observables",
        "num_phenomenological_variables",
        "num_dem_mechanisms",
        "num_unique_spatial_rhs",
        "num_zero_projection_mechanisms",
        "num_osd_spatial_solutions",
        "max_spatial_solution_weight",
        "num_unique_projection_rhs",
        "num_improved_initial_projections",
        "max_initial_projection_weight",
        "max_projection_weight",
    )
    positive_model_counts = {
        "num_detectors",
        "num_observables",
        "num_phenomenological_variables",
        "num_dem_mechanisms",
        "num_unique_spatial_rhs",
        "num_unique_projection_rhs",
    }
    for key in circuit_count_keys:
        _require(
            int(circuit_metadata.get(key, 0)) > 0,
            f"immutable batch {batch_id} has invalid {key}",
        )
    for key in model_count_keys:
        minimum = 1 if key in positive_model_counts else 0
        _require(
            int(model_metadata.get(key, -1)) >= minimum,
            f"immutable batch {batch_id} has invalid {key}",
        )
    for key in (
        "num_detectors",
        "num_observables",
        "num_dem_mechanisms",
        "num_phenomenological_variables",
    ):
        _require(
            int(terminal.get(key, -1)) == int(model_metadata[key]),
            f"commit {batch_id} differs from its result in {key}",
        )
    _require(
        int(circuit_metadata["num_detectors"])
        == int(model_metadata["num_detectors"])
        and int(circuit_metadata["num_observables"])
        == int(model_metadata["num_observables"]),
        f"immutable batch {batch_id} graph dimensions differ from the circuit",
    )
    for key in ("code_n", "code_k", "rounds"):
        _require(
            int(code_metadata.get(key, 0)) > 0,
            f"immutable batch {batch_id} has invalid {key}",
        )
    _require(
        int(model_metadata["num_observables"]) == int(code_metadata["code_k"]),
        f"immutable batch {batch_id} does not expose one observable per logical qubit",
    )
    expected_model_settings = {
        "projection_version": PROJECTION_VERSION,
        "spatial_solver": SPATIAL_SOLVER,
        "graph_projection_solver": GRAPH_PROJECTION_SOLVER,
        "spatial_solver_error_prior": SPATIAL_SOLVER_ERROR_PRIOR,
        "spatial_solver_max_iter": SPATIAL_SOLVER_MAX_ITER,
    }
    for key, expected in expected_model_settings.items():
        _require(
            model_metadata.get(key) == expected,
            f"immutable batch {batch_id} has the wrong {key}",
        )
    _require(
        model_metadata.get("all_spatial_solutions_minimum_weight_certified")
        is True,
        f"immutable batch {batch_id} has uncertified spatial solutions",
    )
    _require(
        model_metadata.get("all_projections_minimum_weight_certified") is True,
        f"immutable batch {batch_id} has uncertified graph projections",
    )
    expected_schedule_seed = schedule_seed(str(plan["variant"]))
    _require(
        int(code_metadata.get("schedule_seed", -1)) == expected_schedule_seed,
        f"immutable batch {batch_id} has the wrong schedule seed",
    )

    fingerprint = {
        **{key: circuit_metadata[key] for key in circuit_hash_keys},
        **{key: model_metadata[key] for key in model_hash_keys},
        **{key: int(circuit_metadata[key]) for key in circuit_count_keys},
        **{
            key: int(model_metadata[key])
            for key in model_count_keys
            if key not in ("num_detectors", "num_observables")
        },
        **{
            key: int(code_metadata[key])
            for key in ("code_n", "code_k", "rounds", "schedule_seed")
        },
    }
    return (
        Observation(failures, shots),
        fingerprint,
        _sha256_file(result_path),
        dict(runtime_versions),
    )


def _range(values: Iterable[int]) -> List[int]:
    values = list(values)
    return [int(min(values)), int(max(values))]


def _curve_summary(
    curve_tasks: Sequence[Task],
    observations: Mapping[str, Observation],
    fingerprints: Mapping[str, Mapping[str, Any]],
    effective_targets: Mapping[str, float],
    table_sha256: str,
) -> Dict[str, Any]:
    task_rows = [(task, observations[task.task_id], fingerprints[task.task_id])
                 for task in curve_tasks]
    ratios = [obs.relative_se / effective_targets[task.task_id]
              for task, obs, _ in task_rows]
    circuit_map = {f"{task.p:.17g}": meta["circuit_sha256"]
                   for task, _, meta in task_rows}
    priors_map = {f"{task.p:.17g}": meta["priors_sha256"]
                  for task, _, meta in task_rows}
    projection_map = {f"{task.p:.17g}": meta["projection_sha256"]
                      for task, _, meta in task_rows}
    return {
        "tasks": len(task_rows),
        "p_range": [min(task.p for task, _, _ in task_rows),
                    max(task.p for task, _, _ in task_rows)],
        "min_shots": min(obs.shots for _, obs, _ in task_rows),
        "max_shots": max(obs.shots for _, obs, _ in task_rows),
        "total_shots": sum(obs.shots for _, obs, _ in task_rows),
        "total_failures": sum(obs.failures for _, obs, _ in task_rows),
        "max_achieved_target_ratio": max(ratios),
        "num_detectors_range": _range(meta["num_detectors"] for _, _, meta in task_rows),
        "num_observables_range": _range(meta["num_observables"] for _, _, meta in task_rows),
        "num_dem_mechanisms_range": _range(
            meta["num_dem_mechanisms"] for _, _, meta in task_rows
        ),
        "num_phenomenological_variables_range": _range(
            meta["num_phenomenological_variables"] for _, _, meta in task_rows
        ),
        "hashes": {
            "table_sha256": table_sha256,
            "circuit_map_sha256": _sha256_json(circuit_map),
            "priors_map_sha256": _sha256_json(priors_map),
            "projection_map_sha256": _sha256_json(projection_map),
        },
    }


def _regenerate_task_fingerprint(task: Task) -> Dict[str, Any]:
    """Rebuild one circuit and projected phenomenological model."""

    data_directory_variable = "HGP_REDUCTION_DATA_DIR"
    previous_data_directory = os.environ.get(data_directory_variable)
    sim_common_module_name = "functions.sim_common"
    previous_sim_common_module = sys.modules.get(sim_common_module_name)
    previous_sim_common_data_directory = (
        getattr(previous_sim_common_module, "DATA_DIR", None)
        if previous_sim_common_module is not None
        else None
    )
    os.environ[data_directory_variable] = str(Path(__file__).resolve().parent)
    try:
        from phenom_dem_data_collection import build_circuit_and_model

        with contextlib.redirect_stdout(io.StringIO()):
            (
                _circuit,
                _model,
                code_metadata,
                circuit_metadata,
                model_metadata,
            ) = build_circuit_and_model(
                task.code,
                task.variant,
                task.p,
                schedule_seed(task.variant),
            )
    except Exception as exc:
        raise PhenomDemAuditError(
            f"cannot regenerate the circuit/model for {task.task_id}: {exc}"
        ) from exc
    finally:
        if previous_data_directory is None:
            os.environ.pop(data_directory_variable, None)
        else:
            os.environ[data_directory_variable] = previous_data_directory
        sim_common_module = sys.modules.get(sim_common_module_name)
        if sim_common_module is not None:
            restored_data_directory = (
                previous_sim_common_data_directory
                if previous_sim_common_module is not None
                else (
                    Path(previous_data_directory)
                    if previous_data_directory is not None
                    else Path(__file__).resolve().parent / "data"
                )
            )
            sim_common_module.DATA_DIR = restored_data_directory

    circuit_hash_keys = ("circuit_sha256", "cnot_order_sha256")
    model_hash_keys = (
        "graph_sha256",
        "observables_sha256",
        "priors_sha256",
        "projection_sha256",
        "metadata_sha256",
    )
    circuit_count_keys = (
        "num_qubits",
        "num_measurements",
        "num_detectors",
        "num_observables",
        "cnot_pair_count",
        "cnot_layer_count",
    )
    model_count_keys = (
        "num_phenomenological_variables",
        "num_dem_mechanisms",
        "num_unique_spatial_rhs",
        "num_zero_projection_mechanisms",
        "num_osd_spatial_solutions",
        "max_spatial_solution_weight",
        "num_unique_projection_rhs",
        "num_improved_initial_projections",
        "max_initial_projection_weight",
        "max_projection_weight",
    )
    return {
        **{key: circuit_metadata[key] for key in circuit_hash_keys},
        **{key: model_metadata[key] for key in model_hash_keys},
        **{key: int(circuit_metadata[key]) for key in circuit_count_keys},
        **{key: int(model_metadata[key]) for key in model_count_keys},
        **{
            key: int(code_metadata[key])
            for key in ("code_n", "code_k", "rounds", "schedule_seed")
        },
    }


def audit_results(
    data_root: Path,
    *,
    regenerate_fingerprints: bool = False,
) -> Dict[str, Any]:
    """Validate one completed fixed sweep without modifying any files."""

    data_root = data_root.expanduser().resolve()
    _require(data_root.is_dir(), f"data root is not a directory: {data_root}")
    manifest_path = data_root / DEFAULT_MANIFEST_NAME
    _require(manifest_path.is_file(), f"manifest does not exist: {manifest_path}")
    try:
        events = read_jsonl(manifest_path)
        state = parse_state(events)
        seed_namespace = int(state.header["seed_namespace"])
        validate_header(state.header, seed_namespace)
    except Exception as exc:
        raise PhenomDemAuditError(f"invalid fixed manifest: {exc}") from exc

    _require(len(TASKS) == EXPECTED_TASK_COUNT,
             f"fixed configuration has {len(TASKS)} tasks, expected {EXPECTED_TASK_COUNT}")
    _require(
        sum(len(level) for level in TASK_LEVELS) == EXPECTED_TASK_COUNT,
        "fixed per-code grids do not define 70 tasks",
    )
    current_source_sha256 = relevant_source_hash()
    _require(
        state.header.get("source_sha256") == current_source_sha256,
        "manifest source hash does not match the current fixed simulation source",
    )
    _require(not state.pending,
             f"manifest has {len(state.pending)} pending batch plan(s)")

    tasks_by_id = {task.task_id: task for task in TASKS}
    _require(len(tasks_by_id) == EXPECTED_TASK_COUNT, "fixed task ids are not unique")
    task_level = {
        task.task_id: level_index
        for level_index, level in enumerate(TASK_LEVELS)
        for task in level
    }
    _require(
        len(task_level) == EXPECTED_TASK_COUNT,
        "fixed task levels do not cover every task exactly once",
    )
    running = {task_id: Observation() for task_id in tasks_by_id}
    effective_targets = {
        task_id: task.target for task_id, task in tasks_by_id.items()
    }
    next_indices = defaultdict(int)
    outstanding: Dict[str, str] = {}
    wave_terminal_started = False
    plans: Dict[str, Mapping[str, Any]] = {}
    plan_paths: Dict[str, Path] = {}
    fingerprints_by_task: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    batch_file_hashes: Dict[str, str] = {}
    runtime_versions_by_batch: Dict[str, Dict[str, str]] = {}
    committed_result_paths: set[Path] = set()
    committed_batches = 0
    abandoned_batches = 0

    for event in state.events[1:]:
        kind = event.get("event")
        batch_id = str(event.get("batch_id", ""))
        if kind == "batch_planned":
            task_id = str(event.get("task_id", ""))
            _require(task_id in tasks_by_id, f"plan {batch_id} references an unknown task")
            _require(task_id not in outstanding,
                     f"task {task_id} has overlapping outstanding plans")
            _require(not wave_terminal_started,
                     f"plan {batch_id} begins before the preceding wave terminates")
            task = tasks_by_id[task_id]
            level_index = task_level[task_id]
            for earlier_level in TASK_LEVELS[:level_index]:
                for earlier_task in earlier_level:
                    _require(
                        running[earlier_task.task_id].relative_se
                        <= earlier_task.target,
                        f"plan {batch_id} starts before all earlier grid levels are complete",
                    )
            index = next_indices[task_id]
            plan_paths[batch_id] = _validate_plan(
                event, task, data_root, seed_namespace, index, running[task_id]
            )
            if event.get("sampling_mode", "adaptive") == REFINEMENT_SAMPLING_MODE:
                effective_targets[task_id] = min(
                    effective_targets[task_id],
                    float(event["refinement_target_relative_se"]),
                )
            next_indices[task_id] += 1
            plans[batch_id] = event
            outstanding[task_id] = batch_id
            outstanding_plans = [plans[value] for value in outstanding.values()]
            sites = {str(value["execution_site"]) for value in outstanding_plans}
            _require(len(sites) == 1,
                     f"concurrent wave containing {batch_id} mixes execution sites")
            wave_levels = {
                task_level[str(value["task_id"])] for value in outstanding_plans
            }
            _require(
                len(wave_levels) == 1,
                f"concurrent wave containing {batch_id} mixes p-grid levels",
            )
            site = next(iter(sites))
            concurrent_processes = sum(int(value["processes"])
                                       for value in outstanding_plans)
            _require(concurrent_processes <= worker_limit(site),
                     f"concurrent wave containing {batch_id} exceeds the {site} worker cap")
            continue

        _require(kind in ("batch_committed", "batch_abandoned"),
                 f"unknown manifest event {kind!r}")
        _require(batch_id in plans, f"terminal event references unknown plan {batch_id}")
        plan = plans[batch_id]
        task_id = str(plan["task_id"])
        _require(outstanding.get(task_id) == batch_id,
                 f"terminal order is inconsistent for {batch_id}")
        wave_terminal_started = True
        outstanding.pop(task_id)
        if not outstanding:
            wave_terminal_started = False
        if kind == "batch_abandoned":
            _require(int(event.get("schema_version", -1)) == SCHEMA_VERSION,
                     f"abandoned batch {batch_id} has the wrong schema version")
            abandoned_batches += 1
            continue

        observation, fingerprint, file_hash, runtime_versions = _validate_committed_batch(
            plan, event, plan_paths[batch_id]
        )
        current = running[task_id]
        running[task_id] = Observation(
            current.failures + observation.failures,
            current.shots + observation.shots,
        )
        fingerprints_by_task[task_id].append(fingerprint)
        batch_file_hashes[batch_id] = file_hash
        runtime_versions_by_batch[batch_id] = runtime_versions
        committed_result_paths.add(plan_paths[batch_id])
        committed_batches += 1

    _require(not outstanding, "manifest event order leaves outstanding task plans")
    _require(
        abandoned_batches == 0,
        f"manifest contains {abandoned_batches} abandoned batch(es)",
    )
    _require(bool(runtime_versions_by_batch), "no committed runtime provenance")
    runtime_version_hashes = {
        _sha256_json(value) for value in runtime_versions_by_batch.values()
    }
    _require(
        len(runtime_version_hashes) == 1,
        "immutable batches use inconsistent runtime versions",
    )
    canonical_runtime_versions = next(iter(runtime_versions_by_batch.values()))

    batch_root = data_root / "batches"
    _require(not batch_root.is_symlink(), f"batch root must not be a symlink: {batch_root}")
    expected_result_paths = set(committed_result_paths)
    expected_lock_paths = {
        path.with_suffix(path.suffix + ".lock") for path in expected_result_paths
    }
    discovered_files: set[Path] = set()
    if batch_root.exists():
        _require(batch_root.is_dir(), f"batch root is not a directory: {batch_root}")
        for directory, child_directories, filenames in os.walk(
            batch_root,
            followlinks=False,
        ):
            directory_path = Path(directory)
            for name in child_directories:
                child = directory_path / name
                _require(
                    not child.is_symlink(),
                    f"batch directory must not be a symlink: {child}",
                )
            for name in filenames:
                path = directory_path / name
                _require(
                    not path.is_symlink(),
                    f"batch file must not be a symlink: {path}",
                )
                _require(path.is_file(), f"batch entry is not a regular file: {path}")
                discovered_files.add(path.resolve())
    expected_files = expected_result_paths | expected_lock_paths
    missing_results = sorted(expected_result_paths - discovered_files)
    missing_locks = sorted(expected_lock_paths - discovered_files)
    unexpected_files = sorted(discovered_files - expected_files)
    _require(
        not missing_results and not missing_locks and not unexpected_files,
        "batch file inventory differs from committed provenance: "
        f"missing_results={missing_results}, missing_locks={missing_locks}, "
        f"unexpected={unexpected_files}",
    )

    canonical_fingerprints: Dict[str, Mapping[str, Any]] = {}
    for task in TASKS:
        observation = running[task.task_id]
        _require(observation.shots > 0,
                 f"task {task.task_id} has no committed shots")
        _require(observation.failures > 0,
                 f"task {task.task_id} has zero failures")
        _require(observation.failures <= observation.shots,
                 f"task {task.task_id} has invalid cumulative counts")
        ratio = observation.relative_se / effective_targets[task.task_id]
        _require(ratio <= 1.0 + 1e-12,
                 f"task {task.task_id} misses its recorded relative-error target")
        task_fingerprints = fingerprints_by_task[task.task_id]
        _require(bool(task_fingerprints),
                 f"task {task.task_id} has no immutable model metadata")
        first = task_fingerprints[0]
        _require(all(value == first for value in task_fingerprints[1:]),
                 f"task {task.task_id} changed circuit or phenomenological-model fingerprints")
        canonical_fingerprints[task.task_id] = first

    if regenerate_fingerprints:
        for task in TASKS:
            regenerated = _regenerate_task_fingerprint(task)
            _require(
                regenerated == canonical_fingerprints[task.task_id],
                f"regenerated circuit/model fingerprint differs for {task.task_id}",
            )

    table_hashes: Dict[str, str] = {}
    curve_summaries: Dict[str, Any] = {}
    for curve in MANUSCRIPT_CURVES:
        path = table_path(data_root, curve.code, curve.variant)
        _require(path.is_file(), f"result table does not exist: {path}")
        try:
            table = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise PhenomDemAuditError(f"cannot load result table {path}: {exc}") from exc
        curve_tasks = sorted(
            (task for task in TASKS
             if task.code == curve.code and task.variant == curve.variant),
            key=lambda task: task.p,
        )
        curve_fingerprints = [
            canonical_fingerprints[task.task_id] for task in curve_tasks
        ]
        for key in (
            "cnot_order_sha256",
            "graph_sha256",
            "observables_sha256",
            "projection_sha256",
        ):
            _require(
                len({fingerprint[key] for fingerprint in curve_fingerprints}) == 1,
                f"curve {curve.curve_id} changes {key} across p",
            )
        for key in ("circuit_sha256", "priors_sha256"):
            _require(
                len({fingerprint[key] for fingerprint in curve_fingerprints})
                == len(curve_fingerprints),
                f"curve {curve.curve_id} does not vary {key} across p",
            )
        expected = np.asarray(
            [[task.p, running[task.task_id].failures, running[task.task_id].shots]
             for task in curve_tasks],
            dtype=float,
        )
        _require(table.shape == expected.shape and np.array_equal(table, expected),
                 f"table {path} is not exactly the sum of committed immutable batches")
        curve_id = curve.curve_id
        table_hashes[curve_id] = _sha256_file(path)
        curve_summaries[curve_id] = _curve_summary(
            curve_tasks,
            running,
            canonical_fingerprints,
            effective_targets,
            table_hashes[curve_id],
        )

    all_observations = [running[task.task_id] for task in TASKS]
    all_fingerprints = [canonical_fingerprints[task.task_id] for task in TASKS]
    ratios = [
        running[task.task_id].relative_se / effective_targets[task.task_id]
        for task in TASKS
    ]
    circuit_map = {task.task_id: canonical_fingerprints[task.task_id]["circuit_sha256"]
                   for task in TASKS}
    priors_map = {
        task.task_id: canonical_fingerprints[task.task_id]["priors_sha256"]
        for task in TASKS
    }
    projection_map = {
        task.task_id: canonical_fingerprints[task.task_id]["projection_sha256"]
        for task in TASKS
    }
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "decoder_config": DECODER_CONFIG,
        "tasks": EXPECTED_TASK_COUNT,
        "curves": curve_summaries,
        "global": {
            "committed_batches": committed_batches,
            "abandoned_batches": abandoned_batches,
            "batch_result_files": len(expected_result_paths),
            "batch_lock_files": len(expected_lock_paths),
            "regenerated_fingerprints": bool(regenerate_fingerprints),
            "runtime_versions": canonical_runtime_versions,
            "p_range": [
                min(min(values) for values in P_VALUES_BY_CODE_DESCENDING.values()),
                max(max(values) for values in P_VALUES_BY_CODE_DESCENDING.values()),
            ],
            "min_shots": min(value.shots for value in all_observations),
            "max_shots": max(value.shots for value in all_observations),
            "total_shots": sum(value.shots for value in all_observations),
            "total_failures": sum(value.failures for value in all_observations),
            "max_achieved_target_ratio": max(ratios),
            "num_detectors_range": _range(meta["num_detectors"] for meta in all_fingerprints),
            "num_observables_range": _range(meta["num_observables"] for meta in all_fingerprints),
            "num_dem_mechanisms_range": _range(
                meta["num_dem_mechanisms"] for meta in all_fingerprints
            ),
            "num_phenomenological_variables_range": _range(
                meta["num_phenomenological_variables"]
                for meta in all_fingerprints
            ),
            "hashes": {
                "source_sha256": state.header["source_sha256"],
                "current_source_sha256": current_source_sha256,
                "manifest_sha256": _sha256_file(manifest_path),
                "tables_map_sha256": _sha256_json(table_hashes),
                "batch_results_map_sha256": _sha256_json(batch_file_hashes),
                "circuit_map_sha256": _sha256_json(circuit_map),
                "priors_map_sha256": _sha256_json(priors_map),
                "projection_map_sha256": _sha256_json(projection_map),
            },
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit of the completed DEM-informed phenomenological sweep."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--regenerate-fingerprints",
        action="store_true",
        help=(
            "Rebuild all 70 circuits and phenomenological models from the current "
            "fixed source and compare their fingerprints with the ledger."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        summary = audit_results(
            args.data_root,
            regenerate_fingerprints=args.regenerate_fingerprints,
        )
    except PhenomDemAuditError as exc:
        print(f"Phenomenological-DEM result audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
