#!/usr/bin/env python3
"""Run restart-safe adaptive sweeps with DEM-informed phenomenological priors."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from functions.phenom_dem_config import (
    BATCH_QUANTUM,
    CPU200_MAX_WORKERS,
    DECODER_CONFIG,
    INITIAL_SHOTS,
    LOCAL_MAX_WORKERS,
    MANUSCRIPT_CURVES,
    MODEL_NAME,
    P_VALUES_BY_CODE_DESCENDING,
    RELAY_GAMMA0,
    RELAY_GAMMA_INTERVAL,
    RELAY_MAX_ITER,
    RELAY_NUM_SETS,
    RELAY_PRE_ITER,
    RELAY_STOP_NCONV,
    TARGET_RELATIVE_SE,
    SCHEDULE_SEEDS_BY_VARIANT,
    p_values_for_code,
    relative_se_target,
    schedule_seed,
    worker_limit,
)


SCRIPT_DIR = Path(__file__).resolve().parent
COLLECTOR = SCRIPT_DIR / "phenom_dem_data_collection.py"
SCHEMA_VERSION = 1
DEFAULT_MANIFEST_NAME = "phenom_dem_adaptive_provenance.jsonl"
DEFAULT_LOCK_NAME = ".phenom_dem_adaptive_collection.lock"
DEFAULT_SEED_NAMESPACE = 2026083101


class AdaptivePhenomDemError(RuntimeError):
    """Raised when the adaptive ledger cannot advance safely."""


@dataclasses.dataclass(frozen=True, order=True)
class Task:
    code: str
    variant: str
    p: float

    @property
    def task_id(self) -> str:
        return f"{self.code}|{self.variant}|p={self.p:.17g}"

    @property
    def target(self) -> float:
        return relative_se_target(self.p)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "code": self.code,
            "variant": self.variant,
            "p": self.p,
            "target_relative_se": self.target,
        }


_GRID_LENGTHS = {len(values) for values in P_VALUES_BY_CODE_DESCENDING.values()}
if len(_GRID_LENGTHS) != 1:
    raise RuntimeError("all manuscript p grids must have the same number of points")
GRID_POINT_COUNT = next(iter(_GRID_LENGTHS))
TASK_LEVELS: Tuple[Tuple[Task, ...], ...] = tuple(
    tuple(
        Task(
            curve.code,
            curve.variant,
            p_values_for_code(curve.code, descending=True)[level],
        )
        for curve in MANUSCRIPT_CURVES
    )
    for level in range(GRID_POINT_COUNT)
)
TASKS: Tuple[Task, ...] = tuple(
    task for level in TASK_LEVELS for task in level
)


@dataclasses.dataclass(frozen=True)
class Observation:
    failures: int = 0
    shots: int = 0

    @property
    def rate(self) -> float:
        return self.failures / self.shots if self.shots else math.nan

    @property
    def relative_se(self) -> float:
        return relative_standard_error(self.failures, self.shots)


@dataclasses.dataclass
class ManifestState:
    header: Mapping[str, Any]
    events: List[Mapping[str, Any]]
    plans: Dict[str, Mapping[str, Any]]
    terminals: Dict[str, Mapping[str, Any]]
    seeds: Dict[int, str]

    @property
    def pending(self) -> List[Mapping[str, Any]]:
        return [
            plan
            for batch_id, plan in self.plans.items()
            if batch_id not in self.terminals
        ]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def relative_standard_error(failures: int, shots: int) -> float:
    failures = int(failures)
    shots = int(shots)
    if shots < 0 or failures < 0 or failures > shots:
        raise ValueError("invalid failure/shot counts")
    if shots == 0 or failures == 0:
        return math.inf
    return math.sqrt((shots - failures) / (shots * failures))


def choose_batch_shots(
    observation: Observation,
    target: float,
    *,
    initial: int = INITIAL_SHOTS,
    quantum: int = BATCH_QUANTUM,
) -> int:
    """Choose the next rounded batch, allowing cumulative shots to double."""

    if target <= 0 or initial <= 0 or quantum <= 0:
        raise ValueError("target, initial, and quantum must be positive")
    if observation.shots == 0:
        return int(initial)
    if observation.relative_se <= target:
        return 0
    if observation.failures == 0:
        return int(max(initial, observation.shots))

    estimated_total = math.ceil(
        (observation.shots - observation.failures)
        / (target * target * observation.failures)
    )
    additional = max(1, estimated_total - observation.shots)
    rounded = max(quantum, math.ceil(additional / quantum) * quantum)
    return int(min(max(initial, observation.shots), rounded))


def allocate_workers(total: int, batch_shots: Sequence[int]) -> List[int]:
    """Divide one global worker budget across concurrent curve batches."""

    if total <= 0 or not batch_shots or any(shots <= 0 for shots in batch_shots):
        raise ValueError("worker budget and batch sizes must be positive")
    if total < len(batch_shots):
        raise ValueError("worker budget must give every concurrent batch one worker")

    base, remainder = divmod(total, len(batch_shots))
    allocation = [
        min(int(shots), base + (1 if index < remainder else 0))
        for index, shots in enumerate(batch_shots)
    ]
    spare = total - sum(allocation)
    while spare:
        progressed = False
        for index, shots in enumerate(batch_shots):
            if spare == 0:
                break
            if allocation[index] < shots:
                allocation[index] += 1
                spare -= 1
                progressed = True
        if not progressed:
            break
    return allocation


def task_map(tasks: Sequence[Task] = TASKS) -> Dict[str, Task]:
    mapping = {task.task_id: task for task in tasks}
    if len(mapping) != len(tasks):
        raise AdaptivePhenomDemError("task identifiers are not unique")
    return mapping


def derive_batch_seed(seed_namespace: int, task: Task, batch_index: int) -> int:
    payload = f"{int(seed_namespace)}|{task.task_id}|batch={int(batch_index)}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, byteorder="little", signed=False)
    return seed or 1


def make_batch_id(task: Task, batch_index: int) -> str:
    digest = hashlib.blake2b(task.task_id.encode("utf-8"), digest_size=8).hexdigest()
    return f"{digest}-b{batch_index:04d}"


def batch_result_path(data_root: Path, task: Task, batch_id: str) -> Path:
    p_tag = f"p{task.p:.4f}".replace(".", "p")
    return (
        data_root
        / "batches"
        / p_tag
        / task.code
        / task.variant
        / f"{batch_id}.json"
    )


def table_path(data_root: Path, code: str, variant: str) -> Path:
    return data_root / DECODER_CONFIG / variant / f"{code}.npy"


def relevant_source_hash() -> str:
    roots = (SCRIPT_DIR / "codes", SCRIPT_DIR / "functions")
    paths = [COLLECTOR, Path(__file__).resolve()]
    for root in roots:
        paths.extend(sorted(root.glob("*.py")))
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda item: str(item)):
        relative = path.relative_to(SCRIPT_DIR)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def header_value(seed_namespace: int) -> Dict[str, Any]:
    return {
        "event": "controller_initialized",
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "model": MODEL_NAME,
        "decoder_config": DECODER_CONFIG,
        "source_sha256": relevant_source_hash(),
        "seed_namespace": int(seed_namespace),
        "schedule_seeds_by_variant": dict(SCHEDULE_SEEDS_BY_VARIANT),
        "p_values_by_code_descending": {
            code: list(values)
            for code, values in P_VALUES_BY_CODE_DESCENDING.items()
        },
        "relative_se_rule": {
            "kind": "constant",
            "target": TARGET_RELATIVE_SE,
        },
        "initial_shots": INITIAL_SHOTS,
        "batch_quantum": BATCH_QUANTUM,
        "relay_parameters": {
            "gamma0": RELAY_GAMMA0,
            "pre_iter": RELAY_PRE_ITER,
            "num_sets": RELAY_NUM_SETS,
            "set_max_iter": RELAY_MAX_ITER,
            "gamma_dist_interval": list(RELAY_GAMMA_INTERVAL),
            "stop_nconv": RELAY_STOP_NCONV,
        },
        "tasks": [task.as_dict() for task in TASKS],
    }


def append_jsonl(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(event), sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    if not path.exists():
        return []
    events: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise AdaptivePhenomDemError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise AdaptivePhenomDemError(f"non-object manifest event at {path}:{line_number}")
            events.append(value)
    return events


def parse_state(events: Sequence[Mapping[str, Any]]) -> ManifestState:
    if not events or events[0].get("event") != "controller_initialized":
        raise AdaptivePhenomDemError("manifest must begin with controller_initialized")
    plans: Dict[str, Mapping[str, Any]] = {}
    terminals: Dict[str, Mapping[str, Any]] = {}
    seeds: Dict[int, str] = {}
    for event in events[1:]:
        kind = event.get("event")
        batch_id = str(event.get("batch_id", ""))
        if kind == "batch_planned":
            if not batch_id or batch_id in plans:
                raise AdaptivePhenomDemError(f"duplicate or missing batch plan: {batch_id!r}")
            seed = int(event["sampler_seed"])
            if seed in seeds:
                raise AdaptivePhenomDemError(
                    f"sampler seed {seed} is reused by {seeds[seed]} and {batch_id}"
                )
            plans[batch_id] = event
            seeds[seed] = batch_id
        elif kind in ("batch_committed", "batch_abandoned"):
            if batch_id not in plans or batch_id in terminals:
                raise AdaptivePhenomDemError(f"invalid terminal event for {batch_id!r}")
            terminals[batch_id] = event
        else:
            raise AdaptivePhenomDemError(f"unknown manifest event: {kind!r}")
    return ManifestState(events[0], list(events), plans, terminals, seeds)


def validate_header(header: Mapping[str, Any], seed_namespace: int) -> None:
    expected = header_value(seed_namespace)
    keys = (
        "schema_version",
        "model",
        "decoder_config",
        "source_sha256",
        "seed_namespace",
        "schedule_seeds_by_variant",
        "p_values_by_code_descending",
        "relative_se_rule",
        "initial_shots",
        "batch_quantum",
        "relay_parameters",
        "tasks",
    )
    for key in keys:
        if header.get(key) != expected.get(key):
            raise AdaptivePhenomDemError(f"manifest configuration mismatch for {key}")


def observation(state: ManifestState, task: Task) -> Observation:
    failures = 0
    shots = 0
    for terminal in state.terminals.values():
        if terminal.get("event") != "batch_committed":
            continue
        if terminal.get("task_id") != task.task_id:
            continue
        failures += int(terminal["failures"])
        shots += int(terminal["shots"])
    return Observation(failures, shots)


def next_batch_index(state: ManifestState, task: Task) -> int:
    indices = [
        int(plan["batch_index"])
        for plan in state.plans.values()
        if plan.get("task_id") == task.task_id
    ]
    return max(indices, default=-1) + 1


def build_command(
    task: Task,
    batch_id: str,
    result_path: Path,
    shots: int,
    sampler_seed: int,
    processes: int,
    execution_site: str,
    python_executable: str = sys.executable,
) -> List[str]:
    limit = worker_limit(execution_site)
    if processes < 1 or processes > limit:
        raise ValueError(f"processes must lie in [1, {limit}] for {execution_site}")
    return [
        python_executable,
        str(COLLECTOR),
        "--batch-id",
        batch_id,
        "--result",
        str(result_path),
        "--code",
        task.code,
        "--variant",
        task.variant,
        "--p",
        f"{task.p:.17g}",
        "--shots",
        str(int(shots)),
        "--sampler-seed",
        str(int(sampler_seed)),
        "--schedule-seed",
        str(schedule_seed(task.variant)),
        "--execution-site",
        execution_site,
        "--processes",
        str(int(processes)),
    ]


def make_plan(
    state: ManifestState,
    task: Task,
    before: Observation,
    shots: int,
    processes: int,
    execution_site: str,
    data_root: Path,
    seed_namespace: int,
) -> Dict[str, Any]:
    index = next_batch_index(state, task)
    batch_id = make_batch_id(task, index)
    seed = derive_batch_seed(seed_namespace, task, index)
    if seed in state.seeds:
        raise AdaptivePhenomDemError(f"derived sampler seed is already planned: {seed}")
    result_path = batch_result_path(data_root, task, batch_id)
    command = build_command(
        task,
        batch_id,
        result_path,
        shots,
        seed,
        processes,
        execution_site,
    )
    return {
        "event": "batch_planned",
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "batch_id": batch_id,
        "batch_index": index,
        "task_id": task.task_id,
        "code": task.code,
        "variant": task.variant,
        "p": task.p,
        "target_relative_se": task.target,
        "before_failures": before.failures,
        "before_shots": before.shots,
        "before_relative_se": (
            before.relative_se if math.isfinite(before.relative_se) else None
        ),
        "batch_shots": int(shots),
        "sampler_seed": int(seed),
        "processes": int(processes),
        "execution_site": execution_site,
        "result_path": str(result_path),
        "command": command,
    }


def expected_batch_request(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the complete immutable collector request implied by a plan."""

    p = float(plan["p"])
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_id": plan["batch_id"],
        "model": MODEL_NAME,
        "decoder_config": DECODER_CONFIG,
        "code": plan["code"],
        "variant": plan["variant"],
        "p": p,
        "p1": p / 10.0,
        "p2": p,
        "p_spam": p,
        "shots": int(plan["batch_shots"]),
        "sampler_seed": int(plan["sampler_seed"]),
        "schedule_seed": schedule_seed(str(plan["variant"])),
        "processes": int(plan["processes"]),
        "execution_site": plan["execution_site"],
        "relay_parameters": {
            "gamma0": RELAY_GAMMA0,
            "pre_iter": RELAY_PRE_ITER,
            "num_sets": RELAY_NUM_SETS,
            "set_max_iter": RELAY_MAX_ITER,
            "gamma_dist_interval": list(RELAY_GAMMA_INTERVAL),
            "stop_nconv": RELAY_STOP_NCONV,
        },
    }


def read_batch_result(plan: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(str(plan["result_path"]))
    if not path.is_file():
        raise AdaptivePhenomDemError(f"batch result was not created: {path}")
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("request") != expected_batch_request(plan):
        raise AdaptivePhenomDemError(
            f"batch result request does not exactly match plan {plan['batch_id']}"
        )
    failures = int(result.get("failures", -1))
    shots = int(result.get("shots", -1))
    if shots != int(plan["batch_shots"]) or failures < 0 or failures > shots:
        raise AdaptivePhenomDemError(f"invalid counts in result for {plan['batch_id']}")
    return result


def make_terminal(
    plan: Mapping[str, Any],
    kind: str,
    *,
    result: Optional[Mapping[str, Any]] = None,
    returncode: int,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "event": kind,
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "batch_id": plan["batch_id"],
        "batch_index": plan["batch_index"],
        "task_id": plan["task_id"],
        "code": plan["code"],
        "variant": plan["variant"],
        "p": plan["p"],
        "target_relative_se": plan["target_relative_se"],
        "sampler_seed": plan["sampler_seed"],
        "processes": plan["processes"],
        "returncode": int(returncode),
    }
    if result is not None:
        value.update(
            {
                "failures": int(result["failures"]),
                "shots": int(result["shots"]),
                "result_path": plan["result_path"],
                "circuit_sha256": result["circuit_metadata"]["circuit_sha256"],
                "cnot_order_sha256": result["circuit_metadata"][
                    "cnot_order_sha256"
                ],
                "graph_sha256": result["phenomenological_model"][
                    "graph_sha256"
                ],
                "observables_sha256": result["phenomenological_model"][
                    "observables_sha256"
                ],
                "priors_sha256": result["phenomenological_model"][
                    "priors_sha256"
                ],
                "projection_sha256": result["phenomenological_model"][
                    "projection_sha256"
                ],
                "metadata_sha256": result["phenomenological_model"][
                    "metadata_sha256"
                ],
                "num_detectors": result["phenomenological_model"][
                    "num_detectors"
                ],
                "num_observables": result["phenomenological_model"][
                    "num_observables"
                ],
                "num_dem_mechanisms": result["phenomenological_model"][
                    "num_dem_mechanisms"
                ],
                "num_phenomenological_variables": result[
                    "phenomenological_model"
                ]["num_phenomenological_variables"],
            }
        )
    if reason is not None:
        value["reason"] = reason
    return value


def register_event(state: ManifestState, event: Mapping[str, Any]) -> None:
    state.events.append(event)
    batch_id = str(event["batch_id"])
    if event["event"] == "batch_planned":
        state.plans[batch_id] = event
        state.seeds[int(event["sampler_seed"])] = batch_id
    else:
        state.terminals[batch_id] = event


def atomic_save_table(path: Path, table: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        np.save(handle, table, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rebuild_tables(state: ManifestState, data_root: Path) -> None:
    grouped: Dict[Tuple[str, str], Dict[float, List[int]]] = {}
    for terminal in state.terminals.values():
        if terminal.get("event") != "batch_committed":
            continue
        key = (str(terminal["code"]), str(terminal["variant"]))
        p = float(terminal["p"])
        totals = grouped.setdefault(key, {}).setdefault(p, [0, 0])
        totals[0] += int(terminal["failures"])
        totals[1] += int(terminal["shots"])

    for curve in MANUSCRIPT_CURVES:
        rows = [
            [p, counts[0], counts[1]]
            for p, counts in sorted(grouped.get((curve.code, curve.variant), {}).items())
        ]
        table = np.asarray(rows, dtype=float).reshape((-1, 3))
        atomic_save_table(table_path(data_root, curve.code, curve.variant), table)


def _run_subprocess(command: Sequence[str], env: Mapping[str, str]) -> int:
    result = subprocess.run(
        list(command), cwd=SCRIPT_DIR, env=dict(env), check=False
    )
    return int(result.returncode)


def run_plans(
    plans: Sequence[Mapping[str, Any]],
    state: ManifestState,
    manifest_path: Path,
    data_root: Path,
) -> None:
    if not plans:
        return
    sites = {str(plan["execution_site"]) for plan in plans}
    if len(sites) != 1:
        raise AdaptivePhenomDemError("one concurrent wave cannot mix execution sites")
    site = next(iter(sites))
    limit = worker_limit(site)
    planned_workers = [int(plan["processes"]) for plan in plans]
    if any(workers < 1 or workers > limit for workers in planned_workers):
        raise AdaptivePhenomDemError(
            f"one or more plans exceed the {site} per-plan worker cap"
        )
    if sum(planned_workers) > limit:
        raise AdaptivePhenomDemError(
            f"concurrent wave requests {sum(planned_workers)} workers, "
            f"exceeding the {site} cap of {limit}"
        )

    env = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        env[variable] = "1"

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(plans)) as executor:
        futures = {
            executor.submit(_run_subprocess, plan["command"], env): plan
            for plan in plans
        }
        outcomes: List[Tuple[Mapping[str, Any], int]] = []
        for future in concurrent.futures.as_completed(futures):
            plan = futures[future]
            try:
                returncode = future.result()
            except Exception:
                returncode = 255
            outcomes.append((plan, returncode))

    failures: List[str] = []
    for plan, returncode in sorted(outcomes, key=lambda pair: str(pair[0]["batch_id"])):
        if returncode != 0:
            reason = f"collector exited with nonzero status {returncode}"
            terminal = make_terminal(
                plan,
                "batch_abandoned",
                returncode=returncode,
                reason=reason,
            )
            append_jsonl(manifest_path, terminal)
            register_event(state, terminal)
            failures.append(f"{plan['batch_id']}: {reason}")
            continue
        try:
            result = read_batch_result(plan)
            terminal = make_terminal(
                plan, "batch_committed", result=result, returncode=returncode
            )
        except Exception as exc:
            terminal = make_terminal(
                plan,
                "batch_abandoned",
                returncode=returncode,
                reason=str(exc),
            )
            append_jsonl(manifest_path, terminal)
            register_event(state, terminal)
            failures.append(f"{plan['batch_id']}: {exc}")
            continue

        append_jsonl(manifest_path, terminal)
        register_event(state, terminal)
        print(
            f"Committed {plan['task_id']} batch {plan['batch_index']}: "
            f"{result['failures']}/{result['shots']} failures."
        )
    rebuild_tables(state, data_root)
    if failures:
        raise AdaptivePhenomDemError("one or more collectors failed: " + "; ".join(failures))


def initialize_state(
    data_root: Path, manifest_path: Path, seed_namespace: int, *, write: bool
) -> ManifestState:
    events = read_jsonl(manifest_path)
    if not events:
        header = header_value(seed_namespace)
        if write:
            append_jsonl(manifest_path, header)
        events = [header]
    state = parse_state(events)
    validate_header(state.header, seed_namespace)
    return state


@contextlib.contextmanager
def controller_lock(data_root: Path) -> Iterator[None]:
    data_root.mkdir(parents=True, exist_ok=True)
    path = data_root / DEFAULT_LOCK_NAME
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AdaptivePhenomDemError(f"another controller holds {path}") from exc
        yield


def unfinished_at_level(
    state: ManifestState, level_tasks: Sequence[Task]
) -> List[Task]:
    return [
        task
        for task in level_tasks
        if observation(state, task).relative_se > task.target
    ]


def print_level_summary(
    state: ManifestState, level_index: int, level_tasks: Sequence[Task]
) -> None:
    print(f"Completed p-grid level {level_index + 1} of {GRID_POINT_COUNT}:")
    for task in level_tasks:
        current = observation(state, task)
        print(
            f"  {task.code}/{task.variant}: {current.failures}/{current.shots}, "
            f"BLER={current.rate:.6g}, relative SE={current.relative_se:.2%}, "
            f"target={task.target:.2%}"
        )


def run_controller(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else data_root / DEFAULT_MANIFEST_NAME
    )
    limit = worker_limit(args.execution_site)
    processes = limit if args.processes is None else args.processes
    if processes < len(MANUSCRIPT_CURVES) or processes > limit:
        raise AdaptivePhenomDemError(
            f"worker budget must be between {len(MANUSCRIPT_CURVES)} and {limit} "
            f"for {args.execution_site}"
        )

    def run_locked() -> int:
        state = initialize_state(
            data_root, manifest_path, args.seed_namespace, write=not args.dry_run
        )
        if not args.dry_run:
            rebuild_tables(state, data_root)

        if state.pending:
            print(f"Resuming {len(state.pending)} planned batch(es).")
            if args.dry_run:
                for plan in state.pending:
                    print("  " + " ".join(plan["command"]))
                return 0
            run_plans(state.pending, state, manifest_path, data_root)

        waves = 0
        completed_levels = 0
        for level_index, level_tasks in enumerate(TASK_LEVELS):
            while True:
                active = unfinished_at_level(state, level_tasks)
                if not active:
                    print_level_summary(state, level_index, level_tasks)
                    completed_levels += 1
                    if (
                        args.max_levels is not None
                        and completed_levels >= args.max_levels
                    ):
                        return 0
                    break

                batch_sizes = [
                    choose_batch_shots(observation(state, task), task.target)
                    for task in active
                ]
                allocations = allocate_workers(processes, batch_sizes)
                plans = [
                    make_plan(
                        state,
                        task,
                        observation(state, task),
                        shots,
                        workers,
                        args.execution_site,
                        data_root,
                        args.seed_namespace,
                    )
                    for task, shots, workers in zip(active, batch_sizes, allocations)
                ]

                print(
                    f"p-grid level {level_index + 1}: launching {len(plans)} "
                    f"adaptive batch(es) with {sum(allocations)} total workers."
                )
                if args.dry_run:
                    for plan in plans:
                        print("  " + " ".join(plan["command"]))
                    return 0

                for plan in plans:
                    append_jsonl(manifest_path, plan)
                    register_event(state, plan)
                run_plans(plans, state, manifest_path, data_root)
                waves += 1
                if args.max_waves is not None and waves >= args.max_waves:
                    print(f"Stopped after --max-waves={args.max_waves}.")
                    return 0
        return 0

    if args.dry_run:
        return run_locked()
    with controller_lock(data_root):
        return run_locked()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the seven manuscript curves over their ten-point April grids, "
            "adding shots until one-binomial-SE error bars reach the configured "
            "relative target."
        )
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--execution-site", choices=("local", "cpu200"), default="local"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help=(
            f"Global worker budget. Defaults to {LOCAL_MAX_WORKERS} locally or "
            f"{CPU200_MAX_WORKERS} on cpu200."
        ),
    )
    parser.add_argument(
        "--seed-namespace", type=int, default=DEFAULT_SEED_NAMESPACE
    )
    parser.add_argument("--max-waves", type=int, default=None)
    parser.add_argument("--max-levels", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.seed_namespace < 0:
        parser.error("--seed-namespace must be nonnegative")
    if args.max_waves is not None and args.max_waves <= 0:
        parser.error("--max-waves must be positive")
    if args.max_levels is not None and args.max_levels <= 0:
        parser.error("--max-levels must be positive")
    limit = worker_limit(args.execution_site)
    chosen = limit if args.processes is None else args.processes
    if chosen < len(MANUSCRIPT_CURVES) or chosen > limit:
        parser.error(
            f"--processes must be between {len(MANUSCRIPT_CURVES)} and {limit} "
            f"for {args.execution_site}"
        )
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run_controller(parse_args(argv))
    except AdaptivePhenomDemError as exc:
        print(f"adaptive DEM collection error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
