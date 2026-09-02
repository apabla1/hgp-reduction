#!/usr/bin/env python3
"""Add precision-refinement batches to an existing phenomenological-DEM run."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import phenom_dem_adaptive_collection as adaptive


SAMPLING_MODE = "precision_refinement_v1"
DEFAULT_TARGET_RELATIVE_SE = 0.10


def refinement_controller_sha256() -> str:
    """Return the digest recorded on every plan made by this controller."""

    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _same_p(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-13, abs_tol=0.0)


def selected_task_levels(
    code: str,
    variants: Optional[Sequence[str]],
    *,
    p_max: Optional[float],
    p_values: Optional[Sequence[float]],
) -> Tuple[Tuple[adaptive.Task, ...], ...]:
    """Return selected tasks while retaining the fixed high-to-low level order."""

    code_tasks = [task for task in adaptive.TASKS if task.code == code]
    if not code_tasks:
        raise adaptive.AdaptivePhenomDemError(
            f"code {code!r} has no task in the fixed manuscript sweep"
        )

    available_variants = {task.variant for task in code_tasks}
    selected_variants = (
        available_variants if variants is None else set(map(str, variants))
    )
    unknown_variants = selected_variants - available_variants
    if unknown_variants:
        names = ", ".join(sorted(unknown_variants))
        raise adaptive.AdaptivePhenomDemError(
            f"code {code!r} has no fixed task for variant(s): {names}"
        )
    if not selected_variants:
        raise adaptive.AdaptivePhenomDemError("at least one variant must be selected")

    available_p_values = sorted({task.p for task in code_tasks})
    if p_values is not None:
        selected_p_values = set()
        for requested in p_values:
            matches = [
                value for value in available_p_values if _same_p(value, requested)
            ]
            if len(matches) != 1:
                raise adaptive.AdaptivePhenomDemError(
                    f"p={requested:.17g} is not on the fixed grid for {code}"
                )
            selected_p_values.add(matches[0])
    elif p_max is not None:
        selected_p_values = {
            value
            for value in available_p_values
            if value < p_max or _same_p(value, p_max)
        }
    else:
        raise adaptive.AdaptivePhenomDemError(
            "one of p_max or p_values must select refinement points"
        )

    levels: List[Tuple[adaptive.Task, ...]] = []
    for level in adaptive.TASK_LEVELS:
        selected = tuple(
            task
            for task in level
            if task.code == code
            and task.variant in selected_variants
            and task.p in selected_p_values
        )
        if selected:
            levels.append(selected)
    if not levels:
        raise adaptive.AdaptivePhenomDemError(
            "the refinement selectors do not match any fixed task"
        )
    return tuple(levels)


def make_refinement_plan(
    state: adaptive.ManifestState,
    task: adaptive.Task,
    before: adaptive.Observation,
    shots: int,
    processes: int,
    execution_site: str,
    data_root: Path,
    seed_namespace: int,
    target_relative_se: float,
) -> Dict[str, Any]:
    """Make a canonical adaptive plan with explicit refinement provenance."""

    plan = adaptive.make_plan(
        state,
        task,
        before,
        shots,
        processes,
        execution_site,
        data_root,
        seed_namespace,
    )
    plan.update(
        {
            "sampling_mode": SAMPLING_MODE,
            "refinement_target_relative_se": float(target_relative_se),
            "refinement_controller_sha256": refinement_controller_sha256(),
        }
    )
    return plan


def unfinished_refinement_tasks(
    state: adaptive.ManifestState,
    tasks: Sequence[adaptive.Task],
    target_relative_se: float,
) -> List[adaptive.Task]:
    """Return selected tasks whose cumulative estimate remains too imprecise."""

    return [
        task
        for task in tasks
        if adaptive.observation(state, task).relative_se > target_relative_se
    ]


def load_existing_state(
    data_root: Path, manifest_path: Path
) -> Tuple[adaptive.ManifestState, int]:
    """Load and validate an existing fixed-source adaptive ledger."""

    if not manifest_path.is_file():
        raise adaptive.AdaptivePhenomDemError(
            f"refinement requires an existing manifest: {manifest_path}"
        )
    events = adaptive.read_jsonl(manifest_path)
    state = adaptive.parse_state(events)
    try:
        seed_namespace = int(state.header["seed_namespace"])
    except (KeyError, TypeError, ValueError) as exc:
        raise adaptive.AdaptivePhenomDemError(
            "existing manifest has no valid seed namespace"
        ) from exc
    adaptive.validate_header(state.header, seed_namespace)
    return state, seed_namespace


def _print_pending(plans: Sequence[Mapping[str, Any]]) -> None:
    for plan in plans:
        print("  " + " ".join(map(str, plan["command"])))


def _print_level_summary(
    state: adaptive.ManifestState,
    tasks: Sequence[adaptive.Task],
    target_relative_se: float,
) -> None:
    print(f"Refinement level p={tasks[0].p:.17g} is complete:")
    for task in tasks:
        current = adaptive.observation(state, task)
        print(
            f"  {task.code}/{task.variant}: {current.failures}/{current.shots}, "
            f"BLER={current.rate:.6g}, relative SE={current.relative_se:.2%}, "
            f"refinement target={target_relative_se:.2%}"
        )


def run_controller(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else data_root / adaptive.DEFAULT_MANIFEST_NAME
    )
    levels = selected_task_levels(
        args.code,
        args.variants,
        p_max=args.p_max,
        p_values=args.p_values,
    )

    with adaptive.controller_lock(data_root):
        state, seed_namespace = load_existing_state(data_root, manifest_path)
        if not args.dry_run:
            adaptive.rebuild_tables(state, data_root)

        if state.pending:
            print(f"Resuming {len(state.pending)} planned batch(es).")
            if args.dry_run:
                _print_pending(state.pending)
                return 0
            adaptive.run_plans(
                state.pending, state, manifest_path, data_root
            )

        waves = 0
        for level_tasks in levels:
            while True:
                active = unfinished_refinement_tasks(
                    state, level_tasks, args.target_relative_se
                )
                if not active:
                    _print_level_summary(
                        state, level_tasks, args.target_relative_se
                    )
                    break

                # Give every concurrent task at least one worker.  A smaller
                # global budget advances a subset without leaving this p level.
                wave_tasks = active[: args.processes]
                batch_sizes = [
                    adaptive.choose_batch_shots(
                        adaptive.observation(state, task),
                        args.target_relative_se,
                    )
                    for task in wave_tasks
                ]
                allocations = adaptive.allocate_workers(
                    args.processes, batch_sizes
                )
                plans = [
                    make_refinement_plan(
                        state,
                        task,
                        adaptive.observation(state, task),
                        shots,
                        workers,
                        args.execution_site,
                        data_root,
                        seed_namespace,
                        args.target_relative_se,
                    )
                    for task, shots, workers in zip(
                        wave_tasks, batch_sizes, allocations
                    )
                ]

                print(
                    f"Refining p={level_tasks[0].p:.17g}: launching "
                    f"{len(plans)} batch(es) with "
                    f"{sum(allocations)} total workers."
                )
                if args.dry_run:
                    _print_pending(plans)
                    return 0

                # Reserve canonical batch identifiers and sampler seeds before
                # any collector subprocess is allowed to start.
                for plan in plans:
                    adaptive.append_jsonl(manifest_path, plan)
                    adaptive.register_event(state, plan)
                adaptive.run_plans(plans, state, manifest_path, data_root)

                waves += 1
                if args.max_waves is not None and waves >= args.max_waves:
                    print(f"Stopped after --max-waves={args.max_waves}.")
                    return 0
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    codes = sorted({task.code for task in adaptive.TASKS})
    variants = sorted({task.variant for task in adaptive.TASKS})
    parser = argparse.ArgumentParser(
        description=(
            "Add immutable precision-refinement batches to selected points in "
            "an existing phenomenological-DEM ledger."
        )
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--code", choices=codes, default="heawood_cycle")
    parser.add_argument("--variants", nargs="+", choices=variants, default=None)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--p-max", type=float)
    selector.add_argument("--p-values", nargs="+", type=float)
    parser.add_argument(
        "--target-relative-se",
        type=float,
        default=DEFAULT_TARGET_RELATIVE_SE,
    )
    parser.add_argument(
        "--execution-site", choices=("local", "cpu200"), default="local"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Global worker budget; defaults to the selected site's cap.",
    )
    parser.add_argument("--max-waves", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.p_max is not None and not 0 < args.p_max < 1:
        parser.error("--p-max must lie in (0, 1)")
    if args.p_values is not None and any(
        not 0 < value < 1 for value in args.p_values
    ):
        parser.error("every --p-values entry must lie in (0, 1)")
    if not 0 < args.target_relative_se < adaptive.TARGET_RELATIVE_SE:
        parser.error(
            "--target-relative-se must be positive and tighter than the "
            f"fixed {adaptive.TARGET_RELATIVE_SE:.2%} target"
        )
    if args.max_waves is not None and args.max_waves <= 0:
        parser.error("--max-waves must be positive")

    limit = adaptive.worker_limit(args.execution_site)
    if args.processes is None:
        args.processes = limit
    elif not 1 <= args.processes <= limit:
        parser.error(
            f"--processes must be between 1 and {limit} for "
            f"{args.execution_site}"
        )
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run_controller(parse_args(argv))
    except adaptive.AdaptivePhenomDemError as exc:
        print(f"phenomenological-DEM refinement error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
