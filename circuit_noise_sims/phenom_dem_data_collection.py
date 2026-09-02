#!/usr/bin/env python3
"""Collect one immutable batch using Stim-informed phenomenological priors."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np

from functions.H_to_CNOT_circuit import (
    generate_full_circuit,
    generate_full_circuit_cardinal,
    generate_full_circuit_split,
)
from functions.phenom_dem_config import (
    DECODER_CONFIG,
    MODEL_NAME,
    RANDOM_SCHEDULE_SEED,
    RELAY_GAMMA0,
    RELAY_GAMMA_INTERVAL,
    RELAY_MAX_ITER,
    RELAY_NUM_SETS,
    RELAY_PARAMETERS,
    RELAY_PRE_ITER,
    RELAY_STOP_NCONV,
    schedule_seed,
    worker_limit,
)
from functions.phenom_dem_decoding import (
    build_phenomenological_dem_model,
    num_failures_phenom_dem,
)
from functions.reduction_funcs import get_reduced_code
from functions.sim_common import VARIANTS, load_code


SCHEMA_VERSION = 1
RUNTIME_DISTRIBUTIONS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "stim": "stim",
    "ldpc": "ldpc",
    "relay_bp": "relay-bp",
}


def runtime_versions() -> Dict[str, str]:
    """Return versions needed to reproduce one immutable batch."""

    versions = {"python": platform.python_version()}
    for key, distribution in RUNTIME_DISTRIBUTIONS.items():
        versions[key] = importlib.metadata.version(distribution)
    return versions


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derive_sampler_seed(base_seed: int, worker_id: int) -> int:
    payload = f"{int(base_seed)}|worker={int(worker_id)}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) or 1


def _target_value(target: Any) -> int:
    value = getattr(target, "value", None)
    if value is None:
        value = getattr(target, "val")
    return int(value)


def cnot_order_metadata(circuit: Any) -> Dict[str, Any]:
    """Fingerprint the ordered CNOT pairs and their Stim tick layers."""

    tick = 0
    pairs = []
    layer_counts: Dict[int, int] = {}
    for instruction in circuit.flattened():
        name = str(instruction.name)
        if name == "TICK":
            tick += 1
            continue
        if name not in ("CX", "CNOT"):
            continue
        targets = instruction.targets_copy()
        if len(targets) % 2:
            raise ValueError("CNOT instruction has an odd target count")
        for index in range(0, len(targets), 2):
            pair = [tick, _target_value(targets[index]), _target_value(targets[index + 1])]
            pairs.append(pair)
            layer_counts[tick] = layer_counts.get(tick, 0) + 1
    if not pairs:
        raise ValueError("generated circuit contains no CNOT gates")
    return {
        "cnot_order_sha256": sha256_json(pairs),
        "cnot_pair_count": len(pairs),
        "cnot_layer_count": len(layer_counts),
        "cnot_layer_pair_counts": [
            [layer, layer_counts[layer]] for layer in sorted(layer_counts)
        ],
    }


def circuit_metadata(circuit: Any) -> Dict[str, Any]:
    metadata = {
        "circuit_sha256": sha256_text(str(circuit)),
        "num_qubits": int(circuit.num_qubits),
        "num_measurements": int(circuit.num_measurements),
        "num_detectors": int(circuit.num_detectors),
        "num_observables": int(circuit.num_observables),
    }
    metadata.update(cnot_order_metadata(circuit))
    return metadata


def _json_metadata(model: Any) -> Dict[str, Any]:
    metadata = model.metadata()
    if not isinstance(metadata, Mapping):
        raise TypeError("phenomenological model metadata() must return a mapping")
    value = dict(metadata)
    # This also rejects NaN, NumPy scalars, and other nonportable provenance.
    json.dumps(value, sort_keys=True, allow_nan=False)
    return value


def build_circuit_and_model(
    code_name: str,
    variant: str,
    p: float,
    random_schedule_seed: int = RANDOM_SCHEDULE_SEED,
) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build the selected circuit and exactly one reusable decoder model."""

    unreduced_code, h = load_code(code_name)
    (
        hx1,
        hx2,
        hx3,
        hz1,
        hz2,
        hz3,
        reduced_code,
        _,
        _,
        distance,
    ) = get_reduced_code(unreduced_code, h)

    noise = (float(p) / 10.0, float(p), float(p))
    if variant == "unreduced_cardinal":
        selected_code = unreduced_code
        used_seed = schedule_seed(variant)
        circuit = generate_full_circuit_cardinal(
            selected_code, h, h, distance, noise, seed=used_seed
        )
    elif variant == "unreduced_random":
        selected_code = unreduced_code
        used_seed = int(random_schedule_seed)
        circuit = generate_full_circuit(selected_code, distance, noise, used_seed)
    elif variant == "reduced_random":
        selected_code = reduced_code
        used_seed = int(random_schedule_seed)
        circuit = generate_full_circuit(selected_code, distance, noise, used_seed)
    elif variant == "reduced_split":
        selected_code = reduced_code
        used_seed = int(random_schedule_seed)
        circuit = generate_full_circuit_split(
            hx1,
            hx2,
            hx3,
            hz1,
            hz2,
            hz3,
            distance,
            noise,
            used_seed,
            code=selected_code,
        )
    else:
        raise ValueError(f"unknown circuit variant: {variant}")

    model = build_phenomenological_dem_model(circuit, selected_code, int(distance))
    code_meta = {
        "code_n": int(selected_code.N),
        "code_k": int(selected_code.K),
        "rounds": int(distance),
        "schedule_seed": int(used_seed),
    }
    circuit_meta = circuit_metadata(circuit)
    model_meta = _json_metadata(model)
    model_meta["metadata_sha256"] = sha256_json(model_meta)
    return circuit, model, code_meta, circuit_meta, model_meta


def _sample_worker(
    circuit: Any,
    model: Any,
    decoder_params: Sequence[Any],
    shots: int,
    sampler_seed: int,
    worker_id: int,
) -> int:
    return int(
        num_failures_phenom_dem(
            circuit,
            model,
            decoder_params,
            shots,
            sampler_seed=sampler_seed,
            worker_id=worker_id,
        )
    )


def sample_batch(
    circuit: Any,
    model: Any,
    shots: int,
    processes: int,
    sampler_seed: int,
) -> int:
    """Sample one batch, reusing the parent-built model in every worker."""

    workers = min(int(processes), int(shots))
    if workers <= 0:
        raise ValueError("shots and processes must be positive")
    base, remainder = divmod(int(shots), workers)
    params = [
        (
            circuit,
            model,
            RELAY_PARAMETERS,
            base + (1 if index < remainder else 0),
            derive_sampler_seed(sampler_seed, index + 1),
            index + 1,
        )
        for index in range(workers)
    ]
    if len(params) == 1:
        return _sample_worker(*params[0])
    with Pool(processes=len(params)) as pool:
        return int(np.sum(pool.starmap(_sample_worker, params)))


def request_spec(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_id": args.batch_id,
        "model": MODEL_NAME,
        "decoder_config": DECODER_CONFIG,
        "code": args.code,
        "variant": args.variant,
        "p": float(args.p),
        "p1": float(args.p) / 10.0,
        "p2": float(args.p),
        "p_spam": float(args.p),
        "shots": int(args.shots),
        "sampler_seed": int(args.sampler_seed),
        "schedule_seed": int(args.schedule_seed),
        "processes": int(args.processes),
        "execution_site": args.execution_site,
        "relay_parameters": {
            "gamma0": RELAY_GAMMA0,
            "pre_iter": RELAY_PRE_ITER,
            "num_sets": RELAY_NUM_SETS,
            "set_max_iter": RELAY_MAX_ITER,
            "gamma_dist_interval": list(RELAY_GAMMA_INTERVAL),
            "stop_nconv": RELAY_STOP_NCONV,
        },
    }


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"batch result is not a JSON object: {path}")
    return result


def validate_existing_result(path: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    result = read_json(path)
    if result.get("request") != spec:
        raise ValueError(f"existing batch result does not match this request: {path}")
    failures = int(result.get("failures", -1))
    shots = int(result.get("shots", -1))
    if shots != int(spec["shots"]) or failures < 0 or failures > shots:
        raise ValueError(f"existing batch result has invalid counts: {path}")
    return result


@contextlib.contextmanager
def exclusive_result_lock(result_path: Path) -> Iterator[None]:
    lock_path = result_path.with_suffix(result_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def collect(args: argparse.Namespace) -> Dict[str, Any]:
    result_path = args.result.expanduser().resolve()
    spec = request_spec(args)
    with exclusive_result_lock(result_path):
        if result_path.exists():
            return validate_existing_result(result_path, spec)

        circuit, model, code_meta, circuit_meta, model_meta = build_circuit_and_model(
            args.code, args.variant, args.p, args.schedule_seed
        )
        failures = sample_batch(
            circuit,
            model,
            shots=args.shots,
            processes=args.processes,
            sampler_seed=args.sampler_seed,
        )
        result = {
            "request": spec,
            "completed_utc": utc_now(),
            "failures": int(failures),
            "shots": int(args.shots),
            "code_metadata": code_meta,
            "circuit_metadata": circuit_meta,
            "phenomenological_model": model_meta,
            "runtime_versions": runtime_versions(),
        }
        atomic_write_json(result_path, result)
        return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect one Relay-BP batch on a Stim-prior phenomenological graph."
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--code", required=True)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--p", required=True, type=float)
    parser.add_argument("--shots", required=True, type=int)
    parser.add_argument("--sampler-seed", required=True, type=int)
    parser.add_argument("--schedule-seed", type=int, default=None)
    parser.add_argument("--execution-site", choices=("local", "cpu200"), default="local")
    parser.add_argument("--processes", required=True, type=int)
    args = parser.parse_args(argv)
    if not args.batch_id.strip():
        parser.error("--batch-id must not be empty")
    if not 0 < args.p < 1:
        parser.error("--p must lie in (0, 1)")
    if args.shots <= 0:
        parser.error("--shots must be positive")
    if args.sampler_seed < 0:
        parser.error("--sampler-seed must be nonnegative")
    expected_schedule_seed = schedule_seed(args.variant)
    if args.schedule_seed is None:
        args.schedule_seed = expected_schedule_seed
    elif args.schedule_seed < 0 or args.schedule_seed != expected_schedule_seed:
        parser.error(
            f"--schedule-seed must be {expected_schedule_seed} "
            f"for {args.variant}"
        )
    limit = worker_limit(args.execution_site)
    if not 1 <= args.processes <= min(limit, args.shots):
        parser.error(
            f"--processes must be between 1 and min(shots, {limit}) "
            f"for {args.execution_site}"
        )
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = collect(args)
    except Exception as exc:
        print(f"Phenomenological-DEM batch collection failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
