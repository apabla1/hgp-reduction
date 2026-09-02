from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

# The production scripts use ``functions`` as a top-level package when run
# from circuit_noise_sims/.  Add that directory explicitly so the full
# ``pytest`` console entrypoint collects this test from the repository root.
CIRCUIT_NOISE_SIMS_DIR = Path(__file__).resolve().parents[1]
if str(CIRCUIT_NOISE_SIMS_DIR) not in sys.path:
    sys.path.insert(0, str(CIRCUIT_NOISE_SIMS_DIR))

import plot_phenom_dem_manuscript as plotting


def _pdf_media_box(path: Path) -> tuple[float, float, float, float]:
    match = re.search(
        rb"/MediaBox\s*\[\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        path.read_bytes(),
    )
    assert match is not None
    return tuple(float(value) for value in match.groups())


def _table(start: float = 1e-3) -> np.ndarray:
    return np.asarray(
        [
            [2 * start, 20, 10_000],
            [start, 5, 10_000],
            [4 * start, 80, 10_000],
        ],
        dtype=float,
    )


def _write_complete_fixture(decoder_root: Path) -> None:
    sources = (
        ("qc_20_5_9", "unreduced_cardinal", 1e-3),
        ("qc_20_5_9", "reduced_split", 1e-3),
        ("qc_20_5_9", "reduced_random", 1e-3),
        ("qc_24_6_10", "reduced_split", 1e-3),
        ("heawood_cycle", "unreduced_cardinal", 5e-4),
        ("heawood_cycle", "reduced_split", 5e-4),
        ("heawood_cycle", "reduced_random", 5e-4),
    )
    for code, variant, start in sources:
        path = decoder_root / variant / f"{code}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, _table(start))


def test_curve_labels_name_the_actual_schedules() -> None:
    assert [(style.variant, style.label) for style in plotting.MAIN_CURVES] == [
        ("unreduced_cardinal", "original, balanced-cardinal SE"),
        ("reduced_split", "reduced, split SE"),
        ("reduced_random", "reduced, unsplit SE"),
    ]
    assert plotting.QC24_CURVE.label.endswith("reduced, split SE")


def test_load_result_table_sorts_p(tmp_path: Path) -> None:
    path = tmp_path / "table.npy"
    np.save(path, _table())
    loaded = plotting.load_result_table(path)
    assert np.all(np.diff(loaded[:, 0]) > 0)


def test_load_result_table_rejects_duplicate_p(tmp_path: Path) -> None:
    path = tmp_path / "table.npy"
    table = _table()
    table[1, 0] = table[0, 0]
    np.save(path, table)
    with pytest.raises(ValueError, match="duplicate"):
        plotting.load_result_table(path)


def test_generate_plots_writes_both_transparent_pdfs(tmp_path: Path) -> None:
    decoder_root = tmp_path / plotting.DECODER_CONFIG
    _write_complete_fixture(decoder_root)
    output_dir = tmp_path / "plots"

    qc_path, heawood_path = plotting.generate_plots(decoder_root, output_dir)

    assert qc_path.name == plotting.DEFAULT_QC_FILENAME
    assert heawood_path.name == plotting.DEFAULT_HEAWOOD_FILENAME
    assert qc_path.read_bytes().startswith(b"%PDF-")
    assert heawood_path.read_bytes().startswith(b"%PDF-")
    assert _pdf_media_box(qc_path) == (0.0, 0.0, 396.0, 324.0)
    assert _pdf_media_box(heawood_path) == (0.0, 0.0, 396.0, 324.0)


def test_generate_plots_uses_distinct_april_x_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, tuple[float, float]]] = []

    def record_panel(
        _decoder_root: Path,
        code: str,
        _output_path: Path,
        *,
        include_qc24: bool,
        use_tex: bool,
        x_limits: tuple[float, float],
    ) -> None:
        del include_qc24, use_tex
        calls.append((code, x_limits))

    monkeypatch.setattr(plotting, "_plot_panel", record_panel)
    plotting.generate_plots(tmp_path, tmp_path / "plots")

    assert calls == [
        ("qc_20_5_9", plotting.QC_X_LIMITS),
        ("heawood_cycle", plotting.HEAWOOD_X_LIMITS),
    ]
    assert plotting.QC_X_LIMITS[0] < 1e-3
    assert plotting.HEAWOOD_X_LIMITS[0] < 5e-4


def test_main_requires_audit_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    decoder_root = tmp_path / plotting.DECODER_CONFIG
    _write_complete_fixture(decoder_root)
    monkeypatch.setattr(
        plotting,
        "audit_dataset",
        lambda _root: (_ for _ in ()).throw(ValueError("audit sentinel")),
    )
    assert plotting.main(["--data-root", str(tmp_path)]) == 1


def test_main_allows_explicit_unverified_fixture(tmp_path: Path) -> None:
    decoder_root = tmp_path / plotting.DECODER_CONFIG
    _write_complete_fixture(decoder_root)
    output_dir = tmp_path / "plots"
    assert (
        plotting.main(
            [
                "--data-root",
                str(tmp_path),
                "--output-dir",
                str(output_dir),
                "--allow-unverified-data",
            ]
        )
        == 0
    )
