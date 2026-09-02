"""QUITS-compatible cardinal schedules for original HGP codes.

The construction follows the balanced-sign assignment and E/N/S/W ordering
used by QUITS' ``cardinal`` strategy.  Qubit indices are converted to the
ordering used by ``bposd.hgp`` and by this repository:

    data, X-check ancillas, Z-check ancillas.

The default seed is fixed to zero because it gives depth eight for the
``qc_20_5_9`` and row-reduced Heawood examples used in the manuscript.
"""

import random
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np

from functions.edge_coloring import edge_color_bipartite


Cnot = Tuple[int, int]
CnotLayer = List[Cnot]
DIRECTION_ORDER: Tuple[str, str, str, str] = ("E", "N", "S", "W")


def _as_binary_array(matrix: Any, name: str) -> np.ndarray:
    array = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional parity-check matrix")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    return array.astype(np.uint8, copy=False)


def balanced_edge_signs(matrix: Any, seed: int = 0) -> Dict[Tuple[int, int], bool]:
    """Assign balanced signs using the current QUITS cardinal heuristic.

    A fresh random stream is used for each classical matrix, matching QUITS,
    which resets the stream to the requested seed for each call.  A random
    number is drawn for every nonzero matrix entry, even when the accumulated
    score already fixes the sign.
    """

    h = _as_binary_array(matrix, "matrix")
    rng = random.Random(seed)
    row_scores: Dict[int, int] = {}
    column_scores: Dict[int, int] = {}
    signs: Dict[Tuple[int, int], bool] = {}

    for row, column in np.argwhere(h == 1):
        row, column = int(row), int(column)
        score = row_scores.get(row, 0) + column_scores.get(column, 0)
        random_value = rng.random()
        positive = score > 0 or (score == 0 and random_value >= 0.5)
        sign = 1 if positive else -1

        signs[row, column] = positive
        row_scores[row] = row_scores.get(row, 0) - sign
        column_scores[column] = column_scores.get(column, 0) - sign

    return signs


def _validate_hgp_dimensions(code: Any, h1: np.ndarray, h2: np.ndarray) -> None:
    m1, n1 = h1.shape
    m2, n2 = h2.shape
    expected_n = n1 * n2 + m1 * m2
    expected_mx = m1 * n2
    expected_mz = n1 * m2

    if int(code.N) != expected_n:
        raise ValueError(
            f"code.N={code.N} is incompatible with H1/H2; expected {expected_n}"
        )
    if code.hx.shape != (expected_mx, expected_n):
        raise ValueError(
            f"code.hx has shape {code.hx.shape}; expected {(expected_mx, expected_n)}"
        )
    if code.hz.shape != (expected_mz, expected_n):
        raise ValueError(
            f"code.hz has shape {code.hz.shape}; expected {(expected_mz, expected_n)}"
        )


def _color_direction(edges: Sequence[Cnot]) -> List[CnotLayer]:
    if not edges:
        return []

    graph = nx.Graph()
    orientation: Dict[frozenset, Cnot] = {}
    for control, target in edges:
        graph.add_edge(control, target)
        orientation[frozenset((control, target))] = (control, target)

    color_classes = edge_color_bipartite(graph)
    return [
        [orientation[frozenset(edge)] for edge in sorted(color_class)]
        for color_class in color_classes
    ]


def get_cardinal_schedule(
    code: Any,
    h1: Any,
    h2: Any,
    seed: int = 0,
) -> Tuple[List[CnotLayer], Dict[str, int]]:
    """Return cardinal CNOT layers and E/N/S/W component depths.

    Each gate is returned as ``(control, target)``.  The flattened schedule
    follows the QUITS direction order E, N, S, W, with an optimal bipartite
    edge coloring within each direction.
    """

    h1 = _as_binary_array(h1, "h1")
    h2 = _as_binary_array(h2, "h2")
    _validate_hgp_dimensions(code, h1, h2)

    m1, n1 = h1.shape
    m2, n2 = h2.shape
    n_data = int(code.N)
    n_xchecks = int(code.hx.shape[0])

    horizontal_signs = balanced_edge_signs(h1, seed)
    vertical_signs = balanced_edge_signs(h2, seed)
    directed_edges: Dict[str, List[Cnot]] = {
        direction: [] for direction in DIRECTION_ORDER
    }

    # H1-generated horizontal edges.
    for i1, j1 in np.argwhere(h1 == 1):
        i1, j1 = int(i1), int(j1)
        for k in range(n2 + m2):
            if k < n2:
                # X-check ancilla -> VV-sector data.
                gate = (n_data + i1 * n2 + k, j1 * n2 + k)
            else:
                i2 = k - n2
                # CC-sector data -> Z-check ancilla.
                gate = (
                    n1 * n2 + i1 * m2 + i2,
                    n_data + n_xchecks + j1 * m2 + i2,
                )

            direction = "E" if ((k < n2) ^ horizontal_signs[i1, j1]) else "W"
            directed_edges[direction].append(gate)

    # H2-generated vertical edges.
    for i2, j2 in np.argwhere(h2 == 1):
        i2, j2 = int(i2), int(j2)
        for k in range(n1 + m1):
            if k < n1:
                # VV-sector data -> Z-check ancilla.
                gate = (
                    k * n2 + j2,
                    n_data + n_xchecks + k * m2 + i2,
                )
            else:
                i1 = k - n1
                # X-check ancilla -> CC-sector data.
                gate = (
                    n_data + i1 * n2 + j2,
                    n1 * n2 + i1 * m2 + i2,
                )

            direction = "N" if ((k < n1) ^ vertical_signs[i2, j2]) else "S"
            directed_edges[direction].append(gate)

    schedule: List[CnotLayer] = []
    component_depths: Dict[str, int] = {}
    for direction in DIRECTION_ORDER:
        direction_layers = _color_direction(directed_edges[direction])
        component_depths[direction] = len(direction_layers)
        schedule.extend(direction_layers)

    return schedule, component_depths
