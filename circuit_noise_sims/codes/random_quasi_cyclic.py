import numpy as np
import ldpc.protograph as pt
import networkx as nx
from bposd.hgp import hgp
from ldpc.code_util import compute_code_parameters


def get_check_adj_graph(H):
    A = (H @ H.T != 0).astype(int)
    np.fill_diagonal(A, 0)
    return nx.from_numpy_array(A, create_using=nx.MultiGraph())


def get_check_coloring(H):
    G = get_check_adj_graph(H)
    color_dict = nx.greedy_color(G, strategy="independent_set")
    num_colors = max(color_dict.values()) + 1
    coloring = []
    for i in range(num_colors):
        coloring.append([k for k, v in color_dict.items() if v == i])
    return coloring


def get_random_qc_code(
    rows=3,
    cols=4,
    row_weight=3,
    lift=6,
    min_dist=11,
    max_coloring=3,
    tries=200,
    seed=None,
):
    rng = np.random.default_rng(seed)

    B = [["" for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        inds = rng.choice(np.arange(cols), size=row_weight, replace=False)
        for pos in inds:
            B[row][int(pos)] = 1

    H = None
    for _ in range(tries):
        M = rng.integers(1, lift, size=(rows, cols))
        A = [r.copy() for r in B]
        for i in range(rows):
            for j in range(cols):
                if B[i][j] != "":
                    A[i][j] = int(M[i, j])

        candidate_H = pt.array(A).to_binary(lift_parameter=lift).astype(int)
        n, k, d = compute_code_parameters(candidate_H)
        if d >= min_dist:
            coloring = get_check_coloring(candidate_H)
            if len(coloring) <= max_coloring:
                H = candidate_H
                break
    else:
        raise RuntimeError(
            f"Failed to find random QC code with d>={min_dist} and "
            f"coloring<={max_coloring} after {tries} tries."
        )

    print(f"\t\tRandom quasi-cyclic code: [n, k, d] = {compute_code_parameters(H)}")

    code = hgp(h1=H, h2=H, compute_distance=True)
    code.name = "Random Quasi-Cyclic Code HGP"
    print(f"\t\tHGP Code: [[{code.N}, {code.K}, {code.D}]]")

    return code, H
