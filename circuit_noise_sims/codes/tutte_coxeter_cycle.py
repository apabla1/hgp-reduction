import numpy as np
from bposd.hgp import hgp
from ldpc.code_util import compute_code_parameters


def get_tutte_coxeter_cycle(remove_redundant_row=True):
    checks = [
        [1, 4, 7],
        [2, 10, 13],
        [3, 16, 19],
        [11, 22, 25],
        [14, 28, 31],
        [12, 34, 37],
        [15, 40, 43],
        [5, 23, 29],
        [6, 35, 41],
        [8, 26, 32],
        [9, 38, 44],
        [17, 24, 45],
        [20, 30, 39],
        [21, 27, 42],
        [18, 33, 36],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
        [22, 23, 24],
        [25, 26, 27],
        [28, 29, 30],
        [31, 32, 33],
        [34, 35, 36],
        [37, 38, 39],
        [40, 41, 42],
        [43, 44, 45],
    ]

    H = np.zeros((30, 45), dtype=int)
    for i in range(30):
        H[i, np.array(checks[i]) - 1] = [1, 1, 1]

    if remove_redundant_row:
        H = H[:-1]

    print(f"\t\tTutte-Coxeter code: [n, k, d] = {compute_code_parameters(H)}")

    code = hgp(h1=H, h2=H, compute_distance=True)
    code.name = "Tutte-Coxeter Code HGP"
    print(f"\t\tHGP Code: [[{code.N}, {code.K}, {code.D}]]")

    return code, H
