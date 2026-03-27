import numpy as np
from bposd.hgp import hgp
from ldpc.code_util import compute_code_parameters


def get_petersen_cycle():
    checks = [
        [1, 5, 6],
        [4, 5, 10],
        [10, 14, 15],
        [6, 11, 12],
        [7, 13, 14],
        [1, 2, 7],
        [3, 4, 9],
        [9, 11, 13],
        [8, 12, 15],
        [2, 3, 8],
    ]

    H = np.zeros((10, 15), dtype=int)
    for i in range(10):
        H[i, np.array(checks[i]) - 1] = [1, 1, 1]

    print(f"\t\tPetersen code: [n, k, d] = {compute_code_parameters(H)}")

    code = hgp(h1=H, h2=H, compute_distance=True)
    code.name = "Petersen Code HGP"
    print(f"\t\tHGP Code: [[{code.N}, {code.K}, {code.D}]]")

    return code, H
