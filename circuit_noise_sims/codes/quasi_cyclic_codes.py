import numpy as np
import ldpc.protograph as pt
from bposd.hgp import hgp
from ldpc.code_util import compute_code_parameters


def _from_protograph(array_like, lift_parameter, label):
    H = pt.array(array_like).to_binary(lift_parameter=lift_parameter).astype(int)
    print(f"\t\t{label}: [n, k, d] = {compute_code_parameters(H)}")

    code = hgp(h1=H, h2=H, compute_distance=True)
    code.name = f"{label} HGP"
    print(f"\t\tHGP Code: [[{code.N}, {code.K}, {code.D}]]")

    return code, H


def get_qc_20_5_9():
    return _from_protograph(
        [[4, "", 4, 3], ["", 3, 3, 4], [3, 4, "", 3]],
        lift_parameter=5,
        label="[20,5,9] quasi-cyclic code",
    )


def get_qc_24_6_10():
    return _from_protograph(
        [[5, "", 3, 3], ["", 4, 2, 1], [2, 1, "", 1]],
        lift_parameter=6,
        label="[24,6,10] quasi-cyclic code",
    )


def get_qc_28_7_11():
    return _from_protograph(
        [[1, "", 2, 3], ["", 5, 6, 1], [4, 5, "", 5]],
        lift_parameter=7,
        label="[28,7,11] quasi-cyclic code",
    )


def get_qc_20_4_9():
    return _from_protograph(
        [
            ["", 2, "", 3, 3],
            ["", "", 3, 1, 1],
            [3, 2, "", "", 1],
            [2, "", 2, 1, ""],
        ],
        lift_parameter=4,
        label="[20,4,9] quasi-cyclic code",
    )
