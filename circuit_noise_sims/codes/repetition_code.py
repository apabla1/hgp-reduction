import numpy as np
from bposd.hgp import hgp
from ldpc.codes import rep_code
from ldpc.code_util import compute_code_parameters


def get_rep_code(length=5):
    H = rep_code(length).toarray().astype(int)
    print(f"\t\tRepetition code: [n, k, d] = {compute_code_parameters(H)}")

    code = hgp(h1=H, h2=H, compute_distance=True)
    code.name = f"Repetition({length}) Code HGP"
    print(f"\t\tHGP Code: [[{code.N}, {code.K}, {code.D}]]")
    return code, H