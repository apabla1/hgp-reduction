"""Canonical logical operators for the left sector of an HGP code."""

import numpy as np
import scipy.sparse as sp
from ldpc import mod2


def _canonical_classical_pair(check_matrix):
    """Return generator/complement matrices ``G, E`` with ``E G.T = I``."""

    generator = mod2.nullspace(sp.csr_matrix(check_matrix)).toarray().astype(np.uint8)
    generator, rank, _, pivot_columns = mod2.row_echelon(generator, full=True)
    generator = np.asarray(generator[:rank], dtype=np.uint8) % 2
    pivot_columns = np.asarray(pivot_columns[:rank], dtype=int)

    complement = np.zeros_like(generator, dtype=np.uint8)
    complement[np.arange(rank), pivot_columns] = 1
    if not np.array_equal(
        (complement @ generator.T) % 2,
        np.eye(rank, dtype=np.uint8),
    ):
        raise ValueError("Failed to construct a dual classical-code basis.")

    return sp.csr_matrix(generator), sp.csr_matrix(complement)


def canonical_hgp_left_logicals(h1, h2=None):
    """Return the canonical left-sector ``(Lx, Lz)`` logical matrices.

    The qubit order is the one used by :class:`bposd.hgp.hgp`: the
    ``n1*n2`` left-sector qubits followed by the ``m1*m2`` right-sector
    qubits.
    """

    h1 = sp.csr_matrix(h1, dtype=np.uint8)
    h2 = h1 if h2 is None else sp.csr_matrix(h2, dtype=np.uint8)
    m1, _ = h1.shape
    m2, _ = h2.shape

    g1, e1 = _canonical_classical_pair(h1)
    g2, e2 = _canonical_classical_pair(h2)
    num_left_logicals = g1.shape[0] * g2.shape[0]
    zero_right = sp.csr_matrix(
        (num_left_logicals, m1 * m2),
        dtype=np.uint8,
    )

    lx = sp.hstack((sp.kron(e1, g2, format="csr"), zero_right), format="csr")
    lz = sp.hstack((sp.kron(g1, e2, format="csr"), zero_right), format="csr")
    return lx.astype(np.uint8), lz.astype(np.uint8)


def canonical_z_logicals(code):
    """Return all canonical Z observables for a simulated HGP code."""

    if hasattr(code, "canonical_lz"):
        logicals = sp.csr_matrix(code.canonical_lz, dtype=np.uint8)
    elif hasattr(code, "h1") and hasattr(code, "h2"):
        _, logicals = canonical_hgp_left_logicals(code.h1, code.h2)
    else:
        raise ValueError(
            "The code does not retain the HGP inputs or a restricted canonical basis."
        )

    if logicals.shape != (int(code.K), int(code.N)):
        raise ValueError(
            "The simulation requires a complete left-sector logical basis; "
            f"got {logicals.shape}, expected {(int(code.K), int(code.N))}."
        )
    return logicals
