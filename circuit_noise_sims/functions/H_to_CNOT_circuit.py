import numpy as np
import stim
from networkx import relabel_nodes
from networkx.algorithms import bipartite
from functions.edge_coloring import edge_color_bipartite


def generate_synd_circuit(H, checks, stab_type, p1, p2, seed):
    """
    Stim (X or Z) syndrome extraction circuit given a PCM H.
    Utilizes edge coloring of the Tanner graph to enforce order of parallel CNOTs.
    (An edge in the Tanner graph corresponds to a CNOT gate; edges of the same
    color can be operated in parallel.)

    :param H: Hx or Hz
    :param checks: qubit indices for the m(X or Z) syndrome qubits
    :param stab_type: T/F (T = X stabilizers; F = Z stabilizers)
    :param p1: single-qubit gate error probability
    :param p2: two-qubit gate error probability
    :param seed: randomizing the order of parallel CNOTs based on coloring
    """
    m, n = H.shape
    tanner_graph = bipartite.from_biadjacency_matrix(H)
    mapping = {i: checks[i] for i in range(m)}
    mapping.update({i: i - m for i in range(m, n + m)})
    tanner_graph = relabel_nodes(tanner_graph, mapping)
    coloring = edge_color_bipartite(tanner_graph)
    if seed != 0:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(coloring, axis=0)

    c = stim.Circuit()

    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p1)

    for r in coloring:
        data_qbts = set(np.arange(H.shape[1]))
        for g in r:
            data_qbts.remove(g[0])
            targets = g[::-1] if stab_type else g
            c.append("CX", targets)
            c.append("DEPOLARIZE2", targets, p2)
        # c.append("DEPOLARIZE1", data_qbts, p1)

    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p1)
    return c



def generate_full_circuit(code, rounds, noise_pars, seed):
    """
    Non-ordered syndrome extraction.

    :param code: code to generate circuit for
    :param rounds: rounds of measurement
    :param noise_pars: tuple of (p1, p2, p_spam)
    :param seed: seed forwarded to generate_synd_circuit
    """
    p1, p2, p_spam = noise_pars
    mx, n = code.hx.shape
    mz = code.hz.shape[0]
    data_qubits = range(n)
    x_checks = range(n, n + mx)
    z_checks = range(n + mx, n + mx + mz)
    c = stim.Circuit()
    z_synd_circuit = generate_synd_circuit(code.hz, z_checks, 0, p1, p2, seed)
    x_synd_circuit = generate_synd_circuit(code.hx, x_checks, 1, p1, p2, seed)
    # ancilla initialization errors
    c.append("X_ERROR", z_checks, p_spam)
    c.append("X_ERROR", x_checks, p_spam)

    # syndrome extraction rounds
    c_se = stim.Circuit()
    # Z syndrome measurement
    c_se += z_synd_circuit
    c_se.append("X_ERROR", z_checks, p_spam)
    c_se.append("MR", z_checks)
    c_se.append("X_ERROR", z_checks, p_spam)
    # X syndrome measurement
    c_se += x_synd_circuit
    c_se.append("R", x_checks)
    c_se.append("X_ERROR", x_checks, p_spam)

    c += c_se * rounds

    # Final transversal measurement
    c.append("X_ERROR", data_qubits, p_spam)
    c.append("MR", data_qubits)
    return c



def generate_full_circuit_split(Hx1, Hx2, Hx3, Hz1, Hz2, Hz3, rounds, noise_pars, seed):
    """
    Order-enforced syndrome extraction implementing a three-way split

      repeat `rounds` times {
        (1) project Z syndromes for Hz1
        (2) project Z syndromes for Hz2
        (3) project Z syndromes for Hz3
        (4) measure Z syndromes and reset ancillas
        (5) project X syndromes for Hx1
        (6) project X syndromes for Hx2
        (7) project X syndromes for Hx3
        (8) measure X syndromes and reset ancillas
      }
      (9) measure data qubits

    The intended interpretation is:
      - part 1: uncombined checks in full + the first bit-type row/column of each combined check
      - part 2: the residual check-type support of combined checks
      - part 3: the second bit-type row/column of each combined check

    :param Hx1: part 1 of Hx
    :param Hx2: part 2 of Hx
    :param Hx3: part 3 of Hx
    :param Hz1: part 1 of Hz
    :param Hz2: part 2 of Hz
    :param Hz3: part 3 of Hz
    :param rounds: rounds of measurement
    :param noise_pars: tuple of (p1, p2, p_spam)
    :param seed: seed forwarded to generate_synd_circuit
    """
    p1, p2, p_spam = noise_pars
    # n equal
    assert Hx1.shape[1] == Hx2.shape[1] == Hx3.shape[1] == Hz1.shape[1] == Hz2.shape[1] == Hz3.shape[1]

    # mx, mz equal
    assert Hx1.shape[0] == Hx2.shape[0] == Hx3.shape[0]
    assert Hz1.shape[0] == Hz2.shape[0] == Hz3.shape[0]

    n = Hx1.shape[1]
    mx = Hx1.shape[0]
    mz = Hz1.shape[0]

    data_qubits = range(n)
    x_synd_qubits = range(n, n + mx)
    z_synd_qubits = range(n + mx, n + mx + mz)

    # entire circuit
    c = stim.Circuit()

    # ancilla initialization errors
    c.append("X_ERROR", z_synd_qubits, p_spam)
    c.append("X_ERROR", x_synd_qubits, p_spam)

    # CNOT syndrome extraction circuit for the three-way split schedule
    c_se = stim.Circuit()

    # Z syndrome extraction: part 1, then residual support, then part 3.
    c_se += generate_synd_circuit(Hz1, z_synd_qubits, stab_type=0, p1=p1, p2=p2, seed=seed + 0)
    c_se += generate_synd_circuit(Hz2, z_synd_qubits, stab_type=0, p1=p1, p2=p2, seed=seed + 1)
    c_se += generate_synd_circuit(Hz3, z_synd_qubits, stab_type=0, p1=p1, p2=p2, seed=seed + 2)

    # measure Z syndromes and reset ancillas
    c_se.append("X_ERROR", z_synd_qubits, p_spam)
    c_se.append("MR", z_synd_qubits)
    c_se.append("X_ERROR", z_synd_qubits, p_spam)

    # X syndrome extraction: part 1, then residual support, then part 3.
    c_se += generate_synd_circuit(Hx1, x_synd_qubits, stab_type=1, p1=p1, p2=p2, seed=seed + 3)
    c_se += generate_synd_circuit(Hx2, x_synd_qubits, stab_type=1, p1=p1, p2=p2, seed=seed + 4)
    c_se += generate_synd_circuit(Hx3, x_synd_qubits, stab_type=1, p1=p1, p2=p2, seed=seed + 5)

    # measure X syndromes and reset ancillas
    c_se.append("X_ERROR", x_synd_qubits, p_spam)
    c_se.append("MR", x_synd_qubits)
    c_se.append("X_ERROR", x_synd_qubits, p_spam)

    c += c_se * rounds

    # final data measurement
    c.append("X_ERROR", data_qubits, p_spam)
    c.append("MR", data_qubits)

    return c
