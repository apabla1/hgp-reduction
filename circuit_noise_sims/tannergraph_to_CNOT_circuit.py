import numpy as np
from edge_coloring import edge_color_bipartite
import stim
from networkx import relabel_nodes
from networkx.algorithms import bipartite

def generate_synd_circuit(H, checks, stab_type, p1, p2, seed):
    """
    Stim (X or Z) syndrome extraction circuit for given an edge-colored Tanner graph
    
    :param H: Hx or Hz
    :param checks: qubit indices for the m(X or Z) syndrome qubits 
    :param stab_type: T/F (T = X stabilizers; F = Z stabilizers)
    :param p1: single-qubit depolarizing probability
    :param p2: two-qubit depolarizing probability
    :param seed: randomizing the order of parallel CNOTs based on coloring
    """
    m, n = H.shape
    tanner_graph = bipartite.from_biadjacency_matrix(H)
    mapping = {i: checks[i] for i in range(m)}
    mapping.update({i:i-m for i in range(m,n+m)})
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
        c.append("DEPOLARIZE1", data_qbts, p1)

    # combine the two together

    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p1)
    return c

# Only tracks Z syndrome measurements
def generate_full_circuit(code, rounds, p1, p2, p_spam, seed):
    """
    Full Stim circuit of repeated syndrome extraction and a final measurement of data qubits
    
    :param code: CSS code with code.hx and code.hz
    :param rounds: # of rounds of syndrome extraction
    :param p1: single-qubit depolarizing probability
    :param p2: two-qubit depolarizing probability
    :param p_spam: syndrome qubit depoarizing probability
    :param seed: seed forwarded to generate_synd_circuit
    """
    mx, n = code.hx.shape
    mz = code.hz.shape[0]
    data_qubits = range(n)
    x_checks = range(n, n+mx)
    z_checks = range(n+mx, n+mx+mz)
    c = stim.Circuit()
    z_synd_circuit = generate_synd_circuit(code.hz, z_checks, 0, p1, p2, seed)
    x_synd_circuit = generate_synd_circuit(code.hx, x_checks, 1, p1, p2, seed)
    
    # ancilla initialization errors
    c.append("X_ERROR", z_checks, p_spam)
    c.append("X_ERROR", x_checks, p_spam)

    ### syndrome extraction rounds
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

def generate_full_circuit_split(Hx1, Hx2, Hz1, Hz2, rounds, p1, p2, p_spam, seed, measure_x=False):
    """
    Like generate_full_circuit, but performs two-stage syndrome extraction:
      - measure syndromes for H1 (stage 1) then H?2 (stage 2)
      - MR between stages (so ancillas reset)
      - effective syndrome is XOR(stage1, stage2) in post-processing.
    If measure_x=False, X checks are still extracted in two stages but NOT recorded.
    """
    mx, n = Hx1.shape
    mz = Hz1.shape[0]

    data_qubits = range(n)
    x_checks = range(n, n + mx)
    z_checks = range(n + mx, n + mx + mz)

    c = stim.Circuit()

    # ancilla initialization errors (once at start, like your original)
    c.append("X_ERROR", z_checks, p_spam)
    c.append("X_ERROR", x_checks, p_spam)

    # build the four subcircuits
    z1 = generate_synd_circuit(Hz1, z_checks, stab_type=0, p1=p1, p2=p2, seed=seed)
    z2 = generate_synd_circuit(Hz2, z_checks, stab_type=0, p1=p1, p2=p2, seed=seed + 1)

    x1 = generate_synd_circuit(Hx1, x_checks, stab_type=1, p1=p1, p2=p2, seed=seed + 2)
    x2 = generate_synd_circuit(Hx2, x_checks, stab_type=1, p1=p1, p2=p2, seed=seed + 3)

    c_se = stim.Circuit()
    for _ in range(rounds):
        # --- Z stage 1 ---
        c_se += z1
        c_se.append("X_ERROR", z_checks, p_spam)
        c_se.append("MR", z_checks)              # records + resets
        c_se.append("X_ERROR", z_checks, p_spam)

        # --- Z stage 2 ---
        c_se += z2
        c_se.append("X_ERROR", z_checks, p_spam)
        c_se.append("MR", z_checks)              # records + resets
        c_se.append("X_ERROR", z_checks, p_spam)

        # --- X stage 1 ---
        c_se += x1
        c_se.append("X_ERROR", x_checks, p_spam)
        if measure_x:
            c_se.append("MR", x_checks)          # records + resets
            c_se.append("X_ERROR", x_checks, p_spam)
        else:
            c_se.append("R", x_checks)           # reset only (no record)
            c_se.append("X_ERROR", x_checks, p_spam)

        # --- X stage 2 ---
        c_se += x2
        c_se.append("X_ERROR", x_checks, p_spam)
        if measure_x:
            c_se.append("MR", x_checks)
            c_se.append("X_ERROR", x_checks, p_spam)
        else:
            c_se.append("R", x_checks)
            c_se.append("X_ERROR", x_checks, p_spam)

    c += c_se

    # final data measurement
    c.append("X_ERROR", data_qubits, p_spam)
    c.append("MR", data_qubits)

    return c
