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
    x_synd_qubits = range(n, n+mx)
    z_synd_qubits = range(n+mx, n+mx+mz)
    c = stim.Circuit()
    z_synd_circuit = generate_synd_circuit(code.hz, z_synd_qubits, 0, p1, p2, seed)
    x_synd_circuit = generate_synd_circuit(code.hx, x_synd_qubits, 1, p1, p2, seed)
    
    # ancilla initialization errors
    c.append("X_ERROR", z_synd_qubits, p_spam)
    c.append("X_ERROR", x_synd_qubits, p_spam)

    ### syndrome extraction rounds
    c_se = stim.Circuit()
    # Z syndrome measurement
    c_se += z_synd_circuit
    c_se.append("X_ERROR", z_synd_qubits, p_spam)
    c_se.append("MR", z_synd_qubits)
    c_se.append("X_ERROR", z_synd_qubits, p_spam)
    # X syndrome measurement
    c_se += x_synd_circuit
    c_se.append("R", x_synd_qubits)
    c_se.append("X_ERROR", x_synd_qubits, p_spam)

    c += c_se * rounds

    # Final transversal measurement
    c.append("X_ERROR", data_qubits, p_spam)
    c.append("MR", data_qubits)
    return c

def generate_full_circuit_split(Hx1, Hx2, Hz1, Hz2, rounds, p1, p2, p_spam, seed):
    """
    Order-enforced syndrome extraction
    
      repeat `rounds` times { 
        (1) project Z syndromes for Hz1
        (2) project Z syndromes for Hz2
        (3) measure Z syndromes and reset ancillas 
        (4) project X syndromes for Hx1
        (5) project X syndromes for Hx2
        (6) measure X syndromes and reset ancillas
      }
      (7) measure data qubits
      
    p1 : single-qubit depolarizing probability
    p2 : two-qubit depolarizing probability
    p_spam : syndrome qubit depolarizing probability
    seed : seed forwarded to generate_synd_circuit
    """
    # n equal
    assert Hx1.shape[1] == Hx2.shape[1]
    assert Hx2.shape[1] == Hz1.shape[1]
    assert Hz1.shape[1] == Hz2.shape[1]
    
    # mx, mz equal
    assert Hx1.shape[0] == Hx2.shape[0]
    assert Hz1.shape[0] == Hz2.shape[0]
                                     
    n = Hx1.shape[1]
    mx = Hx1.shape[0]
    mz = Hz1.shape[0]

    data_qubits = range(n) # [0, n-1]
    x_synd_qubits = range(n, n + mx) # [n, n+mx- 1]
    z_synd_qubits = range(n + mx, n + mx + mz) # [n+mx, n+mx+mz-1]

    # entire circuit
    c = stim.Circuit()

    # ancilla initialization errors
    c.append("X_ERROR", z_synd_qubits, p_spam)
    c.append("X_ERROR", x_synd_qubits, p_spam)

    # CNOT syndrome extraction circuit for (1)-(6)
    c_se = stim.Circuit()
        
    # (1) project Z syndromes for Hz1
    c_se += generate_synd_circuit(Hz1, z_synd_qubits, stab_type=0, p1=p1, p2=p2, seed=seed+0)

    # (2) project Z syndromes for Hz2
    c_se += generate_synd_circuit(Hz2, z_synd_qubits, stab_type=0, p1=p1, p2=p2, seed=seed+1)

    # (3) measure Z syndromes and reset ancillas
    c_se.append("X_ERROR", z_synd_qubits, p_spam)   # ancilla error before measurement
    c_se.append("MR", z_synd_qubits)                # measure Z syndrome qubits + reset to |0>
    c_se.append("X_ERROR", z_synd_qubits, p_spam)   # ancilla initialization error after resetting
    
    # (4) project X syndromes for Hx1
    c_se += generate_synd_circuit(Hx1, x_synd_qubits, stab_type=1, p1=p1, p2=p2, seed=seed+2)
    
    # (5) project X syndromes for Hx2
    c_se += generate_synd_circuit(Hx2, x_synd_qubits, stab_type=1, p1=p1, p2=p2, seed=seed+3)
    
    # (6) measure X syndromes and reset ancillas
    c_se.append("R", x_synd_qubits)                # reset X syndrome q

    # repeat (1)-(6) `rounds` times
    c += c_se * rounds 

    # final data measurement
    c.append("X_ERROR", data_qubits, p_spam)
    c.append("MR", data_qubits)

    return c
