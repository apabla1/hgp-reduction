import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import networkx as nx
import math
import warnings
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from bposd.hgp import hgp
from bposd.css import css_code
from ldpc.code_util import compute_code_parameters, compute_exact_code_distance
from ldpc.mod2 import rank
from networkx.algorithms.bipartite import configuration_model, biadjacency_matrix
from matplotlib import rcParams
from codes.random_codes import get_check_adj_graph, get_check_coloring
from functions.matrix_funcs import add

# suppress exact_code_distance warning
warnings.filterwarnings("ignore", category=UserWarning, module=r"ldpc\.code_util\.code_util")

def get_reduced_code(code, H):
    """    
    Return a reduced HGP code given a HGP code made from two of the same parity-check matrix.
    
    :param code: HGP code
    :param H: the PCM used to create `code`
    """
    coloring = get_check_coloring(H) # coloring of the code
    
### Create color groups of HGP check-type qubits that come from the coloring of the classical checks
    m, n = H.shape
    
    # bit- and check-type qubit indices
    bit_type = np.arange(0, n*n)
    check_type = np.arange(n*n, n*n + m*m)
    
    # `color_groups` holds the qubit coordinates of the color groups, starting at
    # (0, 0) from the top-left and going left-to-right/top-to-bottom
    color_groups = {}
    for group1 in coloring:
        for group2 in coloring:
            length = len(color_groups)
            color_groups[length] = []
            for c1 in group1:
                for c2 in group2:
                    color_groups[length].append((c1, c2))
                    
### ---------------- Reduction ---------------- ###
    def stabs_touching_qubit(H, q):
        """
        Returns the indices of the stabilizers in H that have support on qubit at index q.
        """
        return H.getcol(q).nonzero()[0]
    
### Optimizing size of color groups.
    def color_groups_to_bipartite_graph(color_groups):
        """
        Build the bipartite graph used in the reduction.
        """
        k = int(round(np.sqrt(len(color_groups))))
        assert k * k == len(color_groups)
        G = nx.Graph()

        # 2k + k^2 bipartite nodes
        rowcol_nodes = [f"X{i}" for i in range(k)] + [f"Z{i}" for i in range(k)]
        color_nodes = [i for i in range(k**2)]
        G.add_nodes_from(rowcol_nodes, bipartite=0)
        G.add_nodes_from(color_nodes, bipartite=1)

        # X (row) edges
        for i in range(k): # for each X-node Xi
            for j in range(k): # will connect to k color nodes
                color_idx = (i * k) + j
                w = len(color_groups[color_idx])
                G.add_edge(f"X{i}", color_idx, weight=w)

        # Z (column) edges
        for i in range(k): # for each Z-node Zi
            for j in range(k): # will connect to k color nodes
                color_idx = (j * k) + i
                w = len(color_groups[color_idx])
                G.add_edge(f"Z{i}", color_idx, weight=w)

        return G

    def max_weight_matching(G):
        """
        Find the maximum weight matching of G.
        """
        matching = nx.max_weight_matching(G, maxcardinality=True)
        # Format: e.g., {("X0", 7), ("Z1", 3), ...} (order not guaranteed)
        return matching

    def extract_combine_schedule_from_matching(matching):
        """
        Extract the schedule of which stabilizers to combine given a matching.
        """
        k = len(matching) // 2
        mate = {}
        for u, v in matching:
            mate[u] = v
            mate[v] = u
        # {("X0", 7), ("Z1", 3), ...} -> {X0: 7, Z1: 3, ...}
        Xcombines = [mate[f"X{i}"] for i in range(k)]
        Zcombines = [mate[f"Z{j}"] for j in range(k)]
        return Xcombines, Zcombines

    Xcombines, Zcombines = extract_combine_schedule_from_matching(max_weight_matching(color_groups_to_bipartite_graph(color_groups)))
    
### **Combining stabilizers** 
### Hxnew1 will contain the original rows of stabilizers, and Hxnew2 will contain the combined rows
### Together, Hxnew = Hxnew1 + Hxnew2
### Example: Xcombines = [1, 2, 4],  Hx = [ 0 1 0 0   <- row 0
###                                         0 0 1 0   <- row 1
###                                         1 0 1 0   <- row 2
###                                         1 0 0 1   <- row 3
###                                         1 0 1 1   <- row 4
###                                         1 1 0 1 ] <- row 5 
### =====================================================================================================================
### Hxnew1 = [ 0 1 0 0   <- row 0     +     Hxnew2 = [ 0 0 0 0   <- 0        =     Hxnew = [ 0 1 0 0   <- row 0 
###            0 0 1 0   <- row 1                      1 0 1 0   <- row 2                    1 0 0 0   <- row 1 + row 2
###            1 0 0 1   <- row 3                      0 0 0 0   <- 0                        1 0 0 1   <- row 3
###            1 0 1 0   <- row 2                      1 0 1 1   <- row 4                    0 0 0 1   <- row 2 + row 4
###            1 1 0 1 ] <- row 5                      0 0 0 0 ] <- 0                        1 1 0 1 ] <- row 5 
###                                                                                              ↑ support removed here
    def split_chain_repetition_style(H1, H2, chain):
        """
        Given a decomposition Htot = H1 + H2 and a chain [r0, r1, ..., rk],
        perform repetition-style combines on the total:
            r0 <- r0 + r1
            r1 <- r1 + r2
            ...
            r(k-1) <- r(k-1) + rk
        and then remove row rk.
        """
        chain = list(map(int, chain))
        if len(chain) < 2:
            return H1.tocsr(), H2.tocsr()
    
        H1 = H1.tocsr()
        H2 = H2.tocsr()
        m, n = H1.shape
        assert H2.shape == (m, n)
    
        r_last = chain[-1]
        keep_rows = [r for r in range(m) if r != r_last]
    
        needed_second_rows = set(chain[1:])  # all j's in (i,j)
        T = {}
        for r in needed_second_rows:
            T[r] = add(H1.getrow(r), H2.getrow(r))
    
        # for each i, collect which j rows should be XOR'ed into H2_i
        addmap = {}
        for t in range(len(chain) - 1):
            i = chain[t]
            j = chain[t + 1]
            if i == r_last:
                continue 
            addmap.setdefault(i, []).append(j)
    
        # H1_new only contains deleted rows from H1
        H1_new = H1[keep_rows, :].tocsr()
    
        # H2_new is built row-by-row with XORs
        H2_rows = []
        for r in keep_rows:
            row = H2.getrow(r)
            for j in addmap.get(r, []):
                row = add(row, T[j])
            H2_rows.append(row)
    
        H2_new = sp.vstack(H2_rows, format="csr")
        return H1_new, H2_new

    # X
    Hxnew1 = code.hx.tocsr(copy=True)
    Hxnew2 = sp.csr_matrix(Hxnew1.shape, dtype=int)
    for Xcolorgroup in Xcombines:
        for (c1, c2) in color_groups[Xcolorgroup]:
            Hx_tot = add(Hxnew1, Hxnew2)
            chain = list(map(int, stabs_touching_qubit(Hx_tot, n**2 + m*c1 + c2)))
            Hxnew1, Hxnew2 = split_chain_repetition_style(Hxnew1, Hxnew2, chain)
    
    # Z
    Hznew1 = code.hz.tocsr(copy=True)
    Hznew2 = sp.csr_matrix(Hznew1.shape, dtype=int)
    for Zcolorgroup in Zcombines:
        for (c1, c2) in color_groups[Zcolorgroup]:
            Hz_tot = add(Hznew1, Hznew2)
            chain = list(map(int, stabs_touching_qubit(Hz_tot, n**2 + m*c1 + c2)))
            Hznew1, Hznew2 = split_chain_repetition_style(Hznew1, Hznew2, chain)

### Cutting other support
    Hxnew1 = Hxnew1.tolil(copy=True)
    Hxnew2 = Hxnew2.tolil(copy=True)
    Hznew1 = Hznew1.tolil(copy=True)
    Hznew2 = Hznew2.tolil(copy=True)
    for Zcolorgroup in Zcombines:
        for (c1, c2) in color_groups[Zcolorgroup]:
            q = n**2 + (m*c1 + c2)
            Hxnew1[:, q] = 0
            Hxnew2[:, q] = 0

    for Xcolorgroup in Xcombines:
        for (c1, c2) in color_groups[Xcolorgroup]:
            q = n**2 + (m*c1 + c2)
            Hznew1[:, q] = 0
            Hznew2[:, q] = 0
    Hxnew1 = Hxnew1.tocsr()
    Hxnew2 = Hxnew2.tocsr()
    Hznew1 = Hznew1.tocsr()
    Hznew2 = Hznew2.tocsr()

### Remove qubits that lost support
    def remove_supportless(Hx1, Hx2, Hz1, Hz2):
        keep = np.where((add(Hx1, Hx2).getnnz(axis=0) + add(Hz1, Hz2).getnnz(axis=0)) > 0)[0]
        return (Hx1[:, keep].tocsr(), Hx2[:, keep].tocsr(),
                Hz1[:, keep].tocsr(), Hz2[:, keep].tocsr())
    Hxnew1, Hxnew2, Hznew1, Hznew2 = remove_supportless(Hxnew1, Hxnew2, Hznew1, Hznew2)

    
### Reduced code
    Hxnew = add(Hxnew1, Hxnew2)
    Hznew = add(Hznew1, Hznew2)
    newcode = css_code(hx=Hxnew, hz=Hznew)
    newcode.name = 'Transformed code'
    print(f"\t\tReduced code: [[n', k', d']] = [[{newcode.N}, {newcode.K}, {code.D}]]")
    
    return Hxnew1, Hxnew2, Hznew1, Hznew2, newcode, newcode.N, newcode.K, code.D