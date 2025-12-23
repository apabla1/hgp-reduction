import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from bposd.hgp import hgp
from bposd.css import css_code
from ldpc.code_util import compute_code_parameters, compute_exact_code_distance
from ldpc.mod2 import rank
from networkx.algorithms.bipartite import configuration_model, biadjacency_matrix
from matplotlib import rcParams

def get_reduced_random_code(n, d_v, d_c, min_dist, max_coloring):
    """    
    Return a random reduced HGP code.
    
    :param n: #bits
    :param d_v: how many checks each bit participates in
    :param d_c: how many bits each check involves
    :param min_dist: minimum distance of classical code 
    :param max_coloring: maximum number of color groups in classical code
    """

### Helpers to create the random classical code
    def get_check_adj_graph(H):
        A = (H @ H.T != 0).astype(int) # #checks x #checks; 1 if checks share a bit; 0 otherwise
        np.fill_diagonal(A, 0)
        G = nx.from_numpy_array(A, create_using=nx.MultiGraph())
        return G


    def get_check_coloring(H):
        G = get_check_adj_graph(H)
        color_dict = nx.greedy_color(G, strategy='independent_set')
        
        num_colors = max(list(color_dict.values())) + 1
        coloring = []
        for i in range(num_colors):
            coloring.append([key for key, value in color_dict.items() if value == i])
        return coloring

### Create random classical code
    tries = 10000
    coloring = []
    
    for i in range(tries):
        
        # number of checks
        m = int(n * d_v / d_c)
        
        # random bipartite graph that we model as the Tanner graph
        graph = configuration_model(
            d_v*np.ones(n,dtype=int), 
            d_c*np.ones(m,dtype=int), 
            create_using=nx.Graph())
        
        # create PCM from Tanner graph
        H = biadjacency_matrix(graph, row_order=np.arange(n)).T.toarray()
        
        # (1) *ensure the H is full rank, (2) make sure that the code distance is
        # >= `min_dist`, (3) make sure that the coloring is <= `max_coloring`
        d = compute_exact_code_distance(H)
        #print(f"distance: {d}")
        if d >= min_dist and rank(H) == H.shape[0]:
            coloring = get_check_coloring(H)
            #print(f"# colors: {len(coloring)}")
            if len(coloring) <= max_coloring:
                break
    print(f"Classical code: [n, k, d] = {compute_code_parameters(H)}")
    
### Create HGP code from two of the random classical code
    code = hgp(h1=H, h2=H, compute_distance=True)
    code.name = 'Random Code HGP'
    print(f"HGP Code: [[{code.N}, {code.K}, {code.D}]]")
    
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

    def add(H1, H2):
        """
        Adds two binary csr matrices over F2. 
        """
        H = (H1 + H2).tocsr()
        H.data %= 2
        H.eliminate_zeros()
        return H
    
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
### Hxnew1 = [ 0 1 0 0   <- row 0     +    Hxnew2 = [ 0 0 0 0   <- 0        =    Hxnew = [ 0 1 0 0   <- row 0 
###            0 0 1 0   <- row 1                     1 0 1 0   <- row 2                   1 0 0 0   <- row 1 + row 2
###            1 0 0 1   <- row 3                     0 0 0 0   <- 0                       1 0 0 1   <- row 3
###            1 0 1 0   <- row 2                     1 0 1 1   <- row 4                   0 0 0 1   <- row 2 + row 4
###            1 1 0 1 ] <- row 5                     0 0 0 0 ] <- 0                       1 1 0 1 ] <- row 5 
###                                                                                            ↑ support removed here
    
    def split_chain_repetition_style(H, chain):
        """
        Given H and chain [r0, r1, ..., rk] (row indices in H),
        return (H1, H2) with shape (m-1, n) such that H1 + H2 equals
        H with repetition-style row-combines along the chain.
        """
        chain = list(map(int, chain))
        if len(chain) < 2:
            return H.tocsr(), sp.csr_matrix(H.shape, dtype=int)

        H = H.tocsr()
        m, n = H.shape

        keep_rows = [r for r in range(m) if r != chain[-1]]
        old_to_new = {old: new for new, old in enumerate(keep_rows)}

        H1 = H[keep_rows, :].tocsr()
        H2 = sp.lil_matrix(H1.shape, dtype=int)

        for t in range(len(chain) - 1):
            i = chain[t]
            j = chain[t + 1]
            if i == chain[-1]:
                continue  # target row got removed
            H2[old_to_new[i], :] = H.getrow(j)

        return H1, H2.tocsr()

    # X
    Hxnew1 = code.hx.tocsr(copy=True)
    Hxnew2 = sp.csr_matrix(Hxnew1.shape, dtype=int)

    for Xcolorgroup in Xcombines:
        for (c1, c2) in color_groups[Xcolorgroup]:
            Hx_tot = add(Hxnew1, Hxnew2)
            chain = list(map(int, stabs_touching_qubit(Hx_tot, n**2 + m*c1 + c2)))
            Hxnew1, Hxnew2 = split_chain_repetition_style(Hx_tot, chain)

    # Z
    Hznew1 = code.hz.tocsr(copy=True)
    Hznew2 = sp.csr_matrix(Hznew1.shape, dtype=int)

    for Zcolorgroup in Zcombines:
        for (c1, c2) in color_groups[Zcolorgroup]:
            Hz_tot = add(Hznew1, Hznew2)

            chain = list(map(int, stabs_touching_qubit(Hz_tot, n**2 + m*c1 + c2)))
            Hznew1, Hznew2 = split_chain_repetition_style(Hz_tot, chain)   

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
    print(f"Reduced code: [[n', k', d']] = [[{newcode.N}, {newcode.K}, {code.D}]]")
    
    return Hxnew1, Hxnew2, Hznew1, Hznew2, newcode