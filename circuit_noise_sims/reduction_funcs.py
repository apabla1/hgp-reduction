import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix, identity, diags
import scipy.sparse as sp
from scipy.io import mmwrite
from bposd.hgp import hgp
from bposd.css import css_code
from ldpc.code_util import construct_generator_matrix, compute_code_parameters, estimate_code_distance, compute_exact_code_distance
from ldpc.mod2 import rank
import networkx as nx
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
    
    def get_bit_adj_graph(H):
        """
        Returns the bit-adjacency graph of H, where edges 
        between bits exist iff bits share a common check.
        """
        A = (H @ H.T != 0).astype(int) # #checks x #checks; 1 if checks share a bit; 0 otherwise
        np.fill_diagonal(A, 0)
        G = nx.from_numpy_array(A, create_using=nx.Graph())
        return G


    def get_check_coloring(H):
        """
        Colors the nodes of the bit-adjacency graph, which is 
        equivalent to coloring the checks of H based on which 
        bits share mutual support.
        """
        G = get_bit_adj_graph(H)
        color_dict = nx.greedy_color(G, strategy='independent_set')
        
        num_colors = max(list(color_dict.values())) + 1
        coloring = []
        for i in range(num_colors):
            coloring.append([key for key, value in color_dict.items() if value == i])
        return coloring

### Create random classical code
    tries = 10000
    coloring = []
    H = None
    
    # `tries` chances to get the classical code compliant with `min_dist` and `max_coloring`
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
    code.name = 'HGP Random'
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
    
### Optimizing size of color groups. Since we only remove check-type qubits in select color 
### groups, we should make the size of those color groups the largest
    Zcombines = [i*len(coloring) + i for i in range(len(coloring))]
    Xcombines = [i*len(coloring) + (i+1 if i%2 == 0 else i-1) for i in range(len(coloring) - 1)]

    old_groups = color_groups.copy()
    all_indices = list(color_groups.keys())

    # indices where we want the largest groups to live
    targets = list(dict.fromkeys(Xcombines + Zcombines)) 
    num_targets = len(targets)

    # sort group indices by group size
    sorted_indices = sorted(all_indices, key=lambda idx: len(color_groups[idx]), reverse=True)

    top = sorted_indices[:num_targets]
    rest = sorted_indices[num_targets:]
    new_color_groups = {}

    # send biggest groups to target indices 
    for src_idx, target_idx in zip(top, targets):
        new_color_groups[target_idx] = color_groups[src_idx]

    # send remaining groups to the non-target indices
    non_targets = [idx for idx in all_indices if idx not in targets]
    for src_idx, target_idx in zip(rest, non_targets):
        new_color_groups[target_idx] = color_groups[src_idx]

    color_groups = new_color_groups
       
### Combining stabilizers 
    Hznew1 = code.hz.tocsr(copy=True)
    Hznew2 = sp.csr_matrix(Hznew1.shape, dtype=int)
    # Z
    for Zcolorgroup in Zcombines:
        for (c1, c2) in color_groups[Zcolorgroup]:
            Hz_tot = add(Hznew1, Hznew2)
            q = n**2 + m*c1 + c2
            combinedstabs = list(map(int, stabs_touching_qubit(Hz_tot, q)))
            if len(combinedstabs) < 2:
                continue
            last = combinedstabs[-1]
            keep_rows = [r for r in range(Hz_tot.shape[0]) if r != last]
            old_to_new = {old: new for new, old in enumerate(keep_rows)}
            Hznew1 = Hz_tot[keep_rows, :].tocsr()
            Hznew2_lil = sp.lil_matrix(Hznew1.shape, dtype=int)
            for t in range(len(combinedstabs) - 1):
                i = combinedstabs[t]
                j = combinedstabs[t + 1]
                Hznew2_lil[old_to_new[i], :] = Hz_tot.getrow(j)
            Hznew2 = Hznew2_lil.tocsr()
    # X
    Hxnew1 = code.hx.tocsr(copy=True)
    Hxnew2 = sp.csr_matrix(Hxnew1.shape, dtype=int)
    for Xcolorgroup in Xcombines:
        for (c1, c2) in color_groups[Xcolorgroup]:
            Hx_tot = add(Hxnew1, Hxnew2)
            q = n**2 + m*c1 + c2
            combinedstabs = list(map(int, stabs_touching_qubit(Hx_tot, q)))
            if len(combinedstabs) < 2:
                continue
            last = combinedstabs[-1]
            keep_rows = [r for r in range(Hx_tot.shape[0]) if r != last]
            old_to_new = {old: new for new, old in enumerate(keep_rows)}
            Hxnew1 = Hx_tot[keep_rows, :].tocsr()
            Hxnew2_lil = sp.lil_matrix(Hxnew1.shape, dtype=int)
            for t in range(len(combinedstabs) - 1):
                i = combinedstabs[t]
                j = combinedstabs[t + 1]
                Hxnew2_lil[old_to_new[i], :] = Hx_tot.getrow(j)
            Hxnew2 = Hxnew2_lil.tocsr()

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
    def remove_supportless_split(H1, H2):
        Htot = add(H1, H2)
        col_nnz = Htot.getnnz(axis=0)
        keep_idx = np.where(col_nnz > 0)[0]
        return H1[:, keep_idx].tocsr(), H2[:, keep_idx].tocsr()
    Hxnew1, Hxnew2 = remove_supportless_split(Hxnew1, Hxnew2)
    Hznew1, Hznew2 = remove_supportless_split(Hznew1, Hznew2)
    
### Reduced code
    Hxnew = add(Hxnew1, Hxnew2)
    Hznew = add(Hznew1, Hznew2)
    newcode = css_code(hx=Hxnew, hz=Hznew)
    newcode.name = 'Transformed code'
    print(f"Reduced code: [[n', k', d']] = [[{newcode.N}, {newcode.K}, {code.D}]]")
    
    return Hxnew1, Hxnew2, Hznew1, Hznew2, newcode