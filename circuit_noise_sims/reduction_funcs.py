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
    
    def plus(H1, H2):
        """
        Adds matrices H1 and H2 over F2.
        """
        H = H1 + H2
        H.data %= 2
        H.eliminate_zeros()
        return H
    
    def combinerowsF2(H, stab_indices):
        """ 
        Combines adjacent rows in `stab_indices` of H,
        as a new matrix Hnew.
        For example, if `stab_indices` = [3, 5, 7, 8]:
        - H[3] + H[5] is a row of Hnew
        - H[5] + H[7] is a row of Hnew
        - H[7] + H[8] is a row of Hnew
        - Other rows of Hnew are the same as H (H[1], H[2], H[4], H[6], etc.)
        
        Additionally, Hnew is returned as two matrices Hnew1 and Hnew2 such that
        Hnew = Hnew1 oplus Hnew2. They are defined as follows:
        
        Init: Hnew1 = H, Hnew2 = all zeros
        Combining H[i] and H[j] ==> Hnew1.removerow(i); Hnew2.appendrow(j)
        
        Example: Combining rows i = 2 and j = 3 of
        H = [ 0 0 0 0
              0 1 0 1
              1 0 1 0 ]
        yields:
        Hnew1 = [ 0 0 0 0      Hnew2 = [ 0 0 0 0 
                  0 1 0 1  ]             1 0 1 0 ]
        so that
        Hnew1 + Hnew2 = [ 0 0 0 0
                          1 1 1 1 ] = Hnew
        """
        # no combining needed
        if len(stab_indices) < 2:
            return H, sp.lil_matrix(H.shape, dtype=int)
    
        # init: Hnew1 = H, Hnew2 = all zeros
        Hnew1 = H.tolil(copy=True)
        Hnew2 = sp.lil_matrix(H.shape, dtype=int)

        # iterate and combine rows of adjacent elements in `stab_indices`
        for k in range(len(stab_indices) - 1):
            
            # row indices of H to combine
            i = stab_indices[k]
            j = stab_indices[k + 1]

            # Hnew[j] = Hnew1[j] oplus Hnew2[j] 
            rj_total = (Hnew1.getrow(j).toarray()[0] ^ Hnew2.getrow(j).toarray()[0]).astype(int) 
            
            # Hnew2[i] = Hnew2[i] oplus Hnew[j]
            ri_extra = Hnew2.getrow(i).toarray()[0].astype(int)
            Hnew2[i, :] = (ri_extra ^ rj_total).astype(int)

        # delete last stabilizer row
        last_row = stab_indices[-1]
        m, _ = H.shape
        keep_rows = [r for r in range(m) if r != last_row]

        Hnew1 = Hnew1[keep_rows, :]
        Hnew2 = Hnew2[keep_rows, :]
        return Hnew1, Hnew2
    
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
    Zgroups = [n**2 + (m*c1 + c2) for g in Zcombines for (c1, c2) in color_groups[g]]
    Hznew = code.hz.tolil(copy=True)
    Hznew1 = Hznew.copy()
    Hznew2 = sp.lil_matrix(code.hz.shape, dtype=int)
    for Zcolorgroup in Zcombines:
        for (c1, c2) in color_groups[Zcolorgroup]:
            # qubit coord (a, b) is index (n**2) + (ma + b)
            Hznew = plus(Hznew1, Hznew2)
            Hznew1, Hznew2 = combinerowsF2(Hznew, stabs_touching_qubit(Hznew, n**2 + m*c1 + c2))
            Hznew = plus(Hznew1, Hznew2)
    Xgroups = [n**2 + (m*c1 + c2) for g in Xcombines for (c1, c2) in color_groups[g]]
    Hxnew = code.hx.tolil(copy=True)
    Hxnew1 = Hxnew.copy()
    Hxnew2 = sp.lil_matrix(code.hz.shape, dtype=int)
    for Xcolorgroup in Xcombines:
        for (c1, c2) in color_groups[Xcolorgroup]:
            Hxnew = plus(Hxnew1, Hxnew2)
            Hxnew1, Hxnew2 = combinerowsF2(Hxnew, stabs_touching_qubit(Hxnew, n**2 + m*c1 + c2))
            Hxnew = plus(Hxnew1, Hxnew2)

### Cutting other support
    for Zcolorgroup in Zcombines:
        for (c1, c2) in color_groups[Zcolorgroup]:
            q = n**2 + (m*c1 + c2)
            #print(f"qubit coordinate = ({c1}, {c2})")
            #print(f"qubit index = {q}")
            # qubit idx at coord (a, b) is (n**2) + (ma + b)
            Hxnew[:, q] = 0

    for Xcolorgroup in Xcombines:
        for (c1, c2) in color_groups[Xcolorgroup]:
            q = n**2 + (m*c1 + c2)
            #print(f"qubit coordinate = ({c1}, {c2})")
            #print(f"qubit index = {q}")
            # qubit idx at coord (a, b) is (n**2) + (ma + b)
            Hznew[:, q] = 0

### Remove qubits that lost support
    def remove_supportless(H):
        H.eliminate_zeros()
        col_nnz = H.getnnz(axis=0) 
        keep_idx = np.where(col_nnz > 0)[0]
        return H[:, keep_idx]
    Hxnew = remove_supportless(Hxnew)
    Hznew = remove_supportless(Hznew)
    
### Reduced code
    newcode = css_code(hx=Hxnew, hz=Hznew)
    newcode.name = 'Transformed code'
    print(f"Reduced code: [[n', k', d']] = [[{newcode.N}, {newcode.K}, {code.D}]]")
    
    return Hxnew, Hznew