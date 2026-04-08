import numpy as np
import scipy.sparse as sp
import warnings
import networkx as nx
from bposd.css import css_code


def add(H1, H2):
    """
    Adds two binary csr matrices over F2.
    """
    H = (H1 + H2).tocsr()
    H.data %= 2
    H.eliminate_zeros()
    return H


# suppress exact_code_distance warning
warnings.filterwarnings("ignore", category=UserWarning, module=r"ldpc\.code_util\.code_util")


### Helpers to create the random classical code

def get_check_adj_graph(H):
    A = (H @ H.T != 0).astype(int)  # #checks x #checks; 1 if checks share a bit; 0 otherwise
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



def get_reduced_code(code, H):
    """  
    Return a reduced HGP code given a symmetric HGP code made from two of the same parity-check matrix.

    The returned matrices satisfy
        Hxnew = Hxnew1 + Hxnew2 + Hxnew3,
        Hznew = Hznew1 + Hznew2 + Hznew3,
    where Hxnew/Hznew define the same reduced HGP code as before.

    For X checks:
      - Hxnew1 contains every non-combined check in full, and for combined
        checks it contains one entire bit-type column.
      - Hxnew2 contains the residual support (the check-type support) of the
        combined checks.
      - Hxnew3 contains the second bit-type column of each combined check.

    For Z checks, the analogous row-based partition is used. This enforces the split syndrome extraction schedule.

    :param code: HGP code
    :param H: the PCM used to create `code`
    """
    coloring = get_check_coloring(H)

    ### Create color groups of HGP check-type qubits that come from the coloring of the classical checks
    m, n = H.shape

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
    def stabs_touching_qubit(Hmat, q):
        """
        Returns the indices of the stabilizers in Hmat that have support on qubit at index q.
        """
        return Hmat.getcol(q).nonzero()[0]

    ### Optimizing size of color groups.
    def color_groups_to_bipartite_graph(color_groups_dict):
        """
        Build the bipartite graph used in the reduction.
        """
        k = int(round(np.sqrt(len(color_groups_dict))))
        assert k * k == len(color_groups_dict)
        G = nx.Graph()

        # 2k + k^2 bipartite nodes
        rowcol_nodes = [f"X{i}" for i in range(k)] + [f"Z{i}" for i in range(k)]
        color_nodes = [i for i in range(k**2)]
        G.add_nodes_from(rowcol_nodes, bipartite=0)
        G.add_nodes_from(color_nodes, bipartite=1)

        # X (row) edges
        for i in range(k):
            for j in range(k):
                color_idx = (i * k) + j
                w = len(color_groups_dict[color_idx])
                G.add_edge(f"X{i}", color_idx, weight=w)

        # Z (column) edges
        for i in range(k):
            for j in range(k):
                color_idx = (j * k) + i
                w = len(color_groups_dict[color_idx])
                G.add_edge(f"Z{i}", color_idx, weight=w)

        return G

    def max_weight_matching(G):
        """
        Find the maximum weight matching of G.
        """
        matching = nx.max_weight_matching(G, maxcardinality=True)
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
        Xcombines = [mate[f"X{i}"] for i in range(k)]
        Zcombines = [mate[f"Z{j}"] for j in range(k)]
        return Xcombines, Zcombines

    Xcombines, Zcombines = extract_combine_schedule_from_matching(
        max_weight_matching(color_groups_to_bipartite_graph(color_groups))
    )

    def split_chain_repetition_style_decomp(H1, H2, chain):
        """
        Given a decomposition Htot = H1 + H2 and a chain [r0, r1, ..., rk],
        perform repetition-style combines on the *total*:
            r0 <- r0 + r1
            r1 <- r1 + r2
            ...
            r(k-1) <- r(k-1) + rk
        and then remove row rk.

        This helper is only used internally to build the reduced total code.  A
        separate post-processing step below repartitions the final reduced rows
        into the three Algorithm-2 pieces.
        """
        chain = list(map(int, chain))
        if len(chain) < 2:
            return H1.tocsr(), H2.tocsr()

        H1 = H1.tocsr()
        H2 = H2.tocsr()
        m_rows, n_cols = H1.shape
        assert H2.shape == (m_rows, n_cols)

        r_last = chain[-1]
        keep_rows = [r for r in range(m_rows) if r != r_last]

        needed_second_rows = set(chain[1:])
        T = {}
        for r in needed_second_rows:
            T[r] = add(H1.getrow(r), H2.getrow(r))

        addmap = {}
        for t in range(len(chain) - 1):
            i = chain[t]
            j = chain[t + 1]
            if i == r_last:
                continue
            addmap.setdefault(i, []).append(j)

        H1_new = H1[keep_rows, :].tocsr()

        H2_rows = []
        for r in keep_rows:
            row = H2.getrow(r)
            for j in addmap.get(r, []):
                row = add(row, T[j])
            H2_rows.append(row)

        H2_new = sp.vstack(H2_rows, format="csr")
        return H1_new, H2_new

    # First build the reduced total checks exactly as before.
    Hxwork1 = code.hx.tocsr(copy=True)
    Hxwork2 = sp.csr_matrix(Hxwork1.shape, dtype=int)

    for Xcolorgroup in Xcombines:
        for (c1, c2) in color_groups[Xcolorgroup]:
            Hx_tot = add(Hxwork1, Hxwork2)
            chain = list(map(int, stabs_touching_qubit(Hx_tot, n**2 + m * c1 + c2)))
            Hxwork1, Hxwork2 = split_chain_repetition_style_decomp(Hxwork1, Hxwork2, chain)

    Hzwork1 = code.hz.tocsr(copy=True)
    Hzwork2 = sp.csr_matrix(Hzwork1.shape, dtype=int)

    for Zcolorgroup in Zcombines:
        for (c1, c2) in color_groups[Zcolorgroup]:
            Hz_tot = add(Hzwork1, Hzwork2)
            chain = list(map(int, stabs_touching_qubit(Hz_tot, n**2 + m * c1 + c2)))
            Hzwork1, Hzwork2 = split_chain_repetition_style_decomp(Hzwork1, Hzwork2, chain)

    ### Cutting other support
    Hxwork1 = Hxwork1.tolil(copy=True)
    Hxwork2 = Hxwork2.tolil(copy=True)
    Hzwork1 = Hzwork1.tolil(copy=True)
    Hzwork2 = Hzwork2.tolil(copy=True)
    for Zcolorgroup in Zcombines:
        for (c1, c2) in color_groups[Zcolorgroup]:
            q = n**2 + (m * c1 + c2)
            Hxwork1[:, q] = 0
            Hxwork2[:, q] = 0

    for Xcolorgroup in Xcombines:
        for (c1, c2) in color_groups[Xcolorgroup]:
            q = n**2 + (m * c1 + c2)
            Hzwork1[:, q] = 0
            Hzwork2[:, q] = 0
    Hxwork1 = Hxwork1.tocsr()
    Hxwork2 = Hxwork2.tocsr()
    Hzwork1 = Hzwork1.tocsr()
    Hzwork2 = Hzwork2.tocsr()

    ### Remove qubits that lost support
    def remove_supportless(Hx1, Hx2, Hz1, Hz2):
        keep = np.where((add(Hx1, Hx2).getnnz(axis=0) + add(Hz1, Hz2).getnnz(axis=0)) > 0)[0]
        return (
            Hx1[:, keep].tocsr(),
            Hx2[:, keep].tocsr(),
            Hz1[:, keep].tocsr(),
            Hz2[:, keep].tocsr(),
            keep,
        )

    Hxwork1, Hxwork2, Hzwork1, Hzwork2, kept_cols = remove_supportless(
        Hxwork1, Hxwork2, Hzwork1, Hzwork2
    )

    Hxnew = add(Hxwork1, Hxwork2)
    Hznew = add(Hzwork1, Hzwork2)

    def split_tripartition(Htot, kept_original_cols, axis):
        """
        Repartition the final reduced checks into the three pieces.

        axis='X': classify bit-type support by original bit-type columns.
        axis='Z': classify bit-type support by original bit-type rows.
        """
        Htot = Htot.tocsr()
        part1 = sp.lil_matrix(Htot.shape, dtype=int)
        part2 = sp.lil_matrix(Htot.shape, dtype=int)
        part3 = sp.lil_matrix(Htot.shape, dtype=int)

        for r in range(Htot.shape[0]):
            row = Htot.getrow(r)
            cols = row.indices
            if cols.size == 0:
                continue

            orig_cols = kept_original_cols[cols]
            orig_bit_cols = orig_cols[orig_cols < n**2]

            if axis == 'X':
                bit_groups = sorted({int(q % n) for q in orig_bit_cols})
            elif axis == 'Z':
                bit_groups = sorted({int(q // n) for q in orig_bit_cols})
            else:
                raise ValueError("axis must be 'X' or 'Z'")

            # Non-combined checks stay entirely in part 1
            if len(bit_groups) <= 1:
                part1.rows[r] = cols.tolist()
                part1.data[r] = [1] * len(cols)
                continue

            if len(bit_groups) != 2:
                raise ValueError(
                    f"Reduced {axis}-check row {r} touches {len(bit_groups)} bit-type "
                    f"{'columns' if axis == 'X' else 'rows'}; expected at most 2."
                )

            first_group, second_group = bit_groups
            p1_cols = []
            p2_cols = []
            p3_cols = []
            for local_col, orig_col in zip(cols, orig_cols):
                if orig_col >= n**2:
                    p2_cols.append(int(local_col))
                else:
                    coord = int(orig_col % n) if axis == 'X' else int(orig_col // n)
                    if coord == first_group:
                        p1_cols.append(int(local_col))
                    elif coord == second_group:
                        p3_cols.append(int(local_col))
                    else:
                        raise ValueError(
                            f"Unexpected bit-type coordinate {coord} in reduced {axis}-check row {r}."
                        )

            part1.rows[r] = p1_cols
            part1.data[r] = [1] * len(p1_cols)
            part2.rows[r] = p2_cols
            part2.data[r] = [1] * len(p2_cols)
            part3.rows[r] = p3_cols
            part3.data[r] = [1] * len(p3_cols)

        return part1.tocsr(), part2.tocsr(), part3.tocsr()

    Hxnew1, Hxnew2, Hxnew3 = split_tripartition(Hxnew, kept_cols, axis='X')
    Hznew1, Hznew2, Hznew3 = split_tripartition(Hznew, kept_cols, axis='Z')

    ### Reduced code (unchanged total checks)
    newcode = css_code(hx=Hxnew, hz=Hznew)
    newcode.name = 'Transformed code'
    print(f"\tReduced code: [[n', k', d']] = [[{newcode.N}, {newcode.K}, {code.D}]]")

    return (
        Hxnew1,
        Hxnew2,
        Hxnew3,
        Hznew1,
        Hznew2,
        Hznew3,
        newcode,
        newcode.N,
        newcode.K,
        code.D,
    )
