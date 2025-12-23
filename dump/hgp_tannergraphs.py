import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def XZ_ctype_tannergraph(choice, hx, hz, ctype_start_idx, ctypes, draw=False):
    """
    Print the (X or  Z) Tanner graph for check-type qubits of an HGP code.

    :param choice: "X" or "Z" 
    :param hx: Hx
    :param hz: Hz
    :param ctype_start_idx: Qubit index of first check-type qubit (usually n**2)
    :param ctypes: list of check-type qubits that we actually draw edges for (invalid indices are not drawn)
    :param draw: T/F to draw the graph before returning (default F)
    """
    if choice not in ["X", "Z"]:
        return
    if hx.shape[1] != hz.shape[1]:
        return
    
    N = hx.shape[1]
    mx = hx.shape[0]
    mz = hz.shape[0]
    
    ctype_cols = np.arange(ctype_start_idx, N)
    ctype_set = set(ctype_cols.tolist())
    
    G = nx.Graph()
    q_nodes = list(ctype_cols)
    G.add_nodes_from(q_nodes,  bipartite=0, kind='qubit')
        
    if choice == "X":
        x_nodes = list(range(N, N + mx))
        G.add_nodes_from(x_nodes,  bipartite=1, kind='X')
        
        # hx edges
        coords = hx.tocoo()
        for r, c, v in zip(coords.row, coords.col, coords.data): 
            if v % 2 != 0 and c in ctype_set and c in ctypes:
                G.add_edge(N + r, c)
        
    elif choice == "Z":
        z_nodes = list(range(N + mx, N + mx + mz))
        G.add_nodes_from(z_nodes,  bipartite=1, kind='Z')
        
        # hz edges
        coords = hz.tocoo()
        for r, c, v in zip(coords.row, coords.col, coords.data):
            if v % 2 != 0 and c in ctype_set and c in ctypes: # just looking at one qubit
                G.add_edge(N + mx + r, c)
    
    if draw:           
        pos = nx.bipartite_layout(G, q_nodes)
        nx.draw_networkx_nodes(G, pos, nodelist=q_nodes, node_shape='o', node_color='lightgreen', label='check-type qubits')
        if choice == "X":
            nx.draw_networkx_nodes(G, pos, nodelist=x_nodes, node_shape='s', node_color='salmon', label='X stabilizers')
        elif choice == "Z":
            nx.draw_networkx_nodes(G, pos, nodelist=z_nodes, node_shape='s', node_color='lightblue', label='Z stabilizers')
        nx.draw_networkx_edges(G, pos, alpha=0.6)
        labels = {q: f"q{q}" for q in q_nodes}
        labels.update({N + r: f"X{r}" for r in range(mx)})
        labels.update({N + mx + r: f"Z{r}" for r in range(mz)})
        nx.draw_networkx_labels(G, pos, labels, font_size=6)
        plt.show()
    
    return G
    