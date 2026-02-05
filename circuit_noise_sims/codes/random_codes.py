import numpy as np
import scipy.sparse as sp
import networkx as nx
import math
from bposd.hgp import hgp
from ldpc.code_util import compute_code_parameters, compute_exact_code_distance
from ldpc.mod2 import rank
from networkx.algorithms.bipartite import configuration_model, biadjacency_matrix

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

def get_random_code(n, d_v, d_c, min_dist, max_coloring):
    """    
    Return a random HGP code.
    
    :param n: #bits
    :param d_v: how many checks each bit participates in
    :param d_c: how many bits each check involves
    :param min_dist: minimum distance of classical code 
    :param max_coloring: maximum number of color groups in classical code
    """

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
    print(f"\t\tRandom code: [n, k, d] = {compute_code_parameters(H)}")
    
    ### Create HGP code from two of the classical code
    code = hgp(h1=H, h2=H, compute_distance=True)
    code.name = 'Random Code HGP'
    print(f"\t\tHGP Code: [[{code.N}, {code.K}, {code.D}]]")
    
    return code, H
