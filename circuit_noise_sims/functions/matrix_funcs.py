from scipy.sparse import csr_matrix

def add(H1, H2):
    """
    Adds two binary csr matrices over F2. 
    """
    H = (H1 + H2).tocsr()
    H.data %= 2
    H.eliminate_zeros()
    return H