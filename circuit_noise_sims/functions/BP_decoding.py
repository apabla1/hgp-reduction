import numpy as np
from scipy.sparse import csr_matrix
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder

def get_BP_failures(code, dec, circ, params, p2, iters, rounds):
    """
    code: css code object
    dec: decoder type ('OSD' or 'LSD')
    circ: stim circuit for syndrome extraction
    params: decoder parameters -- [bp_iters, osd_order / lsd_order]
    p2: two-qubit gate error probability
    """
    
    if dec != 'OSD' and dec != 'LSD':
        raise ValueError('Invalid decoder type')
    if len(params) != 2:
        raise ValueError('Invalid decoder paramsameters')
    
    # params = [bp_iters, osd_sweeps]
    H = code.hz.toarray()
    m, n = H.shape
    
    # Construct spacetime decoding graph
    H_dec = np.kron(np.eye(rounds+1,dtype=int), H)
    H_dec = np.concatenate((H_dec,np.zeros([m*(rounds+1),m*rounds],dtype=int)), axis=1)
    for j in range(m*rounds):
        H_dec[j,n*(rounds+1)+j] = 1
        H_dec[m+j,n*(rounds+1)+j] = 1
    H_dec = csr_matrix(H_dec)
    
    if dec == 'OSD':
        decoder = BpOsdDecoder(H_dec, error_rate=float(5*p2), max_iter=params[0], bp_method='ms', osd_method='osd_cs', osd_order=params[1])
    elif dec == 'LSD':
        decoder = BpLsdDecoder(H_dec, error_rate=float(5*p2), max_iter=params[0], bp_method='ms', lsd_method='lsd_cs', lsd_order=params[1], schedule='serial')
    
    sampler = circ.compile_sampler()
    failures = 0
    outer_reps = iters//256    # Stim samples a minimum of 256 shots at a time
    remainder = iters % 256
    for j in range(outer_reps+1):
        print(f"\tIteration {j} of {outer_reps+1}")
        num_shots = 256
        if j == outer_reps:
            num_shots = remainder
        output = sampler.sample(shots=num_shots)
        for i in range(num_shots):
            print(f"\t\tShot {i} of {num_shots}") if i % 50 == 0 else None
            syndromes = np.zeros([rounds+1,m], dtype=int)
            syndromes[:rounds] = output[i,:-n].reshape([rounds,m])
            syndromes[-1] = H @ output[i,-n:] % 2
            syndromes[1:] = syndromes[1:] ^ syndromes[:-1]   # Difference syndrome
            decoder_output = np.reshape(decoder.decode(np.ravel(syndromes))[:n*(rounds+1)], [rounds+1,n])
            correction = decoder_output.sum(axis=0) % 2
            final_state = output[i,-n:] ^ correction
            if (code.lz@final_state%2).any():
                failures += 1
                
    return failures