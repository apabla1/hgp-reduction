import numpy as np
import time
from scipy.sparse import csr_matrix
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder

# For printing time (ignore)
def _fmt_secs(sec: float) -> str:
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def num_failures_BP(code, dec, circ, params, p2, p_data, p_meas, shots, rounds):
    """
    code: css code object
    dec: decoder type ('OSD' or 'LSD')
    circ: stim circuit for syndrome extraction
    params: decoder parameters -- [bp_shots, osd_order / lsd_order]
    p2: two-qubit gate error probability
    p_data: prior per data time-slice variable
    p_meas: prior per measurement variable
    shots: number of shots to sample
    rounds: number of rounds in the circuit
    """
    
    if dec != 'OSD' and dec != 'LSD':
        raise ValueError('Invalid decoder type')
    if len(params) != 2:
        raise ValueError('Invalid decoder paramsameters')
    
    H = code.hz.toarray()
    m, n = H.shape
    
### Construct spacetime decoding graph
    H_dec = np.kron(np.eye(rounds+1,dtype=int), H)
    H_dec = np.concatenate((H_dec,np.zeros([m*(rounds+1),m*rounds],dtype=int)), axis=1)
    for j in range(m*rounds):
        H_dec[j,n*(rounds+1)+j] = 1
        H_dec[m+j,n*(rounds+1)+j] = 1
    H_dec = csr_matrix(H_dec)
    
### Decoder
    err_channel = np.concatenate([np.full(n*(rounds+1), p_data, dtype=float), np.full(m*rounds, p_meas, dtype=float)])
    if dec == 'OSD':
        decoder = BpOsdDecoder(H_dec, error_channel=err_channel.tolist(), max_iter=params[0], bp_method='ms', osd_method='osd_cs', osd_order=params[1], schedule='parallel')
    elif dec == 'LSD':
        decoder = BpLsdDecoder(H_dec, error_channel=err_channel.tolist(), max_iter=params[0], bp_method='ms', lsd_method='lsd_cs', lsd_order=params[1], schedule='serial')

### Sampling
    sampler = circ.compile_sampler()
    num_failures = 0
    shot_num = 0

    # For sampling -- Stim samples a minimum of 256 shots at a time
    # batch_sizes = [256, 256, ..., shots // 256, shots % 256]
    batch_sizes = [256] * (shots // 256)
    remainder = shots % 256
    if remainder:
        batch_sizes.append(remainder)
       
    # Timer    
    t0 = time.perf_counter()
    
    for num_shots in batch_sizes:
        output = sampler.sample(shots=num_shots)
        for i in range(num_shots):
            print(f"\tShot 0 of {shots} (elapsed 0:00)") if shot_num == 0 else None
            shot_num += 1
            if shot_num % max(1, shots // 25) == 0 or shot_num == shots:
                elapsed = time.perf_counter() - t0
                rate = shot_num / elapsed
                eta = (shots - shot_num) / rate if rate > 0 else float("inf")
                print(f"\tShot {shot_num} of {shots}; {num_failures} failed so far (elapsed {_fmt_secs(elapsed)}, eta {_fmt_secs(eta)})")
            syndromes = np.zeros([rounds+1,m], dtype=int) 
            meas = output[i, :-n]  # all ancilla measurement bits (Z then X each round)
            per_round = meas.size // rounds  # should be mz + mx
            meas = meas.reshape(rounds, per_round)
            z_meas = meas[:, :m]
            syndromes[:rounds] = z_meas
            syndromes[-1] = H @ output[i,-n:] % 2
            syndromes[1:] = syndromes[1:] ^ syndromes[:-1]   # Difference syndrome
            decoder_output = np.reshape(decoder.decode(np.ravel(syndromes))[:n*(rounds+1)], [rounds+1,n])
            correction = decoder_output.sum(axis=0) % 2
            final_state = output[i,-n:] ^ correction
            if (code.lz@final_state%2).any():
                num_failures += 1
                
    return num_failures