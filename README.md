# Hypergraph Product (HGP) Physical Qubit Reduction ()

This repository contains 

### Directory structure:
```python
 hgp-reduction
    ├── Figs
    │   ├── ...
    ├── README.md
    ├── circuit_noise_sims
    │   ├── README.md
    │   ├── functions
    │   │   ├── BP_decoding.py
    │   │   ├── H_to_CNOT_circuit.py
    │   │   ├── circuit_utils.py
    │   │   ├── edge_coloring.py
    │   │   ├── matrix_funcs.py
    │   │   └── reduction_funcs.py
    │   └── noise_sims.py
    └── transform_random_codes.ipynb
```
---

Main files:
- `transform_random_codes.ipynb`: Implements the reduction procedure of Section _ and runs it on random classical LDPC codes. Walks through the entire procedure of creating the codes and doing the reduction. Used to obtain the results in Table _.
- `circuit_noise_sims/noise_sims.py`: Runs noise simulations to test the effectiveness of the reduction. Used to obtain the results in Section _. More details in `noise_funcs/README.md`