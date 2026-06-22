# Hypergraph Product (HGP) Physical Qubit Reduction (2605.11318)

This repository contains the code for reducing HGP code circuits and testing the reduction with noisy simulations.

### Directory structure:
```python
hgp-reduction
├── Figs                         # generated figures and saved plots
│   ├── ...
├── README.md                    # this file
├── circuit_noise_sims           # circuit-level noise simulations (Section 5)
│   ├── README.md
│   ├── codes                    # codes that we sample noise on
│   │   ├── K33_cycle.py
│   │   ├── __pycache__
│   │   ├── heawood_cycle.py
│   │   ├── petersen_cycle.py
│   │   ├── quasi_cyclic_codes.py
│   │   ├── random_codes.py 
│   │   ├── random_quasi_cyclic.py
│   │   ├── repetition_code.py
│   │   └── tutte_coxeter_cycle.py
│   ├── data                      # saved simulation data
│   │   └── ...
│   ├── data_collection.py        # runs noise simulations and saves data
│   ├── functions                 # helper functions
│   │   ├── H_to_CNOT_circuit.py 
│   │   ├── decoding.py 
│   │   ├── edge_coloring.py
│   │   ├── matrix_funcs.py 
│   │   ├── reduction_funcs.py
│   │   └── sim_common.py  
│   ├── plots                     # generated plots
│   │   └── ...
│   ├── plotting.py               # loads data and makes plots
│   └── requirements              # installs requirements
│       ├── install_requirements.sh
│       ├── python.txt
│       └── system-ubuntu.txt
└── transform_random_codes.ipynb   # reduction walkthrough (Section 3.1)
```
