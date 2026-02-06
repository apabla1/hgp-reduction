```python
usage: noise_sims.py [-h] --shots SHOTS --decode {BPOSD,BPLSD,RelayBP} [--order ORDER] [--max-iter MAX_ITER] [--gamma0 GAMMA0] [--pre-iter PRE_ITER] [--num-sets NUM_SETS] [--gamma-dist-interval GAMMA_DIST_INTERVAL GAMMA_DIST_INTERVAL] [--stop-nconv STOP_NCONV]

options:
  -h, --help            show this help message and exit
  --shots SHOTS         Number of circuit samples to decode (e.g., 10000)
  --decode {BPOSD,BPLSD,RelayBP}
                        Decoder type: BPOSD, BPLSD, RelayBP
  --order ORDER         (OSD/LSD) OSD order (if OSD) or LSD neighborhood size (if LSD) (e.g., OSD-2 or LSD-6)
  --max-iter MAX_ITER   (OSD/LSD) Maximum number of iterations in BP decoding
  --gamma0 GAMMA0       (RelayBP) Initial gamma value for Relay BP
  --pre-iter PRE_ITER   (RelayBP) Number of pre-iterations for Relay BP
  --num-sets NUM_SETS   (RelayBP) Number of sets for Relay BP
  --gamma-dist-interval GAMMA_DIST_INTERVAL GAMMA_DIST_INTERVAL
                        (RelayBP) Gamma distribution interval for Relay BP (e.g., -0.24 0.66)
  --stop-nconv STOP_NCONV
                        (RelayBP) Number of consecutive iterations without improvement to stop Relay BP
```
