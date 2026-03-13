Example runs:

```bash
# Relay-BP decoder (default parameters)
python3 noise_sims.py --shots 5000 --decoder Relay

# BP-OSD decoder with custom order
python3 noise_sims.py --shots 5000 --decoder OSD --bp-max-iter 100 --bp-max-order 10

# Override Relay defaults
python3 noise_sims.py --shots 5000 --decoder Relay --relay-num-sets 200 --relay-stop-nconv 10
```