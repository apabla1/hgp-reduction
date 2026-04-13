## Requirements

Install Python dependencies:

python3 -m pip install -r requirements/python.txt

Install Python dependencies and optional LaTeX system packages (Ubuntu/Debian):

bash requirements/install_requirements.sh --with-latex

If you only want Python dependencies:

bash requirements/install_requirements.sh

# Simulation and Plotting

Simulation and plotting are now split into two scripts:

- `data_collection.py`: runs noisy simulations and appends to saved data.
- `plotting.py`: loads saved data and generates plots.

---
## Data collection (`data_collection.py`)

Common arguments:
- `--shots`: number of circuit samples to decode.
- `--decoder`: `OSD`, `LSD`, or `Relay`.
- `--codes`: one or more codes to simulate.
- `--list-codes`: print the available codes (see below) and exit.
- `--processes`: number of worker processes.
- `--p-values`: explicit values separated by commas or a range with "low high step" format.
- `--bp-max-iter`, `--bp-max-order`: BP/OSD/LSD settings.
- `--relay-gamma0`, `--relay-pre-iter`, `--relay-num-sets`, `--relay-max-iter`, `--relay-gamma-dist-interval`, `--relay-stop-nconv`: Relay-specific settings.

Available codes:
- `heawood_cycle`
- `K33_cycle`
- `petersen_cycle`
- `quasi_cyclic_codes`
- `random_codes`
- `random_quasi_cyclic`
- `repetition_code`
- `tutte_coxeter_cycle`

### Examples

```bash
# Relay-BP decoder, default p sweep (0.5e-3 to 1.0e-2 in 0.5e-3 steps)
python3 data_collection.py --shots 5000 --decoder Relay

# BP-OSD decoder with custom order
python3 data_collection.py --shots 5000 --decoder OSD --bp-max-iter 100 --bp-max-order 10

# p-value range: lower upper step (space-separated)
python3 data_collection.py --shots 5000 --decoder Relay --p-values 5e-4 1e-2 5e-4

# Explicit p values (comma-separated)
python3 data_collection.py --shots 5000 --decoder Relay --p-values 5e-4,1e-3,2e-3
```

---
## Plotting (`plotting.py`)

Common arguments:
- `--decoder`: `OSD`, `LSD`, or `Relay`.
- `--codes`: one or more codes to plot.
- `--list-codes`: print the available codes (see below) and exit.
- `--p-min`, `--p-max`: p-range to display.
- `--bp-max-iter`, `--bp-max-order`: BP/OSD/LSD settings.
- `--relay-gamma0`, `--relay-pre-iter`, `--relay-num-sets`, `--relay-max-iter`, `--relay-gamma-dist-interval`, `--relay-stop-nconv`: Relay-specific settings.

Available codes:
- `heawood_cycle`
- `K33_cycle`
- `petersen_cycle`
- `quasi_cyclic_codes`
- `random_codes`
- `random_quasi_cyclic`
- `repetition_code`
- `tutte_coxeter_cycle`

### Examples

```bash
# Plot all available p points for Relay default parameters
python3 plotting.py --decoder Relay

# Plot only p in [1e-3, 6e-3]
python3 plotting.py --decoder Relay --p-min 1e-3 --p-max 6e-3

# Plot selected codes for BP-OSD custom settings
python3 plotting.py --decoder OSD --bp-max-iter 100 --bp-max-order 10 --codes heawood_cycle K33_cycle

```

## Data layout

Data files are now organized by decoder configuration:

```text
data/
	<decoder_config>/
		unreduced_random/
			<code_name>.npy
		reduced_random/
			<code_name>.npy
		reduced_split/
			<code_name>.npy
```

Each file stores rows of `[p, failures, total_shots]`.
When `data_collection.py` is re-run with the same decoder configuration, it appends shots to matching `p` rows by default.
