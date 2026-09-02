import numpy as np
import stim
from networkx import relabel_nodes
from networkx.algorithms import bipartite
from functions.cardinal_schedule import get_cardinal_schedule
from functions.edge_coloring import edge_color_bipartite
from functions.logical_basis import canonical_z_logicals


def _ordered_qubits(qubits):
    return sorted(map(int, qubits))


def _append_hadamard_layer(circuit, targets, all_qubits, p1):
    circuit.append("H", _ordered_qubits(targets))
    # Active H gates and idle qubits receive the same one-qubit channel.
    circuit.append("DEPOLARIZE1", _ordered_qubits(all_qubits), p1)
    circuit.append("TICK")


def _append_cnot_layer(circuit, targets, all_qubits, p1, p2):
    targets = list(map(int, targets))
    idle_qubits = set(map(int, all_qubits)) - set(targets)

    circuit.append("CX", targets)
    circuit.append("DEPOLARIZE2", targets, p2)
    if idle_qubits:
        circuit.append("DEPOLARIZE1", _ordered_qubits(idle_qubits), p1)
    circuit.append("TICK")


def _append_z_memory_annotations(circuit, code, rounds):
    """Append the Z-detector model and canonical logical-Z observables."""

    n = int(code.N)
    mx = int(code.hx.shape[0])
    mz = int(code.hz.shape[0])
    total_measurements = int(circuit.num_measurements)
    data_measurement_start = total_measurements - n
    recorded_round_measurements = data_measurement_start
    if recorded_round_measurements == rounds * mz:
        measurements_per_round = mz
    elif recorded_round_measurements == rounds * (mz + mx):
        measurements_per_round = mz + mx
    else:
        raise ValueError(
            "Unexpected measurement layout for Z-memory annotations: "
            f"got {total_measurements} total measurements."
        )

    def rec_target(measurement_index):
        return stim.target_rec(int(measurement_index) - total_measurements)

    # The initial Z-check outcomes are fixed by |0> data preparation.
    for check in range(mz):
        circuit.append("DETECTOR", [rec_target(check)])

    # Compare each later Z-check outcome with its previous-round value.
    for round_index in range(1, rounds):
        current_start = round_index * measurements_per_round
        previous_start = current_start - measurements_per_round
        for check in range(mz):
            circuit.append(
                "DETECTOR",
                [
                    rec_target(current_start + check),
                    rec_target(previous_start + check),
                ],
            )

    # Close the Z boundary using the final transversal data measurement.
    last_round_start = (rounds - 1) * measurements_per_round
    hz = code.hz.tocsr()
    for check in range(mz):
        targets = [rec_target(last_round_start + check)]
        targets.extend(
            rec_target(data_measurement_start + qubit)
            for qubit in hz.indices[hz.indptr[check] : hz.indptr[check + 1]]
        )
        circuit.append("DETECTOR", targets)

    logicals = canonical_z_logicals(code).tocsr()
    for observable in range(logicals.shape[0]):
        targets = [
            rec_target(data_measurement_start + qubit)
            for qubit in logicals.indices[
                logicals.indptr[observable] : logicals.indptr[observable + 1]
            ]
        ]
        circuit.append("OBSERVABLE_INCLUDE", targets, observable)


def generate_synd_circuit(
    H,
    checks,
    stab_type,
    p1,
    p2,
    seed,
    include_basis_change=True,
    all_qubits=None,
):
    """
    Stim (X or Z) syndrome extraction circuit given a PCM H.
    Utilizes edge coloring of the Tanner graph to enforce order of parallel CNOTs.
    (An edge in the Tanner graph corresponds to a CNOT gate; edges of the same
    color can be operated in parallel.)

    :param H: Hx or Hz
    :param checks: qubit indices for the m(X or Z) syndrome qubits
    :param stab_type: T/F (T = X stabilizers; F = Z stabilizers)
    :param p1: single-qubit gate error probability
    :param p2: two-qubit gate error probability
    :param seed: randomizing the order of parallel CNOTs based on coloring
    :param include_basis_change: wrap X-check CNOTs in Hadamard layers
    :param all_qubits: physical qubits that idle during each gate layer
    """
    m, n = H.shape
    tanner_graph = bipartite.from_biadjacency_matrix(H)
    mapping = {i: checks[i] for i in range(m)}
    mapping.update({i: i - m for i in range(m, n + m)})
    tanner_graph = relabel_nodes(tanner_graph, mapping)
    coloring = edge_color_bipartite(tanner_graph)
    if seed != 0:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(coloring, axis=0)

    c = stim.Circuit()
    if all_qubits is None:
        all_qubits = set(range(n)) | set(map(int, checks))
    else:
        all_qubits = set(map(int, all_qubits))

    if stab_type and include_basis_change:
        _append_hadamard_layer(c, checks, all_qubits, p1)

    for r in coloring:
        targets = []
        for g in r:
            targets.extend(g[::-1] if stab_type else g)
        _append_cnot_layer(c, targets, all_qubits, p1, p2)

    if stab_type and include_basis_change:
        _append_hadamard_layer(c, checks, all_qubits, p1)
    return c


def generate_cardinal_synd_circuit(code, h1, h2, p1, p2, seed=0):
    """Generate one mixed-X/Z cardinal syndrome-extraction block."""

    n = int(code.N)
    mx = int(code.hx.shape[0])
    mz = int(code.hz.shape[0])
    x_checks = range(n, n + mx)
    all_qubits = range(n + mx + mz)
    schedule, component_depths = get_cardinal_schedule(code, h1, h2, seed=seed)

    c = stim.Circuit()
    _append_hadamard_layer(c, x_checks, all_qubits, p1)
    for layer in schedule:
        targets = [qubit for gate in layer for qubit in gate]
        _append_cnot_layer(c, targets, all_qubits, p1, p2)
    _append_hadamard_layer(c, x_checks, all_qubits, p1)

    return c, component_depths



def generate_full_circuit(code, rounds, noise_pars, seed):
    """
    Non-ordered syndrome extraction.

    :param code: code to generate circuit for
    :param rounds: rounds of measurement
    :param noise_pars: tuple of (p1, p2, p_spam)
    :param seed: seed forwarded to generate_synd_circuit
    """
    p1, p2, p_spam = noise_pars
    mx, n = code.hx.shape
    mz = code.hz.shape[0]
    data_qubits = range(n)
    x_checks = range(n, n + mx)
    z_checks = range(n + mx, n + mx + mz)
    all_qubits = range(n + mx + mz)
    c = stim.Circuit()
    z_synd_circuit = generate_synd_circuit(
        code.hz, z_checks, 0, p1, p2, seed, all_qubits=all_qubits
    )
    x_synd_circuit = generate_synd_circuit(
        code.hx, x_checks, 1, p1, p2, seed, all_qubits=all_qubits
    )

    # Data and ancilla preparation errors.
    c.append("X_ERROR", data_qubits, p_spam)
    c.append("X_ERROR", z_checks, p_spam)
    c.append("X_ERROR", x_checks, p_spam)

    # syndrome extraction rounds
    c_se = stim.Circuit()
    # Z syndrome measurement
    c_se += z_synd_circuit
    c_se.append("X_ERROR", z_checks, p_spam)
    c_se.append("MR", z_checks)
    c_se.append("X_ERROR", z_checks, p_spam)
    # X syndrome measurement
    c_se += x_synd_circuit
    c_se.append("R", x_checks)
    c_se.append("X_ERROR", x_checks, p_spam)

    c += c_se * rounds

    # Final transversal measurement
    c.append("X_ERROR", data_qubits, p_spam)
    c.append("MR", data_qubits)
    _append_z_memory_annotations(c, code, rounds)
    return c


def generate_full_circuit_cardinal(code, h1, h2, rounds, noise_pars, seed=0):
    """Generate an original-HGP memory circuit with cardinal extraction.

    The default seed is fixed to zero.  Both X- and Z-check CNOTs occur in the
    same E/N/S/W schedule; only Z-check outcomes are recorded for the Z-memory
    experiment, matching :func:`generate_full_circuit`.
    """

    p1, p2, p_spam = noise_pars
    mx, n = code.hx.shape
    mz = code.hz.shape[0]
    data_qubits = range(n)
    x_checks = range(n, n + mx)
    z_checks = range(n + mx, n + mx + mz)

    syndrome_circuit, _ = generate_cardinal_synd_circuit(
        code, h1, h2, p1=p1, p2=p2, seed=seed
    )

    c = stim.Circuit()
    c.append("X_ERROR", data_qubits, p_spam)
    c.append("X_ERROR", z_checks, p_spam)
    c.append("X_ERROR", x_checks, p_spam)

    c_se = stim.Circuit()
    c_se += syndrome_circuit
    c_se.append("X_ERROR", z_checks, p_spam)
    c_se.append("MR", z_checks)
    c_se.append("X_ERROR", z_checks, p_spam)
    c_se.append("R", x_checks)
    c_se.append("X_ERROR", x_checks, p_spam)
    c += c_se * rounds

    c.append("X_ERROR", data_qubits, p_spam)
    c.append("MR", data_qubits)
    _append_z_memory_annotations(c, code, rounds)
    return c



def generate_full_circuit_split(
    Hx1,
    Hx2,
    Hx3,
    Hz1,
    Hz2,
    Hz3,
    rounds,
    noise_pars,
    seed,
    *,
    code,
):
    """
    Order-enforced syndrome extraction implementing a three-way split

      repeat `rounds` times {
        (1) project Z syndromes for Hz1
        (2) project Z syndromes for Hz2
        (3) project Z syndromes for Hz3
        (4) measure Z syndromes and reset ancillas
        (5) project X syndromes for Hx1
        (6) project X syndromes for Hx2
        (7) project X syndromes for Hx3
        (8) measure X syndromes and reset ancillas
      }
      (9) measure data qubits

    The intended interpretation is:
      - part 1: uncombined checks in full and the first left-sector segment
        of each combined check
      - part 2: the residual right-sector support of combined checks
      - part 3: the second left-sector segment of each combined check

    :param Hx1: part 1 of Hx
    :param Hx2: part 2 of Hx
    :param Hx3: part 3 of Hx
    :param Hz1: part 1 of Hz
    :param Hz2: part 2 of Hz
    :param Hz3: part 3 of Hz
    :param rounds: rounds of measurement
    :param noise_pars: tuple of (p1, p2, p_spam)
    :param seed: seed forwarded to generate_synd_circuit
    :param code: reduced CSS code carrying the restricted canonical basis
    """
    p1, p2, p_spam = noise_pars
    # n equal
    assert Hx1.shape[1] == Hx2.shape[1] == Hx3.shape[1] == Hz1.shape[1] == Hz2.shape[1] == Hz3.shape[1]

    # mx, mz equal
    assert Hx1.shape[0] == Hx2.shape[0] == Hx3.shape[0]
    assert Hz1.shape[0] == Hz2.shape[0] == Hz3.shape[0]

    n = Hx1.shape[1]
    mx = Hx1.shape[0]
    mz = Hz1.shape[0]

    if (
        int(code.N) != n
        or code.hx.shape != (mx, n)
        or code.hz.shape != (mz, n)
    ):
        raise ValueError("The supplied reduced code does not match the split matrices.")

    data_qubits = range(n)
    x_synd_qubits = range(n, n + mx)
    z_synd_qubits = range(n + mx, n + mx + mz)
    all_qubits = range(n + mx + mz)

    # entire circuit
    c = stim.Circuit()

    # Data and ancilla preparation errors.
    c.append("X_ERROR", data_qubits, p_spam)
    c.append("X_ERROR", z_synd_qubits, p_spam)
    c.append("X_ERROR", x_synd_qubits, p_spam)

    # CNOT syndrome extraction circuit for the three-way split schedule
    c_se = stim.Circuit()

    # Z syndrome extraction: part 1, then residual support, then part 3.
    for phase, phase_seed in (
        (Hz1, seed + 0),
        (Hz2, seed + 1),
        (Hz3, seed + 2),
    ):
        c_se += generate_synd_circuit(
            phase,
            z_synd_qubits,
            stab_type=0,
            p1=p1,
            p2=p2,
            seed=phase_seed,
            all_qubits=all_qubits,
        )

    # measure Z syndromes and reset ancillas
    c_se.append("X_ERROR", z_synd_qubits, p_spam)
    c_se.append("MR", z_synd_qubits)
    c_se.append("X_ERROR", z_synd_qubits, p_spam)

    # Prepare X ancillas once, execute all three phases, and return to the
    # measurement basis once.  An empty middle phase adds no gates or noise.
    _append_hadamard_layer(c_se, x_synd_qubits, all_qubits, p1)
    for phase, phase_seed in (
        (Hx1, seed + 3),
        (Hx2, seed + 4),
        (Hx3, seed + 5),
    ):
        c_se += generate_synd_circuit(
            phase,
            x_synd_qubits,
            stab_type=1,
            p1=p1,
            p2=p2,
            seed=phase_seed,
            include_basis_change=False,
            all_qubits=all_qubits,
        )
    _append_hadamard_layer(c_se, x_synd_qubits, all_qubits, p1)

    # measure X syndromes and reset ancillas
    c_se.append("X_ERROR", x_synd_qubits, p_spam)
    c_se.append("MR", x_synd_qubits)
    c_se.append("X_ERROR", x_synd_qubits, p_spam)

    c += c_se * rounds

    # final data measurement
    c.append("X_ERROR", data_qubits, p_spam)
    c.append("MR", data_qubits)

    _append_z_memory_annotations(c, code, rounds)

    return c
