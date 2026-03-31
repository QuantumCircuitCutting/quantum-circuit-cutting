"""Feature 2: Circuit Cutting - Locate cut positions and generate subcircuits."""

from qiskit import QuantumCircuit


def cut(qc, mode="classical", max_qubits=5, memory_limit=None,
        **quantum_options):
    """
    Split a quantum circuit into smaller subcircuits.

    Parameters
    ----------
    qc : QuantumCircuit
        Input quantum circuit.
    mode : str
        "classical" or "quantum".
    max_qubits : int
        Maximum qubit count per subcircuit.
    memory_limit : float, optional
        Memory limit in GB (classical path, alternative to max_qubits).
    **quantum_options
        Additional options for quantum path:
        - method (int): Partitioning method (default 1).
        - gatecut_only (bool): Gate cuts only (default False).
        - seed (int): Random seed.

    Returns
    -------
    tuple
        classical: (subcircuits, cut_info)
        quantum:   (subcircuits, cut_meta)
    """
    if mode == "classical":
        return _cut_classical(qc, max_qubits, memory_limit)
    elif mode == "quantum":
        return _cut_quantum(qc, max_qubits, **quantum_options)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'classical' or 'quantum'.")


def _cut_classical(qc, max_qubits, memory_limit):
    """Classical path: partition + cut."""
    from qcc_classical.locator import partition_qubits
    from qcc_classical.cut import cut_two_qubit_gates_for_statevec

    if memory_limit is not None:
        groups = partition_qubits(qc, max_memory_gb=memory_limit)
    else:
        groups = partition_qubits(qc, max_segment_qubits=max_qubits)

    subcircuits, cut_info = cut_two_qubit_gates_for_statevec(qc, groups)
    return subcircuits, cut_info


def _cut_quantum(qc, max_qubits, method=1, gatecut_only=False, seed=None):
    """Quantum path: partition + split into subcircuits.

    Returns
    -------
    subcircuits : list[MyQuantumCircuit]
        Leaf subcircuits with roles assigned.
    cut_meta : dict
        Metadata needed for reconstruction (cutloc, original circuit, etc.)
    """
    import re
    from qiskit.qasm3 import dumps

    from circuit_cut_point_locator.cut_locator import exec_partitioning
    from circuit_cutter import (
        MyQuantumCircuit,
        split_all_circuit,
        DataClassCutLoc,
        filter_leaf_subcircuits,
        dedup_circuits_with_map,
        assign_roles_inplace,
    )

    # Step 1: Find cut positions via graph partitioning
    result = exec_partitioning(qc, device_nqubit_limit=max_qubits,
                               method=method, gatecut_only=gatecut_only,
                               seed=seed)
    cut_lst = result["cut_lst"]

    if not cut_lst:
        raise ValueError("No cuts found. Circuit may already fit within max_qubits.")

    # Step 2: Convert Qiskit circuit to MyQuantumCircuit
    import math as _math
    from qiskit import QuantumCircuit as _QC

    # Decompose cx/cz to rzz at qiskit circuit level for gate cut compatibility
    _qc_rzz = _QC(qc.num_qubits)
    _orig_count = {}    # qubit -> gate count in original
    _rzz_count = {}     # qubit -> gate count in rzz version
    _idx_map = {}       # (qubit_idx, orig_count) -> rzz_count of the rzz gate

    for _inst in qc.data:
        _qubits = [_qb._index for _qb in _inst.qubits]
        if _inst.operation.name == 'cx':
            _ctrl, _tgt = _qubits[0], _qubits[1]
            _orig_count[_ctrl] = _orig_count.get(_ctrl, 0) + 1
            _orig_count[_tgt] = _orig_count.get(_tgt, 0) + 1
            # ry(tgt)
            _rzz_count[_tgt] = _rzz_count.get(_tgt, 0) + 1
            # rx(tgt)
            _rzz_count[_tgt] += 1
            # rzz(ctrl, tgt)
            _rzz_count[_ctrl] = _rzz_count.get(_ctrl, 0) + 1
            _idx_map[(_ctrl, _orig_count[_ctrl])] = _rzz_count[_ctrl]
            _rzz_count[_tgt] += 1
            _idx_map[(_tgt, _orig_count[_tgt])] = _rzz_count[_tgt]
            # rz(ctrl)
            _rzz_count[_ctrl] += 1
            # rz(tgt)
            _rzz_count[_tgt] += 1
            # ry(tgt)
            _rzz_count[_tgt] += 1
            _qc_rzz.ry(_math.pi / 2, _tgt)
            _qc_rzz.rx(_math.pi, _tgt)
            _qc_rzz.rzz(_math.pi / 2, _ctrl, _tgt)
            _qc_rzz.rz(-_math.pi / 2, _ctrl)
            _qc_rzz.rz(_math.pi / 2, _tgt)
            _qc_rzz.ry(_math.pi / 2, _tgt)
        else:
            for _qb in _qubits:
                _orig_count[_qb] = _orig_count.get(_qb, 0) + 1
                _rzz_count[_qb] = _rzz_count.get(_qb, 0) + 1
            _qc_rzz.append(_inst.operation, _qubits)

    # Remap cut_lst idx_on_qubit
    for _c in cut_lst:
        _c["orig_idx_on_qubit"] = _c["idx_on_qubit"]
        _key = (_c["qubit"], _c["idx_on_qubit"])
        if _key in _idx_map:
            _c["idx_on_qubit"] = _idx_map[_key]

    qasm_str = dumps(_qc_rzz)
    qasm_str = MyQuantumCircuit.rewrite_qasm3_measure_to_scalar_y(
        qasm_str, n_qubits=qc.num_qubits)

    import importlib.resources as pkg_resources
    yaml_path = str(pkg_resources.files("circuit_cutter").joinpath("qubit_gate_info.yaml"))

    myqc = MyQuantumCircuit(yaml_path=yaml_path)
    myqc.parse_qasm3(qasm_str)
    qubit_pattern = re.compile(myqc.params["qubit_pattern"])

    # Step 3: Build DataClassCutLoc from exec_partitioning output
    cut_positions = []
    is_gate_or_wire = []
    targets = []
    controls = []

    for ci in cut_lst:
        q_idx = ci["qubit"]
        orig_idx = ci["orig_idx_on_qubit"]  # before rzz remap
        idx_on_qubit = ci["idx_on_qubit"]   # after rzz remap
        is_gate = (ci["type"] == "gate")

        cut_positions.append({f"q[{q_idx}]": idx_on_qubit})
        is_gate_or_wire.append(is_gate)

        qubit_obj = myqc.get_qubit_by_name_index(f"q[{q_idx}]")

        if is_gate:
            # Use original idx to find the cx qubit pair in the original circuit
            ctrl_q, tgt_q = _resolve_cx_roles(qc, q_idx, orig_idx, myqc)
            targets.append(tgt_q)
            controls.append(ctrl_q)
        else:
            targets.append(qubit_obj)
            controls.append(qubit_obj)

    cutloc = DataClassCutLoc(
        cut_positions=cut_positions,
        IsGateOrWireList=is_gate_or_wire,
        qubit_pattern=qubit_pattern,
        qubits_dict=None,
    )

    # Step 4: Split circuit
    theta_by_space_cut_id = {}
    all_subs = split_all_circuit(
        original_circuit=myqc,
        cutloc=cutloc,
        targets=targets,
        controls=controls,
        yaml_path=yaml_path,
        theta_by_space_cut_id=theta_by_space_cut_id,
    )

    subcircuits = filter_leaf_subcircuits(all_subs, cutloc)
    dedup_res = dedup_circuits_with_map(subcircuits)
    subcircuits = dedup_res.unique_circuits
    assign_roles_inplace(subcircuits)

    # Convert MyQuantumCircuit -> DataClassSubQCParams for downstream use
    import re as _re
    from circuit_cutter import DataClassSubQCParams
    subqc_params_list = []
    for myqc_sub in subcircuits:
        sid = myqc_sub.subqc_id or f"subcircuit_{len(subqc_params_list)}"
        qasm_str = myqc_sub.patched_qasm3_for_qiskit()
        # Remove custom cut gate lines (gate_cut_*, wire_cut_*) for qasm3.loads compatibility
        lines = qasm_str.split("\n")
        lines = [l for l in lines if not _re.match(r"\s*(gate_cut_|wire_cut_)", l)]
        qasm_str = "\n".join(lines)
        subqc_params_list.append(DataClassSubQCParams(
            subqc_id=sid,
            subcircuit_role=myqc_sub.subcircuit_role,
            qasm3=qasm_str,
        ))

    cut_meta = {
        "cutloc": cutloc,
        "original": myqc,
        "subcircuits_myqc": subcircuits,
        "yaml_path": yaml_path,
        "exec_partitioning_result": result,
        "theta_by_space_cut_id": theta_by_space_cut_id,
    }

    return subqc_params_list, cut_meta


def _resolve_cx_roles(qc_qiskit, target_q_idx, idx_on_qubit, myqc):
    """Determine which qubit is CX control and which is CX target."""
    gate_count = {q: 0 for q in range(qc_qiskit.num_qubits)}
    for inst in qc_qiskit.data:
        qubits = [q._index for q in inst.qubits]
        for q in qubits:
            gate_count[q] += 1
        if target_q_idx in qubits and gate_count[target_q_idx] == idx_on_qubit:
            if len(qubits) == 2:
                ctrl_idx, tgt_idx = qubits[0], qubits[1]
                ctrl_obj = myqc.get_qubit_by_name_index(f"q[{ctrl_idx}]")
                tgt_obj = myqc.get_qubit_by_name_index(f"q[{tgt_idx}]")
                return ctrl_obj, tgt_obj
    # fallback
    obj = myqc.get_qubit_by_name_index(f"q[{target_q_idx}]")
    return obj, obj


def _find_control_qubit(qc_qiskit, myqc, target_q_idx, idx_on_qubit):
    """Find the control qubit of a 2-qubit gate at a given position."""
    gate_count = {q: 0 for q in range(qc_qiskit.num_qubits)}
    for inst in qc_qiskit.data:
        qubits = [q._index for q in inst.qubits]
        for q in qubits:
            gate_count[q] += 1
        if target_q_idx in qubits and gate_count[target_q_idx] == idx_on_qubit:
            other = [q for q in qubits if q != target_q_idx]
            if other:
                return myqc.get_qubit_by_name_index(f"q[{other[0]}]")
    return myqc.get_qubit_by_name_index(f"q[{target_q_idx}]")
