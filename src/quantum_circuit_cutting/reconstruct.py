"""Feature 5: Result Reconstruction - Recover original circuit results from subcircuit results."""


def reconstruct(subcircuits_or_results, cut_info,
                mode="classical", output="statevector",
                qubit_indices=None, pauli_map=None):
    """
    Reconstruct original circuit results from subcircuit results.

    Parameters
    ----------
    subcircuits_or_results : list
        classical: subcircuits from cut()
        quantum:   List[DataClassSubQCRes] from execute()
    cut_info : dict
        Cut metadata from cut() (required for both paths).
    mode : str
        "classical" or "quantum".
    output : str
        classical: "statevector" / "probability" / "expectation"
        quantum:   "expectation" / "distribution"
    qubit_indices : list[int], optional
        Qubit indices for probability computation (classical, output="probability").
        If None, uses all qubits.
    pauli_map : dict, optional
        Pauli operator map for expectation (classical, output="expectation").
        e.g. {0: 'Z', 1: 'Z'} for <Z_0 Z_1>.

    Returns
    -------
    Result in the requested format.
    """
    if mode == "classical":
        return _reconstruct_classical(subcircuits_or_results, cut_info,
                                      output, qubit_indices, pauli_map)
    elif mode == "quantum":
        return _reconstruct_quantum(subcircuits_or_results, cut_info, output)
    else:
        raise ValueError(f"Unknown mode: {mode}.")


def _reconstruct_classical(subcircuits, cut_info, output, qubit_indices, pauli_map):
    """Classical path reconstruction.

    Flow:
      1. run_subcircuits_to_statevecs(subcircuits) -> sv_subs
      2. merge_statevec_subs(sv_subs, cut_info) -> statevector
      3. Output format conversion
    """
    from qcc_classical.reconstruct import (
        run_subcircuits_to_statevecs,
        merge_statevec_subs,
        compute_joint_probs,
        compute_all_single_qubit_probs,
        expect_pauli,
    )

    # Step 1: Execute subcircuits via statevector simulation
    sv_subs = run_subcircuits_to_statevecs(subcircuits)

    # Step 2: Merge statevectors
    statevector = merge_statevec_subs(sv_subs, cut_info)

    # Step 3: Convert to requested output format
    if output == "statevector":
        return statevector
    elif output == "probability":
        if qubit_indices is None:
            all_qubits = []
            for g in cut_info["qubit_groups"]:
                all_qubits.extend(g)
            qubit_indices = sorted(all_qubits)
        return compute_joint_probs(cut_info, qubit_indices)
    elif output == "expectation":
        if pauli_map is None:
            return compute_all_single_qubit_probs(cut_info)
        return expect_pauli(cut_info, pauli_map)
    else:
        raise ValueError(f"Unknown output format: {output}")


def _reconstruct_quantum(results, cut_meta, output):
    """Quantum path reconstruction.

    Uses tensor network contraction to reconstruct results
    from subcircuit measurement outcomes.
    """
    import math
    from circuit_cutter.TensorNetworkData2 import (
        build_counts_layout_lists_from_results,
        CutSystem,
        contract_expectation,
        contract_distribution,
        postprocess_distribution,
        build_local_tensor_expectation,
        build_local_tensor_distribution,
        tensor_distribution_to_prob_dict,
    )
    from circuit_cutter.MyQuantumCircuit import (
        build_specs_and_out_order,
        replace_all_logates_in_circuit,
        assign_roles_inplace,
    )

    cutloc = cut_meta["cutloc"]
    original = cut_meta["original"]
    subcircuits_myqc = cut_meta["subcircuits_myqc"]

    space_cut_ids = tuple(
        i for i, is_gate in enumerate(cutloc.IsGateOrWireList) if is_gate)
    time_cut_ids = tuple(
        i for i, is_gate in enumerate(cutloc.IsGateOrWireList) if not is_gate)

    theta_by_space_cut_id = cut_meta.get("theta_by_space_cut_id",
        {cid: 0.0 for cid in space_cut_ids})
    system = CutSystem(
        space_cut_ids=space_cut_ids,
        time_cut_ids=time_cut_ids,
        theta_by_space_cut_id=theta_by_space_cut_id,
    )

    # Rebuild bundle_circuits for specs (same as execute step3)
    bundles = replace_all_logates_in_circuit(subcircuits_myqc)
    bundle_circuits = [c for b in bundles for c in b.circuits]
    assign_roles_inplace(bundle_circuits)

    specs, role_to_spec_index, out_order = build_specs_and_out_order(
        original, bundle_circuits)

    out_order_qiskit = list(reversed(out_order))

    counts_list, layout_list = build_counts_layout_lists_from_results(
        system, specs, results, role_to_spec_index=role_to_spec_index)

    if output == "expectation":
        # Z...Z parity: (-1)^(number of 1s in out_bits)
        def z_parity(bits):
            return (-1.0) ** sum(bits)

        local_tensors = []
        for si, spec in enumerate(specs):
            lt = build_local_tensor_expectation(
                system, spec, counts_list[si], layout_list[si], f_local=z_parity)
            local_tensors.append(lt)
        return contract_expectation(system, specs, local_tensors)
    elif output == "distribution":
        local_tensors = []
        for si, spec in enumerate(specs):
            lt = build_local_tensor_distribution(
                system, spec, counts_list[si], layout_list[si])
            local_tensors.append(lt)
        raw_dist = contract_distribution(system, specs, local_tensors, out_order_qiskit)
        return postprocess_distribution(raw_dist)
    else:
        raise ValueError(f"Unknown output format: {output}.")
