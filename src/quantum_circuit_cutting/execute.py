"""
Feature 4: Quantum Execution
============================
Execute subcircuits on quantum devices and return measurement results.

Three-step serial pipeline:
  Step 1: Treewidth-aware gate cut analysis (optional)
  Step 2: Annealing-based SWAP optimization (optional)
  Step 3: Quantum device execution + result reconstruction
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qiskit.qasm3 import loads
from circuit_cutter import MyQuantumCircuit


# ════════════════════════════════════════════════════════════════
# Intermediate data structures
# ════════════════════════════════════════════════════════════════
@dataclass
class CutDecision:
    """
    Result of treewidth cut analysis. Produced by Step 1, consumed by Step 3.
    Step 2 (Annealing) does not modify this.

    Attributes
    ----------
    should_cut : bool
        Whether to apply QPD cutting.
    gate_idx : int or None
        Index of the gate to cut (None if should_cut=False).
    backend : Any
        Qiskit backend (AerSimulator or real device).
    seed_transpiler : int
        Seed for transpilation.
    optimization_level : int
        Optimization level for transpilation.
    """
    should_cut: bool = False
    gate_idx: Optional[int] = None
    backend: Any = None
    seed_transpiler: int = 42
    optimization_level: int = 1


# ════════════════════════════════════════════════════════════════
# Utility functions
# ════════════════════════════════════════════════════════════════
def _get_device_n_qubit(device: Dict[str, Any]) -> int:
    """Get the number of qubits from a device dict."""
    if "n_qubit" in device:
        return device["n_qubit"]
    if "backend" in device:
        return device["backend"].num_qubits
    if "fake_backend" in device:
        mod = importlib.import_module("qiskit_ibm_runtime.fake_provider")
        fake = getattr(mod, device["fake_backend"])()
        return fake.num_qubits
    raise KeyError("device must have 'n_qubit', 'backend', or 'fake_backend'")


def _select_device(
    device_info: Any,
    n_qubits_required: int,
) -> Dict[str, Any]:
    """Select a device with enough qubits from device_info."""
    devices = device_info if isinstance(device_info, list) else [device_info]
    candidates = [d for d in devices if _get_device_n_qubit(d) >= n_qubits_required]
    if not candidates:
        raise ValueError(
            f"No device has enough qubits ({n_qubits_required} required). "
            f"Available: {[_get_device_n_qubit(d) for d in devices]}"
        )
    return min(candidates, key=_get_device_n_qubit)


def _build_backend(device: Dict[str, Any], use_simulator: bool):
    """
    Build a Qiskit backend from a device dict.

    Priority:
      1. 'backend' key (real IBMBackend, etc.)
      2. 'fake_backend' key (class name string)
      3. 'coupling_map' key (fallback)
    """
    if "backend" in device:
        real_backend = device["backend"]
        if use_simulator:
            from qiskit_aer import AerSimulator
            return AerSimulator.from_backend(real_backend)
        return real_backend

    if "fake_backend" in device:
        mod = importlib.import_module("qiskit_ibm_runtime.fake_provider")
        fake = getattr(mod, device["fake_backend"])()
        if use_simulator:
            from qiskit_aer import AerSimulator
            return AerSimulator.from_backend(fake)
        return fake

    from qiskit.transpiler import CouplingMap
    n_qubit = device["n_qubit"]
    coupling_map = CouplingMap(device["coupling_map"])
    if use_simulator:
        from qiskit_aer import AerSimulator
        return AerSimulator(coupling_map=coupling_map)
    from qiskit.providers.fake_provider import GenericBackendV2
    return GenericBackendV2(
        num_qubits=n_qubit,
        coupling_map=device["coupling_map"],
    )


# ════════════════════════════════════════════════════════════════
# Main API
# ════════════════════════════════════════════════════════════════
def execute(
    subcircuits,                            # List[DataClassSubQCParams]
    device_info=None,                       # Feature 3 output (direct)
    available_devices=None,                 # device list for internal recommend()
    cut_meta=None,                          # Feature 2 quantum path metadata
    shots: int = 8192,
    use_simulator: bool = True,
    use_treewidth: bool = False,
    use_annealing: bool = False,
    treewidth_options: Optional[dict] = None,
    annealing_options: Optional[dict] = None,
) -> list:                                  # List[DataClassSubQCRes]
    """
    Optimize and execute subcircuits, returning a list of DataClassSubQCRes.

    Applies three steps in sequence:
      Step 1: Treewidth gate cut analysis (optional)
      Step 2: Annealing SWAP optimization (optional)
      Step 3: Quantum device execution + result reconstruction

    Parameters
    ----------
    subcircuits : List[DataClassSubQCParams]
        Output from Feature 2 (circuit cutting).
    device_info : dict or list of dict, optional
        Output from Feature 3 (device recommendation).
        If not provided, available_devices is used to call recommend() internally.
        Each dict should have keys:
        - name (str), gate_speed (float)
        - backend / fake_backend / coupling_map (one of these)
    available_devices : list, optional
        List of available devices. Used to call recommend() internally
        when device_info is not provided.
    shots : int
        Number of measurement shots.
    use_simulator : bool
        If True, use AerSimulator.
    use_treewidth : bool
        If True, apply Step 1 (Treewidth gate cut optimization).
    use_annealing : bool
        If True, apply Step 2 (Annealing SWAP optimization).
    treewidth_options : dict, optional
        Options passed to step1_treewidth_optimize.
    annealing_options : dict, optional
        Options passed to step2_annealing_optimize. Required keys:
        - client: amplify.BaseClient
        device_info is automatically injected from execute() arguments.

    Returns
    -------
    List[DataClassSubQCRes]
        Measurement results for each subcircuit. Input for Feature 5 (reconstruction).
    """
    if treewidth_options is None:
        treewidth_options = {}
    if annealing_options is None:
        annealing_options = {}

    # Resolve device_info: call recommend() internally if not provided
    if device_info is None:
        if available_devices is None:
            raise ValueError(
                "Either device_info or available_devices must be provided."
            )
        from device_recommender import match_subqc_to_device
        subqc_lst = [
            {
                "name": params.subqc_id,
                "qasm": params.qasm3,
                "nshot": shots,
            }
            for params in subcircuits
        ]
        device_info = match_subqc_to_device(subqc_lst, available_devices)

    # Step 1: Treewidth gate cut analysis
    if cut_meta is not None:
        cut_decisions = []  # Step 3 handles assignment expansion via cut_meta
    elif use_treewidth:
        cut_decisions = step1_treewidth_optimize(
            subcircuits, device_info, use_simulator, **treewidth_options
        )
    else:
        cut_decisions = _no_cut_decisions(subcircuits, device_info, use_simulator)

    # Step 2: Annealing SWAP optimization
    if use_annealing:
        if "client" not in annealing_options:
            raise ValueError(
                "use_annealing=True requires an Amplify client. "
                "Pass annealing_options={'client': your_amplify_client}. "
                "See https://amplify.fixstars.com/ for setup instructions."
            )
        # Ensure device_info has backend objects for annealing transpiler
        _ann_device_info = []
        for d in device_info:
            d_copy = dict(d)
            if "backend" not in d_copy and "fake_backend" in d_copy:
                mod = importlib.import_module("qiskit_ibm_runtime.fake_provider")
                d_copy["backend"] = getattr(mod, d_copy["fake_backend"])()
            _ann_device_info.append(d_copy)
        annealing_options["device_info"] = _ann_device_info
        subcircuits = step2_annealing_optimize(subcircuits, **annealing_options)

    # Step 3: Execute and reconstruct
    results = step3_run_on_device(subcircuits, cut_decisions, shots, cut_meta=cut_meta, device_info=device_info, use_simulator=use_simulator)

    return results


def _no_cut_decisions(
    subcircuits,        # List[DataClassSubQCParams]
    device_info,
    use_simulator: bool,
) -> List[CutDecision]:
    """Build default CutDecisions (no cut) when treewidth optimization is skipped."""
    decisions = []
    for params in subcircuits:
        qasm3 = MyQuantumCircuit.ensure_rzz_gate_defined_for_qiskit_loads(params.qasm3)
        qc = loads(qasm3)
        device = _select_device(device_info, qc.num_qubits)
        backend = _build_backend(device, use_simulator)
        decisions.append(CutDecision(
            should_cut=False,
            backend=backend,
        ))
    return decisions


# ════════════════════════════════════════════════════════════════
# Step 1: Treewidth gate cut analysis
# ════════════════════════════════════════════════════════════════
def step1_treewidth_optimize(
    subcircuits,            # List[DataClassSubQCParams]
    device_info,
    use_simulator: bool,
    **options,
) -> List[CutDecision]:
    """
    Analyze each subcircuit for treewidth-based gate cut optimization.

    Determines whether QPD cutting is beneficial for each subcircuit
    based on break-even analysis. Does not execute circuits.

    Parameters
    ----------
    subcircuits : List[DataClassSubQCParams]
        Output from Feature 2 (circuit cutting).
    device_info : dict or list of dict
        Device information from Feature 3 (device recommendation).
    use_simulator : bool
        If True, use AerSimulator instead of real hardware.
    **options
        Additional options passed to analyze_cuts():
        shots (int), H0 (float), p_gate (float), sigma_shot (float).

    Returns
    -------
    List[CutDecision]
        Cut decisions for each subcircuit.
    """
    from treewidth_gate_cut.treewidth_cut import analyze_cuts
    return analyze_cuts(subcircuits, device_info, use_simulator, **options)


# ════════════════════════════════════════════════════════════════
# Step 2: Annealing SWAP optimization
# ════════════════════════════════════════════════════════════════
def step2_annealing_optimize(
    subcircuits,            # List[DataClassSubQCParams]
    **options,
) -> list:                  # List[DataClassSubQCParams]
    """
    Apply annealing-based SWAP optimization to each subcircuit.

    Input: List[DataClassSubQCParams]
    Output: List[DataClassSubQCParams] (qasm3 updated with transpiled circuits)

    Parameters (via **options)
    -------------------------
    device_info : dict or list of dict
        Device information from execute(). Each dict contains:
        name, n_qubit, coupling_map, basis_gates, etc.
    client : amplify.BaseClient
        Annealing solver client.
    annealing_mode : str (default "best")
        One of "best", "qiskit", "annealing".
    max_bits : int (default 8192)
        Maximum number of variables for the annealing solver.
    """
    from annealing_transpiler.transpiler import annealing_transpile
    from circuit_cutter import MyQuantumCircuit
    from dataclasses import replace
    # Ensure rzz gate is defined in QASM for qiskit loads compatibility
    subcircuits = [
        replace(p, qasm3=MyQuantumCircuit.ensure_rzz_gate_defined_for_qiskit_loads(p.qasm3))
        for p in subcircuits
    ]
    transpiled_subcircuits = annealing_transpile(
        subcircuits,
        options["device_info"],
        options["client"],
        options.get("annealing_mode", "best"),
        options.get("max_bits", 8192)
    )
    return transpiled_subcircuits


# ════════════════════════════════════════════════════════════════
# Step 3: Execute subcircuits on quantum devices
# ════════════════════════════════════════════════════════════════
def step3_run_on_device(
    subcircuits,                    # List[DataClassSubQCParams]
    cut_decisions: List[CutDecision],
    shots: int,
    cut_meta: Optional[dict] = None,
    device_info=None,
    use_simulator: bool = True,
) -> list:                          # List[DataClassSubQCRes]
    """
    Execute each subcircuit and return measurement results.

    When cut_meta is provided (quantum path), expands subcircuits into
    all assignment combinations using replace_all_logates_in_circuit,
    executes each on AerSimulator, and returns DataClassSubQCRes with
    assignment and layout information for tensor network reconstruction.

    Parameters
    ----------
    subcircuits : List[DataClassSubQCParams]
        Subcircuits to execute (may be transpiled by Step 2).
    cut_decisions : List[CutDecision]
        Cut decisions from Step 1 (or default no-cut decisions).
    shots : int
        Number of measurement shots per subcircuit.
    cut_meta : dict, optional
        Metadata from cut() quantum path. Required for assignment expansion.

    Returns
    -------
    List[DataClassSubQCRes]
        Measurement results for each subcircuit.
    """
    if cut_meta is None:
        # Fallback: simple execution without assignment expansion
        from treewidth_gate_cut.treewidth_cut import run_with_decisions
        return run_with_decisions(subcircuits, cut_decisions, shots)

    # Quantum path: expand logates into all assignment combinations
    from qiskit.qasm3 import loads as qasm3_loads
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from circuit_cutter import (
        replace_all_logates_in_circuit,
        MyQuantumCircuit,
        DataClassSubQCRes,
    )
    from circuit_cutter.MyQuantumCircuit import (
        circuit_identity_key,
        infer_role_from_tags,
        build_layout_from_circuit_spec_assignment,
        build_specs_and_out_order,
        assign_roles_inplace,
        dedup_circuits_with_map,
    )

    subcircuits_myqc = cut_meta["subcircuits_myqc"]
    original = cut_meta["original"]

    # Expand logates into bundles (all assignment combinations)
    bundles = replace_all_logates_in_circuit(subcircuits_myqc)
    bundle_circuits = [c for b in bundles for c in b.circuits]
    assign_roles_inplace(bundle_circuits)

    # Build specs for layout construction
    specs, role_to_spec_index, out_order = build_specs_and_out_order(
        original, bundle_circuits)

    # Execute each unique circuit
    # Build backend from device_info
    if device_info and len(device_info) > 0:
        backend = _build_backend(device_info[0], use_simulator)
    else:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()

    counts_by_key = {}
    for c in bundle_circuits:
        k = circuit_identity_key(c)
        if k in counts_by_key:
            continue
        qasm_str = c.patched_qasm3_for_qiskit()
        qasm_str = MyQuantumCircuit.ensure_rzz_gate_defined_for_qiskit_loads(qasm_str)
        qc = qasm3_loads(qasm_str)
        tqc = transpile(qc, backend)
        job = backend.run(tqc, shots=shots)
        result = job.result()
        try:
            counts = result.get_counts(0)
        except Exception:
            counts = {"0" * qc.num_clbits: shots}
        counts_by_key[k] = {str(key).replace(" ", ""): v for key, v in counts.items()}

    # Build DataClassSubQCRes with assignment and layout
    results = []
    for bundle in bundles:
        for c in bundle.circuits:
            role = getattr(c, "subcircuit_role", None) or infer_role_from_tags(
                getattr(c, "tags", None))
            if role not in role_to_spec_index:
                continue
            si = int(role_to_spec_index[role])
            spec = specs[si]
            layout = build_layout_from_circuit_spec_assignment(
                c, spec=spec, assignment=bundle.assignment)

            cid = circuit_identity_key(c)
            results.append(DataClassSubQCRes(
                job_id="execute",
                subqc_id=f"{role}__{cid}",
                subcircuit_role=role,
                assignment=bundle.assignment,
                counts=counts_by_key.get(cid, {}),
                layout=layout,
                metadata={"circuit_key": cid},
            ))

    return results
