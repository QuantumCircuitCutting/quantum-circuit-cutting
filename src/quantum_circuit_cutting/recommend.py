"""Feature 3: Device Recommendation - Select optimal devices for subcircuits."""


def recommend(subcircuit_params, available_devices, shots=8192):
    """
    Recommend optimal quantum devices for each subcircuit.

    Parameters
    ----------
    subcircuit_params : list
        List of subcircuit parameter dicts or DataClassSubQCParams.
    available_devices : list[dict]
        Available devices, each with keys:
        name, n_qubit, coupling_map, gate_speed, etc.
    shots : int
        Number of shots per subcircuit.

    Returns
    -------
    list[dict]
        List of device dicts (one per subcircuit, in the same order).
        Each dict contains the full device info from available_devices.
    """
    from device_recommender import match_subqc_to_device

    subqc_lst = []
    for p in subcircuit_params:
        if hasattr(p, "subqc_id"):
            subqc_lst.append({
                "name": p.subqc_id,
                "qasm": p.qasm3,
                "nshot": shots,
            })
        elif isinstance(p, dict):
            subqc_lst.append(p)
        else:
            raise TypeError(f"Unsupported subcircuit type: {type(p)}")

    # Remove custom cut gate lines and ensure rzz gate is defined
    import re
    from circuit_cutter import MyQuantumCircuit
    for entry in subqc_lst:
        if "qasm" in entry and entry["qasm"]:
            lines = entry["qasm"].split("\n")
            cleaned = [l for l in lines if not re.match(r"\s*(gate_cut_|wire_cut_)", l)]
            entry["qasm"] = "\n".join(cleaned)
            entry["qasm"] = MyQuantumCircuit.ensure_rzz_gate_defined_for_qiskit_loads(entry["qasm"])

    recommendation = match_subqc_to_device(subqc_lst, available_devices)

    device_name_map = {d["name"]: d for d in available_devices}
    device_info = []
    for p in subqc_lst:
        rec = recommendation.get(p["name"], {})
        device_name = rec.get("device")
        if device_name and device_name in device_name_map:
            device_info.append(device_name_map[device_name])
        elif available_devices:
            device_info.append(available_devices[0])
        else:
            name = p["name"]
            raise ValueError(f"No device found for {name}")

    return device_info
