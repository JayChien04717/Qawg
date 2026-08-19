"""Refactored QAWG data logging module.

Serializes experiment results to a temporary .npz file and triggers a sub-process
running in the Labber Python 3.9 environment to create the HDF5 using the official Labber API.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any
import numpy as np


def write_result_to_hdf5(
    result: Any,
    filepath: str,
    comment: str = "",
    project: str = "",
    user: str = "",
    average_mode: bool = True,
    cfg: dict[str, Any] | None = None,
    rename_sweeps: dict[str, str] | None = None,
) -> None:
    """Serialize the ExperimentResult to .npz and invoke labber_converter.py.

    Parameters
    ----------
    result : ExperimentResult
        The experiment result containing acquired traces and sweep axes.
    filepath : str
        The path where the HDF5 file will be written.
    comment : str, optional
        A user comment to attach to the log.
    project : str, optional
        The project name for tagging.
    user : str, optional
        The user name for tagging.
    average_mode : bool, optional
        If True, averages the traces and shots over repetitions.
        If False, treats repetitions as the first sweep dimension.
    cfg : dict, optional
        The experiment configuration settings to record in HDF5 config metadata.
    rename_sweeps : dict, optional
        A dictionary mapping original sweep axis names to new names.
    """
    # 1. Determine repetitions and steps
    n_average = 1
    if hasattr(result, "iq_shots") and result.iq_shots is not None and result.iq_shots.size > 0:
        n_average = result.iq_shots.shape[0]
    elif hasattr(result, "iq_traces") and result.iq_traces is not None and result.iq_traces.size > 0:
        n_average = result.iq_traces.shape[0]
    elif hasattr(result, "raw") and result.raw is not None and result.raw.size > 0:
        n_average = result.raw.shape[0]

    orig_sweep_names = list(result.axes.keys())
    if rename_sweeps:
        orig_sweep_names = [rename_sweeps.get(name, name) for name in orig_sweep_names]
    orig_sweep_values = [np.asarray(val) for val in result.axes.values()]
    orig_sweep_shapes = [len(val) for val in orig_sweep_values]

    # Build sweep dimensions based on mode
    if not average_mode:
        sweep_names = ["Repetition"] + orig_sweep_names
        sweep_values = [np.arange(n_average, dtype=np.float64)] + orig_sweep_values
        sweep_shapes = [n_average] + orig_sweep_shapes
    else:
        sweep_names = orig_sweep_names
        sweep_values = orig_sweep_values
        sweep_shapes = orig_sweep_shapes

    # 2. Collect payload data
    payload: dict[str, Any] = {
        "sweep_names": np.array(sweep_names, dtype=object),
        "sweep_shapes": np.array(sweep_shapes, dtype=np.int32),
        "average_mode": average_mode,
        "readout_name": getattr(result, "readout_name", "ro"),
        "comment": comment,
        "project": project,
        "user": user,
        "filepath": filepath,
    }

    # Save individual sweep arrays
    for name, val in zip(sweep_names, sweep_values):
        payload[f"sweep_val_{name}"] = val

    # Save traces/shots if present and not empty
    has_iq_traces = (
        hasattr(result, "iq_traces")
        and result.iq_traces is not None
        and result.iq_traces.size > 0
    )
    has_raw = hasattr(result, "raw") and result.raw is not None and result.raw.size > 0
    has_iq_shots = (
        hasattr(result, "iq_shots")
        and result.iq_shots is not None
        and result.iq_shots.size > 0
        and not np.all(np.isnan(result.iq_shots))
    )

    if has_iq_traces:
        payload["iq_traces"] = result.iq_traces
        payload["iq_time_s"] = result.iq_time_s
    if has_raw:
        payload["raw"] = result.raw
        payload["raw_time_s"] = result.raw_time_s
    if has_iq_shots:
        payload["iq_shots"] = result.iq_shots

    # Save extra metadata attributes if present
    if (
        hasattr(result, "initial_trigger_delay_s")
        and result.initial_trigger_delay_s is not None
    ):
        payload["initial_trigger_delay_s"] = result.initial_trigger_delay_s
    if hasattr(result, "acquire_window_s") and result.acquire_window_s is not None:
        payload["acquire_window_s"] = result.acquire_window_s
    if hasattr(result, "remove_dc_offset") and result.remove_dc_offset is not None:
        payload["remove_dc_offset"] = result.remove_dc_offset

    # Save configuration dictionary if present
    if cfg is not None and isinstance(cfg, dict):
        import json

        payload["cfg_json"] = json.dumps(cfg)

    # 3. Write to temporary .npz file
    npz_path = os.path.join(os.path.dirname(filepath), "temp_measure_data.npz")
    np.savez(npz_path, **payload)

    # 4. Trigger the subprocess converter script
    # Look for labber_converter.py in the same folder as this script
    converter_script = os.path.join(os.path.dirname(__file__), "labber_converter.py")

    # Locate Labber's Python 3.9 interpreter from environment variable or standard path
    python_exe = os.environ.get("LABBER_PYTHON")
    if python_exe is None:
        standard_paths = [
            r"C:\Users\cluster\anaconda3\envs\hoiqel\python.exe",
            r"C:\Program Files (x86)\Keysight\Labber\python-labber\python.exe",
            r"C:\Program Files (x86)\Keysight\Labber\python-labber\python.exe",
        ]
        for p in standard_paths:
            if os.path.exists(p):
                python_exe = p
                break
    if python_exe is None:
        python_exe = sys.executable  # fallback to active environment
        print(
            "Warning: Labber Python interpreter not found. Using current Python environment."
        )
    else:
        print(f"Using Labber Python interpreter at: {python_exe}")

    # Invoke converter subprocess
    cmd = [python_exe, converter_script, npz_path]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)

        print(f"Labber HDF5 converter output:\n{res.stdout}")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Labber HDF5 converter failed with exit code {exc.returncode}.\n"
            f"Stdout: {exc.stdout}\n"
            f"Stderr: {exc.stderr}"
        ) from exc
    finally:
        # Clean up the temporary file
        if os.path.exists(npz_path):
            os.remove(npz_path)
