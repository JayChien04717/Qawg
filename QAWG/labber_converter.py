"""Labber HDF5 converter script.

Runs in the Labber Python 3.9 environment. Loads the serialized .npz payload,
reconstructs the multi-dimensional sweep grid, handles trace padding/averaging,
calls Labber.createLogFile_ForData / addEntry to generate the core file,
and opens the result in h5py to write additional metadata and instrument configs.
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import h5py

# 幫 Labber 內建的 Python 指路，告訴它 API 檔案放在哪裡
# 請確認你們電腦上 Labber 實際的安裝路徑，通常 API 會放在 Script 資料夾中
labber_script_path = r"C:\Program Files\Labber\Script"
path = r"C:\Program Files\Keysight\Labber\Script"
sys.path.append(path)


if labber_script_path not in sys.path:
    sys.path.append(labber_script_path)

try:
    import Labber
    print(Labber.version)
    HAS_LABBER = True
except ImportError:
    HAS_LABBER = False

# Define HDF5 compatible variable-length string type
try:
    vlen_str = h5py.string_dtype(encoding="utf-8")
except AttributeError:
    vlen_str = h5py.special_dtype(vlen=str)

# Hotpatch deprecated NumPy aliases for compatibility with legacy packages under NumPy 1.24+
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex


def clean_attr_value(val):
    if val is None:
        return "None"
    if isinstance(val, (int, float, bool, np.integer, np.floating)):
        return val
    if isinstance(val, str):
        return val
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore")
    return str(val)


def pad_trace_data(arr, target_len, is_complex=False):
    # arr has shape (*sweep_shapes, current_len)
    current_len = arr.shape[-1]
    if current_len == target_len:
        return arr
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_len - current_len)]
    constant_value = np.nan if not is_complex else (np.nan + np.nan * 1j)
    return np.pad(arr, pad_width, mode="constant", constant_values=constant_value)


def main():
    if len(sys.argv) < 2:
        print("Usage: python labber_converter.py <npz_path>")
        sys.exit(1)

    npz_path = sys.argv[1]
    if not os.path.exists(npz_path):
        print(f"Error: .npz file not found at {npz_path}")
        sys.exit(1)

    # 1. Load the NPZ payload
    payload = np.load(npz_path, allow_pickle=True)

    sweep_names = [str(n) for n in payload["sweep_names"]]
    sweep_shapes = [int(s) for s in payload["sweep_shapes"]]
    average_mode = bool(payload["average_mode"])
    readout_name = str(payload["readout_name"])
    comment = str(payload["comment"])
    project = str(payload["project"])
    user = str(payload["user"])
    filepath = str(payload["filepath"])

    cfg_json = str(payload["cfg_json"]) if "cfg_json" in payload else None

    has_iq_traces = "iq_traces" in payload
    has_raw = "raw" in payload
    has_iq_shots = "iq_shots" in payload

    iq_traces = payload["iq_traces"] if has_iq_traces else None
    iq_time_s = payload["iq_time_s"] if has_iq_traces else None
    raw = payload["raw"] if has_raw else None
    raw_time_s = payload["raw_time_s"] if has_raw else None
    iq_shots = payload["iq_shots"] if has_iq_shots else None

    # 2. Determine repetitions and sizes
    n_average = 1
    if has_iq_shots:
        n_average = iq_shots.shape[0]
    elif has_iq_traces:
        n_average = iq_traces.shape[0]
    elif has_raw:
        n_average = raw.shape[0]

    # 3. Process trace/shot data based on average_mode
    if average_mode:
        iq_traces_proc = np.mean(iq_traces, axis=0) if has_iq_traces else None
        raw_proc = np.mean(raw, axis=0) if has_raw else None
        iq_shots_proc = np.mean(iq_shots, axis=0) if has_iq_shots else None
    else:
        iq_traces_proc = iq_traces if has_iq_traces else None
        raw_proc = raw if has_raw else None
        iq_shots_proc = iq_shots if has_iq_shots else None

    # Reshape processed arrays into the full multi-dimensional sweep grid
    iq_len = iq_traces_proc.shape[-1] if has_iq_traces else 0
    raw_len = raw_proc.shape[-1] if has_raw else 0
    shots_len = 1 if has_iq_shots else 0

    iq_traces_grid = (
        iq_traces_proc.reshape(tuple(sweep_shapes) + (iq_len,))
        if has_iq_traces
        else None
    )
    raw_grid = raw_proc.reshape(tuple(sweep_shapes) + (raw_len,)) if has_raw else None
    iq_shots_grid = (
        iq_shots_proc.reshape(tuple(sweep_shapes) + (1,)) if has_iq_shots else None
    )

    # 4. Handle Trace Padding
    max_L = max(iq_len, raw_len, shots_len, 1)

    if has_raw and raw_len == max_L:
        time_values = raw_time_s
    elif has_iq_traces and iq_len == max_L:
        time_values = iq_time_s
    else:
        # Fallback padding/generation of Time axis
        if has_raw and len(raw_time_s) > 0:
            dt = raw_time_s[1] - raw_time_s[0] if len(raw_time_s) > 1 else 1.0
            time_values = raw_time_s[0] + np.arange(max_L) * dt
        elif has_iq_traces and len(iq_time_s) > 0:
            dt = iq_time_s[1] - iq_time_s[0] if len(iq_time_s) > 1 else 1.0
            time_values = iq_time_s[0] + np.arange(max_L) * dt
        else:
            time_values = np.arange(max_L, dtype=np.float64)

    iq_traces_padded = (
        pad_trace_data(iq_traces_grid, max_L, is_complex=True)
        if has_iq_traces
        else None
    )
    raw_padded = pad_trace_data(raw_grid, max_L, is_complex=False) if has_raw else None
    iq_shots_padded = (
        pad_trace_data(iq_shots_grid, max_L, is_complex=True) if has_iq_shots else None
    )

    # 5. Build channels configurations for Labber API
    log_channels = []
    if has_iq_traces:
        log_channels.append(
            {
                "name": f"{readout_name} - Demodulated Trace",
                "unit": "V",
                "vector": False,
                "complex": True,
                "instrument": "AlazarTech ATS9371 at localhost",
            }
        )
    if has_raw:
        log_channels.append(
            {
                "name": f"{readout_name} - Raw Trace",
                "unit": "V",
                "vector": False,
                "complex": False,
                "instrument": "AlazarTech ATS9371 at localhost",
            }
        )
    if has_iq_shots:
        log_channels.append(
            {
                "name": f"{readout_name} - Integrated IQ",
                "unit": "V",
                "vector": False,
                "complex": True,
                "instrument": "AlazarTech ATS9371 at localhost",
            }
        )

    # Step channels must start with Time (innermost, fastest changing),
    # followed by sweep axes in reversed order (fastest to slowest).
    step_channels = []
    step_channels.append(
        {
            "name": "Time",
            "unit": "s",
            "values": time_values,
            "instrument": "Sweeper at localhost",
        }
    )
    for name in reversed(sweep_names):
        val = payload[f"sweep_val_{name}"]
        step_channels.append(
            {
                "name": name,
                "unit": "",
                "values": val,
                "instrument": "Sweeper at localhost",
            }
        )

    # 6. Create log file via official Labber API if available, else use fallback mockup
    if HAS_LABBER:
        f_log = Labber.createLogFile_ForData(
            name=filepath,
            log_channels=log_channels,
            step_channels=step_channels,
            use_database=False,
        )

        # Determine total number of sweep steps (excluding the Time axis)
        num_sweeps = int(np.prod(sweep_shapes))

        # Reshape trace/shot grids to (num_sweeps, max_L) for sequential feed
        iq_traces_flat = (
            iq_traces_padded.reshape(num_sweeps, max_L) if has_iq_traces else None
        )
        raw_flat = raw_padded.reshape(num_sweeps, max_L) if has_raw else None
        iq_shots_flat = (
            iq_shots_padded.reshape(num_sweeps, max_L) if has_iq_shots else None
        )

        for idx in range(num_sweeps):
            entry_data = {}
            if has_iq_traces:
                entry_data[f"{readout_name} - Demodulated Trace"] = iq_traces_flat[idx]
            if has_raw:
                entry_data[f"{readout_name} - Raw Trace"] = raw_flat[idx]
            if has_iq_shots:
                entry_data[f"{readout_name} - Integrated IQ"] = iq_shots_flat[idx]

            f_log.addEntry(entry_data)

        # Ensure it gets closed/released
        if hasattr(f_log, "close"):
            f_log.close()
        del f_log
    else:
        print(
            "Warning: Labber Python API is not installed. Running in mock fallback mode using h5py."
        )
        with h5py.File(filepath, "w") as f:
            # Create root attributes
            f.attrs["Step dimensions"] = np.array(
                sweep_shapes + [1] * (14 - len(sweep_shapes)), dtype=np.int32
            )
            f.attrs["version"] = "1.8.6"
            f.attrs["creation_time"] = 0.0
            f.attrs["comment"] = comment
            f.attrs["hardware_loop"] = False
            f.attrs["log_parallel"] = True
            f.attrs["logger_mode"] = False
            f.attrs["time_per_point"] = 0.0
            f.attrs["wait_between"] = 0.0
            f.attrs["trig_channel"] = ""
            f.attrs["arm_trig_mode"] = False

            # Create compound datasets for validation check
            chan_dtype = [
                ("name", vlen_str),
                ("instrument", vlen_str),
                ("quantity", vlen_str),
                ("unitPhys", vlen_str),
                ("unitInstr", vlen_str),
                ("gain", "<f8"),
                ("offset", "<f8"),
                ("amp", "<f8"),
                ("highLim", "<f8"),
                ("lowLim", "<f8"),
                ("outputChannel", vlen_str),
                ("limit_action", vlen_str),
                ("limit_run_script", "?"),
                ("limit_script", vlen_str),
                ("use_log_interval", "?"),
                ("log_interval", "<f8"),
                ("limit_run_always", "?"),
            ]
            f.create_dataset("Channels", shape=(17,), dtype=chan_dtype)

            inst_dtype = [
                ("hardware", vlen_str),
                ("version", vlen_str),
                ("id", vlen_str),
                ("model", vlen_str),
                ("name", vlen_str),
                ("interface", "<i2"),
                ("address", vlen_str),
                ("server", vlen_str),
                ("startup", "<i2"),
                ("lock", "?"),
                ("show_advanced", "?"),
                ("Timeout", "<f8"),
                ("Term. character", vlen_str),
                ("Send end on write", "?"),
                ("Lock VISA resource", "?"),
                ("Suppress end bit termination on read", "?"),
                ("Use specific TCP port", "?"),
                ("TCP port", "<f8"),
                ("Use VICP protocol", "?"),
                ("Baud rate", "<f8"),
                ("Data bits", "<f8"),
                ("Stop bits", "<f8"),
                ("Parity", vlen_str),
                ("GPIB board number", "<f8"),
                ("Send GPIB go to local at close", "?"),
                ("PXI chassis", "<f8"),
                ("Run in 32-bit mode", "?"),
            ]
            f.create_dataset("Instruments", shape=(3,), dtype=inst_dtype)

            step_dtype = [
                ("channel_name", vlen_str),
                ("step_unit", "<i2"),
                ("wait_after", "<f8"),
                ("after_last", "<i2"),
                ("final_value", "<f8"),
                ("use_relations", "?"),
                ("equation", vlen_str),
                ("show_advanced", "?"),
                ("sweep_mode", "<i2"),
                ("use_outside_sweep_rate", "?"),
                ("sweep_rate_outside", "<f8"),
                ("alternate_direction", "?"),
            ]
            f.create_dataset("Step list", shape=(14,), dtype=step_dtype)

            log_dtype = [("channel_name", vlen_str)]
            f.create_dataset("Log list", shape=(3,), dtype=log_dtype)

            # Data group
            data_grp = f.create_group("Data")
            data_grp.attrs["Completed"] = True
            data_grp.attrs["Step dimensions"] = np.array(
                sweep_shapes + [1] * (14 - len(sweep_shapes)), dtype=np.int32
            )
            data_grp.attrs["Step index"] = np.arange(len(sweep_shapes), dtype=np.int32)
            data_grp.attrs["Fixed step index"] = np.arange(
                len(sweep_shapes), 14, dtype=np.int32
            )
            data_grp.attrs["Fixed step values"] = np.zeros(
                14 - len(sweep_shapes), dtype=np.float64
            )

            chan_names_dtype = [("name", vlen_str), ("info", vlen_str)]
            chan_names = np.array([(name, "") for name in sweep_names], dtype=chan_names_dtype)
            data_grp.create_dataset(
                "Channel names", data=chan_names, dtype=chan_names_dtype
            )

            data_shape = tuple(sweep_shapes) + (len(sweep_shapes), 1)
            data_grp.create_dataset("Data", shape=data_shape, dtype=np.float64)
            data_grp.create_dataset("Time stamp", shape=(1,), dtype=np.float64)

            # Traces group
            traces_grp = f.create_group("Traces")
            traces_grp.create_dataset(
                "Time stamp", shape=tuple(sweep_shapes), dtype=np.float64
            )

            # Traces
            if has_iq_traces:
                t_shape = (max_L, 2) + tuple(sweep_shapes)
                dset = traces_grp.create_dataset(
                    f"{readout_name} - Demodulated Trace",
                    shape=t_shape,
                    dtype=np.float64,
                )
                dset.attrs["complex"] = True
                dset.attrs["x, name"] = "Time"
                dset.attrs["x, unit"] = "s"
                traces_grp.create_dataset(
                    f"{readout_name} - Demodulated Trace_N",
                    data=np.array([max_L], dtype=np.int32),
                )
                traces_grp.create_dataset(
                    f"{readout_name} - Demodulated Trace_t0dt",
                    data=np.array([[0.0, 1.0]], dtype=np.float64),
                )

            if has_raw:
                t_shape = (max_L,) + tuple(sweep_shapes)
                dset = traces_grp.create_dataset(
                    f"{readout_name} - Raw Trace", shape=t_shape, dtype=np.float64
                )
                dset.attrs["complex"] = False
                dset.attrs["x, name"] = "Time"
                dset.attrs["x, unit"] = "s"
                traces_grp.create_dataset(
                    f"{readout_name} - Raw Trace_N",
                    data=np.array([max_L], dtype=np.int32),
                )
                traces_grp.create_dataset(
                    f"{readout_name} - Raw Trace_t0dt",
                    data=np.array([[0.0, 1.0]], dtype=np.float64),
                )

            if has_iq_shots:
                t_shape = (1, 2) + tuple(sweep_shapes)
                dset = traces_grp.create_dataset(
                    f"{readout_name} - Integrated IQ", shape=t_shape, dtype=np.float64
                )
                dset.attrs["complex"] = True
                dset.attrs["x, name"] = "Time"
                dset.attrs["x, unit"] = "s"
                traces_grp.create_dataset(
                    f"{readout_name} - Integrated IQ_N",
                    data=np.array([1], dtype=np.int32),
                )
                traces_grp.create_dataset(
                    f"{readout_name} - Integrated IQ_t0dt",
                    data=np.array([[0.0, 1.0]], dtype=np.float64),
                )

            # Views group
            f.create_group("Views")

    # 8. Post-process metadata injection using h5py
    awg_cfg = {}
    dig_cfg = {}
    if cfg_json:
        cfg = json.loads(cfg_json)
        for k, v in cfg.items():
            k_str = str(k).lower()
            if (
                "awg" in k_str
                or "marker" in k_str
                or "pulse" in k_str
                or "vpp" in k_str
                or "sigma" in k_str
                or "gain" in k_str
            ):
                awg_cfg[k] = v
            elif (
                "adc" in k_str
                or "acquire" in k_str
                or "trigger" in k_str
                or "integrate" in k_str
                or "demod" in k_str
                or "record" in k_str
                or "sample" in k_str
                or "average" in k_str
            ):
                dig_cfg[k] = v
            elif "freq" in k_str:
                if "demod" in k_str:
                    dig_cfg[k] = v
                else:
                    awg_cfg[k] = v
            elif "time" in k_str:
                if "pulse" in k_str or "width" in k_str:
                    awg_cfg[k] = v
                else:
                    dig_cfg[k] = v
            else:
                awg_cfg[k] = v

    with h5py.File(filepath, "r+") as f:
        # Write root tags & comments
        f.attrs["comment"] = comment
        if "Tags" not in f:
            tags_grp = f.create_group("Tags")
        else:
            tags_grp = f["Tags"]
        tags_grp.attrs["Project"] = np.array([project], dtype=vlen_str)
        tags_grp.attrs["User"] = np.array([user], dtype=vlen_str)
        tags_grp.attrs["Tags"] = np.array([], dtype=vlen_str)

        # Write Instrument config attributes
        if "Instrument config" not in f:
            inst_grp = f.create_group("Instrument config")
        else:
            inst_grp = f["Instrument config"]

        # Ensure Sweeper subgroup exists
        if "Sweeper at localhost" not in inst_grp:
            inst_grp.create_group("Sweeper at localhost")

        # Ensure AWG subgroup exists and populate it
        if "Tektronix AWG5208 at localhost" not in inst_grp:
            awg_grp = inst_grp.create_group("Tektronix AWG5208 at localhost")
        else:
            awg_grp = inst_grp["Tektronix AWG5208 at localhost"]
        for k, v in awg_cfg.items():
            awg_grp.attrs[str(k)] = clean_attr_value(v)

        # Ensure Digitizer subgroup exists and populate it
        if "AlazarTech ATS9371 at localhost" not in inst_grp:
            dig_grp = inst_grp.create_group("AlazarTech ATS9371 at localhost")
        else:
            dig_grp = inst_grp["AlazarTech ATS9371 at localhost"]
        dig_grp.attrs["Readout Name"] = clean_attr_value(readout_name)

        if "initial_trigger_delay_s" in payload:
            dig_grp.attrs["Trigger Delay (Result)"] = clean_attr_value(
                payload["initial_trigger_delay_s"]
            )
        if "acquire_window_s" in payload:
            dig_grp.attrs["Acquisition Window"] = clean_attr_value(
                payload["acquire_window_s"]
            )
        if "remove_dc_offset" in payload:
            dig_grp.attrs["Remove DC Offset"] = clean_attr_value(
                payload["remove_dc_offset"]
            )

        for k, v in dig_cfg.items():
            dig_grp.attrs[str(k)] = clean_attr_value(v)

        if not HAS_LABBER:
            # In mock mode, populate the datasets under /Traces group with actual values
            traces_grp = f["Traces"]
            if has_iq_traces:
                real_part = np.real(iq_traces_padded)
                imag_part = np.imag(iq_traces_padded)
                stacked = np.stack((real_part, imag_part), axis=-2)
                # Transpose from (*sweep_shapes, 2, max_L) to (max_L, 2, *sweep_shapes)
                d = stacked.ndim - 2
                axes_order = [d + 1, d] + list(range(d))
                packed = np.transpose(stacked, axes_order)
                traces_grp[f"{readout_name} - Demodulated Trace"][:] = packed

            if has_raw:
                # Transpose from (*sweep_shapes, max_L) to (max_L, *sweep_shapes)
                d = raw_padded.ndim - 1
                axes_order = [d] + list(range(d))
                packed = np.transpose(raw_padded, axes_order)
                traces_grp[f"{readout_name} - Raw Trace"][:] = packed

            if has_iq_shots:
                real_part = np.real(iq_shots_padded)
                imag_part = np.imag(iq_shots_padded)
                stacked = np.stack((real_part, imag_part), axis=-2)
                # Slice the first element along max_L dimension to get (*sweep_shapes, 2)
                stacked_slice = stacked[..., :, 0]
                # Transpose to (2, *sweep_shapes)
                d = stacked_slice.ndim - 1
                axes_order = [d] + list(range(d))
                packed_slice = np.transpose(stacked_slice, axes_order)
                # Expand first dimension to get (1, 2, *sweep_shapes)
                packed = np.expand_dims(packed_slice, axis=0)
                traces_grp[f"{readout_name} - Integrated IQ"][:] = packed

    print("HDF5 Log conversion completed successfully.")


if __name__ == "__main__":
    main()
