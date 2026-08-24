import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np

_STEP_NAME_API = "Step index API"
vlen_bytes = h5py.special_dtype(vlen=bytes)

# HDF5 Enum datatypes matching official Labber SDK representation
interface_enum = h5py.special_dtype(
    enum=(
        np.int16,
        {
            "GPIB": 0,
            "None": 7,
            "Other": 6,
            "PXI": 3,
            "Serial": 4,
            "TCPIP": 1,
            "USB": 2,
            "VISA": 5,
        },
    )
)
startup_enum = h5py.special_dtype(
    enum=(np.int16, {"Do nothing": 2, "Get config": 1, "Set config": 0})
)
range_type_enum = h5py.special_dtype(
    enum=(np.int16, {"Center - Span": 2, "Single": 0, "Start - Stop": 1})
)
step_type_enum = h5py.special_dtype(
    enum=(np.int16, {"Fixed # of pts": 1, "Fixed step": 0})
)
interp_enum = h5py.special_dtype(
    enum=(np.int16, {"Linear": 0, "Log": 1, "Log, #/decade": 2, "Lorentzian": 3})
)
step_unit_enum = h5py.special_dtype(enum=(np.int16, {"Instrument": 0, "Physical": 1}))
after_last_enum = h5py.special_dtype(
    enum=(np.int16, {"Goto first point": 0, "Goto value...": 2, "Stay at final": 1})
)
sweep_mode_enum = h5py.special_dtype(
    enum=(np.int16, {"Between points": 1, "Continuous": 2, "Off": 0})
)


def _get_database_folder():
    """Return database folder.

    Returns
    -------
    Any
        Result of the operation.
    """
    appdata = os.environ.get("APPDATA")
    if appdata:
        prefs_path = os.path.join(appdata, "Labber", "LabberPrefs.json")
        if os.path.exists(prefs_path):
            try:
                with open(prefs_path, "r", encoding="utf-8") as f:
                    prefs = json.load(f)
                    db_folder = prefs.get("Database folder")
                    if db_folder:
                        return db_folder
            except Exception:
                pass
    return os.path.expanduser("~/Labber/Data")


def _create_log_path(
    sLogName,
    dateObj=None,
    logger_mode=False,
    bConfigFile=False,
    bCreatePath=True,
    database_folder=None,
):
    """Create log path.

    Parameters
    ----------
    sLogName : Any
        Value for ``sLogName``.
    dateObj : Any, default: None
        Value for ``dateObj``.
    logger_mode : Any, default: False
        Value for ``logger_mode``.
    bConfigFile : Any, default: False
        Value for ``bConfigFile``.
    bCreatePath : Any, default: True
        Value for ``bCreatePath``.

    Returns
    -------
    Any
        Result of the operation.
    """
    if dateObj is None:
        import datetime

        dateObj = datetime.datetime.today()

    preferences_db = (
        _get_database_folder()
        if database_folder is None
        else os.fspath(database_folder)
    )

    sYear = dateObj.strftime("%Y")
    sMonth = dateObj.strftime("%m")
    sDay = dateObj.strftime("Data_%m%d")

    sPath = os.path.join(preferences_db, sYear, sMonth, sDay)
    if bCreatePath:
        os.makedirs(sPath, exist_ok=True)

    return os.path.join(sPath, sLogName + ".hdf5")


def getTraceDict(value=[], x0=0.0, dx=1.0, x1=None, logX=False, x=None):
    """Create a Labber-compatible dictionary describing an ``(x, y)`` trace.

    Parameters
    ----------
    value : Any, default: []
        Dependent-variable values.
    x0 : Any, default: 0.0
        Value for ``x0``.
    dx : Any, default: 1.0
        Value for ``dx``.
    x1 : Any, default: None
        Value for ``x1``.
    logX : bool, default: False
        Whether to interpolate logarithmically between ``x0`` and ``x1``.
    x : Any, default: None
        Explicit independent-variable values.

    Returns
    -------
    Any
        Result of the operation.
    """
    y = np.asarray(value)
    d = {"y": y}
    if x is not None:
        d["x"] = np.asarray(x)
        d["t0"] = 0.0
        d["dt"] = 1.0
    elif x1 is not None and logX:
        d["t0"] = x0
        d["dt"] = dx
        d["x"] = np.geomspace(x0, x1, len(y))
    else:
        if x1 is not None and len(y) > 1:
            dx = (x1 - x0) / (len(y) - 1)
        d["t0"] = x0
        d["dt"] = dx
    return d


def _to_bytes(s):
    """Convert the value to bytes.

    Parameters
    ----------
    s : Any
        Value for ``s``.

    Returns
    -------
    Any
        Result of the operation.
    """
    if isinstance(s, str):
        return s.encode("utf-8")
    return s


def _to_str(s):
    """Convert the value to str.

    Parameters
    ----------
    s : Any
        Value for ``s``.

    Returns
    -------
    Any
        Result of the operation.
    """
    if isinstance(s, bytes):
        return s.decode("utf-8")
    return s


class LogFile(object):
    def __init__(self, file_name, instrument_units=False):
        """Initialize the LogFile instance.

        Parameters
        ----------
        file_name : Any
            Name of the file.
        """
        base, ext = os.path.splitext(file_name)
        if ext.lower() != ".hdf5":
            file_name = base + ".hdf5"
        self.file_name = os.path.abspath(file_name)
        self.instrument_units = bool(instrument_units)

    @staticmethod
    def _log_group_names(h5):
        names = [""]
        numbered = [name for name in h5 if name.startswith("Log_")]
        names.extend(sorted(numbered, key=lambda name: int(name.split("_", 1)[1])))
        return names

    def _select_log_group(self, h5, log=-1):
        names = self._log_group_names(h5)
        if log is None:
            log = -1
        try:
            name = names[int(log)]
        except (IndexError, TypeError, ValueError) as exc:
            raise IndexError(f"Log index {log!r} is out of range") from exc
        return h5 if name == "" else h5[name]

    @staticmethod
    def _channel_name(channels, channel, *, kind):
        if not channels:
            raise ValueError(f"No {kind} channels found in log file.")
        if channel is None:
            return channels[0]["name"]
        if isinstance(channel, (int, np.integer)):
            try:
                return channels[int(channel)]["name"]
            except IndexError as exc:
                raise ValueError(f"{kind.title()} channel index {channel} not found.") from exc
        name = _to_str(channel)
        if name not in {item["name"] for item in channels}:
            raise ValueError(f"{kind.title()} channel {name!r} not found.")
        return name

    def getFilePath(self):
        """Return the getFilePath result.

        Returns
        -------
        Any
            Result of the operation.
        """
        return self.file_name

    def setComment(self, comment, log=-1, set_all=True):
        """Return the setComment result.

        Parameters
        ----------
        comment : Any
            Value for ``comment``.
        """
        with h5py.File(self.file_name, "r+") as f:
            if set_all:
                groups = [
                    f if name == "" else f[name] for name in self._log_group_names(f)
                ]
            else:
                groups = [self._select_log_group(f, log)]
            for grp in groups:
                grp.attrs["comment"] = _to_bytes(comment)

    def getComment(self, log=-1):
        """Return the getComment result.

        Returns
        -------
        Any
            Result of the operation.
        """
        with h5py.File(self.file_name, "r") as f:
            grp = self._select_log_group(f, log)
            return _to_str(grp.attrs.get("comment", f.attrs.get("comment", b"")))

    def setProject(self, project):
        """Return the setProject result.

        Parameters
        ----------
        project : Any
            Value for ``project``.
        """
        with h5py.File(self.file_name, "r+") as f:
            if "Tags" in f:
                f["Tags"].attrs["Project"] = np.array(
                    [_to_bytes(project)], dtype=object
                )

    def getProject(self):
        """Return the getProject result.

        Returns
        -------
        Any
            Result of the operation.
        """
        with h5py.File(self.file_name, "r") as f:
            if "Tags" in f:
                proj = f["Tags"].attrs.get("Project", [b""])
                if len(proj) > 0:
                    return _to_str(proj[0])
            return ""

    def setUser(self, user):
        """Return the setUser result.

        Parameters
        ----------
        user : Any
            Value for ``user``.
        """
        with h5py.File(self.file_name, "r+") as f:
            if "Tags" in f:
                f["Tags"].attrs["User"] = np.array([_to_bytes(user)], dtype=object)

    def getUser(self):
        """Return the getUser result.

        Returns
        -------
        Any
            Result of the operation.
        """
        with h5py.File(self.file_name, "r") as f:
            if "Tags" in f:
                usr = f["Tags"].attrs.get("User", [b""])
                if len(usr) > 0:
                    return _to_str(usr[0])
            return ""

    def setTags(self, tags):
        """Return the setTags result.

        Parameters
        ----------
        tags : Any
            Value for ``tags``.
        """
        if tags is None:
            tags = []
        elif isinstance(tags, (str, bytes)):
            tags = [tags]

        with h5py.File(self.file_name, "r+") as f:
            if "Tags" in f:
                if len(tags) == 0:
                    f["Tags"].attrs["Tags"] = np.array([])
                else:
                    f["Tags"].attrs["Tags"] = np.array(
                        [_to_bytes(t) for t in tags], dtype=object
                    )

    def getTags(self):
        """Return the getTags result.

        Returns
        -------
        Any
            Result of the operation.
        """
        with h5py.File(self.file_name, "r") as f:
            if "Tags" in f:
                tags = f["Tags"].attrs.get("Tags", [])
                return [_to_str(t) for t in tags]
            return []

    def getNumberOfEntries(self, name=None, log=None):
        # In a completed or in-progress file, this corresponds to the size of Data/Time stamp
        # (or Log_N/Data/Time stamp).
        """Return the getNumberOfEntries result.

        Returns
        -------
        Any
            Result of the operation.
        """
        del name  # Channels written here always share the same entry count.
        with h5py.File(self.file_name, "r") as f:
            if log is None:
                groups = [
                    f if item == "" else f[item]
                    for item in self._log_group_names(f)
                ]
            else:
                groups = [self._select_log_group(f, log)]
            total = 0
            for grp in groups:
                if "Data/Time stamp" in grp:
                    total += grp["Data/Time stamp"].shape[0]
                elif "Traces/Time stamp" in grp:
                    total += grp["Traces/Time stamp"].shape[0]
            return total

    def getStepChannels(self):
        """Return the getStepChannels result.

        Returns
        -------
        Any
            Result of the operation.
        """
        with h5py.File(self.file_name, "r") as f:
            channels = []
            if "Channels" in f:
                for row in f["Channels"]:
                    name = _to_str(row["name"])
                    if name == _STEP_NAME_API:
                        continue
                    is_step = False
                    if "Step list" in f:
                        for step_row in f["Step list"]:
                            if _to_str(step_row["channel_name"]) == name:
                                is_step = True
                                break
                    if is_step:
                        values = np.array([])
                        cfg_path = f"Step config/{name}/Step items"
                        if cfg_path in f:
                            step_item = f[cfg_path][0]
                            # range_type: 1 (STARTSTOP), 0 (SINGLE)
                            # step_type: 1 (N_PTS)
                            range_type = step_item["range_type"]
                            if range_type == 1:
                                values = np.linspace(
                                    step_item["start"],
                                    step_item["stop"],
                                    step_item["n_pts"],
                                )
                            elif range_type == 0:
                                values = np.array([step_item["single"]])

                        channels.append(
                            {
                                "name": name,
                                "unit": _to_str(
                                    row["unitInstr"]
                                    if self.instrument_units
                                    else row["unitPhys"]
                                ),
                                "values": values,
                                "complex": False,  # step channels are always real
                                "vector": False,  # step channels are always scalar
                            }
                        )
            return channels

    def getAxis(self):
        """Return all sweep axes in Labber step-channel order.

        Returns
        -------
        list of dict
            Each dictionary contains ``axis_name`` and ``axis_values``.  The
            values are returned as independent NumPy arrays so callers can
            modify them without changing the channel metadata.
        """
        return [
            {
                "axis_name": channel["name"],
                "axis_values": np.array(channel["values"], copy=True),
            }
            for channel in self.getStepChannels()
        ]

    def getLogChannels(self):
        """Return the getLogChannels result.

        Returns
        -------
        Any
            Result of the operation.
        """
        with h5py.File(self.file_name, "r") as f:
            channels = []
            if "Channels" in f:
                log_names = set()
                if "Log list" in f:
                    for log_row in f["Log list"]:
                        log_names.add(_to_str(log_row["channel_name"]))

                for row in f["Channels"]:
                    name = _to_str(row["name"])
                    if name in log_names:
                        is_vector = False
                        is_complex = False
                        if "Traces" in f and name in f["Traces"]:
                            is_vector = True
                            is_complex = bool(
                                f["Traces"][name].attrs.get("complex", False)
                            )
                        else:
                            if "Data/Channel names" in f:
                                infos = [
                                    (_to_str(r["name"]), _to_str(r["info"]))
                                    for r in f["Data/Channel names"]
                                ]
                                name_infos = [
                                    info for name_str, info in infos if name_str == name
                                ]
                                if "Real" in name_infos or "Imaginary" in name_infos:
                                    is_complex = True

                        channels.append(
                            {
                                "name": name,
                                "unit": _to_str(
                                    row["unitInstr"]
                                    if self.instrument_units
                                    else row["unitPhys"]
                                ),
                                "complex": is_complex,
                                "vector": is_vector,
                            }
                        )
            return channels

    def getData(self, name=None, entry=None, inner=None, log=-1):
        """Retrieve channel data using the public Labber ``LogFile`` API.

        ``name`` is optional, matching Labber: omitting it selects the first
        log channel.  Vector traces are returned at their stored maximum
        length, with shorter entries padded by the HDF5 dataset fill value.
        """
        channel_name = self._channel_name(self.getLogChannels(), name, kind="log")
        with h5py.File(self.file_name, "r") as f:
            grp = self._select_log_group(f, log)
            values = None

            if "Data/Channel names" in grp and "Data/Data" in grp:
                chns = [
                    (_to_str(row["name"]), _to_str(row["info"]))
                    for row in grp["Data/Channel names"]
                ]
                indices = [idx for idx, item in enumerate(chns) if item[0] == channel_name]
                if len(indices) == 1:
                    values = grp["Data/Data"][:, indices[0], :].T
                elif len(indices) == 2:
                    real_idx = next(idx for idx in indices if chns[idx][1] == "Real")
                    imag_idx = next(idx for idx in indices if chns[idx][1] == "Imaginary")
                    ds = grp["Data/Data"]
                    values = ds[:, real_idx, :].T + 1j * ds[:, imag_idx, :].T

            if values is None and "Traces" in grp and channel_name in grp["Traces"]:
                ds = grp[f"Traces/{channel_name}"]
                is_complex = bool(ds.attrs.get("complex", False))
                if is_complex:
                    values = ds[:, 0, :].T + 1j * ds[:, 1, :].T
                elif ds.shape[1] == 1:
                    values = ds[:, 0, :].T
                else:
                    values = ds[:, 1, :].T

            if values is None:
                raise ValueError(f"Channel {channel_name} not found in log file.")
            if entry is not None:
                values = values[entry]
            if inner is not None:
                values = values[..., inner]
            return values

    def getTraceXY(self, y_channel=None, x_channel=None, entry=-1):
        """Retrieve one ``(x, y)`` trace, defaulting to the first log channel."""
        channel_name = self._channel_name(
            self.getLogChannels(), y_channel, kind="log"
        )
        with h5py.File(self.file_name, "r") as f:
            grp = self._select_log_group(f, -1)
            if "Traces" in grp and channel_name in grp["Traces"]:
                ds = grp[f"Traces/{channel_name}"]
                n_values = np.asarray(grp[f"Traces/{channel_name}_N"])
                n_index = entry if len(n_values) > 1 else 0
                trace_len = int(n_values[n_index])
                is_complex = bool(ds.attrs.get("complex", False))
                components = ds.shape[1]
                has_custom_x = (components == 2 and not is_complex) or (
                    components == 3 and is_complex
                )

                if is_complex:
                    y = ds[:trace_len, 0, entry] + 1j * ds[:trace_len, 1, entry]
                elif components == 1:
                    y = ds[:trace_len, 0, entry]
                else:
                    y = ds[:trace_len, 1, entry]

                if has_custom_x:
                    x_component = 2 if is_complex else 0
                    x = ds[:trace_len, x_component, entry]
                else:
                    t0dt_path = f"Traces/{channel_name}_t0dt"
                    if t0dt_path in grp:
                        t0dt = grp[t0dt_path]
                        t0dt_index = entry if len(t0dt) > 1 else 0
                        t0, dt = t0dt[t0dt_index]
                    else:
                        t0, dt = 0.0, 1.0
                    x = t0 + np.arange(trace_len) * dt
                return x, y

        y = np.asarray(self.getData(channel_name, entry=entry, log=-1))
        steps = self.getStepChannels()
        step_name = self._channel_name(steps, x_channel, kind="step")
        step_index = next(idx for idx, item in enumerate(steps) if item["name"] == step_name)
        step_values = np.asarray(steps[step_index]["values"])
        if step_index == 0:
            x = step_values
        else:
            outer_dims = [len(item["values"]) for item in steps[1:]]
            flat_entry = entry if entry >= 0 else int(np.prod(outer_dims)) + entry
            multi_index = np.unravel_index(flat_entry, outer_dims)
            x = np.full(y.shape, step_values[multi_index[step_index - 1]])
        return x, y

    def addEntry(self, data):
        """Return the addEntry result.

        Parameters
        ----------
        data : Any
            Input data to process.
        """
        with h5py.File(self.file_name, "r+") as f:
            step_dims = list(f.attrs["Step dimensions"])
            if len(step_dims) == 0:
                step_dims = [1]
            dim1 = step_dims[0]
            M = np.prod(step_dims[1:]) if len(step_dims) > 1 else 1

            latest_grp_name = ""
            for key in sorted(f.keys()):
                if key.startswith("Log_"):
                    latest_grp_name = key

            grp = f
            if latest_grp_name:
                grp = f[latest_grp_name]

            col = 0
            if "Data/Time stamp" in grp:
                col = grp["Data/Time stamp"].shape[0]
            elif "Traces/Time stamp" in grp:
                col = grp["Traces/Time stamp"].shape[0]

            if col >= M:
                if latest_grp_name:
                    N = int(latest_grp_name.split("_")[1]) + 1
                else:
                    N = 2

                new_grp_name = f"Log_{N}"
                new_grp = f.create_group(new_grp_name)

                for attr_key, attr_val in f.attrs.items():
                    new_grp.attrs[attr_key] = attr_val

                for key in [
                    "Channels",
                    "Instrument config",
                    "Instruments",
                    "Log list",
                    "Step config",
                    "Step list",
                    "Tags",
                ]:
                    if key in f:
                        f.copy(key, f"{new_grp_name}/{key}")

                grp = new_grp
                col = 0

            if "Data" not in grp:
                step_names = []
                if "Step list" in f:
                    for step_row in f["Step list"]:
                        step_names.append(_to_str(step_row["channel_name"]))

                log_names = []
                if "Log list" in f:
                    for log_row in f["Log list"]:
                        log_names.append(_to_str(log_row["channel_name"]))

                channels_dict = {}
                if "Channels" in f:
                    for row in f["Channels"]:
                        name = _to_str(row["name"])
                        channels_dict[name] = {
                            "complex": False,  # Default
                            "vector": False,
                        }
                scalar_chns = []
                for name in step_names:
                    scalar_chns.append((name, ""))

                for name in log_names:
                    val = data.get(name)
                    is_vector = False
                    is_complex = False
                    if val is not None:
                        if isinstance(val, dict) and ("y" in val):
                            is_vector = True
                            if np.iscomplexobj(val["y"]):
                                is_complex = True
                        elif isinstance(val, (list, np.ndarray)) and not isinstance(
                            val, (str, bytes)
                        ):
                            pass

                    inst_grp_path = (
                        "Instrument config/Generic - GPIB: , Log channels at localhost"
                    )
                    if inst_grp_path in grp:
                        inst_grp = grp[inst_grp_path]
                        if f"___{name}___x_name" in inst_grp.attrs:
                            is_vector = True
                        if name in inst_grp.attrs:
                            default_val = inst_grp.attrs[name]
                            if isinstance(
                                default_val, (complex, np.complexfloating)
                            ) or np.iscomplexobj(default_val):
                                is_complex = True

                    if not is_vector:
                        if is_complex:
                            scalar_chns.append((name, "Real"))
                            scalar_chns.append((name, "Imaginary"))
                        else:
                            scalar_chns.append((name, ""))

                data_grp = grp.create_group("Data")
                data_grp.attrs["Completed"] = False
                data_grp.attrs["Step dimensions"] = step_dims
                data_grp.attrs["Step index"] = np.arange(len(step_dims), dtype=int)
                data_grp.attrs["Fixed step index"] = np.array([], dtype=int)
                data_grp.attrs["Fixed step values"] = np.array([], dtype=float)
                if len(step_dims) > 1:
                    data_grp.attrs["Entries, last trace"] = dim1

                dtype_chn_names = np.dtype([("name", vlen_bytes), ("info", vlen_bytes)])
                chn_names_arr = np.array(
                    [(_to_bytes(n), _to_bytes(i)) for n, i in scalar_chns],
                    dtype=dtype_chn_names,
                )
                data_grp.create_dataset("Channel names", data=chn_names_arr)

                num_channels = len(scalar_chns)
                data_grp.create_dataset(
                    "Data",
                    shape=(dim1, num_channels, 0),
                    maxshape=(dim1, num_channels, M),
                    chunks=(dim1, num_channels, M),
                    dtype="f8",
                )

                data_grp.create_dataset(
                    "Time stamp", shape=(0,), maxshape=(M,), chunks=(M,), dtype="f8"
                )

            data_grp = grp["Data"]
            ds_data = data_grp["Data"]
            ds_time = data_grp["Time stamp"]

            ds_data.resize((dim1, ds_data.shape[1], col + 1))
            ds_time.resize((col + 1,))

            curr_timestamp = time.time()
            # Yes! creation_time is a float timestamp, and Time stamp is the elapsed time since creation_time!
            creation_time = f.attrs.get("creation_time", curr_timestamp)
            elapsed_time = curr_timestamp - creation_time
            ds_time[col] = elapsed_time

            outer_dims = step_dims[1:]
            if len(outer_dims) > 0:
                multi_idx = np.unravel_index(col, outer_dims)
            else:
                multi_idx = ()

            chns = [
                (_to_str(row["name"]), _to_str(row["info"]))
                for row in data_grp["Channel names"]
            ]

            step_names = []
            if "Step list" in f:
                for step_row in f["Step list"]:
                    step_names.append(_to_str(step_row["channel_name"]))

            for idx, name in enumerate(step_names):
                values = np.array([])
                cfg_path = f"Step config/{name}/Step items"
                if cfg_path in f:
                    step_item = f[cfg_path][0]
                    range_type = step_item["range_type"]
                    if name == _STEP_NAME_API:
                        values = np.array([1.0])
                    elif range_type == 1:
                        values = np.linspace(
                            step_item["start"], step_item["stop"], step_item["n_pts"]
                        )
                    elif range_type == 0:
                        values = np.array([step_item["single"]])

                ch_idx = [i for i, (n, info) in enumerate(chns) if n == name][0]
                if idx == 0:
                    ds_data[:, ch_idx, col] = values
                else:
                    val = values[multi_idx[idx - 1]]
                    ds_data[:, ch_idx, col] = val

            for name, info in chns:
                if name in step_names:
                    continue
                val = data.get(name)
                if val is not None:
                    if info == "":
                        val_arr = np.asarray(val)
                        if val_arr.ndim == 0:
                            ds_data[:, chns.index((name, "")), col] = float(val)
                        else:
                            ds_data[:, chns.index((name, "")), col] = val_arr
                    elif info == "Real":
                        val_arr = np.asarray(val)
                        real_idx = chns.index((name, "Real"))
                        imag_idx = chns.index((name, "Imaginary"))
                        if val_arr.ndim == 0:
                            ds_data[:, real_idx, col] = np.real(val)
                            ds_data[:, imag_idx, col] = np.imag(val)
                        else:
                            ds_data[:, real_idx, col] = np.real(val_arr)
                            ds_data[:, imag_idx, col] = np.imag(val_arr)

            log_names = []
            if "Log list" in f:
                for log_row in f["Log list"]:
                    log_names.append(_to_str(log_row["channel_name"]))

            vector_names = []
            inst_grp_path = (
                "Instrument config/Generic - GPIB: , Log channels at localhost"
            )
            if inst_grp_path in grp:
                inst_grp = grp[inst_grp_path]
                for name in log_names:
                    if f"___{name}___x_name" in inst_grp.attrs:
                        vector_names.append(name)

            for name in vector_names:
                val = data.get(name)
                if val is not None:
                    if not isinstance(val, dict) or "y" not in val:
                        val = getTraceDict(val)

                    y_data = np.asarray(val["y"])
                    trace_len = len(y_data)
                    t0 = val.get("t0", 0.0)
                    dt = val.get("dt", 1.0)
                    x_data = val.get("x", None)
                    is_complex = np.iscomplexobj(y_data)

                    if not is_complex and x_data is None:
                        C = 1
                    elif (not is_complex and x_data is not None) or (
                        is_complex and x_data is None
                    ):
                        C = 2
                    else:
                        C = 3

                    if "Traces" not in grp:
                        grp.create_group("Traces")
                    traces_grp = grp["Traces"]

                    ds_trace_path = name
                    if ds_trace_path not in traces_grp:
                        traces_grp.create_dataset(
                            name,
                            shape=(trace_len, C, 0),
                            maxshape=(None, C, M),
                            chunks=(trace_len, C, max(1, M)),
                            dtype="f8",
                            fillvalue=np.nan,
                        )
                        traces_grp.create_dataset(
                            name + "_N",
                            shape=(0,),
                            maxshape=(M,),
                            chunks=(max(1, M),),
                            dtype="i4",
                        )
                        traces_grp.create_dataset(
                            name + "_t0dt",
                            shape=(0, 2),
                            maxshape=(M, 2),
                            chunks=(max(1, M), 2),
                            dtype="f8",
                        )

                        ds_trace = traces_grp[name]
                        ds_trace.attrs["complex"] = is_complex

                        x_name = _to_str(
                            inst_grp.attrs.get(f"___{name}___x_name", b"Index")
                        )
                        x_unit = _to_str(inst_grp.attrs.get(f"___{name}___x_unit", b""))
                        ds_trace.attrs["x, name"] = _to_bytes(x_name)
                        ds_trace.attrs["x, unit"] = _to_bytes(x_unit)

                    ds_trace = traces_grp[name]
                    ds_N = traces_grp[name + "_N"]
                    ds_t0dt = traces_grp[name + "_t0dt"]

                    current_trace_len = ds_trace.shape[0]
                    if trace_len > current_trace_len:
                        ds_trace.resize((trace_len, C, ds_trace.shape[2]))

                    ds_trace.resize((ds_trace.shape[0], C, col + 1))
                    ds_N.resize((col + 1,))
                    ds_t0dt.resize((col + 1, 2))

                    if not is_complex and x_data is None:
                        ds_trace[:trace_len, 0, col] = y_data
                    elif not is_complex and x_data is not None:
                        ds_trace[:trace_len, 0, col] = x_data
                        ds_trace[:trace_len, 1, col] = y_data
                    elif is_complex and x_data is None:
                        ds_trace[:trace_len, 0, col] = np.real(y_data)
                        ds_trace[:trace_len, 1, col] = np.imag(y_data)
                    else:
                        ds_trace[:trace_len, 0, col] = np.real(y_data)
                        ds_trace[:trace_len, 1, col] = np.imag(y_data)
                        ds_trace[:trace_len, 2, col] = x_data

                    ds_N[col] = trace_len
                    if x_data is not None:
                        ds_t0dt[col, 0] = 0.0
                        ds_t0dt[col, 1] = 0.0
                    else:
                        ds_t0dt[col, 0] = t0
                        ds_t0dt[col, 1] = dt

            if "Traces" in grp:
                traces_grp = grp["Traces"]
                if "Time stamp" not in traces_grp:
                    traces_grp.create_dataset(
                        "Time stamp",
                        shape=(0,),
                        maxshape=(M,),
                        chunks=(max(1, M),),
                        dtype="f8",
                    )
                ds_trace_time = traces_grp["Time stamp"]
                ds_trace_time.resize((col + 1,))
                ds_trace_time[col] = elapsed_time

            if col == M - 1:
                if "Data" in grp:
                    grp["Data"].attrs["Completed"] = True

                time_stamps = (
                    ds_time[:] if "Data" in grp else grp["Traces/Time stamp"][:]
                )
                if len(time_stamps) > 0:
                    time_sweep = np.diff(np.r_[0.0, time_stamps])
                    median_dt = np.median(time_sweep)
                    f.attrs["time_per_point"] = float(median_dt / dim1)


def createLogFile_ForData(
    name,
    log_channels,
    step_channels=[],
    use_database=True,
    database_folder=None,
    instrument_settings=None,
):
    """Create LogFile ForData.

    Parameters
    ----------
    name : Any
        Name of the target object.
    log_channels : Any
        Value for ``log_channels``.
    step_channels : Any, default: []
        Value for ``step_channels``.
    use_database : Any, default: True
        Whether to use database.
    database_folder : path-like, default: None
        Labber database root. Uses Labber preferences when omitted.
    instrument_settings : list of dict, default: None
        Fixed instrument settings. Each dict contains ``name``, ``value``, and
        optionally ``unit``. Settings appear in Labber's instrument
        configuration without becoming step or log channels.

    Returns
    -------
    Any
        Result of the operation.
    """
    if use_database:
        resolved_path = _create_log_path(
            name,
            database_folder=database_folder,
        )
    else:
        base, ext = os.path.splitext(name)
        if ext.lower() != ".hdf5":
            name = base + ".hdf5"
        resolved_path = os.path.abspath(name)
        parent_dir = os.path.dirname(resolved_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    for ch in log_channels:
        if "vector" not in ch:
            ch["vector"] = True

    if instrument_settings is None:
        instrument_settings = []

    bAllVector = all([ch.get("vector", True) for ch in log_channels])

    actual_step_channels = list(step_channels)
    if bAllVector:
        step_index_api_ch = {"name": _STEP_NAME_API, "values": np.array([1.0])}
        actual_step_channels = [step_index_api_ch] + actual_step_channels

    with h5py.File(resolved_path, "w") as f:
        step_dims = [len(ch["values"]) for ch in actual_step_channels]
        f.attrs["Step dimensions"] = step_dims
        f.attrs["arm_trig_mode"] = False
        f.attrs["comment"] = b""
        f.attrs["creation_time"] = time.time()
        f.attrs["hardware_loop"] = False
        f.attrs["log_name"] = _to_bytes(
            os.path.splitext(os.path.basename(resolved_path))[0]
        )
        f.attrs["log_parallel"] = True
        f.attrs["logger_mode"] = False
        f.attrs["time_per_point"] = 0.0
        f.attrs["trig_channel"] = b""
        f.attrs["version"] = b"1.8.6"
        f.attrs["wait_between"] = 0.01

        tags_grp = f.create_group("Tags")
        tags_grp.attrs["Project"] = np.array([b""], dtype=object)
        tags_grp.attrs["Tags"] = np.array([])
        tags_grp.attrs["User"] = np.array([b""], dtype=object)

        f.create_group("Settings")
        inst_cfg_grp = f.create_group("Instrument config")

        dtype_instruments = np.dtype(
            [
                ("hardware", vlen_bytes),
                ("version", vlen_bytes),
                ("id", vlen_bytes),
                ("model", vlen_bytes),
                ("name", vlen_bytes),
                ("interface", interface_enum),
                ("address", vlen_bytes),
                ("server", vlen_bytes),
                ("startup", startup_enum),
                ("lock", "?"),
                ("show_advanced", "?"),
                ("Timeout", "<f8"),
                ("Term. character", vlen_bytes),
                ("Send end on write", "?"),
                ("Lock VISA resource", "?"),
                ("Suppress end bit termination on read", "?"),
                ("Use specific TCP port", "?"),
                ("TCP port", "<f8"),
                ("Use VICP protocol", "?"),
                ("Baud rate", "<f8"),
                ("Data bits", "<f8"),
                ("Stop bits", "<f8"),
                ("Parity", vlen_bytes),
                ("GPIB board number", "<f8"),
                ("Send GPIB go to local at close", "?"),
                ("PXI chassis", "<f8"),
                ("Run in 32-bit mode", "?"),
            ]
        )

        step_inst_id = b"Generic - GPIB: , Step channels at localhost"
        log_inst_id = b"Generic - GPIB: , Log channels at localhost"
        settings_inst_id = b"Generic - GPIB: , Instrument settings at localhost"

        inst_data = [
            (
                b"Generic",
                b"1.0",
                step_inst_id,
                b"",
                b"Step channels",
                0,
                b"",
                b"",
                0,
                False,
                False,
                10.0,
                b"Auto",
                True,
                False,
                False,
                False,
                0.0,
                False,
                9600.0,
                8.0,
                1.0,
                b"No parity",
                0.0,
                False,
                1.0,
                False,
            ),
            (
                b"Generic",
                b"1.0",
                log_inst_id,
                b"",
                b"Log channels",
                0,
                b"",
                b"",
                0,
                False,
                False,
                10.0,
                b"Auto",
                True,
                False,
                False,
                False,
                0.0,
                False,
                9600.0,
                8.0,
                1.0,
                b"No parity",
                0.0,
                False,
                1.0,
                False,
            ),
        ]
        if instrument_settings:
            settings_instrument = list(inst_data[-1])
            settings_instrument[2] = settings_inst_id
            settings_instrument[4] = b"Instrument settings"
            inst_data.append(tuple(settings_instrument))
        f.create_dataset(
            "Instruments", data=np.array(inst_data, dtype=dtype_instruments)
        )

        dtype_channels = np.dtype(
            [
                ("name", vlen_bytes),
                ("instrument", vlen_bytes),
                ("quantity", vlen_bytes),
                ("unitPhys", vlen_bytes),
                ("unitInstr", vlen_bytes),
                ("gain", "<f8"),
                ("offset", "<f8"),
                ("amp", "<f8"),
                ("highLim", "<f8"),
                ("lowLim", "<f8"),
                ("outputChannel", vlen_bytes),
                ("limit_action", vlen_bytes),
                ("limit_run_script", "?"),
                ("limit_script", vlen_bytes),
                ("use_log_interval", "?"),
                ("log_interval", "<f8"),
                ("limit_run_always", "?"),
            ]
        )

        chn_data = []
        for ch in actual_step_channels:
            name_bytes = _to_bytes(ch["name"])
            unit_bytes = _to_bytes(ch.get("unit", ""))
            chn_data.append(
                (
                    name_bytes,
                    step_inst_id,
                    name_bytes,
                    unit_bytes,
                    unit_bytes,
                    1.0,
                    0.0,
                    1.0,
                    float("inf"),
                    float("-inf"),
                    b"",
                    b"Nothing",
                    False,
                    b"",
                    False,
                    1.0,
                    False,
                )
            )

        for ch in log_channels:
            name_bytes = _to_bytes(ch["name"])
            unit_bytes = _to_bytes(ch.get("unit", ""))
            is_vector = ch.get("vector", True)
            high_lim = 0.0 if is_vector else float("inf")
            low_lim = 0.0 if is_vector else float("-inf")
            chn_data.append(
                (
                    name_bytes,
                    log_inst_id,
                    name_bytes,
                    unit_bytes,
                    unit_bytes,
                    1.0,
                    0.0,
                    1.0,
                    high_lim,
                    low_lim,
                    b"",
                    b"Nothing",
                    False,
                    b"",
                    False,
                    1.0,
                    False,
                )
            )

        for setting in instrument_settings:
            name_bytes = _to_bytes(setting["name"])
            unit_bytes = _to_bytes(setting.get("unit", ""))
            chn_data.append(
                (
                    name_bytes,
                    settings_inst_id,
                    name_bytes,
                    unit_bytes,
                    unit_bytes,
                    1.0,
                    0.0,
                    1.0,
                    float("inf"),
                    float("-inf"),
                    b"",
                    b"Nothing",
                    False,
                    b"",
                    False,
                    1.0,
                    False,
                )
            )

        f.create_dataset("Channels", data=np.array(chn_data, dtype=dtype_channels))

        dtype_log_list = np.dtype([("channel_name", vlen_bytes)])
        log_list_data = [(_to_bytes(ch["name"]),) for ch in log_channels]
        f.create_dataset("Log list", data=np.array(log_list_data, dtype=dtype_log_list))

        dtype_step_list = np.dtype(
            [
                ("channel_name", vlen_bytes),
                ("step_unit", step_unit_enum),
                ("wait_after", "<f8"),
                ("after_last", after_last_enum),
                ("final_value", "<f8"),
                ("use_relations", "?"),
                ("equation", vlen_bytes),
                ("show_advanced", "?"),
                ("sweep_mode", sweep_mode_enum),
                ("use_outside_sweep_rate", "?"),
                ("sweep_rate_outside", "<f8"),
                ("alternate_direction", "?"),
            ]
        )
        step_list_data = []
        for ch in actual_step_channels:
            step_list_data.append(
                (
                    _to_bytes(ch["name"]),
                    0,
                    float(ch.get("wait_after", 0.0)),
                    0,
                    0.0,
                    False,
                    b"x",
                    False,
                    0,
                    False,
                    0.0,
                    False,
                )
            )
        f.create_dataset(
            "Step list", data=np.array(step_list_data, dtype=dtype_step_list)
        )

        step_cfg_grp = f.create_group("Step config")

        step_inst_cfg = inst_cfg_grp.create_group(
            "Generic - GPIB: , Step channels at localhost"
        )
        step_inst_cfg.attrs["Installed options"] = np.array([])
        for ch in actual_step_channels:
            name = ch["name"]
            step_inst_cfg.attrs[name] = 0.0

            single_step_grp = step_cfg_grp.create_group(name)

            opt_grp = single_step_grp.create_group("Optimizer")
            opt_grp.attrs["Enabled"] = False

            vals = ch["values"]
            if name == _STEP_NAME_API:
                vals = np.linspace(1.0, 2.0, 51)

            start = vals[0]
            stop = vals[-1]
            span = abs(stop - start)
            opt_grp.attrs["Initial step size"] = float(0.2 * span) if span > 0 else 1.0
            opt_grp.attrs["Max value"] = float(max(start, stop))
            opt_grp.attrs["Min value"] = float(min(start, stop))
            opt_grp.attrs["Precision"] = float(1e-4 * span) if span > 0 else 1e-4
            opt_grp.attrs["Start value"] = float(start)

            dtype_rel = np.dtype(
                [
                    ("variable", vlen_bytes),
                    ("channel_name", vlen_bytes),
                    ("use_lookup", "?"),
                ]
            )
            single_step_grp.create_dataset(
                "Relation parameters",
                data=np.array([(b"x", b"Step values", False)], dtype=dtype_rel),
            )

            dtype_items = np.dtype(
                [
                    ("range_type", range_type_enum),
                    ("step_type", step_type_enum),
                    ("single", "<f8"),
                    ("start", "<f8"),
                    ("stop", "<f8"),
                    ("center", "<f8"),
                    ("span", "<f8"),
                    ("step", "<f8"),
                    ("n_pts", "<i4"),
                    ("interp", interp_enum),
                    ("sweep_rate", "<f8"),
                ]
            )
            if len(vals) > 1 and name != _STEP_NAME_API:
                item_data = (1, 1, start, start, stop, 0.0, 0.0, 0.0, len(vals), 0, 0.0)
            else:
                item_data = (
                    0,
                    1,
                    start,
                    start,
                    start + 1.0,
                    0.0,
                    0.0,
                    0.0,
                    len(vals),
                    0,
                    0.0,
                )
            single_step_grp.create_dataset(
                "Step items", data=np.array([item_data], dtype=dtype_items)
            )

        log_inst_cfg = inst_cfg_grp.create_group(
            "Generic - GPIB: , Log channels at localhost"
        )
        log_inst_cfg.attrs["Installed options"] = np.array([])
        for ch in log_channels:
            name = ch["name"]
            is_vector = ch.get("vector", True)
            is_complex = ch.get("complex", False)
            if is_vector:
                x_name = ch.get(
                    "x_name", "Time" if "x_unit" in ch or "x_name" in ch else "Index"
                )
                x_unit = ch.get(
                    "x_unit", "s" if "x_unit" in ch or "x_name" in ch else ""
                )
                log_inst_cfg.attrs[f"___{name}___x_name"] = _to_bytes(x_name)
                log_inst_cfg.attrs[f"___{name}___x_unit"] = _to_bytes(x_unit)
                log_inst_cfg.attrs[name] = np.array(
                    [], dtype=complex if is_complex else float
                )
            else:
                log_inst_cfg.attrs[name] = 0.0j if is_complex else 0.0

        if instrument_settings:
            settings_inst_cfg = inst_cfg_grp.create_group(
                "Generic - GPIB: , Instrument settings at localhost"
            )
            settings_inst_cfg.attrs["Installed options"] = np.array([])
            for setting in instrument_settings:
                settings_inst_cfg.attrs[setting["name"]] = setting["value"]

    return LogFile(resolved_path)

EMBEDDED_GROUP = "metagroup"
PARTIAL_GROUP = "metagroup__partial"


def _preview_png(result) -> bytes:
    """Render a hardware-independent preview without using pyplot state.

    Parameters
    ----------
    result : Any
        Experiment result to process.

    Returns
    -------
    bytes
        Result of the operation.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    raw = getattr(result, "raw_iq", None)
    if isinstance(raw, dict):
        raw = next(iter(raw.values()), None)
    values = np.asarray(raw) if raw is not None else np.asarray([])
    fig = Figure(figsize=(8.2, 4.8), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    title = str(getattr(result, "experiment_type", "Experiment"))

    if values.size == 0:
        ax.text(0.5, 0.5, "No raw IQ data", ha="center", va="center")
        ax.set_axis_off()
    else:
        squeezed = np.squeeze(values)
        if squeezed.ndim == 1:
            x = getattr(result, "x_axis", None)
            if x is None or len(np.asarray(x).reshape(-1)) != squeezed.size:
                x = np.arange(squeezed.size)
            ax.plot(x, np.real(squeezed), lw=1.2, label="I")
            ax.plot(x, np.imag(squeezed), lw=1.2, label="Q")
            ax.plot(x, np.abs(squeezed), lw=1.4, label="|IQ|")
            ax.legend(frameon=False, ncols=3)
            ax.set_xlabel(getattr(result, "x_name", "") or "Sweep")
            ax.set_ylabel("ADC unit")
        elif str(getattr(result, "data_kind", "")).startswith("single_shot"):
            clouds = squeezed.reshape((-1, squeezed.shape[-1]))
            for index, cloud in enumerate(clouds[:8]):
                ax.scatter(cloud.real, cloud.imag, s=5, alpha=0.35, label=str(index))
            ax.set(xlabel="I", ylabel="Q")
            ax.legend(title="state/grid", frameon=False)
        else:
            image = np.abs(squeezed)
            while image.ndim > 2:
                image = np.nanmean(image, axis=-1)
            if image.ndim == 1:
                ax.plot(image)
            else:
                plotted = ax.imshow(image, origin="lower", aspect="auto", cmap="viridis")
                fig.colorbar(plotted, ax=ax, label="|IQ|")
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.18)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=130, metadata={"Software": "QickworkspaceV2"})
    return buffer.getvalue()


def _figure_payloads(figures) -> dict[str, bytes]:
    """Normalise named PNG bytes or Matplotlib figures for HDF5 embedding."""
    if isinstance(figures, Mapping):
        return dict(figures)
    payloads = {}
    for index, figure in enumerate(figures or []):
        if figure is None or not hasattr(figure, "savefig"):
            continue
        buffer = BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=150,
            bbox_inches="tight",
            metadata={"Software": "QickworkspaceV2"},
        )
        name = "analysis.png" if not payloads else f"analysis_{index + 1}.png"
        payloads[name] = buffer.getvalue()
    return payloads


def _write_native_group(
    root: h5py.Group,
    result,
    *,
    comment: str,
    tags: list[str],
    utc_time,
    save_plots: bool,
    figures: Mapping[str, bytes] | None,
) -> None:
    """Write native group.

    Parameters
    ----------
    root : h5py.Group
        Value for ``root``.
    result : Any
        Experiment result to process.
    comment : str
        Value for ``comment``.
    tags : list[str]
        Value for ``tags``.
    utc_time : Any
        Value for ``utc_time``.
    save_plots : bool
        Value for ``save_plots``.
    figures : Mapping[str, bytes] | None
        Value for ``figures``.
    """
    from . import hdf5_store as native

    local_time = utc_time.astimezone(native.LOCAL_TIMEZONE)
    data_kind, analysis_id, plot_id = native._dispatch_ids(result)
    metadata = dict(getattr(result, "metadata", {}) or {})
    session_id = str(getattr(result, "session_id", "") or metadata.get("session_id", ""))
    lineage = {
        "session_id": session_id or None,
        "parent_id": getattr(result, "parent_id", None),
        "children": getattr(result, "children", []) or [],
    }
    root.attrs.update(
        {
            "schema_name": native.SCHEMA_NAME,
            "schema_version": native.SCHEMA_VERSION,
            "write_complete": False,
            "container_kind": "labber_hybrid",
            "experiment_id": result.experiment_id,
            "experiment_type": str(getattr(result, "experiment_type", "")),
            "timestamp_utc": utc_time.isoformat(),
            "timestamp_local": local_time.isoformat(),
            "data_kind": data_kind,
            "analysis_id": analysis_id,
            "plot_id": plot_id,
            "quality": native._quality_value(result),
            "interrupted": bool(getattr(result, "interrupted", False)),
        }
    )

    meta = root.create_group("meta")
    native._write_text(meta, "comment", comment)
    meta.create_dataset("tags", data=np.asarray(tags, dtype=native._STRING_DTYPE))
    native._write_text(meta, "config_json", native._json_dumps(getattr(result, "config", {}) or {}))
    native._write_text(meta, "metadata_json", native._json_dumps(metadata))
    native._write_text(meta, "provenance_json", native._json_dumps(native._provenance()))
    native._write_text(meta, "lineage_json", native._json_dumps(lineage))

    axes = root.create_group("axes")
    for name, payload in native._axis_entries(result).items():
        values, attrs = native._normalise_dataset_payload(payload)
        axis = axes.create_group(str(name).replace("/", "_"))
        array = np.asarray(values)
        if array.dtype.kind in {"O", "U", "S"}:
            strings = [item.decode() if isinstance(item, bytes) else str(item) for item in array.reshape(-1)]
            axis.create_dataset(
                "values", data=np.asarray(strings, dtype=native._STRING_DTYPE).reshape(array.shape)
            )
        else:
            axis.create_dataset("values", data=array, **native._dataset_options(array))
        for key in ("label", "unit", "description", "scale"):
            value = attrs.get(key)
            if value not in (None, ""):
                axis.attrs[key] = value

    raw_group = root.create_group("raw")
    raw_iq = getattr(result, "raw_iq", None)
    dims = getattr(result, "dataset_dims", {}) or {}
    if isinstance(raw_iq, dict):
        native._write_tree(raw_group, native._decorate_tree_with_dims(raw_iq, dims))
    elif raw_iq is not None:
        native._write_tree(raw_group, {"iq": {"values": raw_iq, "dims": dims.get("iq", [])}})
    native._write_tree(
        raw_group,
        native._decorate_tree_with_dims(getattr(result, "raw_data", {}) or {}, dims),
    )

    analysis = root.create_group("analysis")
    native._write_tree(analysis, getattr(result, "analysis_data", {}) or {})

    results = root.create_group("results")
    native._write_text(results, "fit_result_json", native._json_dumps(getattr(result, "fit_result", {}) or {}))
    if getattr(result, "fit_params", None) is not None:
        native._write_tree(results, {"fit_params": result.fit_params})
    if getattr(result, "fit_errors", None) is not None:
        native._write_tree(results, {"fit_errors": result.fit_errors})
    native._write_text(
        results,
        "summary_json",
        native._json_dumps(
            {
                "scalar_result": getattr(result, "scalar_result", None),
                "quality_message": getattr(result, "quality_message", ""),
                "avg_count": int(getattr(result, "avg_count", 0)),
            }
        ),
    )

    if save_plots or figures:
        plots = root.create_group("plots")
        payloads = _figure_payloads(figures)
        if save_plots:
            payloads.setdefault("main.png", _preview_png(result))
            payloads.setdefault("analysis.png", payloads["main.png"])
            payloads.setdefault("preview.png", payloads["analysis.png"])
        for name, payload in payloads.items():
            safe_name = str(name).replace("/", "_")
            data = payload.getvalue() if hasattr(payload, "getvalue") else bytes(payload)
            dataset = plots.create_dataset(
                safe_name,
                data=np.frombuffer(data, dtype=np.uint8),
                compression="gzip",
                compression_opts=4,
            )
            dataset.attrs["mime_type"] = "image/svg+xml" if safe_name.lower().endswith(".svg") else "image/png"

    root.attrs.modify("write_complete", True)


def embed_result_in_labber(
    labber_path,
    result,
    *,
    comment: str = "",
    tags: Iterable[str] | str = (),
    data_root=None,
    save_plots: bool = True,
    figures: Mapping[str, bytes] | None = None,
    catalog: bool = True,
) -> Path:
    """Append a complete native experiment under ``/metagroup``.

    Parameters
    ----------
    labber_path : Any
        Filesystem location for labber path.
    result : Any
        Experiment result to process.
    comment : str, default: ''
        Value for ``comment``.
    tags : Iterable[str] | str, default: ()
        Value for ``tags``.
    data_root : Any, default: None
        Value for ``data_root``.
    save_plots : bool, default: True
        Value for ``save_plots``.
    figures : Mapping[str, bytes] | None, default: None
        Value for ``figures``.
    catalog : bool, default: True
        Value for ``catalog``.

    Returns
    -------
    Path
        Result of the operation.

    Raises
    ------
    FileExistsError
        If the operation cannot be completed.
    FileNotFoundError
        If the operation cannot be completed.
    """
    from . import hdf5_store as native

    path = Path(labber_path).expanduser().resolve()
    if path.suffix.lower() != ".hdf5":
        path = path.with_suffix(".hdf5")
    if not path.is_file():
        raise FileNotFoundError(path)

    utc_time = native._as_utc(getattr(result, "timestamp", None))
    if not native.validate_experiment_id(getattr(result, "experiment_id", "")):
        result.experiment_id = native.generate_experiment_id(utc_time)
    result.timestamp = utc_time
    result.comment = str(comment if comment != "" else getattr(result, "comment", ""))
    result.tags = native._normalise_tags(
        tags if tags else getattr(result, "tags", []),
        getattr(result, "experiment_type", ""),
    )

    with h5py.File(path, "r+") as h5:
        if EMBEDDED_GROUP in h5:
            raise FileExistsError(f"/{EMBEDDED_GROUP} already exists in {path}")
        if PARTIAL_GROUP in h5:
            del h5[PARTIAL_GROUP]
        partial = h5.create_group(PARTIAL_GROUP)
        try:
            _write_native_group(
                partial,
                result,
                comment=result.comment,
                tags=result.tags,
                utc_time=utc_time,
                save_plots=save_plots,
                figures=figures,
            )
            h5.flush()
            h5.move(PARTIAL_GROUP, EMBEDDED_GROUP)
            h5.flush()
        except Exception:
            if PARTIAL_GROUP in h5:
                del h5[PARTIAL_GROUP]
            h5.flush()
            raise

    if catalog:
        root = Path(data_root).expanduser().resolve() if data_root is not None else path.parent
        native._register_file(path, root)
    return path


def read_embedded_plot(path, name: str = "main.png") -> bytes:
    """Read an embedded plot without loading experiment arrays.

    Parameters
    ----------
    path : Any
        Filesystem path.
    name : str, default: 'main.png'
        Name of the target object.

    Returns
    -------
    bytes
        Result of the operation.
    """
    with h5py.File(Path(path).expanduser().resolve(), "r") as h5:
        dataset = h5[f"{EMBEDDED_GROUP}/plots/{name}"]
        return np.asarray(dataset, dtype=np.uint8).tobytes()


class LabberHDF5Saver:
    """Write a Labber log and native experiment metadata into one HDF5 file.

    ``filename_mode="random"`` (default) appends the same sortable experiment
    ID used by the native HDF5 store. ``"sequential"`` keeps the caller's
    numbered Labber filename.
    """

    VALID_FILENAME_MODES = {"random", "sequential"}

    def __init__(self, filename_mode="random", embed_native=True, save_plots=True):
        """Initialize the LabberHDF5Saver instance.

        Parameters
        ----------
        filename_mode : Any, default: 'random'
            Value for ``filename_mode``.
        embed_native : Any, default: True
            Value for ``embed_native``.
        save_plots : Any, default: True
            Value for ``save_plots``.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
        if filename_mode not in self.VALID_FILENAME_MODES:
            raise ValueError("filename_mode must be 'random' or 'sequential'")
        self.filename_mode = filename_mode
        self.embed_native = bool(embed_native)
        self.save_plots = bool(save_plots)

    def _output_path(self, filepath, result=None):
        """Return the output path result.

        Parameters
        ----------
        filepath : Any
            Value for ``filepath``.
        result : Any, default: None
            Experiment result to process.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        FileExistsError
            If the operation cannot be completed.
        """
        from pathlib import Path
        import re

        path = Path(filepath).expanduser().resolve()
        if path.suffix.lower() != ".hdf5":
            path = path.with_suffix(".hdf5")
        if self.filename_mode == "sequential":
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing Labber file: {path}")
            return path

        from . import hdf5_store as native
        experiment_id = getattr(result, "experiment_id", "") if result is not None else ""
        if not native.validate_experiment_id(experiment_id):
            experiment_id = native.generate_experiment_id(getattr(result, "timestamp", None))
        # get_next_filename_labber historically adds _001; random mode replaces it.
        stem = re.sub(r"_\d{3}$", "", path.stem)
        candidate = path.with_name(f"{stem}_{experiment_id}.hdf5")
        while candidate.exists():
            experiment_id = native.generate_experiment_id(getattr(result, "timestamp", None))
            candidate = path.with_name(f"{stem}_{experiment_id}.hdf5")
        if result is not None:
            result.experiment_id = experiment_id
        return candidate

    def save(
        self,
        filepath,
        x_info,
        z_info,
        y_info=None,
        comment=None,
        tag=None,
        *,
        result=None,
        data_root=None,
        figures=None,
        catalog=True,
    ):
        """Save the operation.

        Parameters
        ----------
        filepath : Any
            Value for ``filepath``.
        x_info : Any
            Value for ``x_info``.
        z_info : Any
            Value for ``z_info``.
        y_info : Any, default: None
            Value for ``y_info``.
        comment : Any, default: None
            Value for ``comment``.
        tag : Any, default: None
            Value for ``tag``.
        result : Any, default: None
            Experiment result to process.
        data_root : Any, default: None
            Value for ``data_root``.
        figures : Any, default: None
            Value for ``figures``.
        catalog : Any, default: True
            Value for ``catalog``.

        Returns
        -------
        Any
            Result of the operation.
        """
        final_path = self._output_path(filepath, result=result)
        zdata = z_info["values"]
        channel = dict(z_info)
        channel.update({"complex": True, "vector": False})
        log = createLogFile_ForData(
            str(final_path), [channel], list(filter(None, [x_info, y_info])), use_database=False
        )
        if y_info:
            for trace in zdata:
                log.addEntry({channel["name"]: trace})
        else:
            log.addEntry({channel["name"]: zdata})
        if comment:
            log.setComment(comment)
        if tag:
            log.setTags(tag)

        if self.embed_native and result is not None:
            embedded_tags = list(getattr(result, "tags", []) or [])
            supplied_tags = [tag] if isinstance(tag, (str, bytes)) else list(tag or [])
            for item in supplied_tags:
                if item not in embedded_tags:
                    embedded_tags.append(item)
            embed_result_in_labber(
                log.getFilePath(),
                result,
                comment=str(comment or ""),
                tags=embedded_tags,
                data_root=data_root,
                save_plots=self.save_plots,
                figures=figures,
                catalog=catalog,
            )
        return log.getFilePath()
