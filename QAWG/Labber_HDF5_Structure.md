# Labber HDF5 File Format Structure

Labber stores measurement data, instrument configurations, and sweep parameters in a hierarchical HDF5 format. This document describes the structure of Labber-compatible HDF5 files based on the analysis of Labber's output data files.

---

## 1. Root Level Attributes `/`

The root level of the HDF5 file contains metadata attributes about the measurement run:

| Attribute Name | Data Type | Description |
| :--- | :--- | :--- |
| `Step dimensions` | `int32` array | Represents the number of sweep steps for each channel in the `Step list`. For a 1D sweep with 126 points, it looks like: `[126, 1, 1, 1, ...]` (14 elements). |
| `version` | `str` | Labber version (e.g., `'1.8.6'`). |
| `log_name` | `str` | Name of the log file/measurement. |
| `creation_time` | `float64` | UNIX epoch timestamp when the file was created. |
| `comment` | `str` | Multiline user notes or additional context from the measurement setup. |
| `time_per_point` | `float64` | Time taken per sweep point in seconds. |
| `wait_between` | `float64` | Delay between sweep steps in seconds. |
| `hardware_loop` | `bool` | Whether hardware-controlled looping was used. |
| `log_parallel` | `bool` | Whether parallel logging was enabled. |
| `logger_mode` | `bool` | True if run in logger mode. |
| `arm_trig_mode` | `bool` | Trigger arming mode flag. |
| `trig_channel` | `str` | Name of the channel used to trigger. |

---

## 2. Root Datasets

### 2.1. `Channels` (Compound Dataset)
Contains metadata for all active channels (quantities monitored or changed).
- **Shape**: `(num_channels,)`
- **Fields (Compound Dtype)**:
  - `name` (`str`/`object`): Full channel name (e.g. `'DC - coil - Current'`).
  - `instrument` (`str`/`object`): Name of the instrument managing the channel.
  - `quantity` (`str`/`object`): Physical quantity name (e.g. `'Current'`).
  - `unitPhys` (`str`/`object`): Physical unit (e.g. `'A'`, `'V'`, `'s'`).
  - `unitInstr` (`str`/`object`): Instrument-level unit.
  - `gain` (`float64`): Scaling gain multiplier.
  - `offset` (`float64`): Scaling offset.
  - `amp` (`float64`): Amplification multiplier.
  - `highLim` (`float64`): Upper safety limit.
  - `lowLim` (`float64`): Lower safety limit.
  - `outputChannel` (`str`/`object`): Output mapping channel.
  - `limit_action` (`str`/`object`): Action to take if safety limits are exceeded (e.g., `'Nothing'`).
  - `limit_run_script` (`bool`): Whether to run a script on limit exceedance.
  - `limit_script` (`str`/`object`): Path to script to execute.
  - `use_log_interval` (`bool`): Log interval enable.
  - `log_interval` (`float64`): Log interval.
  - `limit_run_always` (`bool`): Run script always.

### 2.2. `Instruments` (Compound Dataset)
Lists the physical and virtual instruments loaded in the measurement configuration.
- **Shape**: `(num_instruments,)`
- **Fields (Compound Dtype)**:
  - `hardware` (`str`/`object`): Name of the driver (e.g. `'Yokogawa GS200 DC Source'`).
  - `version` (`str`/`object`): Driver/hardware version.
  - `id` (`str`/`object`): Unique identifier (e.g. `'Yokogawa GS200 DC Source - USB: 0x0B21::0x0039::91S522309, DC - coil at localhost'`).
  - `model` (`str`/`object`): Instrument model (e.g. `'GS210'`).
  - `name` (`str`/`object`): Name designated in Labber (e.g. `'DC - coil'`).
  - `interface` (`int16`): Connection interface code (e.g., `1` for GPIB, `2` for USB, `6` for TCPIP).
  - `address` (`str`/`object`): Instrument connection string or IP address.
  - `server` (`str`/`object`): Server IP/localhost.
  - Connection protocol configurations: `Timeout`, `Term. character`, `Send end on write`, `Baud rate`, etc.

### 2.3. `Step list` (Compound Dataset)
Lists channels configured for stepping or scanning, corresponding to dimensions in the sweep space.
- **Shape**: `(num_step_items,)` (Usually matches the length of `Step dimensions` attribute, typically `14`).
- **Fields (Compound Dtype)**:
  - `channel_name` (`str`/`object`): Channel name to sweep (e.g. `'RF - qubit - Power'`).
  - `step_unit` (`int16`): Step unit indicator.
  - `wait_after` (`float64`): Time to wait after setting parameter.
  - `after_last` (`int16`): Behavior after the last step is completed (e.g. return to start).
  - `final_value` (`float64`): Final parameter value.
  - `use_relations` (`bool`): If parameter values are calculated based on another channel.
  - `equation` (`str`/`object`): Expression used for relation calculation (e.g. `'x'`).
  - `show_advanced` (`bool`)
  - `sweep_mode` (`int16`)
  - `use_outside_sweep_rate` (`bool`)
  - `sweep_rate_outside` (`float64`)
  - `alternate_direction` (`bool`)

### 2.4. `Log list` (Compound Dataset)
Defines which channels are being active targets for logging/recording.
- **Shape**: `(num_logged_channels,)`
- **Fields (Compound Dtype)**:
  - `channel_name` (`str`/`object`): Channel name whose results are recorded (e.g. `'Digitizer - Channel A - Average piecewise demodulated values'`).

---

## 3. Subgroups

### 3.1. `/Data` Group
Manages coordinates and coordinates indexing for the current sweep coordinates.
- **Attributes**:
  - `Completed` (`bool`): True if measurement is completed.
  - `Step dimensions` (`int32` array): Same as root `Step dimensions`.
  - `Step index` (`int32` array): Indices indicating which indices in `Step list` are active sweeps.
  - `Fixed step index` (`int32` array): Indices of channels in `Step list` that are fixed (non-swept, size `1`).
  - `Fixed step values` (`float64` array): Pre-set values for the fixed channels.
- **Datasets**:
  - `/Data/Channel names` (Compound Dtype `[('name', 'O'), ('info', 'O')]`): Contains name and info of swept channels.
  - `/Data/Data`: Coordinate arrays. Shape: `(swept_pts_dim1, swept_pts_dim2, ..., 1)`. Contains coordinate values for the swept channels at each sweep point.
  - `/Data/Time stamp`: Floats containing measurement execution elapsed time.

### 3.2. `/Traces` Group
Contains the measured/logged data records.

For each logged channel `MyChannel` in the `Log list`, the following datasets are created under `/Traces`:
1. **`/Traces/MyChannel`**: The data array containing measurement outcomes.
   - **Shape**: `(trace_pts, 2, swept_pts_dim1)` if complex, or `(trace_pts, swept_pts_dim1)` if real.
     - `trace_pts`: Number of points in the waveform record (e.g., 981).
     - `2`: Dimension representing complex values (Index 0 is Real, Index 1 is Imaginary) if attribute `complex` is True.
     - `swept_pts_dim1`: The sweep dimension size (e.g., 126).
   - **Attributes**:
     - `complex` (`bool`): Whether the data is complex.
     - `x, name` (`str`): Name of trace x-axis (e.g. `'Time'`).
     - `x, unit` (`str`): Unit of trace x-axis (e.g. `'s'`).
2. **`/Traces/MyChannel_N`**:
   - **Shape**: `(1,)`
   - **Data**: Single integer representing the trace size (`trace_pts`).
3. **`/Traces/MyChannel_t0dt`**:
   - **Shape**: `(1, 2)`
   - **Data**: `[[t0, dt]]` representing start coordinate and step size for trace x-axis (e.g., `[[0.0, 1e-9]]` for start at 0s with 1ns resolution).

Additional dataset:
- **`/Traces/Time stamp`**: Float timestamps for each swept point. Shape: `(swept_pts_dim1, ...)`.

### 3.3. `/Instrument config` Group
Contains subgroups named after instrument IDs from the `Instruments` list.
- **Subgroup `/Instrument config/<Instrument_ID>` Attributes**:
  - The key-value attributes represent all settings/parameters of the instrument at runtime (e.g. `Frequency: 4.82e9`, `Power: -5.0`, `Output: True`).

### 3.4. `/Step config` Group
Contains subgroups for each stepped channel.
- **Subgroup `/Step config/<Channel_Name>`**:
  - **`/Optimizer` Group**: Attributes: `Enabled`, `Initial step size`, `Max value`, `Min value`, `Precision`, `Start value`.
  - **`Relation parameters` Dataset**: Structured array listing relationships.
  - **`Step items` Dataset**: Structured array detailing start, stop, span, step size, n_pts, sweep_rate.

### 3.5. `/Tags` Group
Contains project tagging info.
- **Attributes**: `Project` (array of strings), `Tags` (array of strings), `User` (array of strings).

### 3.6. `/Views` Group
Visual configuration data for Labber Log Browser.
- **Attributes**: `selected_view`.
- **Subgroup `/Views/Current view`**: Layout configuration parameters (e.g. colormap, cursors position, scaling settings).
