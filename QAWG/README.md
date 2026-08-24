# QAWG Experiment Compiler

`QAWG` is a host-side experiment compiler for the Tektronix AWG5208 and
ATS9371 stack in this repository.

The package owns the complete hardware stack:

```text
QAWG/
    timeline.py    Hardware-independent waveform timing helpers
    awg5200/       AWG5208 waveform upload, marker, sequence, and driver
    alazar/        ATS9371 acquisition and signal processing
    awg_alazar.py  Public AWG/Alazar coordination entry point
    compiler.py    Experiment rules and compiled sequence plan
    tomography.py  Heterodyne tomography helpers
    Labber_saver.py Raw ADC and FIR-IQ Labber HDF5 writer
```

Common imports:

```python
from QAWG import AWGAlazar, ExperimentProgram, waveform, delay_auto
from QAWG.awg5200 import gaussian_square_ns
from QAWG.alazar import Alazar9371, CaptureConfig, downconvert, fir_filter, integrate
```

## Layer responsibilities

`QAWG.timeline` is the hardware-independent timeline helper layer. It is useful
when you want to hand-build a waveform timeline from envelopes:

```python
readout = waveform(envelope, fc=50e6, ch=3, gain=0.02)
drive = waveform(envelope, fc=100e6, ch=4, gain=0.02)
timeline = drive / delay_auto(40 * ns) / readout
```

`ExperimentProgram` is the higher-level compiler DSL. `add_pulse()` records a
pulse definition; it does not call `waveform()` immediately. During
`compile()`, the compiler schedules `play()` / `trigger()` events, renders
envelopes, modulates AWG waveforms, creates marker waveforms, and produces a
hardware-independent `CompiledExperiment`.

`AWGAlazar` is the hardware coordinator. It applies readout settings to the
ATS9371, uploads the compiled AWG sequence when needed, arms Alazar, starts the
AWG, captures records, demodulates records, and returns an `ExperimentResult`.
It keeps full acquire-window debug data from the most recent acquisition on:

```python
experiment.last_records_volts
experiment.last_downconverted_iq
experiment.last_time_s
```

## Hardware vs experiment configuration

Connection-time settings describe fixed hardware capabilities:

```python
experiment = AWGAlazar.connect(
    "TCPIP0::192.168.10.171::inst0::INSTR",
    awg_sample_rate_hz=2.5e9,
    alazar_sample_rate_hz=1e9,
    acquire_window_s=1.5 * us,
    trigger_slope="rising",
    trigger_level=140,
)
```

`acquire_window_s` is the maximum raw ATS record requested for every shot.
It is independent of IQ integration.

`trigger_slope` and `trigger_level` are fixed ATS external-trigger electrical
settings. `trigger_level` is the Alazar SDK threshold code from 0 to 255, not
a voltage value.

ATS9371/ATS9373 replace each 12-bit ADC sample's least-significant bit with
the external-trigger state. `AWGAlazar` therefore masks that corrupted bit by
default and maps both trigger states to the center of the same effective
11-bit quantization bin. Pass `mask_external_trigger_lsb=False` to
`AWGAlazar.connect(...)` only for raw hardware diagnostics.

Readout processing settings belong to each experiment declaration:

```python
self.declare_readout(
    "ro",
    adc_channel=cfg["adc_channel"],
    length=cfg["ro_len"],
    demod_freq=cfg["f_res"],
    waveform_ch=cfg["res_ch"],
    marker_length=40 * ns,
    marker_lead=500 * ns,
    integrate_time=cfg["integrate_time"],
)
```

ATS trigger configuration belongs in the program body:

```python
self.trigger("ro")
```

Mark the pulse that is measured and demodulated with `readout=True`:

```python
self.add_pulse(
    "readout",
    gen="res",
    style="const",
    length=1 * us,
    frequency=cfg["f_res"],
    gain=0.02,
    readout=True,
)
```

The trigger marker is a fixed pulse at the beginning of every sequence step;
its width is controlled by `marker_length` and defaults to `40 ns`.
`marker_lead` independently controls the minimum delay from the marker edge to
the tagged readout pulse. If a pulse starts too early, the compiler shifts the
entire shot later while preserving all relative pulse timing. Keeping marker
width separate from readout timing is important on ATS9371/ATS9373 because the
external-trigger state replaces the ADC sample LSB. `self.trigger("ro")` uses
`marker_lead` as the default integration delay. Pass `trigger_delay=...`
explicitly after measuring the actual hardware time of flight.

`waveform_ch` may be used together with `marker_length`: the former tracks the
readout waveform/window and the latter controls only the trigger-pulse width.
ATS9371 begins the acquire window immediately after receiving the marker.
`trigger_delay` configures the integration-window offset inside that acquired
record. The delay must be identical for every sequence step.

When `compiled.acquire(...)` starts, its compatibility wrapper delegates to
`AWGAlazar`. The coordinator applies the ADC channel, demodulation frequency,
integration delay, and integration time, uploads the compiled AWG sequence
when needed, and collects the result.
`integrate_time` averages IQ from `trigger_delay` to
`trigger_delay + integrate_time`; that interval must fit inside the hardware
acquisition window.

It intentionally resembles QICK's program style:

```python
class MyProgram(ExperimentProgram):
    def _initialize(self, cfg):
        ...

    def _body(self, cfg):
        ...
```

The execution model is different from FPGA firmware. Sweeps are expanded at
compile time into AWG waveform assets and sequence-list steps. ATS records are
captured in the same interleaved order and reshaped using the compiler's record
layout.

The normal lifecycle is:

```text
ExperimentProgram.add_pulse()/play()/trigger()
    -> compile()
    -> CompiledExperiment(channel_waveforms, marker_waveforms, readout timing)
    -> compiled.upload() or compiled.acquire()
    -> AWGAlazar uploads AWG sequence once when needed
    -> AWGAlazar captures ATS acquire-window records
    -> Alazar DSP integrates shots; ExperimentResult performs reductions
```

## Data contract

The only acquisition averaging parameter is `n_average`:

```python
result = compiled.acquire(n_average=1000)
```

For `P` sequence points, the result keeps the unaveraged records:

```python
result.raw.shape
# (n_average, P, adc_sample)

result.iq_traces.shape
# (n_average, P, iq_sample)

result.iq_shots.shape
# (n_average, P)
```

Averaging is explicit:

```python
raw_average = result.raw_average("ro")
iq_average = result.iq_average("ro")
s21 = result.s21("ro")
```

This is important for single-shot experiments because the compiler does not
discard shot-level information.

## Basic program

```python
from QAWG import ExperimentProgram, LinearSweep, MHz, ns, us


class SpectroscopyProgram(ExperimentProgram):
    def _initialize(self, cfg):
        self.declare_gen("qubit", ch=4, amplitude_vpp=0.5)
        self.declare_gen("res", ch=3, amplitude_vpp=0.5)

        self.declare_readout(
            "ro",
            adc_channel="CHA",
            length=1 * us,
            demod_freq=cfg["f_res"],
            waveform_ch=3,
            marker_channel=1,
            integrate_time=800 * ns,
        )

        frequency = self.add_sweep(
            "frequency",
            LinearSweep(2920 * MHz, 3000 * MHz, 101),
        )

        self.add_pulse(
            "probe",
            gen="qubit",
            style="const",
            length=15 * us,
            frequency=frequency,
            gain=0.01,
        )
        self.add_pulse(
            "readout",
            gen="res",
            style="const",
            length=2 * us,
            frequency=cfg["f_res"],
            gain=0.02,
            readout=True,
        )

    def _body(self, cfg):
        self.play("probe", at=0)
        self.play("readout", at=0)
        self.trigger("ro")
```

Compile without hardware to inspect generated assets:

```python
program = SpectroscopyProgram(
    {"f_res": 50 * MHz},
    final_delay_s=1 * us,
)

compiled = program.compile(sample_rate_hz=2.5e9)
frequencies = compiled.axis("frequency")
qubit_steps = compiled.preview(channel=4)
```

Compile and bind to an existing `AWGAlazar` instance:

```python
compiled = experiment.compile(program)
result = compiled.acquire(n_average=1000)
```

One constant DC offset is removed from every acquired trace by default before
downconversion and integration. Disable it for a specific program only when
the raw record mean must be preserved:

```python
program = MyProgram(cfg, remove_dc_offset=False)
compiled = experiment.compile(program)
```

The default behavior subtracts each trace's full-record mean independently.

## Relative timing

`play()` without `at` advances a program cursor:

```python
self.play("pi")
self.delay_auto(40 * ns)
self.play("readout")
self.trigger("ro")
```

This compiles to:

```text
pi pulse -> 40 ns delay -> readout + marker
```

Explicit `at` allows overlapping pulses:

```python
self.play("probe", at=0)
self.play("readout", at=0)
self.trigger("ro", trigger_delay=650 * ns)
```

Here `at` controls pulse placement. The fixed marker begins the sequence step;
`marker_lead` controls pulse placement relative to it, while `trigger_delay`
controls the integration window inside the acquired record.

## Power Rabi

Power Rabi sweeps waveform gain. AWG5208 cannot update an FPGA gain register
per shot, so the compiler renders one waveform asset per gain and places those
assets in the sequence list.

```python
from QAWG import PowerRabiProgram

program = PowerRabiProgram({
    "f_res": 50e6,
    "f_ge": 100e6,
    "gain_start": 0.0,
    "gain_stop": 0.1,
    "steps": 51,
    "qubit_len": 100 * ns,
    "qubit_sigma": 15 * ns,
    "res_len": 1 * us,
    "res_gain": 0.02,
    "ro_len": 1 * us,
})

compiled = experiment.compile(program)
result = compiled.acquire(n_average=1000)

gain = result.axis("gain")
iq = result.s21("ro")
```

## T1

T1 sweeps `delay_auto()` between the pi pulse and readout. Every compiled
sequence step is padded to the same total duration, so trigger cadence does not
depend on the delay value.

```python
from QAWG import T1Program

program = T1Program({
    "f_res": 50e6,
    "f_ge": 100e6,
    "delay_start": 0,
    "delay_stop": 100 * us,
    "steps": 101,
    "pi_len": 100 * ns,
    "pi_sigma": 15 * ns,
    "pi_gain": 0.02,
    "res_len": 1 * us,
    "res_gain": 0.02,
    "ro_len": 1 * us,
})

result = experiment.compile(program).acquire(n_average=1000)
delay = result.axis("delay")
iq = result.s21("ro")
```

## Single shot

Single-shot calibration uses a categorical `state` axis. The `g` step skips
the pi pulse; the `e` step plays it.

```python
from QAWG import SingleShotProgram

program = SingleShotProgram(cfg)
compiled = experiment.compile(program)
result = compiled.acquire(n_average=10_000)

states = result.axis("state")
shots = result.iq_shots

ground = shots[:, 0]
excited = shots[:, 1]
```

No automatic average is performed by `shots()`.

## Logging to Labber HDF5

Acquisition and file I/O are deliberately separate. `AWGAlazar` only returns
data; `QAWG.Labber_saver` writes it afterward.

Save time-resolved IF records and complex downconverted waveforms:

```python
from QAWG.Labber_saver import createLogFile_ForData, getTraceDict

raw_name = "IF raw average"
iq_name = "Downconverted IQ average"
dt = np.median(np.diff(result.raw_time_s))
log = createLogFile_ForData(
    "tof_waveform",
    [
        {"name": raw_name, "unit": "V", "x_name": "Time", "x_unit": "s"},
        {"name": iq_name, "unit": "V", "complex": True,
         "x_name": "Time", "x_unit": "s"},
    ],
    database_folder=DATA_FOLDER,
)
log.addEntry({
    raw_name: getTraceDict(result.raw_average()[0], x0=0, dx=dt),
    iq_name: getTraceDict(result.iq_average()[0], x0=0, dx=dt),
})
```

Save a scalar sweep axis and complex S21:

```python
log = createLogFile_ForData(
    "resonator_s21",
    [{"name": "S21", "unit": "V", "complex": True, "vector": False}],
    [{"name": "SGS frequency", "unit": "Hz", "values": frequency_hz}],
    database_folder=DATA_FOLDER,
)
log.addEntry({"S21": resonator_iq})
```

`database_folder=DATA_FOLDER` makes Labber create:
`DATA_FOLDER/YYYY/MM/Data_MMDD/<name>.hdf5`. When omitted, Labber's configured
database folder is used.

## Current limits

- Sweeps are compile-time AWG sequence expansion, not FPGA runtime loops.
- Programs without sweeps still use the same backend as a one-step AWG
  sequence. This keeps upload, acquisition, and result shapes consistent.
- The current backend supports one readout named `"ro"`.
- The compiler produces a hardware-independent plan. Hardware upload and
  acquisition execution are owned by `AWGAlazar`.
- Sweep axes use Cartesian expansion when more than one axis is declared.
- Conditional playback currently supports equality checks such as
  `when=("state", "e")`.
- Waveform upload caching and sequence chunking are future optimizations.
- Hardware acquisition requires an already configured `AWGAlazar` instance
  whose sample rates and readout settings match the compiled program.
