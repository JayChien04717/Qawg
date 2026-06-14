# Tektronix AWG5208

The package separates instrument state from waveform calculations:

- `waveforms.py`: pure NumPy waveform, marker, and WFMX functions.
- `transport.py`: the small PyVISA boundary.
- `driver.py`: AWG5208 SCPI operations.

Multi-channel timing, `delay`, `delay_auto`, and `parallel` examples are in
[`TIMELINE_TUTORIAL.md`](TIMELINE_TUTORIAL.md).

Install hardware communication support:

```powershell
pip install pyvisa pyvisa-py
```

Example using a LAN VISA resource:

```python
import numpy as np

from awg5200 import AWG5208

sample_rate = 2.5e9
number_of_samples = 25_000
sample_index = np.arange(number_of_samples)
center_sample = 12_500
sigma_samples = 250
envelope = 0.2 * np.exp(
    -0.5 * ((sample_index - center_sample) / sigma_samples) ** 2
)

with AWG5208.connect("TCPIP0::192.168.1.50::inst0::INSTR") as awg:
    awg.set_awg_mode()
    awg.set_sample_rate(sample_rate)

    name = awg.upload_waveform(
        envelope,
        fc=100e6,
        ch=3,
        name="readout_100mhz",
        amplitude_vpp=0.5,
    )
    awg.marker(waveform_ch=3, marker_ch=1)
    awg.run()
```

`AWGControl:RUN` starts or arms the current asset. `force_trigger()` sends a
software A or B trigger. Triggered, repeated, or multi-channel experiments
normally use an AWG sequence whose steps wait for trigger A/B; sequence upload
is intentionally separate from basic waveform upload.
