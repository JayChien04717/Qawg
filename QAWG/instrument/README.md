# External instruments

`QAWG.instrument` contains control drivers for laboratory instruments that are
not part of AWG waveform rendering or Alazar acquisition.

## SGS100A

The SGS100A supplies the microwave carrier. The AWG supplies the lower
frequency I/Q waveform, while the SGS100A performs IQ upconversion.

```python
from QAWG.instrument import InstrumentManager

inst = InstrumentManager()
sgs = inst.add_sgs100a("readout_lo", "TCPIP::192.168.0.10::INSTR")

inst.configure_sgs100a(
    "readout_lo",
    reference_source="EXT",
    reference_frequency_hz=10e6,
    lo_output=True,
    iq_modulation=True,
)

sgs.frequency = 6e9
sgs.power = 10
sgs.on()
```

The configuration above:

1. locks the SGS100A to the external 10 MHz reference input;
2. routes the carrier to the rear `REF/LO OUT` connector;
3. enables external analog I/Q modulation.

The `REF/LO OUT` connector is shared. It can output either LO, reference, or
nothing:

```python
sgs.configure_lo_output(True, mode="LO")
sgs.configure_lo_output(
    True,
    mode="REF",
    reference_frequency_hz=10e6,
)
sgs.configure_lo_output(False)
```

Reference input can also be configured directly:

```python
sgs.configure_reference_clock("EXT", external_frequency_hz=10e6)
sgs.configure_reference_clock("INT")
```
