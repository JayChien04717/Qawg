# ATS9371 100 MHz capture

This folder is a small, QCoDeS-free driver built directly on the official
AlazarTech `ATSApi.dll`.

## Wiring

```text
Mixer IF output (100 MHz) -> ATS9371 CH A
AWG marker/trigger output -> ATS9371 TRIG IN
```

The first configuration assumes:

- ATS9371 internal clock at 1 GS/s
- Channel A, DC coupled, 50 ohm, +/-400 mV
- External positive-edge TTL trigger
- NPT AutoDMA, one record per AWG trigger
- 2560 samples per record by default (256 cycles of a 100 MHz tone)

Verify the AWG trigger voltage and the ATS9371 TRIG IN termination before
connecting the instruments.

## Run

Run from the parent `AlazarTech` directory on the Windows computer containing
the ATS9371:

```powershell
python -m custmon.capture_100mhz --num-averages 1000 --samples 2560
```

The program arms the ATS9371 before waiting for the AWG trigger. It saves raw
ADC samples, voltage records, the 100 MHz complex amplitude, and the FFT peak
frequency to `capture_100mhz.npz`.

`samples` must currently be at least 256 and a multiple of 128.

For a longer acquisition, configure the record and DMA grouping separately:

```powershell
python -m custmon.capture_100mhz `
  --samples 2560 `
  --num-averages 1000 `
  --records-per-buffer 100 `
  --dma-buffer-count 4
```

Here, `samples` is the length captured after each trigger and `num-averages`
is both the number of repeated shots and the total trigger count. Each DMA
buffer completes after 100 triggers. Four DMA buffers are posted in a ring.
`num-averages` must be divisible by `records-per-buffer`.

## Three different delays

The integration delay skips samples after the trigger before calculating the
single dispersive IQ point. This corresponds to the `int_delay` idea in the
QCoDeS controller example:

```powershell
python -m custmon.capture_100mhz `
  --integration-delay-s 2e-7 `
  --integration-time-s 2e-6
```

At 1 GS/s these values select samples 200 through 2199.

`--trigger-delay-s` uses the hardware `AlazarSetTriggerDelay` call and delays
the start of record capture after the trigger. It changes which waveform is
transferred:

```powershell
python -m custmon.capture_100mhz --trigger-delay-s 2e-7
```

Neither setting slows down the AWG trigger repetition rate. Set the AWG
sequence repetition period separately to prevent the board and DMA pipeline
from being overrun.

## Two demodulation outputs

The saved `dispersive_iq` array contains one complex IQ point per trigger, and
`dispersive_iq_average` contains their final complex average. The
integration window is configured in seconds with `--integration-delay-s` and
`--integration-time-s`.

The saved `emission_iq` array contains one time-resolved complex envelope per
trigger. `emission_iq_average` is the final trace averaged over
`num-averages`. Processing first mixes 100 MHz to complex baseband, then
applies a boxcar moving average. The default window is 50 samples, or 50 ns
at 1 GS/s:

```powershell
python -m custmon.capture_100mhz `
  --integration-delay-s 2e-7 `
  --integration-time-s 2e-6 `
  --moving-average-samples 50
```

Apply the moving average after digital down-conversion. Averaging the original
100 MHz real waveform directly would cancel its positive and negative cycles.
