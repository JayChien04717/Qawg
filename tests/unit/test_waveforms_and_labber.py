from datetime import datetime

import h5py
import numpy as np

from QAWG.awg5200.waveforms import exponential_decay, exponential_rise
from QAWG.Labber_saver import createLogFile_ForData, getTraceDict


def test_exponential_envelopes_use_requested_tau():
    sample_rate = 1e9
    tau = 80e-9
    decay = exponential_decay(400, sample_rate, tau, 1.0)
    rise = exponential_rise(400, sample_rate, tau, 1.0)
    tau_sample = int(round(tau * sample_rate))
    np.testing.assert_allclose(decay[tau_sample], np.exp(-1.0), atol=1e-12)
    np.testing.assert_allclose(rise[tau_sample], 1.0 - np.exp(-1.0), atol=1e-12)


def test_create_log_keeps_averaged_complex_waveforms_in_correct_order(tmp_path):
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(4, 32)) * 1e-3
    iq = raw.astype(complex) + 1j * (0.5 * raw)
    time = np.arange(32) * 1e-9
    raw_average = np.mean(raw, axis=0)
    iq_average = np.mean(iq, axis=0)
    raw_name = "Digitizer - CHB - IF raw average"
    iq_name = "Digitizer - CHB - Downconverted IQ average"
    log = createLogFile_ForData(
        str(tmp_path / "waveforms.hdf5"),
        [
            {"name": raw_name, "unit": "V", "x_name": "Time", "x_unit": "s"},
            {
                "name": iq_name,
                "unit": "V",
                "complex": True,
                "x_name": "Time",
                "x_unit": "s",
            },
        ],
        use_database=False,
    )
    log.addEntry(
        {
            raw_name: getTraceDict(raw_average, x0=time[0], dx=1e-9),
            iq_name: getTraceDict(iq_average, x0=time[0], dx=1e-9),
        }
    )
    path = log.getFilePath()

    np.testing.assert_allclose(log.getData(raw_name, entry=0), raw_average)
    np.testing.assert_allclose(log.getData(iq_name, entry=0), iq_average)
    saved_time, saved_iq = log.getTraceXY(iq_name, entry=0)
    np.testing.assert_allclose(saved_time, time)
    np.testing.assert_allclose(saved_iq, iq_average)

    with h5py.File(path, "r") as h5:
        assert h5["Traces/Time stamp"].shape == (1,)
        assert list(h5["Log list"]["channel_name"]) == [
            raw_name.encode(),
            iq_name.encode(),
        ]
        assert h5[f"Traces/{raw_name}"].shape[1] == 1
        assert h5[f"Traces/{iq_name}"].shape[1] == 2
        assert "QAWG" not in h5


def test_create_log_keeps_axis_and_complex_s21(tmp_path):
    frequency_hz = np.linspace(5.8e9, 6.0e9, 11)
    s21 = 0.2 / (1 + 2j * (frequency_hz - 5.9e9) / 10e6)
    log = createLogFile_ForData(
        str(tmp_path / "resonator.hdf5"),
        [{"name": "S21", "unit": "V", "complex": True, "vector": False}],
        [{"name": "Frequency", "unit": "Hz", "values": frequency_hz}],
        use_database=False,
    )
    log.addEntry({"S21": s21})
    path = log.getFilePath()

    np.testing.assert_allclose(log.getAxis()[0]["axis_values"], frequency_hz)
    np.testing.assert_allclose(log.getData("S21", entry=0), s21)
    with h5py.File(path, "r") as h5:
        assert "Data" in h5
        assert "QAWG" not in h5


def test_create_log_uses_explicit_labber_database_folder(tmp_path):
    database = tmp_path / "LabberData"

    log = createLogFile_ForData(
        "resonator_folder_rule",
        [{"name": "S21", "complex": True, "vector": False}],
        [
            {
                "name": "Frequency",
                "unit": "Hz",
                "values": np.array([5.8e9, 5.9e9]),
            }
        ],
        database_folder=database,
    )
    log.addEntry({"S21": np.array([1 + 0j, 0.5 + 0.2j])})
    path = log.getFilePath()

    today = datetime.today()
    expected_parent = (
        database
        / today.strftime("%Y")
        / today.strftime("%m")
        / today.strftime("Data_%m%d")
    )
    assert path == str(expected_parent / "resonator_folder_rule.hdf5")
    assert expected_parent.joinpath("resonator_folder_rule.hdf5").is_file()


def test_create_log_saves_fixed_instrument_settings(tmp_path):
    settings = [
        {"name": "SGS100A - Frequency", "value": 5.9e9, "unit": "Hz"},
        {"name": "SGS100A - Power", "value": 4.0, "unit": "dBm"},
        {"name": "SGS100A - LO output", "value": "LO"},
        {"name": "AWG - CH1 amplitude", "value": 0.5, "unit": "Vpp"},
    ]
    log = createLogFile_ForData(
        str(tmp_path / "instrument_settings.hdf5"),
        [{"name": "S21", "unit": "V", "complex": True, "vector": False}],
        use_database=False,
        instrument_settings=settings,
    )
    log.addEntry({"S21": np.array([0.1 + 0.2j])})

    with h5py.File(log.getFilePath(), "r") as h5:
        config = h5[
            "Instrument config/Generic - GPIB: , Instrument settings at localhost"
        ]
        assert config.attrs["SGS100A - Frequency"] == 5.9e9
        assert config.attrs["SGS100A - Power"] == 4.0
        assert config.attrs["SGS100A - LO output"] == "LO"
        assert config.attrs["AWG - CH1 amplitude"] == 0.5

        channel_rows = {
            row["name"].decode(): row for row in h5["Channels"][:]
        }
        assert channel_rows["SGS100A - Frequency"]["unitPhys"] == b"Hz"
        assert channel_rows["SGS100A - Power"]["unitPhys"] == b"dBm"
        assert channel_rows["AWG - CH1 amplitude"]["unitPhys"] == b"Vpp"
        assert list(h5["Log list"]["channel_name"]) == [b"S21"]
