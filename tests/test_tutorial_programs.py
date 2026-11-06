"""Compile every Program class shown in the tutorial (no hardware)."""

import re
from pathlib import Path

import numpy as np
import pytest

from QAWG import ExperimentProgram, LinearSweep, MHz, ValuesSweep, ns, us

TUTORIAL = Path(__file__).resolve().parents[1] / "QAWG_TUTORIAL_ZH.md"

DISPERSIVE_CFG = dict(
    ro_len=1 * us, ro_if=50 * MHz, ro_gain=0.3, qubit_if=100 * MHz, pi_gain=0.5,
    f_start=80 * MHz, f_stop=120 * MHz, points=11, probe_gain=0.05,
)
EMISSION_CFG = dict(
    window=2.5 * us, emission_if=50 * MHz, ref_if=50 * MHz,
    qubit_if=100 * MHz, pi_gain=0.5,
)


def _class_blocks():
    text = TUTORIAL.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    return [block for block in blocks if re.search(r"^class \w+", block, re.M)]


def _namespace():
    namespace = {
        "np": np, "ExperimentProgram": ExperimentProgram,
        "LinearSweep": LinearSweep, "ValuesSweep": ValuesSweep,
        "MHz": MHz, "ns": ns, "us": us, "T_MAX": 62 * us,
    }
    for block in _class_blocks():
        # keep only the class definitions; skip hardware/plot statements
        for definition in re.findall(
            r"^class .*?(?=^\S|\Z)", block, flags=re.S | re.M
        ):
            exec(definition, namespace)
    return namespace


@pytest.mark.parametrize(
    "name, cfg",
    [
        ("OneTone", DISPERSIVE_CFG),
        ("TwoTone", DISPERSIVE_CFG),
        ("RabiAmp", DISPERSIVE_CFG),
        ("T1", DISPERSIVE_CFG),
        ("Ramsey", DISPERSIVE_CFG),
        ("TOF", {}),
        ("Emission", EMISSION_CFG),
    ],
)
def test_tutorial_program_compiles(name, cfg):
    namespace = _namespace()
    program = namespace[name](cfg, final_delay_s=1 * us)
    compiled = program.compile(sample_rate_hz=2.5e9)
    assert compiled.number_of_sequence_steps >= 1
    assert compiled.marker_waveforms.any()


def test_t1_marker_follows_the_anchored_readout():
    namespace = _namespace()
    compiled = namespace["T1"](DISPERSIVE_CFG, final_delay_s=1 * us).compile(
        sample_rate_hz=2.5e9
    )
    starts = {int(np.flatnonzero(m)[0]) for m in compiled.marker_waveforms}
    assert len(starts) == 1
    assert starts.pop() == round((62 * us - 200 * ns) * 2.5e9)
