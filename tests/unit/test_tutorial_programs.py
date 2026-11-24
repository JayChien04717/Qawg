"""Compile every Program class shown in TOMOGRAPHY_WIGNER_ZH.md (no hardware)."""

import re
from pathlib import Path

import numpy as np

from QAWG import ExperimentProgram, LinearSweep, MHz, ValuesSweep, ns, us

TUTORIAL = Path(__file__).resolve().parents[2] / "TOMOGRAPHY_WIGNER_ZH.md"

EMISSION_CFG = dict(
    window=2.5 * us,
    emission_if=50 * MHz,
    ref_if=50 * MHz,
    qubit_if=50 * MHz,
    pi_gain=0.5,
)


def _class_blocks():
    text = TUTORIAL.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    return [block for block in blocks if re.search(r"^class \w+", block, re.M)]


def _namespace():
    namespace = {
        "np": np,
        "ExperimentProgram": ExperimentProgram,
        "LinearSweep": LinearSweep,
        "ValuesSweep": ValuesSweep,
        "MHz": MHz,
        "ns": ns,
        "us": us,
    }
    for block in _class_blocks():
        for definition in re.findall(
            r"^class .*?(?=^\S|\Z)", block, flags=re.S | re.M
        ):
            exec(definition, namespace)
    return namespace


def test_tomography_tutorial_emission_program_compiles():
    namespace = _namespace()
    program = namespace["Emission"](EMISSION_CFG, final_delay_s=20 * us)
    compiled = program.compile(sample_rate_hz=2.5e9)
    assert compiled.number_of_sequence_steps == 2
    assert compiled.marker_waveforms.any()
