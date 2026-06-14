import unittest

from QAWG.instrument.manager import BaseInstrumentManager
from QAWG.instrument.sgs100a import RohdeSchwarzSGS100A


class FakeSGS100A(RohdeSchwarzSGS100A):
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.responses: dict[str, str] = {}

    def write(self, cmd: str) -> None:
        self.commands.append(cmd)

    def query(self, cmd: str) -> str:
        return self.responses.get(cmd, "0")


class SGS100ATests(unittest.TestCase):
    def setUp(self) -> None:
        self.sgs = FakeSGS100A()

    def test_configure_external_reference_clock(self) -> None:
        self.sgs.configure_reference_clock("EXT", external_frequency_hz=10e6)

        self.assertEqual(
            self.sgs.commands,
            [
                "SOUR:ROSC:EXT:FREQ 10MHZ",
                "SOUR:ROSC:SOUR EXT",
            ],
        )

    def test_external_reference_requires_frequency(self) -> None:
        with self.assertRaisesRegex(ValueError, "external_frequency_hz"):
            self.sgs.configure_reference_clock("EXT")

    def test_configure_lo_output(self) -> None:
        self.sgs.configure_lo_output(True)
        self.sgs.configure_lo_output(False)

        self.assertEqual(
            self.sgs.commands,
            [
                "CONN:REFL:OUTP LO",
                "CONN:REFL:OUTP OFF",
            ],
        )

    def test_configure_reference_output(self) -> None:
        self.sgs.configure_lo_output(
            True,
            mode="REF",
            reference_frequency_hz=100e6,
        )

        self.assertEqual(
            self.sgs.commands,
            [
                "SOUR:ROSC:OUTP:FREQ 100MHZ",
                "CONN:REFL:OUTP REF",
            ],
        )

    def test_legacy_property_names_remain_available(self) -> None:
        self.sgs.LO_source = "EXT"
        self.sgs.ref_LO_out = "LO"

        self.assertEqual(
            self.sgs.commands,
            [
                "SOUR:LOSC:SOUR EXT",
                "CONN:REFL:OUTP LO",
            ],
        )

    def test_manager_configures_clock_lo_and_iq(self) -> None:
        manager = BaseInstrumentManager()
        manager._instruments["readout"] = type(
            "Spec",
            (),
            {"driver": self.sgs},
        )()

        manager.configure_sgs100a(
            "readout",
            reference_source="EXT",
            reference_frequency_hz=10e6,
            lo_output=True,
            iq_modulation=True,
        )

        self.assertEqual(
            self.sgs.commands,
            [
                "SOUR:ROSC:EXT:FREQ 10MHZ",
                "SOUR:ROSC:SOUR EXT",
                "CONN:REFL:OUTP LO",
                ":IQ:STAT 1",
            ],
        )


if __name__ == "__main__":
    unittest.main()
