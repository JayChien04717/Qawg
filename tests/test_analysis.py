import unittest

import numpy as np

from QAWG import ExperimentResult, calculate_window


class WindowAnalysisTests(unittest.TestCase):
    def test_calculate_window_uses_readout_duration_not_late_transient(self):
        raw_time_s = np.arange(1500) / 1e9
        iq_time_s = np.arange(1481) / 1e9
        envelope = np.zeros(iq_time_s.size)
        envelope[120:720] = 1.0
        envelope[1100:1110] = 2.0
        raw = np.zeros((4, 1, raw_time_s.size))
        iq_traces = np.tile(
            envelope.astype(complex),
            (4, 1, 1),
        )
        result = ExperimentResult(
            axes={},
            point_coordinates=({},),
            raw=raw,
            iq_traces=iq_traces,
            iq_shots=np.mean(iq_traces, axis=2),
            raw_time_s=raw_time_s,
            iq_time_s=iq_time_s,
            initial_trigger_delay_s=500e-9,
            readout_windows_s=np.array([[500e-9, 1100e-9]]),
            marker_windows_s=np.array([[0.0, 1600e-9]]),
            acquire_window_s=1500e-9,
            remove_dc_offset=True,
        )

        analysis = calculate_window(
            result,
            plot=False,
            report=False,
        )

        self.assertAlmostEqual(
            analysis.suggested_trigger_delay_s,
            600e-9,
        )
        self.assertAlmostEqual(
            analysis.integration_stop_s,
            640e-9,
        )
        self.assertTrue(result.remove_dc_offset)


if __name__ == "__main__":
    unittest.main()
