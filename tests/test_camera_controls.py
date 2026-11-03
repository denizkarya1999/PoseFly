"""V4L2 conversions and driver rejection tests; no camera is opened."""
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'PoseFly Desktop App/PoseFly.WinForms/PyBackend'))
from camera import linux_controls as lc


class ControlTests(unittest.TestCase):
    def test_exposure_uses_100_microsecond_units(self):
        controls = Mock()
        controls.query.return_value = (1, 1, 10000, 1)
        controls.get.return_value = 3
        controls.set.side_effect = lambda control, value: value
        report = lc.set_exposure(controls, 1000)
        self.assertIn('Exposure 1 ms', report)
        self.assertEqual(controls.set.call_args_list[0].args, (lc.EXPOSURE_AUTO, 1))
        self.assertEqual(controls.set.call_args_list[1].args, (lc.EXPOSURE_ABSOLUTE, 10))

    def test_exposure_respects_range_and_step_and_reports_actual_value(self):
        controls = Mock()
        controls.query.return_value = (1, 5, 30, 4)
        controls.get.return_value = 3
        controls.set.side_effect = lambda control, value: value
        self.assertIn('Exposure 0.5 ms', lc.set_exposure(controls, 6000))
        lc.set_exposure(controls, 5)
        self.assertEqual(controls.set.call_args.args, (lc.EXPOSURE_ABSOLUTE, 29))

    def test_exposure_failure_restores_previous_auto_mode(self):
        controls = Mock()
        controls.query.return_value = (1, 1, 10000, 1)
        controls.get.return_value = 3
        def set_value(control, value):
            if control == lc.EXPOSURE_ABSOLUTE:
                raise OSError('unsupported')
            return value
        controls.set.side_effect = set_value
        with self.assertRaises(OSError):
            lc.set_exposure(controls, 1000)
        self.assertEqual(controls.set.call_args.args, (lc.EXPOSURE_AUTO, 3))

    def test_iso_selects_supported_integer_menu_index(self):
        controls = Mock()
        controls.menu_values.return_value = {0: 100000, 1: 200000, 2: 400000}
        controls.get.return_value = 1
        controls.set.side_effect = lambda control, value: value
        report = lc.set_iso(controls, 250)
        self.assertIn('ISO 200 (requested 250)', report)
        self.assertEqual(controls.set.call_args.args, (lc.ISO, 1))

    def test_unavailable_controls_are_reported_and_fd_closed(self):
        controls = Mock()
        controls.query.side_effect = OSError('unsupported')
        controls.menu_values.side_effect = OSError('unsupported')
        with patch.object(lc.os, 'open', return_value=5), \
             patch.object(lc.os, 'close') as close, \
             patch.object(lc, 'Controls', return_value=controls):
            report = lc.apply_controls(42, 250, 1000)
        self.assertIn('Manual exposure unavailable', report)
        self.assertIn('ISO control unsupported', report)
        controls.set.assert_not_called()
        close.assert_called_once_with(5)

    def test_ioctl_layout_and_integer_menu_readback(self):
        def driver(fd, request, data, mutate):
            self.assertEqual(fd, 5)
            if request == 0xC0445624:
                data[:] = lc.QUERY.pack(lc.ISO, 9, b'ISO', 0, 1, 1, 0, 0, 0, 0)
            elif request == 0xC02C5625:
                index = lc.MENU.unpack(data)[1]
                lc.struct.pack_into('=q', data, 8, (index + 1) * 100000)
            else:
                self.fail(f'Unexpected ioctl {request:x}')
        with patch.object(lc.fcntl, 'ioctl', side_effect=driver):
            self.assertEqual(lc.Controls(5).menu_values(lc.ISO), {0: 100000, 1: 200000})


if __name__ == '__main__':
    unittest.main()
