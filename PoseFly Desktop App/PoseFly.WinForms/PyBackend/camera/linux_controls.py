"""Native V4L2 exposure/ISO controls, without fabricated pixel effects.

Units: https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html
Exposure uses 100 us ticks; ISO integer-menu values use ISO * 1000.
"""
import fcntl
import os
import struct

EXPOSURE_AUTO = 0x009A0901
EXPOSURE_ABSOLUTE = 0x009A0902
ISO = 0x009A0917
ISO_AUTO = 0x009A0918
QUERY = struct.Struct('=II32siiiiIII')
CONTROL = struct.Struct('=Ii')
MENU = struct.Struct('=II32sI')


def ioctl(fd, number, payload):
    data = bytearray(payload)
    request = (3 << 30) | (len(data) << 16) | (ord('V') << 8) | number
    fcntl.ioctl(fd, request, data, True)
    return data


class Controls:
    def __init__(self, fd):
        self.fd = fd

    def query(self, control):
        values = QUERY.unpack(ioctl(self.fd, 36, QUERY.pack(control, 0, b'', 0, 0, 0, 0, 0, 0, 0)))
        if values[7] & (1 | 4):  # disabled or read-only
            raise OSError('Control is not writable')
        return values[1], values[3], values[4], max(1, values[5])

    def get(self, control):
        return CONTROL.unpack(ioctl(self.fd, 27, CONTROL.pack(control, 0)))[1]

    def set(self, control, value):
        ioctl(self.fd, 28, CONTROL.pack(control, int(value)))
        return self.get(control)

    def menu_values(self, control):
        kind, minimum, maximum, _ = self.query(control)
        if kind != 9:
            raise OSError('ISO control is not an integer menu')
        values = {}
        for index in range(minimum, maximum + 1):
            try:
                data = ioctl(self.fd, 37, MENU.pack(control, index, b'', 0))
                value = struct.unpack_from('=q', data, 8)[0]
                if value > 0:
                    values[index] = value
            except OSError:
                continue
        if not values:
            raise OSError('No supported ISO values')
        return values


def set_exposure(controls, shutter_hz):
    # Query before changing modes, then restore the old mode if exposure fails.
    _, low, high, step = controls.query(EXPOSURE_ABSOLUTE)
    previous_mode = controls.get(EXPOSURE_AUTO)
    try:
        if controls.set(EXPOSURE_AUTO, 1) != 1:
            raise OSError('Camera did not enter manual exposure')
        requested = 10000.0 / shutter_hz
        steps = max(0, min((high - low) // step, round((requested - low) / step)))
        ticks = low + steps * step
        actual = controls.set(EXPOSURE_ABSOLUTE, ticks)
        return f'Exposure {actual / 10:g} ms (requested {1000 / shutter_hz:g} ms)'
    except OSError:
        controls.set(EXPOSURE_AUTO, previous_mode)
        raise


def set_iso(controls, iso):
    choices = controls.menu_values(ISO)
    try:
        previous_mode = controls.get(ISO_AUTO)
    except OSError:
        previous_mode = None  # Some devices expose manual ISO without an auto control.
    try:
        if previous_mode is not None and controls.set(ISO_AUTO, 0) != 0:
            raise OSError('Camera did not enter manual ISO')
        index = min(choices, key=lambda i: abs(choices[i] - iso * 1000))
        actual = controls.set(ISO, index)
        if actual not in choices:
            raise OSError('Camera returned an unknown ISO value')
        return f'ISO {choices[actual] / 1000:g} (requested {iso})'
    except OSError:
        if previous_mode is not None:
            controls.set(ISO_AUTO, previous_mode)
        raise


def apply_controls(index, iso, shutter_hz, manual=True):
    try:
        fd = os.open(f'/dev/video{int(index)}', os.O_RDWR | os.O_NONBLOCK)
    except OSError as exc:
        return f'Camera controls unavailable: {exc.strerror}'
    try:
        controls = Controls(fd)
        if not manual:
            for mode in (3, 0):
                try:
                    if controls.set(EXPOSURE_AUTO, mode) == mode:
                        exposure = 'Auto exposure'
                        break
                except OSError:
                    continue
            else:
                exposure = 'Auto exposure unsupported'
            try:
                iso_text = 'Auto ISO' if controls.set(ISO_AUTO, 1) == 1 else 'Auto ISO unavailable'
            except OSError:
                iso_text = 'ISO control unsupported'
            return f'{exposure} | {iso_text}'
        try:
            exposure = set_exposure(controls, shutter_hz)
        except OSError as exc:
            exposure = f'Manual exposure unavailable: {exc}'
        try:
            iso_text = set_iso(controls, iso)
        except OSError:
            iso_text = 'ISO control unsupported (no artificial gain applied)'
        return f'{exposure} | {iso_text}'
    finally:
        os.close(fd)
