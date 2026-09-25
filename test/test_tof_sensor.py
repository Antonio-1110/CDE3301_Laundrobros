"""ToF read guarding and timestamps (hardware/tof_sensor.py), no I2C."""

from laundry_control.hardware import tof_sensor


class _Sensor:

    def __init__(self, results):
        self.results = results

    def get_distance_cm(self):
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _reader(results, reinit_after=3):
    clock = iter(range(0, 10**6, 36))
    made = []

    def make():
        made.append(_Sensor(results))
        return made[-1]

    reader = tof_sensor.GuardedReader(
        make, now_ns=lambda: next(clock), warn=lambda _t: None,
        info=lambda _t: None, reinit_after=reinit_after,
    )
    return reader, made


def test_reading_is_stamped_mid_measurement():
    reader, _made = _reader([12.5])

    # The clock reads 0 before and 36 after the (blocking) read.
    assert reader.read() == (12.5, 18)


def test_i2c_errors_skip_the_reading_instead_of_raising():
    reader, _made = _reader([OSError(121, 'Remote I/O error'), 10.0])

    assert reader.read() is None
    assert reader.read()[0] == 10.0
    assert reader.failures == 0


def test_repeated_failures_reinitialise_the_sensor():
    shared = [OSError(5, 'EIO')] * 3 + [11.0]
    reader, made = _reader(shared, reinit_after=3)

    for _ in range(3):
        assert reader.read() is None

    assert len(made) == 2
    assert reader.read()[0] == 11.0


def test_driver_timeouts_are_caught_too():
    reader, _made = _reader([RuntimeError('Timeout waiting for VL53L0X!')])

    assert reader.read() is None
