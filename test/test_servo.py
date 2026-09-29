"""The servo driver's hold/release behaviour (hardware/servo.py), no GPIO."""

from laundry_control.hardware.servo import ServoDriver
import pytest


class _FakeServo:
    """Mimics gpiozero AngularServo: value None means no pulses."""

    def __init__(self):
        self.value = None
        self.closed = False
        self.angles = []

    @property
    def angle(self):
        return self.angles[-1] if self.angles else None

    @angle.setter
    def angle(self, value):
        self.angles.append(value)
        self.value = 0.0

    def detach(self):
        self.value = None

    def close(self):
        self.closed = True


def _driver():
    made = []

    def make():
        made.append(_FakeServo())
        return made[-1]

    return ServoDriver(make_servo=make, settle_sec=0.0), made


def test_close_keeps_driving_and_open_releases():
    driver, made = _driver()

    driver.move(100.0, hold=True)
    assert driver.holding

    driver.move(55.0, hold=False)
    assert not driver.holding
    # One servo object across moves, so the hold is never interrupted.
    assert len(made) == 1 and made[0].angles == [100.0, 55.0]


def test_close_frees_the_pin():
    driver, made = _driver()
    driver.move(100.0, hold=True)

    driver.close()

    assert made[0].closed and made[0].value is None
    assert not driver.holding


class _Parameter:

    def __init__(self, value):
        self.value = value


class _GripperNode:
    """gripper_node's service logic over a fake servo, no ROS graph."""

    def __init__(self, hold_closed=None):
        from laundry_control.hardware.gripper_node import GripperNode

        self._close_callback = GripperNode._close_callback.__get__(self)
        self._open_callback = GripperNode._open_callback.__get__(self)
        self._move_to = GripperNode._move_to.__get__(self)
        self._driver, self.made = _driver()
        self.params = {'open_angle_deg': 55.0, 'close_angle_deg': 100.0,
                       'hold_closed': False, 'settle_sec': 0.0}

        if hold_closed is not None:
            self.params['hold_closed'] = hold_closed

    def get_parameter(self, name):
        return _Parameter(self.params[name])


def test_close_stops_the_pulses_by_default():
    from std_srvs.srv import Trigger

    node = _GripperNode()

    response = node._close_callback(Trigger.Request(), Trigger.Response())

    assert response.success
    assert node.made[0].angles == [100.0]
    assert not node._driver.holding


def test_hold_closed_keeps_driving_until_open():
    from std_srvs.srv import Trigger

    node = _GripperNode(hold_closed=True)

    node._close_callback(Trigger.Request(), Trigger.Response())
    assert node._driver.holding

    node._open_callback(Trigger.Request(), Trigger.Response())
    assert not node._driver.holding


def test_out_of_range_angle_never_touches_the_servo():
    driver, made = _driver()

    with pytest.raises(ValueError):
        driver.move(400.0)

    assert made == []


def test_settle_time_is_how_long_pulses_go_out(monkeypatch):
    from laundry_control.hardware import servo

    slept = []
    monkeypatch.setattr(servo.time, 'sleep', slept.append)

    driver, made = _driver()
    driver.move(100.0)
    driver.move(55.0, settle_sec=2.0)

    # The driver's own default (0 here), then the per-move override.
    assert slept == [0.0, 2.0]
    assert servo.SETTLE_SEC >= 1.5


def _fake_pwm_class(tmp_path, device='1f00098000.pwm', exported=True):
    """Build a /sys/class/pwm lookalike: one chip and, maybe, channel 2."""
    platform = tmp_path / 'devices' / device
    platform.mkdir(parents=True)
    chip = tmp_path / 'class' / 'pwmchip3'
    chip.mkdir(parents=True)
    (chip / 'device').symlink_to(platform)

    if exported:
        channel = chip / 'pwm2'
        channel.mkdir()
        for name in ('period', 'duty_cycle', 'enable'):
            (channel / name).write_text('0')

    return str(tmp_path / 'class')


def test_the_hardware_channel_is_found_only_when_set_up(tmp_path):
    from laundry_control.hardware import servo

    assert servo.hardware_pwm_channel(str(tmp_path / 'nothing')) is None

    other = _fake_pwm_class(tmp_path / 'a', device='107d517a80.pwm')
    assert servo.hardware_pwm_channel(other) is None

    unexported = _fake_pwm_class(tmp_path / 'b', exported=False)
    assert servo.hardware_pwm_channel(unexported) is None

    ready = _fake_pwm_class(tmp_path / 'c')
    assert servo.hardware_pwm_channel(ready).endswith('pwmchip3/pwm2')


def test_hardware_servo_writes_the_pulse_width(tmp_path):
    import os

    from laundry_control.hardware import servo

    channel = servo.hardware_pwm_channel(_fake_pwm_class(tmp_path))

    def read(name):
        with open(os.path.join(channel, name)) as handle:
            return int(handle.read())

    motor = servo.HardwarePwmServo(channel)
    assert read('period') == 20_000_000 and read('enable') == 0

    # 0.5-2.5 ms over 0-180 deg: 90 deg is 1.5 ms.
    motor.angle = 90.0
    assert read('duty_cycle') == 1_500_000 and read('enable') == 1
    assert motor.value is not None

    motor.detach()
    assert read('enable') == 0 and motor.value is None

    # ServoDriver works on it unchanged: close holds or releases.
    driver = ServoDriver(make_servo=lambda: servo.HardwarePwmServo(channel),
                         settle_sec=0.0)
    driver.move(100.0)
    assert read('enable') == 0 and not driver.holding
    driver.move(100.0, hold=True)
    assert read('enable') == 1 and driver.holding
