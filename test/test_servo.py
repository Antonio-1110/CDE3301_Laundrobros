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
                       'hold_closed': False}

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
