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

    def __init__(self, hold_closed=None, hold_open=None):
        from laundry_control.hardware.gripper_node import GripperNode

        self._close_callback = GripperNode._close_callback.__get__(self)
        self._open_callback = GripperNode._open_callback.__get__(self)
        self._move_to = GripperNode._move_to.__get__(self)
        self._driver, self.made = _driver()
        self.params = {'open_angle_deg': 55.0, 'close_angle_deg': 100.0,
                       'hold_closed': False, 'hold_open': False,
                       'settle_sec': 0.0}

        if hold_closed is not None:
            self.params['hold_closed'] = hold_closed

        if hold_open is not None:
            self.params['hold_open'] = hold_open

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


def test_hold_open_keeps_the_open_claw_driven():
    from std_srvs.srv import Trigger

    node = _GripperNode(hold_open=True)

    node._open_callback(Trigger.Request(), Trigger.Response())

    assert node.made[0].angles == [55.0]
    assert node._driver.holding


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


def test_first_move_runs_at_real_time_priority_then_returns(monkeypatch):
    from laundry_control.hardware import gripper_node
    from std_srvs.srv import Trigger

    calls = []
    monkeypatch.setattr(
        gripper_node, 'enter_realtime',
        lambda priority: calls.append(('enter', priority)),
    )
    monkeypatch.setattr(
        gripper_node, 'leave_realtime', lambda: calls.append(('leave',)),
    )

    node = _GripperNode()
    node._driver = None
    node._enter_realtime = gripper_node.GripperNode._enter_realtime.__get__(
        node
    )
    node.params['rt_priority'] = 60
    node.get_logger = lambda: type(
        'Logger', (), {'info': print, 'warning': print}
    )()

    import laundry_control.hardware.servo as servo_module

    made = []

    def driver():
        made.append(ServoDriver(make_servo=_FakeServo, settle_sec=0.0))
        return made[-1]

    monkeypatch.setattr(servo_module, 'ServoDriver', driver)

    node._close_callback(Trigger.Request(), Trigger.Response())
    node._open_callback(Trigger.Request(), Trigger.Response())

    # Only around the first move, and back to normal after it.
    assert calls == [('enter', 60), ('leave',)]


def test_real_time_refused_is_reported_not_fatal():
    from laundry_control.hardware import gripper_node

    reason = gripper_node.enter_realtime(60)

    if reason is None:
        gripper_node.leave_realtime()
        pytest.skip('this user may use real-time priority')

    assert 'not permitted' in reason.lower()
