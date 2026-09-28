"""
Tests for scan_recorder_node's wait-for-/tf queue (scan/recorder_node.py).

Driven through a stub rather than a real node: a real one would join
the ROS graph, and on the rig answer the live save_scan calls.
"""

from collections import deque
import math

from builtin_interfaces.msg import Time as TimeMsg
from laundry_control.scan import recorder_node
from laundry_control.scan.recorder_node import ScanRecorderNode, TF_WAIT_SEC
import pytest
from rclpy.time import Time
from sensor_msgs.msg import JointState, Range


def _stamp(seconds):
    return TimeMsg(sec=int(seconds), nanosec=int(round(seconds % 1 * 1e9)))


def _reading(seconds, distance=0.3):
    msg = Range()
    msg.header.stamp = _stamp(seconds)
    msg.header.frame_id = 'tof_sensor_link'
    msg.range = distance
    msg.min_range = 0.03
    msg.max_range = 2.0
    return msg


class _Logger:

    def warning(self, *_a, **_k):
        pass


class _Clock:

    def __init__(self, stub):
        self.stub = stub

    def now(self):
        return Time(nanoseconds=int(self.stub.now * 1e9))


class _Recorder:
    """The node's queue logic over a fake /tf that reaches `tf_latest`."""

    _drain = ScanRecorderNode._drain
    _record_fallback = ScanRecorderNode._record_fallback
    _range_callback = ScanRecorderNode._range_callback
    _joint_state_callback = ScanRecorderNode._joint_state_callback
    _sweep_angle_at = ScanRecorderNode._sweep_angle_at

    def __init__(self, now, tf_latest):
        self.now = now
        self.tf_latest = tf_latest
        self._pending = deque()
        self._sweep_history = deque()
        self._tf_exact_count = 0
        self._tf_fallback_count = 0
        self._tf_dropped_count = 0
        self.recorded = []

    def get_clock(self):
        return _Clock(self)

    def get_logger(self):
        return _Logger()

    def _lookup(self, sensor_frame, when):
        if isinstance(when, Time):
            return 'latest', 'latest'

        seconds = when.sec + when.nanosec * 1e-9

        if seconds > self.tf_latest:
            raise RuntimeError('extrapolation into the future')

        return 'exact', 'exact'

    def _record(self, msg, sensor_frame, tcp_transform, base_transform):
        stamp = msg.header.stamp
        self.recorded.append((stamp.sec + stamp.nanosec * 1e-9, tcp_transform))


def test_a_reading_waits_for_tf_then_uses_its_own_time():
    recorder = _Recorder(now=10.00, tf_latest=9.95)

    recorder._range_callback(_reading(9.98))

    # /tf has not reached 9.98 yet: nothing placed, nothing lost.
    assert recorder.recorded == []
    assert len(recorder._pending) == 1

    recorder.tf_latest = 10.05
    recorder.now = 10.06
    recorder._drain()

    assert recorder.recorded == [(pytest.approx(9.98), 'exact')]
    assert recorder._tf_exact_count == 1
    assert recorder._tf_fallback_count == 0


def test_a_reading_tf_never_reaches_falls_back_after_the_wait():
    recorder = _Recorder(now=10.00, tf_latest=9.00)

    recorder._range_callback(_reading(9.99))
    recorder.now = 9.99 + TF_WAIT_SEC - 0.01
    recorder._drain()

    assert recorder.recorded == []

    recorder.now = 9.99 + TF_WAIT_SEC + 0.01
    recorder._drain()

    assert recorder.recorded == [(pytest.approx(9.99), 'latest')]
    assert recorder._tf_fallback_count == 1


def test_readings_stay_in_capture_order():
    recorder = _Recorder(now=10.00, tf_latest=9.90)

    for stamp in (9.92, 9.96, 9.99):
        recorder._range_callback(_reading(stamp))

    recorder.tf_latest = 9.97
    recorder._drain()

    assert [stamp for stamp, _ in recorder.recorded] == [
        pytest.approx(9.92), pytest.approx(9.96)
    ]
    assert len(recorder._pending) == 1


def test_save_places_everything_still_queued():
    recorder = _Recorder(now=10.00, tf_latest=9.97)

    recorder._range_callback(_reading(9.95))
    recorder._range_callback(_reading(9.99))

    # Already placed: the first, exact. Forced: the second, which
    # /tf has not reached, with the latest transform.
    recorder._drain(force=True)

    assert recorder.recorded == [
        (pytest.approx(9.95), 'exact'), (pytest.approx(9.99), 'latest'),
    ]
    assert not recorder._pending


def test_out_of_range_readings_are_ignored():
    recorder = _Recorder(now=10.00, tf_latest=10.00)

    recorder._range_callback(_reading(9.99, distance=5.0))

    assert recorder.recorded == [] and not recorder._pending


def test_sweep_angle_is_interpolated_at_the_readings_time():
    recorder = _Recorder(now=10.00, tf_latest=10.00)

    assert math.isnan(recorder._sweep_angle_at(9.95))

    for stamp, angle in ((9.9, 1.0), (10.0, 2.0)):
        msg = JointState()
        msg.header.stamp = _stamp(stamp)
        msg.name = ['joint6', recorder_node.SWEEP_JOINT_NAME]
        msg.position = [0.0, angle]
        recorder._joint_state_callback(msg)

    assert recorder._sweep_angle_at(9.95) == pytest.approx(1.5)


def test_a_reading_also_waits_for_the_joint_state_past_it():
    recorder = _Recorder(now=10.00, tf_latest=10.00)

    msg = JointState()
    msg.header.stamp = _stamp(9.90)
    msg.name = [recorder_node.SWEEP_JOINT_NAME]
    msg.position = [1.0]
    recorder._joint_state_callback(msg)

    # /tf has reached 9.95, but /joint_states only 9.90: wait.
    recorder._range_callback(_reading(9.95))

    assert recorder.recorded == []

    msg.header.stamp = _stamp(10.00)
    msg.position = [2.0]
    recorder._joint_state_callback(msg)
    recorder._drain()

    assert recorder.recorded == [(pytest.approx(9.95), 'exact')]
