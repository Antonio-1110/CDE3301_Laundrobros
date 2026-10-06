"""The ESP32 MQTT protocol, clock sync and servo driver, with no broker."""

import json
import threading

from laundry_control.hardware import esp32_protocol as proto
from laundry_control.hardware.mqtt_servo import (
    GripperLinkError,
    MqttServoDriver,
)
import pytest

MS = 1_000_000

# The ESP32's clock reads this many ns less than the Pi's.
TRUE_OFFSET_NS = 1_700_000_000_000 * MS


def _pong(ping_payload, esp_ns, boot='b1'):
    return json.dumps(
        {'id': int(ping_payload), 't_us': esp_ns // 1000, 'boot': boot}
    ).encode()


def _round_trip(sync, t0_ns, out_ms, back_ms, boot='b1'):
    """One ping: out_ms on the way to the ESP32, back_ms on the way back."""
    ping = sync.ping(t0_ns)
    esp_ns = t0_ns + out_ms * MS - TRUE_OFFSET_NS
    return sync.pong(_pong(ping, esp_ns, boot), t0_ns + (out_ms + back_ms) * MS)


def _reading(t_esp_ns, boot='b1', seq=1):
    return proto.RangeReading(seq=seq, t_us=t_esp_ns // 1000, mm=100, boot=boot)


def test_symmetric_round_trip_recovers_the_exact_offset():
    sync = proto.ClockSync()

    assert _round_trip(sync, 0, 4, 4) == 8 * MS
    assert sync.offset_ns == TRUE_OFFSET_NS
    assert sync.uncertainty_ns == 4 * MS


def test_the_shortest_round_trip_wins_over_slow_lopsided_ones():
    sync = proto.ClockSync()

    # A ping that sat 60 ms in the ESP32's queue, then a clean one.
    _round_trip(sync, 0, 62, 3)
    _round_trip(sync, 500 * MS, 3, 3)
    _round_trip(sync, 1000 * MS, 40, 2)

    assert sync.offset_ns == TRUE_OFFSET_NS
    assert sync.uncertainty_ns == 3 * MS


def test_error_is_bounded_by_half_the_round_trip():
    sync = proto.ClockSync()
    _round_trip(sync, 0, 10, 0)  # All the delay on one leg: worst case.

    assert abs(sync.offset_ns - TRUE_OFFSET_NS) <= sync.uncertainty_ns


def test_old_samples_leave_the_window():
    sync = proto.ClockSync(window=2)
    _round_trip(sync, 0, 1, 1)
    _round_trip(sync, 1, 30, 30)
    _round_trip(sync, 2, 30, 30)

    assert sync.uncertainty_ns == 30 * MS


def test_slow_unknown_and_malformed_pongs_are_ignored():
    sync = proto.ClockSync(max_rtt_ns=50 * MS)

    assert _round_trip(sync, 0, 40, 40) is None
    assert sync.pong(b'{"id": 999, "t_us": 1, "boot": "b1"}', 0) is None
    assert sync.pong(b'not json', 0) is None
    assert not sync.synced


def test_a_pong_is_only_used_once():
    sync = proto.ClockSync()
    ping = sync.ping(0)
    pong = _pong(ping, 2 * MS - TRUE_OFFSET_NS)

    assert sync.pong(pong, 4 * MS) is not None
    assert sync.pong(pong, 5 * MS) is None


def test_reading_is_stamped_at_its_capture_time_not_its_arrival():
    sync = proto.ClockSync()
    _round_trip(sync, 0, 3, 3)

    captured_ns = 1000 * MS
    arrived_ns = captured_ns + 45 * MS  # A WiFi hiccup.

    stamp, reason = sync.stamp(
        _reading(captured_ns - TRUE_OFFSET_NS), arrived_ns
    )

    assert reason is None
    assert stamp == captured_ns


def test_readings_wait_for_a_sync_with_the_same_boot():
    sync = proto.ClockSync()

    assert sync.stamp(_reading(0), 0)[0] is None

    _round_trip(sync, 0, 3, 3, boot='b1')

    stamp, reason = sync.stamp(_reading(0, boot='b2'), 10 * MS)
    assert stamp is None and 'not synced' in reason


def test_reboot_resets_the_sync():
    sync = proto.ClockSync()
    _round_trip(sync, 0, 1, 1, boot='b1')
    _round_trip(sync, 500 * MS, 20, 20, boot='b2')

    assert sync.boot == 'b2'
    assert sync.uncertainty_ns == 20 * MS


def test_reading_from_the_future_resets_the_sync():
    sync = proto.ClockSync()
    _round_trip(sync, 0, 1, 1)

    # As if this Pi's clock had been stepped back 1 s by NTP.
    stamp, reason = sync.stamp(_reading(2000 * MS - TRUE_OFFSET_NS), 1000 * MS)

    assert stamp is None and 'future' in reason
    assert not sync.synced


def test_stale_readings_are_dropped():
    sync = proto.ClockSync(max_age_ns=2000 * MS)
    _round_trip(sync, 0, 1, 1)

    stamp, reason = sync.stamp(_reading(0 - TRUE_OFFSET_NS), 2500 * MS)

    assert stamp is None and 'old' in reason


def test_decode_range():
    reading = proto.decode_range(
        b'{"seq": 7, "t_us": 123456, "mm": 143, "boot": "ab"}'
    )

    assert reading == proto.RangeReading(7, 123456, 143.0, 'ab')
    assert proto.decode_range(b'{"seq": 7}') is None
    assert proto.decode_range(b'[1, 2]') is None
    assert proto.decode_range(b'\xff') is None


def test_sequence_gaps_count_lost_readings_but_not_reboots():
    gaps = proto.SequenceGaps()

    assert gaps.update(_reading(0, seq=1)) == 0
    assert gaps.update(_reading(0, seq=2)) == 0
    assert gaps.update(_reading(0, seq=5)) == 2
    assert gaps.update(_reading(0, boot='b2', seq=1)) == 0
    assert gaps.lost == 2


def test_out_of_order_readings_are_not_counted_twice():
    gaps = proto.SequenceGaps()

    for seq in (1, 3, 2, 4):
        gaps.update(_reading(0, seq=seq))

    assert gaps.lost == 1


def test_filling_until_the_window_is_full():
    sync = proto.ClockSync(window=2)
    _round_trip(sync, 0, 1, 1)
    assert sync.filling

    _round_trip(sync, 1, 1, 1)
    assert not sync.filling


def test_topics_share_the_prefix():
    topics = proto.topics('lab/')

    assert topics.range == 'lab/tof/range'
    assert topics.gripper_cmd == 'lab/gripper/cmd'


class _FakeLink:
    """An MqttLink stand-in; `answer` plays the ESP32."""

    def __init__(self, answer=None, connected=True):
        self.answer = answer
        self.connected = connected
        self.handlers = {}
        self.sent = []
        self.stopped = False

    def subscribe(self, topic, callback, qos=0):
        self.handlers[topic] = callback

    def wait_connected(self, timeout_sec):
        return self.connected

    def publish(self, topic, payload, qos=0, retain=False):
        self.sent.append((topic, json.loads(payload), qos, retain))

        if self.answer is not None:
            reply = self.answer(json.loads(payload))

            if reply is not None:
                # Answer from another thread, as paho would.
                threading.Thread(
                    target=self.handlers[TOPICS.gripper_ack],
                    args=(json.dumps(reply).encode(),),
                ).start()

        return True

    def stop(self):
        self.stopped = True


TOPICS = proto.topics('laundrobros')


def _driver(link):
    return MqttServoDriver(link, TOPICS, settle_sec=0.0, ack_margin_sec=0.5)


def test_move_returns_once_the_esp32_acks():
    link = _FakeLink(answer=lambda cmd: {'id': cmd['id'], 'ok': True})

    _driver(link).move(100.0, hold=True, settle_sec=0.2)

    topic, cmd, qos, retain = link.sent[0]
    assert topic == TOPICS.gripper_cmd
    assert cmd['angle'] == 100.0 and cmd['hold'] and cmd['settle'] == 0.2
    assert qos == 1 and not retain


def test_missing_ack_fails_the_move():
    with pytest.raises(GripperLinkError, match='no ack'):
        _driver(_FakeLink()).move(55.0)


def test_ack_for_another_command_is_not_taken():
    link = _FakeLink(answer=lambda cmd: {'id': cmd['id'] - 1, 'ok': True})

    with pytest.raises(GripperLinkError, match='no ack'):
        _driver(link).move(55.0)


def test_refused_move_fails_with_the_esp32s_reason():
    link = _FakeLink(
        answer=lambda cmd: {'id': cmd['id'], 'ok': False, 'msg': 'brownout'}
    )

    with pytest.raises(GripperLinkError, match='brownout'):
        _driver(link).move(55.0)


def test_offline_esp32_or_broker_fails_fast_without_sending():
    link = _FakeLink()
    driver = _driver(link)
    link.handlers[TOPICS.status](b'offline')

    with pytest.raises(GripperLinkError, match='offline'):
        driver.move(55.0)

    with pytest.raises(GripperLinkError, match='broker'):
        _driver(_FakeLink(connected=False)).move(55.0)

    assert link.sent == []


def test_out_of_range_angle_never_reaches_the_esp32():
    link = _FakeLink()

    with pytest.raises(ValueError):
        _driver(link).move(500.0)

    assert link.sent == []
