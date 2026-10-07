"""
The MQTT protocol to the ESP32, and the clock sync for its readings.

The ESP32 (esp32/tof_gripper_bridge) owns the ToF sensor and the
gripper servo; ClockSync stamps its readings in this Pi's time.

Pure logic: no paho, no rclpy, so it is testable anywhere. The MQTT
connection is mqtt_link.py; the nodes using this are tof_sensor.py
(source:=mqtt) and gripper_node.py (backend:=mqtt).

TOPICS (prefix config.MQTT_TOPIC_PREFIX, "laundrobros")
------
    <prefix>/tof/range      ESP32 -> Pi   QoS 0
        {"seq": 812, "t_us": 48211034, "mm": 143, "boot": "3fa2c1d0"}
        One reading: raw millimetres (the Pi applies offset_cm), and
        t_us, the ESP32's esp_timer_get_time() at the MIDDLE of the
        measurement window. boot is random per ESP32 boot.
    <prefix>/time/ping      Pi -> ESP32   QoS 0      "<id>"
    <prefix>/time/pong      ESP32 -> Pi   QoS 0
        {"id": 17, "t_us": 48211102, "boot": "3fa2c1d0"}
        Answered at once, t_us taken when the ping arrived.
    <prefix>/gripper/cmd    Pi -> ESP32   QoS 1, never retained
        {"id": 4, "angle": 100.0, "hold": false, "settle": 1.5}
    <prefix>/gripper/ack    ESP32 -> Pi   QoS 0
        {"id": 4, "ok": true, "msg": "..."}
        Sent once the move has settled (and the pulses stopped,
        unless hold), so a service call returns when the claw is done.
    <prefix>/esp32/status   ESP32 -> Pi   retained   "online"/"offline"
        "offline" is the ESP32's MQTT last will: the broker publishes
        it when the ESP32 drops off.

QoS 0 for the timing-critical topics on purpose: a re-sent reading
or pong would arrive late, and a late pong is a wrong clock sample.
Gripper commands are never retained, or the broker would replay the
last one to an ESP32 that reboots mid-run.

CLOCK SYNC
----------
scan_recorder_node places each reading with the arm's pose at its
header.stamp. During a scan J7 turns ~2 deg in 36 ms (~7 mm at the
bucket wall, ~0.2 mm per ms), so the stamp must say when the light
hit the laundry, in THIS Pi's clock. Stamping on arrival would add
the WiFi latency, which wanders between ~3 ms and 100+ ms (more with
the ESP32's WiFi power save on). So the ESP32 stamps each reading
with its own clock, and ClockSync maps that clock to the Pi's, the
way NTP does:

    t0 = Pi sends ping            t_esp = ESP32 receives it
    t1 = Pi receives the pong
    offset = (t0 + t1) / 2 - t_esp,   error <= (t1 - t0) / 2

A ping that waited somewhere (a busy ESP32 loop, a WiFi retry, a
busy Pi) has a long round trip and a loose bound, so of the last
CLOCK_SYNC_WINDOW pings the one with the SHORTEST round trip wins.
The ESP32's crystal drifts by tens of ppm: over the ~10 s window
that is well under a millisecond, so no drift term. Until the window
is full (after each ESP32 boot) the node pings in a fast burst, so
the estimate is tight within about a second instead of ten.

The ESP32's clock restarts when it reboots; a new boot id resets the
sync, and readings are held back until a pong from the new boot.
"""

from dataclasses import dataclass
import json
import threading

# The ESP32's status payloads (its retained status topic).
ONLINE = 'online'
OFFLINE = 'offline'


@dataclass(frozen=True)
class Topics:
    range: str  # noqa: A003 - the ToF range topic
    ping: str
    pong: str
    gripper_cmd: str
    gripper_ack: str
    status: str


def topics(prefix):
    """Return every topic name under `prefix`."""
    prefix = prefix.rstrip('/')

    return Topics(
        range=f'{prefix}/tof/range',
        ping=f'{prefix}/time/ping',
        pong=f'{prefix}/time/pong',
        gripper_cmd=f'{prefix}/gripper/cmd',
        gripper_ack=f'{prefix}/gripper/ack',
        status=f'{prefix}/esp32/status',
    )


def _json(payload):
    """Decode a JSON object payload; None if it is not one."""
    try:
        data = json.loads(payload)
    except (ValueError, TypeError):
        return None

    return data if isinstance(data, dict) else None


@dataclass(frozen=True)
class RangeReading:
    seq: int
    t_us: int
    mm: float
    boot: str


def decode_range(payload):
    """Return the RangeReading in a range payload; None if malformed."""
    data = _json(payload)

    try:
        return RangeReading(
            seq=int(data['seq']),
            t_us=int(data['t_us']),
            mm=float(data['mm']),
            boot=str(data['boot']),
        )
    except (KeyError, TypeError, ValueError):
        return None


def decode_pong(payload):
    """Return (id, t_us, boot) from a pong payload; None if malformed."""
    data = _json(payload)

    try:
        return int(data['id']), int(data['t_us']), str(data['boot'])
    except (KeyError, TypeError, ValueError):
        return None


def encode_ping(ping_id):
    return str(int(ping_id)).encode()


def encode_gripper_cmd(cmd_id, angle, hold, settle_sec):
    return json.dumps(
        {
            'id': int(cmd_id),
            'angle': float(angle),
            'hold': bool(hold),
            'settle': float(settle_sec),
        }
    ).encode()


def decode_gripper_ack(payload):
    """Return (id, ok, msg) from an ack payload; None if malformed."""
    data = _json(payload)

    try:
        return int(data['id']), bool(data['ok']), str(data.get('msg', ''))
    except (KeyError, TypeError, ValueError):
        return None


class ClockSync:
    """
    Map the ESP32's microsecond clock to this Pi's nanosecond clock.

    ping(now_ns) -> payload to publish on the ping topic.
    pong(payload, now_ns) feeds the answer back. Then stamp(reading,
    now_ns) gives a reading's capture time in Pi nanoseconds, or None
    with the reason it was held back (see stamp()).

    now_ns must be the clock the readings are stamped in (the node's
    ROS clock), read as close to the send/receive as possible.

    Thread-safe: pings go out from a ROS timer, pongs and readings
    come in on paho's network thread.
    """

    def __init__(
        self,
        window=20,
        max_rtt_ns=100_000_000,
        max_age_ns=2_000_000_000,
        future_slack_ns=5_000_000,
    ):
        # Pongs kept; the shortest round trip among them is used.
        self._window = window
        # Round trips longer than this are never used: their bound is
        # too loose to be worth having.
        self._max_rtt_ns = max_rtt_ns
        # Readings older than this on arrival are dropped: they sat in
        # a buffer through a WiFi outage, and the scan has moved on.
        self._max_age_ns = max_age_ns
        # A reading may appear this far in the future (sync error)
        # before the sync is assumed broken.
        self._future_slack_ns = future_slack_ns

        self._lock = threading.RLock()
        self._next_id = 0
        self._sent = {}
        self.reset()

    def reset(self, boot=None):
        with self._lock:
            self.boot = boot
            self._samples = []

    @property
    def synced(self):
        return bool(self._samples)

    @property
    def filling(self):
        """Return True until the window is full (ping fast meanwhile)."""
        return len(self._samples) < self._window

    def _best(self):
        return min(self._samples) if self._samples else None

    @property
    def offset_ns(self):
        """Pi ns minus ESP32 ns (best sample); None before the first pong."""
        best = self._best()
        return None if best is None else best[1]

    @property
    def uncertainty_ns(self):
        """Half the best round trip: the bound on offset_ns's error."""
        best = self._best()
        return None if best is None else best[0] // 2

    def _ping(self, now_ns):
        """Start a round trip; return the ping payload to publish."""
        self._next_id += 1

        # Forget pings that never came back (lost, or the ESP32 was
        # offline), so this never grows.
        stale = now_ns - 10 * self._max_rtt_ns
        self._sent = {k: t for k, t in self._sent.items() if t >= stale}

        self._sent[self._next_id] = now_ns
        return encode_ping(self._next_id)

    def _pong(self, payload, now_ns):
        """
        Use a pong; return the round trip in ns, or None if unusable.

        Unusable: malformed, not one of our pings (or answered
        already), or slower than max_rtt_ns.
        """
        decoded = decode_pong(payload)

        if decoded is None:
            return None

        ping_id, t_us, boot = decoded
        sent_ns = self._sent.pop(ping_id, None)

        if sent_ns is None:
            return None

        if boot != self.boot:
            self.reset(boot)

        rtt_ns = now_ns - sent_ns

        if rtt_ns < 0 or rtt_ns > self._max_rtt_ns:
            return None

        offset_ns = (sent_ns + now_ns) // 2 - t_us * 1000

        self._samples.append((rtt_ns, offset_ns))
        del self._samples[: -self._window]

        return rtt_ns

    def _stamp(self, reading, now_ns):
        """
        Return (stamp_ns, None), or (None, reason) to drop the reading.

        Reasons: not synced with this boot yet; too old; or in the
        future - which means the sync is wrong (e.g. this Pi's clock
        was stepped by NTP), so it is reset and redone.
        """
        if reading.boot != self.boot or not self._samples:
            return None, 'clock not synced with the ESP32 yet'

        stamp_ns = reading.t_us * 1000 + self.offset_ns

        if stamp_ns > now_ns + self.uncertainty_ns + self._future_slack_ns:
            self.reset(self.boot)
            return None, (
                f'reading stamped {(stamp_ns - now_ns) * 1e-6:.1f} ms in the '
                'future: clock sync reset'
            )

        if now_ns - stamp_ns > self._max_age_ns:
            return None, (
                f'reading {(now_ns - stamp_ns) * 1e-9:.2f} s old on arrival'
            )

        return stamp_ns, None

    def ping(self, now_ns):
        """Start a round trip; return the ping payload to publish."""
        with self._lock:
            return self._ping(now_ns)

    def pong(self, payload, now_ns):
        """Use a pong; return its round trip in ns, or None if unusable."""
        with self._lock:
            return self._pong(payload, now_ns)

    def stamp(self, reading, now_ns):
        """Return (stamp_ns, None), or (None, reason) to drop the reading."""
        with self._lock:
            return self._stamp(reading, now_ns)


class SequenceGaps:
    """Count readings lost in transit, from the ESP32's sequence numbers."""

    def __init__(self):
        self.lost = 0
        self._last = None
        self._boot = None

    def update(self, reading):
        """Return how many readings were lost just before this one."""
        if reading.boot != self._boot or self._last is None:
            gap = 0
        elif reading.seq <= self._last:
            return 0  # Out of order: already counted as lost. Rare.
        else:
            gap = reading.seq - self._last - 1

        self._boot = reading.boot
        self._last = reading.seq
        self.lost += gap
        return gap
