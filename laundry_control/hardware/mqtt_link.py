"""
A small MQTT connection for talking to the ESP32 (esp32_protocol.py).

Wraps paho-mqtt: connects in the background and keeps reconnecting,
re-subscribes after every reconnect, and turns Nagle's algorithm off
(TCP_NODELAY), which would otherwise hold small packets - a clock
sync ping - back for up to tens of ms waiting for more to send.

Callbacks run on paho's network thread, not the ROS executor.

Works with paho-mqtt 1.x (Ubuntu's python3-paho-mqtt) and 2.x (pip).
paho is imported in MqttLink(), so importing this module needs none.

The broker is mosquitto on this Pi (setup/mosquitto/README.md).
"""

import socket
import threading


class MqttLink:

    def __init__(
        self,
        host,
        port,
        client_id,
        will=None,
        log_info=print,
        log_warn=print,
    ):
        """
        Set up the client; start() connects.

        will: (topic, payload) published by the broker if this client
        drops off without disconnecting, or None.
        """
        import paho.mqtt.client as mqtt

        self.host = host
        self.port = port
        self._info = log_info
        self._warn = log_warn
        self._handlers = {}
        self._connected = threading.Event()
        self._ever_connected = False

        if hasattr(mqtt, 'CallbackAPIVersion'):
            self._client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2, client_id=client_id
            )
        else:
            self._client = mqtt.Client(client_id=client_id)

        if will is not None:
            self._client.will_set(will[0], will[1], qos=1, retain=True)

        self._client.reconnect_delay_set(min_delay=1, max_delay=5)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def subscribe(self, topic, callback, qos=0):
        """Call callback(payload_bytes) for each message on `topic`."""
        self._handlers[topic] = (callback, qos)

        if self.connected:
            self._client.subscribe(topic, qos)

    def start(self):
        """Connect in the background; reconnects by itself from then on."""
        self._info(f'MQTT: connecting to {self.host}:{self.port}...')
        self._client.connect_async(self.host, self.port, keepalive=10)
        self._client.loop_start()

    def stop(self):
        self._client.disconnect()
        self._client.loop_stop()

    @property
    def connected(self):
        return self._connected.is_set()

    def wait_connected(self, timeout_sec):
        return self._connected.wait(timeout_sec)

    def publish(self, topic, payload, qos=0, retain=False):
        """Publish; False if not connected (nothing is queued then)."""
        if not self.connected:
            return False

        info = self._client.publish(topic, payload, qos=qos, retain=retain)
        return info.rc == 0

    # paho 1.x and 2.x call these with different arguments, hence *rest.

    def _on_connect(self, client, userdata, flags, reason, *rest):
        if _failed(reason):
            self._warn(f'MQTT: broker refused the connection: {reason}')
            return

        try:
            client.socket().setsockopt(
                socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
            )
        except (AttributeError, OSError):
            pass

        for topic, (_callback, qos) in self._handlers.items():
            client.subscribe(topic, qos)

        self._connected.set()
        self._info(
            f'MQTT: {"re" if self._ever_connected else ""}connected to '
            f'{self.host}:{self.port}.'
        )
        self._ever_connected = True

    def _on_disconnect(self, client, userdata, *rest):
        self._connected.clear()
        self._warn('MQTT: disconnected from the broker; reconnecting...')

    def _on_message(self, client, userdata, message):
        handler = self._handlers.get(message.topic)

        if handler is None:
            return

        try:
            handler[0](message.payload)
        except Exception as exc:
            # An exception here would otherwise vanish into paho's thread.
            self._warn(f'MQTT: handler for {message.topic} failed: {exc!r}')


def _failed(reason):
    """Return True if a connect result (int or paho 2 ReasonCode) failed."""
    is_failure = getattr(reason, 'is_failure', None)

    if is_failure is not None:
        return bool(is_failure)

    return reason != 0
