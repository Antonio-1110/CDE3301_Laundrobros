#!/usr/bin/env bash
# Start (if needed) and check the MQTT broker the ESP32 talks to.
#
#   mqtt_broker/broker.sh          make sure mosquitto is up, show status
#   mqtt_broker/broker.sh --watch  same, then print every laundrobros/ message
#
# mosquitto is a systemd service (enabled at boot), so this normally only
# reports; it starts the service (sudo) when it is not running.
set -u

PORT=1883
TOPIC='laundrobros/#'

if ! systemctl is-active --quiet mosquitto; then
    echo "mosquitto is not running, starting it (sudo)..."
    sudo systemctl start mosquitto || { echo "could not start mosquitto" >&2; exit 1; }
    sleep 1
fi

systemctl is-active --quiet mosquitto || { echo "mosquitto failed to start: journalctl -u mosquitto" >&2; exit 1; }
if ss -ltn | grep -q ":$PORT "; then
    echo "mosquitto: running, listening on port $PORT"
else
    echo "mosquitto is running but nothing listens on port $PORT: check $(dirname "$0")/laundrobros.conf in /etc/mosquitto/conf.d/" >&2
    exit 1
fi

echo
echo "Set MQTT_HOST in the ESP32's config.h to one of:"
echo "  $(hostname).local        (mDNS, preferred)"
for ip in $(hostname -I); do echo "  $ip"; done

if [ "${1:-}" = "--watch" ]; then
    echo
    echo "Watching $TOPIC (Ctrl-C to stop). Expect 'esp32/status online', then ~30 tof/range a second."
    exec mosquitto_sub -h localhost -t "$TOPIC" -v
fi
