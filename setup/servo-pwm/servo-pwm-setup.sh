#!/bin/sh
#
# Run as root at boot (servo-pwm.service): export the Pi 5's RP1 PWM
# channel that drives GPIO 18 (the gripper servo) and hand its control
# files to the user who runs gripper_node, so laundry_control can use
# hardware PWM without root. See README.md next to this file.
#
# Usage: servo-pwm-setup.sh <user>

set -eu

USER_NAME="${1:?usage: $0 <user>}"
RP1_PWM_DEVICE=1f00098000.pwm
CHANNEL=2
PERIOD_NS=20000000   # 50 Hz, the servo's frame rate

# The RP1 PWM driver can come up a little after this service starts.
chip=""
for _ in $(seq 20); do
    for candidate in /sys/class/pwm/pwmchip*; do
        case "$(readlink -f "$candidate/device")" in
            *"$RP1_PWM_DEVICE") chip="$candidate" ;;
        esac
    done
    [ -n "$chip" ] && break
    sleep 1
done

if [ -z "$chip" ]; then
    echo "servo-pwm: no PWM chip for $RP1_PWM_DEVICE; is" \
        "'dtoverlay=pwm,pin=18,func=2' in /boot/firmware/config.txt?" >&2
    exit 1
fi

if [ ! -d "$chip/pwm$CHANNEL" ]; then
    echo "$CHANNEL" > "$chip/export"
fi

# The channel's files appear a moment after the export.
for _ in $(seq 20); do
    [ -e "$chip/pwm$CHANNEL/enable" ] && break
    sleep 0.1
done

echo 0 > "$chip/pwm$CHANNEL/enable"
echo "$PERIOD_NS" > "$chip/pwm$CHANNEL/period"

for name in period duty_cycle enable; do
    chown "$USER_NAME" "$chip/pwm$CHANNEL/$name"
done

echo "servo-pwm: $chip/pwm$CHANNEL ready for $USER_NAME"
