# ESP32 bridge: ToF sensor and gripper over MQTT

`tof_gripper_bridge/` is the ESP32 firmware. It reads the VL53L0X,
drives the gripper servo, and talks to the Pi over WiFi through
mosquitto on the Pi (`setup/mosquitto/README.md`). On the Pi, the
same `tof_sensor` and `gripper_node` nodes switch to it with
`tof_source:=mqtt gripper_backend:=mqtt`. Nothing after them
changes: the scan recorder and the pipeline see the same topic and
services.

The protocol (topics and payloads) and the clock sync are documented
in `laundry_control/hardware/esp32_protocol.py`.

## Wiring

| From | To |
|---|---|
| VL53L0X VIN / GND | ESP32 3V3 / GND |
| VL53L0X SDA / SCL | `I2C_SDA_PIN` / `I2C_SCL_PIN` (top of the sketch) |
| Servo signal | `SERVO_PIN` (top of the sketch) |
| Servo V+ / GND | **a separate 5-6 V supply**, its GND tied to the ESP32's GND |

Don't power the servo from the ESP32's 3V3 or 5V pin. A stalling
servo pulls amps, which browns out the board and drops WiFi.

## Build and flash (Arduino IDE 2, or arduino-cli)

1. Install the board package "esp32 by Espressif Systems" and pick
   your board (for example "LOLIN D32" / "MH ET LIVE ESP32MiniKit", or
   "ESP32C3 Dev Module" for a C3 SuperMini).
2. Install these libraries from the Library Manager: **PubSubClient**
   (Nick O'Leary), **VL53L0X** (Pololu), **ESP32Servo** (Kevin
   Harrington), and **ArduinoJson** (Benoit Blanchon, v7).
3. `cp tof_gripper_bridge/config.h.example tof_gripper_bridge/config.h`,
   then set your WiFi and the Pi's LAN IP (`hostname -I` on the Pi).
   Check the pins at the top of the sketch. `config.h` is git-ignored.
4. Flash it, then open the serial monitor at 115200. It should print
   `VL53L0X ready` and `MQTT connected`.

## Timestamps: why the ESP32 stamps its own readings

The scan recorder places each reading where the arm was at the
reading's timestamp. J7 sweeps about 0.2 mm/ms at the bucket wall. If
the Pi stamped readings when they arrived, WiFi latency would go into
the stamp: typically 3-20 ms, with spikes past 100 ms. That is
several mm to 2 cm of error.

Instead, the ESP32 stamps each reading at the middle of its
measurement, using its own clock (`esp_timer_get_time()`). The Pi
works out the offset between the two clocks with NTP-style pings,
keeping the fastest round trip of the last 20. In a test with
simulated WiFi latency (median 20 ms, max 115 ms each way), the stamps
came out within 2 ms of the true capture time. The firmware has to
keep `loop()` non-blocking for this to work. The comment at the top of
the sketch says why.

## Behaviour to know

- **Gripper commands** are acknowledged once the move has settled.
  The Pi's open/close service fails if no ack arrives within
  settle + 2 s.
- **hold:** after a move, the pulses stop (as on the Pi) unless the
  command says `hold`. The ESP32's PWM is hardware-timed, so the shake
  the Pi's software PWM caused with `hold_closed` should be gone.
  That's untested.
- **Disconnects** don't move the servo. The broker publishes
  `offline` on `laundrobros/esp32/status` (the ESP32's last will), and
  `gripper_node` then fails moves at once instead of waiting.
- **Reboot:** a new random boot id makes the Pi redo the clock sync.
  Readings are held back for about a second while it does.
