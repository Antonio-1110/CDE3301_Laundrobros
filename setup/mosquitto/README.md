# MQTT broker for the ESP32 link

The ESP32 (`esp32/tof_gripper_bridge`) owns the ToF sensor and the
gripper servo, and talks to this Pi over MQTT, through mosquitto
running on this Pi. Only needed with `tof_source:=mqtt` or
`gripper_backend:=mqtt` (`config.TOF_SOURCE` / `GRIPPER_BACKEND`).

## 1. Install and configure mosquitto

```
sudo apt install mosquitto mosquitto-clients
sudo cp setup/mosquitto/laundrobros.conf /etc/mosquitto/conf.d/
sudo systemctl restart mosquitto
sudo systemctl enable mosquitto
```

The Python client (`paho-mqtt`) is in `requirements.txt`, so it goes
into the workspace venv with `pip install -r requirements.txt`.

## 2. Check it

Watch everything the ESP32 sends (readings, pongs, acks, status):

```
mosquitto_sub -h localhost -t 'laundrobros/#' -v
```

`laundrobros/esp32/status online` should show up as soon as the ESP32
connects, then about 30 `laundrobros/tof/range` messages a second.

Ping it by hand (it answers on `laundrobros/time/pong`):

```
mosquitto_pub -h localhost -t laundrobros/time/ping -m 1
```

## 3. Use it

```
ros2 launch laundry_control laundry_bringup.launch.py \
    tof_source:=mqtt gripper_backend:=mqtt
```

Or set `TOF_SOURCE = 'mqtt'` / `GRIPPER_BACKEND = 'mqtt'` in
`laundry_control/config.py` to make it the default. That setting also
switches `laundry gripper ANGLE`.

`tof_sensor` logs `Clock synced with ESP32 boot ...: +/- N ms` about a
second after the ESP32 connects. N is the bound on the timestamp
error. It should be a few ms. If it stays above
`config.ESP32_CLOCK_SYNC_WARN_MS` the node keeps warning: check the
ESP32's WiFi signal and that its power save is off
(`WiFi.setSleep(false)`, already in the sketch).

## Firewall

The broker has no password, so only let the rig's own network reach
it. On this Pi, `eth0` is on the rig router (192.168.1.x) and `wlan0`
is on the campus WiFi. With `ufw` on, open the port on `eth0` only:

```
sudo ufw allow in on eth0 to any port 1883 proto tcp
sudo ufw status
```

Don't use a plain `sudo ufw allow 1883/tcp`: that would open the
broker to the whole campus network as well.
