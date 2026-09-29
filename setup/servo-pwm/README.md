# Hardware PWM for the gripper servo (Raspberry Pi 5)

The servo reads its target angle from the length of a pulse sent 50
times a second. By default the pulses come from software (lgpio), and
on a busy Pi they come out uneven: the claw jitters and misses moves.
The Pi 5's RP1 chip has PWM hardware on GPIO 18, which makes exact
pulses whatever the CPU is doing. This sets it up. Once it's done,
`laundry_control` uses it automatically (`hardware/servo.py`), and
falls back to software PWM when it isn't set up.

It needs sudo and one reboot. The servo stays on GPIO 18, so nothing
is rewired. It works the same on Ubuntu 24.04 (this rig) and on Raspberry
Pi OS: both boot through the Pi firmware, which reads
`/boot/firmware/config.txt` and its overlays, and Ubuntu's `raspi` kernel
includes the RP1 PWM driver (`pwm-rp1`).

## 1. Switch on the PWM hardware on GPIO 18

Add this line at the end of `/boot/firmware/config.txt` (the `[all]`
section):

```
dtoverlay=pwm,pin=18,func=2
```

- `dtoverlay=pwm`: switch on the RP1 PWM block, which is off at boot.
- `pin=18,func=2`: connect GPIO 18 to it (PWM0 channel 2), instead of
  plain GPIO.

```bash
echo 'dtoverlay=pwm,pin=18,func=2' | sudo tee -a /boot/firmware/config.txt
```

## 2. Hand the PWM channel to your user at every boot

```bash
cd ~/ros2_ws/src/CDE3301_Laundrobros/setup/servo-pwm
sudo install -m 755 servo-pwm-setup.sh /usr/local/sbin/
sudo install -m 644 servo-pwm.service /etc/systemd/system/
sudo systemctl enable servo-pwm.service
```

- `servo-pwm-setup.sh` runs as root at boot. It activates ("exports")
  channel 2 of that PWM block, sets a 20 ms (50 Hz) period, and makes
  your user the owner of the channel's three control files (`period`,
  `duty_cycle`, `enable`). Nothing else changes owner.
- `servo-pwm.service` runs that script once at every boot.
  `systemctl enable` switches it on for future boots.

If your user isn't `cde3301a`, edit the `ExecStart` line in
`servo-pwm.service` first.

## 3. Reboot and check

```bash
sudo reboot
# after the reboot:
systemctl status servo-pwm.service     # should end "...pwm2 ready for cde3301a"
ls -l /sys/class/pwm/pwmchip*/pwm2/     # period, duty_cycle, enable owned by you
```

Then start the bring-up and try `laundry gripper close` / `open`.

## Undo

```bash
sudo systemctl disable servo-pwm.service
sudo rm /etc/systemd/system/servo-pwm.service /usr/local/sbin/servo-pwm-setup.sh
sudo sed -i '/^dtoverlay=pwm,pin=18,func=2$/d' /boot/firmware/config.txt
sudo reboot
```
