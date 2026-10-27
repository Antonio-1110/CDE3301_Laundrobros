# Performance: what simulation says

How long the scan and retrieve motions take, and how well the detector does, measured without the arm. Measured 2026-10-19 on the Pi 5, at the default settings. To redo it: [`benchmarks/sim_timing.sh`](../benchmarks/sim_timing.sh) (motion times, on an isolated fake controller) and [`benchmarks/sim_metrics.sh`](../benchmarks/sim_metrics.sh) (detection and coverage, plain files).

**What these numbers are.** The fake controller runs MoveIt's planned trajectories in real time, so the arm's motion times match what the rig is given. They leave out the gripper (open or close is 0.5 s on the rig with the ESP32, `ESP32_GRIPPER_SETTLE_SEC`, plus its ack) and the real arm's tracking, so the rig is a few seconds slower per grasp. **They say nothing about whether a grasp picks the laundry up**: simulation has no cloth, and a grasp counts as done when the motions finish. That needs the rig.

## Scan time

From INTER, to INTER, with `--fake-hardware`. A run also spends about 3 s connecting to MoveIt.

| Scan | Time | Strokes | Notes |
|---|---|---|---|
| Quick (default) | **97 s** | 12 in, 12 out | `laundry scan` |
| Full (`--full`) | **140 s** | 12 in, end scan, 12 out | adds the tilting end scan, about 43 s |
| Quick, `--step 0.06` | 57 s | 6 in, 6 out | |
| Quick, `--step 0.12` | 37 s | 3 in, 3 out | misses about half the socks (see `--step` help) |

Where the quick scan's 97 s goes: 7 s from INTER to the mouth, 35 s inward strokes (each stroke is a 150° J7 sweep at velocity 0.03, about 3 s), 7 s turnaround, 35 s outward strokes, 9 s out and back to INTER. The J7 sweep dominates, which is what issue #10 is about (shorter scans). Everything is baked or planned live from the same speeds, so run-to-run variation was under 1 s (a repeat of `--step 0.12` took 36.7 s against 37.3 s).

## Retrieve time

| What | Time (simulation) | On the rig |
|---|---|---|
| Sensorless sweep, all 24 grabs of the full-drum plan (`laundry preplanned`) | 346 s at 45 °/s (2026-10-19); 253.8 s at 60 °/s / 7 rad/s² (2026-10-27, CPU-loaded; 178 s of it motion, see below) | about 7 min at 45 °/s; not yet re-measured |
| One grab of the sweep, INTER to INTER via DROP | 12-17 s, mean 14.4 s | mean about 17.4 s |
| `laundry run`, one item in the bucket | **125 s** | about 130 s |

- **The sweep** is 2 layers of 12 grabs (about 19 and 13 cm above the floor, for a pile top at 25 cm, `config.PILE`): 163 s for the first layer, 173 s for the second. Deeper grabs take longer (13-16 s at 10 cm from the closed end, 13-14 s elsewhere). The rig adds about 3 s per grab, for the claw to close and to open at DROP. A fuller or emptier drum changes the heights and the number of layers; see `docs/operating.md`. (An earlier version of these numbers, 196 s for 12 grabs, was measured on the old plan that aimed 2 cm above the floor.) **2026-10-21, at 60 deg/s:** 213 s for all 24 grabs (8.9 s per grab), with the sweep's moves checked while the claw moves, the route in and the descent (and the lift and the route to DROP) as one trajectory each, and direct grab <-> DROP routes that skip INTER (`arm/direct.py`); 242-246 s without the direct routes. The rig adds the claw's 0.5 s twice per grab, minus the checks that overlap it; today's rig log, before these changes, took about 14 s per grab.
- **`laundry run`, one item (125 s):** scan 93 s (75%), detection 0.4 s, grasp 28 s. The grasp is about 10 s of reachability probing and 18 s of moving, including the drop. The rig adds about 4.5 s for the three gripper moves (open before the scan, close, release).
- **`laundry clear` is an estimate, not a measurement.** Each further item costs a rescan: about 97 s scan, 30 s grasp and 4.5 s of gripper, so **about 2.2 min per item**, plus about 1.6 min for the final scan that finds nothing (about 2.5 min if the end scan runs on its own). With the grab grid first (`--no-grabs` skips it), add the sweep's 3.8 min.

Where the laundry is changes the grasp part, not the scan: each baked route to a grab spot takes about 1-1.5 s each way and DROP 2.36 s each way at 60 °/s / 7 rad/s² (2.2 and 3.6 s at the 45 °/s / 2 rad/s² of 2026-10-19), and the detected item's grab is solved live (the 10 s probing). The scan times above predate the one-trajectory-per-pass scans and the coarse-first `run`/`clear` (2026-10-26): re-measure.

## Detection (no arm needed)

From the 8 empty baselines (collected 2026-09-30, before the new mount), by `laundry evaluate`. **Update 2026-10-26** (issue #22), against the 2026-10-22 baselines (new mount): without a further gate, leave-one-out gave **6 false clusters over 8 folds**, all shallow (8.5-12 mm, 4.0-5.0σ, 12-61 cm³). The strong-or-large gate (`perception/detect.py`, `DEFAULT_MIN_PEAK_SIGMA` 5.5σ or `DEFAULT_LARGE_VOLUME_M3` 70 cm³) brings that to **0 over 8, quick and full**. Synthetic, quick scan, floor and lower wall: 476/480 → **474/480** found (flat cloth on the lower wall 94 → 90%), stray clusters 0.19 → **0.00** per trial. The 2026-10-21 rig scan with laundry still returns its one cluster (44.8σ). The table below is the older set.

| Measure | Quick scan | Full scan |
|---|---|---|
| False clusters, leave-one-out over 8 empty scans | 1 (0.12 per scan) | 1 (0.12 per scan) |
| Synthetic items found, floor and lower wall (5 item types, 480 trials) | **480/480** | |
| The same with dark fabric (0.3 reflectivity) | 477/480 (99.4%); socks 96% floor, 98% lower wall | |
| Synthetic items found, all regions, reachable ones | | 703/782 (89.9%) |
| Grasp point error to the item's apex, median / 90th percentile | 0.7 / 1.6 cm | 1.0 / 6.3 cm (all regions) |
| Time to build the bucket model and detect | 0.4 s | |

Full scan, by region (items found / reached by at least one beam):

| | floor | lower wall | closed end | upper wall | ceiling |
|---|---|---|---|---|---|
| flat cloth | 100% | 98% | 51% | 50% | unseen |
| sock | 88% | 100% | 32% | 17% | unseen |
| shirt | 100% | 100% | 81% | 89% | unseen |
| towel | 100% | 100% | 100% | 84% | unseen |
| bundle | 100% | 100% | 100% | 93% | unseen |

The upper wall and ceiling are outside the scan's design: 418 of 1200 items placed there were never reached by any beam. Measured coverage of the bucket surface (8 scans pooled): floor 100%, lower wall 99%, lower closed end 74%, upper wall 6%, ceiling 0%.

**How far to trust these.** They are an upper bound. The items are injected into real empty scans with a simple ToF model, and the baselines predate the new ToF mount, so a rig run with real fabric and the new offsets can do worse. The false-positive count (now 0 in 8 with the gate) comes from only 8 scans, and the gate was chosen on the same data, so its margins are thin (the strongest false cluster peaked at 5.0σ against the 5.5σ cut). The old-mount empty scans in `validation_scans/empty_quick_20260930/` no longer apply: against the new baselines all 8 show the same 8-12σ cluster at about (−0.04, −0.49, 0.37) on one side wall, a mount/geometry difference rather than noise. New empty validation scans are needed.

## Reachability

`laundry scene check` on the fake controller, with the generated plan for a pile top at 25 cm: all 24 sweep grabs and their approach routes, HOME, DROP, BOTTOM and `drop_return` are collision-free under the 3 cm arm padding. Generating the plan, which tries heights and tilts for each of 12 spots, takes seconds; baking the routes and timing them takes the rest of the 2.5 minutes of `./rebake.sh retrieve`. The reachable heights, from the generator on the fake controller (a pile top at 25 cm): about 19 cm above the floor at 45° tilt near the mouth and 60° deeper, then 13 cm at 45°. Those come from collision checks against the modelled bucket, so confirm the clearance on the rig (HARDWARE_TESTS 17b).

## What only the rig can tell

- **Grasp success.** Nothing here senses whether the claw closed on laundry. Count it by eye or with a camera, and split the failures into motion failed, claw closed on nothing, and item dropped on the way.
- **Detection with real fabric, the new ToF mount and the ESP32 sensor.** Re-measure the mounting offsets and re-collect the baselines first.
- **The gripper's real time per move,** and the ESP32 link under a long run.

## Rig log

- **2026-10-22 18:57: C24 "Speed Exceeds Limit" on the transfer DROP → HOME at 7.0 rad/s².** `TRANSFER_MAX_ACCELERATION_RAD_S2` was raised from 5.0 to 7.0 (jerk 42 → 59) and the routes were re-timed, but not yet run on the arm. On the first real DROP → HOME (via INTER), 3.23 s in (0.87 s into the INTER → HOME leg), the xArm rejected the streamed targets (`set_servo_angle_j ret=1`, 13 times in 130 ms), raised C24 and stopped. The driver deactivated `xarm7_traj_controller`, which cancelled the goal, and MoveIt reported error -4 (CONTROL_FAILED). No joint was too fast: the planned peak is 59°/s against the 60°/s cap, and acceleration stays within 7.0. The **gripper tip** was too fast. FK of the baked timing, with the tip 0.142 m along link7 z, shows:

  | INTER → HOME | Duration | Flange peak | Gripper-tip peak |
  |---|---|---|---|
  | 5.0 rad/s² (ran on the rig) | 3.80 s | 780 mm/s | 975 mm/s |
  | 7.0 rad/s² (C24) | 3.60 s | 841 mm/s | **1050 mm/s** |

  The xArm7's maximum TCP speed is 1000 mm/s. With the higher acceleration, J1, J3 and J7 reach about 59°/s together, 0.8 s into the leg, and their speeds add up at the tool, about 0.6 m from the base. DROP → INTER peaks at about 350 mm/s, so it was fine. **Even the 5.0 timing was only 2.5% under the limit.** Fix: cap the gripper-tip speed when the routes are timed (about 900 mm/s), or go back to 5.0 / 42 and re-time. The arm must have C24 cleared (restart the bring-up, or use UFACTORY Studio) before it will move again. Logs: `~/.ros/log/ros2_control_node_22677_1792666590164.log`, `move_group_22675_*.log`.

## Sweep time vs transfer speed and acceleration (2026-10-27)

The 24-grab sweep as baked on 2026-10-22 (depths 40/30/20/10 cm from the closed end, two layers), re-timed for each setting with the claw kept under `TRANSFER_MAX_TIP_SPEED_M_S` (0.9 m/s), jerk = 8.4 × acceleration. **Motion time** is the sum of the planned durations of every trajectory the real sweep code sends (`laundry preplanned` against a stand-in arm). It matched the simulated execution move by move (1.5 vs 1.6 s, 3.76 vs 3.8 s, 3.25 vs 3.4 s). Two settings were also run in full on the fake controller in real time.

| deg/s \ rad/s² | 3 | 5 | 7 |
|---|---|---|---|
| 45 | 242.0 s | 226.0 s | 226.5 s |
| 60 | 217.2 s | 192.5 s (sim 256.7 s) | **178.3 s** (sim 253.8 s), current |
| 90 | 216.5 s (DROP routes stop at their vias) | 168.2 s | 149.1 s |

Fit: **M(v, a) ≈ 15.8 + 140.5 / v + 202.6 · v / a** seconds, with v the top joint speed in rad/s and a the acceleration in rad/s² (each move ≈ distance / v + v / a). rms 4.2 s, max 9.1 s off. The scatter is real, not noise: which routes keep a rounded timing changes from setting to setting (25–42 of 48 rounded grab↔DROP moves, 21–31 of 31 direct routes), and at 90°/s HOME is slowed for the claw. At 45°/s, more acceleration buys nothing (velocity-bound); at 90°/s and 3 rad/s², the DROP routes lose their rounded timing.

**Not motion: about 64–76 s per sweep (~3 s per grab)** in these simulations. That's the arm standing still while the next moves are prepared and collision-checked. It's CPU time, independent of v and a, and inflated here (load ~10 on 4 cores, the simulations at nice 10). The rig's figure has to be measured; the claw's 0.5 s per move overlaps it.

**Per grab** (motion only): 7.3 s on average at 60°/s / 7 rad/s² (6.4–8.9 s by position), 6.1 s at 90 / 7, 10.0 s at 45 / 3. Grabs deep in the bucket (10 cm from the closed end) cost about 2.3 s more than those near the mouth (40 cm). Adding or removing a grab changes the sweep by its motion time plus ~3 s of overhead (in this sim): about 10 s at the current setting.
