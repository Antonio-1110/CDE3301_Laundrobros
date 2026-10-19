# Performance: what simulation says

How long the scan and retrieve motions take, and how well the detector does, measured without the arm. Measured 2026-10-19 on the Pi 5, at the default settings. To redo it: [`benchmarks/sim_timing.sh`](../benchmarks/sim_timing.sh) (motion times, on an isolated fake controller) and [`benchmarks/sim_metrics.sh`](../benchmarks/sim_metrics.sh) (detection and coverage, plain files).

**What these numbers are.** The fake controller runs MoveIt's planned trajectories in real time, so the arm's motion times match what the rig is given. They leave out the gripper (open or close is 1.5 s on the rig, `SETTLE_SEC`, plus the ESP32's ack) and the real arm's tracking, so the rig is a few seconds slower per grasp. **They say nothing about whether a grasp picks the laundry up**: simulation has no cloth, and a grasp counts as done when the motions finish. That needs the rig.

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
| Sensorless sweep, all 12 grabs (`laundry preplanned`) | 196 s | about 3.8 min |
| One grab of the sweep, INTER to INTER via DROP | 11-22 s, mean 15.7 s | mean about 18.7 s |
| `laundry run`, one item in the bucket | **125 s** | about 130 s |

- **Per grab:** deeper grabs take longer: 11-16 s at 40 and 30 cm from the closed end, 16-22 s at 10 cm. The rig adds about 3 s per grab, for the claw to close and to open at DROP.
- **`laundry run`, one item (125 s):** scan 93 s (75%), detection 0.4 s, grasp 28 s. The grasp is about 10 s of reachability probing and 18 s of moving, including the drop. The rig adds about 4.5 s for the three gripper moves (open before the scan, close, release).
- **`laundry clear` is an estimate, not a measurement.** Each further item costs a rescan: about 97 s scan, 30 s grasp and 4.5 s of gripper, so **about 2.2 min per item**, plus about 1.6 min for the final scan that finds nothing (about 2.5 min if the end scan runs on its own). With the grab grid first (`--no-grabs` skips it), add the sweep's 3.8 min.

Where the laundry is changes the grasp part, not the scan: each baked route to a grab spot takes 2.2 s each way, DROP 3.6 s each way, and the detected item's grab is solved live (the 10 s probing).

## Detection (no arm needed)

From the 8 empty baselines (collected 2026-09-30, before the new mount), by `laundry evaluate`.

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

**How far to trust these.** They are an upper bound. The items are injected into real empty scans with a simple ToF model, and the baselines predate the new ToF mount, so a rig run with real fabric and the new offsets can do worse. The false-positive count is 1 in 8: too few scans to say more than "about one false cluster in eight empty scans".

## Reachability

`laundry scene check` on the fake controller: all 12 generated grab spots, their approach routes, HOME, DROP, BOTTOM and `drop_return` are collision-free under the 3 cm arm padding. One problem is reported: `scan_plans/retrieve.yaml` was solved for other settings (`grab_targets.yaml` and `clearance_m` changed). Re-bake it (`./rebake.sh retrieve`) before treating the sweep times as final, since they were measured with that plan.

## What only the rig can tell

- **Grasp success.** Nothing here senses whether the claw closed on laundry. Count it by eye or with a camera, and split the failures into motion failed, claw closed on nothing, and item dropped on the way.
- **Detection with real fabric, the new ToF mount and the ESP32 sensor.** Re-measure the mounting offsets and re-collect the baselines first.
- **The gripper's real time per move,** and the ESP32 link under a long run.
