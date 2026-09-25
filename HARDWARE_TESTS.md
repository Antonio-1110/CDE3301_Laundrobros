# Hardware tests

These commands need the real rig: arm, ToF sensor and servo. Everything else was verified on a machine with none of them, using the MoveIt fake controller and `--fake-hardware`. Run the tests in order, since later ones assume earlier ones passed.

**Before every session:**

```bash
source ~/ros2_ws/src/CDE3301_Laundrobros/env.sh
```

**Paste back:** each test's **Paste back** line says what to send. For scans, also send the CSV file.

**Safety:** keep a hand on the e-stop for every motion test, especially 10–12, which drive J7 further than the old path ever did.

---

## A. After pulling this branch

**1. Clean rebuild of this package.** This restructure renamed files, so a stale symlink install fails to build.

```bash
cd ~/ros2_ws && rm -rf build/laundry_control install/laundry_control && python3 -m colcon build --symlink-install --packages-select laundry_control && source install/setup.bash && head -1 install/laundry_control/lib/laundry_control/tof_sensor && laundry --help | head -3
```

- **Paste back:** the last 4 lines.
- **Verifies:**
  - The package builds on the rig, and `laundry` is on PATH.
  - The shebang line is the venv's `.venv/bin/python3`, not `/usr/bin/python3`. Otherwise `tof_sensor` and `gripper_node` can't import their hardware libraries.

**2. Bring-up starts every node** (leave it running for everything below).

```bash
ros2 launch laundry_control laundry_bringup.launch.py rviz:=false
```

Then, in a second terminal:

```bash
ros2 node list | grep -E "tof_sensor|scan_recorder_node|gripper_node|move_group" && ros2 param get /move_group planning_pipelines
```

- **Paste back:** the output.
- **Verifies:** the new launch file (replaces `real_arm_scan.launch.py`, and now also starts `gripper_node`).
- **About the second command:** it tells us whether Pilz PTP is loaded. If it prints only `['ompl']`, every joint move falls back to OMPL. That's harmless but slower, and less repeatable.

## B. Motion and devices

**3. Named poses and the flange check.**

```bash
laundry move inter && laundry check-flange
```

- **Paste back:** the check-flange block.
- **Verifies:** named poses through the new CLI. check-flange now reports the insertion axis against the bucket axis fitted from `baseline_scans/`.
- **Expected** (fake-controller FK): elevation within ±1° of horizontal, about 6° to the fitted axis (5.8–6.5° across runs, since MoveIt stops anywhere within its 0.01 rad joint tolerance), 4.3–4.7 cm of drift, and an "OFFSET" verdict. That verdict is expected: the baselines were recorded along this same path. If the rig's numbers differ by more than about 1°, INTER on the rig differs from the fake controller's FK, and the synthetic evaluation's ray geometry needs re-measuring.

**4. Relative moves.** Run at INTER; the arm ends where it started.

```bash
laundry move joint7 30 && laundry move joint7 -30 && laundry move linear 0.03 && laundry move linear -0.03 && laundry move inter
```

- **Paste back:** "ok", or the first error.
- **Verifies:** the J7 and tool-Z moves work exactly as `move_cli.py` did.

**5. Gripper through `gripper_node`, then a raw servo angle.**

```bash
laundry gripper close && laundry gripper open && laundry gripper 80
```

- **Paste back:** the output, and whether the claw physically moved each time.
- **Verifies:**
  - open/close go through `gripper_node`'s services.
  - `ANGLE` drives the servo directly. The GPIO library is now loaded on first use rather than at import, so this is the first real check of that change.

## C. Scanning

**6. Empty-bucket scan through the new CLI.** Empty the bucket and open the gripper first.

```bash
laundry scan --save scan_records/hw_empty_01.csv 2>&1 | tail -5 && ros2 param get /scan_recorder_node csv_path
```

- **Paste back:** the output, plus the CSV.
- **Verifies:**
  - `--save` writes exactly that file.
  - The recorder's `csv_path` is handed back afterwards (the second command must print an empty string).
  - The recorder's `TF: ... exact, ... fallback` tally is in the log.

**7. The new scan detects nothing.**

```bash
laundry detect scan_records/hw_empty_01.csv
```

- **Paste back:** everything from "Loaded candidate" down.
- **Verifies:** on real data, the new detector (hysteresis + low-confidence gate) reports 0 clusters in an empty bucket.

**8. Observed-start-state planning on the real arm.** On the fake controller this was required to finish a scan at all; on the rig it's untested.

```bash
laundry scan --observed-start-state --save scan_records/hw_empty_obs_01.csv 2>&1 | grep -cE "failed|deviates"; laundry detect scan_records/hw_empty_obs_01.csv | grep "Found"
```

- **Paste back:** both lines, plus the CSV.
- **Verifies:** the arm can plan each stroke from `/joint_states` instead of MoveIt's own state. If it scans cleanly and the result matches test 7, I'll make it the default everywhere.

**9. Scan with the ray columns** (a check on test 6's CSV).

```bash
head -1 scan_records/hw_empty_01.csv
```

- **Paste back:** the header line.
- **Verifies:** new scans include `raw_range,ox,oy,oz,j7`. The 8 committed baselines predate these columns, so the synthetic evaluation currently has to rebuild each ray from the scan geometry. Real ray columns replace that approximation.

## D. Wider sweep (the coverage fix)

`laundry evaluate --coverage` shows why this matters. The current 150° sweep is centred on the floor and never sees the top ~120° of the bucket (0% of the ceiling, 3–11% of the upper wall). In simulation, a 360° sweep raises whole-bucket coverage from 45% to 80%, and the ceiling to 76%.

**The main hardware risk is cable wrap.** The ToF and servo cables run past J7, and J7 will turn up to 360° in each stroke. So step up the sweep, and check the cables after each run. `--velocity 0.02` keeps J7 under its 123°/s limit; at the current default of 0.1 it already runs at ~157°/s.

**10. 210° sweep.**

```bash
laundry scan --sweep 210 --velocity 0.02 --acceleration 0.02 --save scan_records/hw_sweep210.csv 2>&1 | grep -E "COMPLETE|failed|twist runs"
```

- **Paste back:** the output, the CSV, and "cables OK / not OK".
- **Verifies:** J7 range, cable slack, and coverage (I'll measure it from the CSV).

**11. 270° sweep.** Only if test 10 was fine; same command with `--sweep 270` and `--save scan_records/hw_sweep270.csv`.

- **Paste back:** the same as test 10.

**12. 360° sweep.** Only if test 11 was fine; same command with `--sweep 360` and `--save scan_records/hw_sweep360.csv`.

- **Paste back:** the same as test 10, plus the scan's wall-clock time (`time laundry scan ...`).

**13. Baselines for the chosen path.** The noise field is path-specific, so a new sweep needs its own baselines. Use an empty bucket and an open gripper, with the sweep you settled on.

```bash
laundry baseline collect --count 8 --dest baseline_scans_sweep360 -- --sweep 360 --velocity 0.02 --acceleration 0.02
laundry evaluate --baseline baseline_scans_sweep360 --coverage --synthetic
```

- **Paste back:** the evaluate output, plus the 8 CSVs.
- **Verifies:** false positives and coverage for the new path. If both look right, it replaces `baseline_scans/`.

## E. Grasping

**14. Grasp offset (measure by hand).** `config.GRIPPER_OFFSET_Z = 0.15` is reported, not verified. It's the distance along link7 +Z (at INTER, horizontally into the bucket) from the flange face to the point where the claw closes.

- **Paste back:** the distance in cm.
- **Also measure** the ToF sensor's offset from the flange axis (7.75 cm expected) and its distance forward of the flange face (2.8 cm expected). The old README had these two swapped relative to the code; the code's values are the ones in use.

**15. Dry run with one item.** Put a small towel on the floor, mid-depth.

```bash
laundry run --dry-run --save scan_records/hw_towel_dry.csv 2>&1 | grep -A12 "Found"
```

- **Paste back:** the output, the CSV, and a photo.
- **Verifies:** detection on real cloth. The printed `grasp=(...)` point should sit on the towel; say how far off it looks.

**16. Full retrieval.**

```bash
laundry run --save scan_records/hw_towel_run.csv 2>&1 | grep -vE "joint[0-9]:|Joint-space target"
```

- **Paste back:** the output, plus whether the towel was picked up and dropped.
- **Verifies:** the whole pipeline end to end on the rig.

---

## F. Real-cloth validation scans I need

Everything detection-related is currently tuned on synthetic items injected into real empty scans. These scans check it against real cloth. They all use the **current** path and `baseline_scans/`, since those baselines exist. Take them before changing the sweep.

**How to take each one:**

1. Place the item and take a photo into the bucket from behind the arm.
2. Run `laundry scan --save validation_scans/<name>.csv`, using the file names below.
3. Add one line to `validation_scans/notes.csv`:

   ```text
   name,item,colour,size_cm_LxWxH,depth_from_mouth_cm,clock_position,notes
   val_towel_floor_mid,small towel folded,white,30x20x4,25,6,
   ```

   - `depth_from_mouth_cm` is measured along the bucket axis from the rim to the item's centre.
   - `clock_position` is looking into the bucket from the arm: 12 = top/ceiling, 6 = floor, 3 = right wall, 9 = left wall.

| # | name | item | placement | why |
|---|---|---|---|---|
| 1 | `val_empty_repeat` | none | — | a 9th empty scan never used in the model: an independent false-positive check |
| 2 | `val_towel_floor_mid` | small towel, folded, light | floor (6), 25 cm deep | easy case, to confirm the basics |
| 3 | `val_towel_floor_mouth` | same towel | floor (6), 8 cm deep | the mouth, where the fit is weakest |
| 4 | `val_towel_closed_end` | same towel | against the closed end | the cap, and the cap/wall corner |
| 5 | `val_shirt_crumpled_wall` | T-shirt, crumpled | lower wall (4 or 8), 20 cm deep | wall, not floor |
| 6 | `val_sock_light_floor` | sock, light | floor (6), 20 cm deep | small item |
| 7 | `val_sock_dark_floor` | sock, **black** | same spot as 6 | dark fabric absorbs IR; the synthetic model assumes ~0.3 reflectivity |
| 8 | `val_sock_dark_wall` | black sock | lower wall (4 or 8), 30 cm deep | small + dark + wall |
| 9 | `val_cloth_flat_floor` | handkerchief / thin cloth, laid flat | floor (6), 20 cm deep | the hardest case (synthetic recall ~40%) |
| 10 | `val_towel_upper_wall` | small towel | upper wall (2 or 10), 25 cm deep | the scan barely sees here: the coverage prediction says this should mostly be MISSED |
| 11 | `val_two_items` | towel + sock | towel floor 20 cm, sock lower wall 35 cm | two clusters, correctly ranked |
| 12 | `val_towel_floor_mid_b` | as 2 | as 2, item re-placed | repeatability of detection and grasp point |

**Paste back:** the 12 CSVs, `notes.csv`, and the photos. I'll then run the following, and compare each reported grasp point with the noted position:

```bash
laundry evaluate --laundry validation_scans/val_*.csv
laundry detect validation_scans/<each>.csv
```
