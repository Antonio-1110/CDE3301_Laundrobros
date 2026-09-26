# Hardware tests

These commands need the real rig: arm, ToF sensor and servo. Everything else was verified on a machine with none of them, using the MoveIt fake controller and `--fake-hardware`. Run the tests in order, since later ones assume earlier ones passed.

**Before every session:**

```bash
source ~/ros2_ws/src/CDE3301_Laundrobros/env.sh
```

**Paste back:** each test's **Paste back** line says what to send. For scans, also send the CSV file.

**Safety:** keep a hand on the e-stop for every motion test.

**ToF timestamps:** readings are now stamped at the middle of each ~33 ms measurement, not at its end. That moves every scan point ~2° of J7 rotation (up to ~7 mm at the wall) compared with the committed baselines, which is one more reason to re-collect them (test 13).

**What changed in the motion:**
- **Scan speed:** the default scan velocity is now **0.03** (was 0.1).
- **End scan:** the closed end is covered by the baked precession end scan (test 7), not the BOTTOM detour.
- **Transfers:** INTER ↔ HOME/DROP use baked paths (test 5).

The committed `baseline_scans/` were recorded with the old speed and the old detour, so **they no longer match the default scan path**. Tests 8–10 check how the old baselines judge the new scans, and section D replaces them.

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

**1b. Switch `xarm_ros2` to the branch without the bucket and table.** The bucket and table are now MoveIt world objects, placed from `config.OBSTACLES`. Branch `world-obstacles` of the fork drops them from the URDF and keeps the gripper.

```bash
cd ~/ros2_ws/src/xarm_ros2 && git fetch origin && git checkout world-obstacles && cd ~/ros2_ws && python3 -m colcon build --symlink-install --packages-select xarm_description xarm_moveit_config && source install/setup.bash
```

- **Paste back:** the last line of the build output.

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
- **Also check** that the bring-up log has `laundry-N` printing "move_group has the obstacles." (the one-shot `laundry scene apply`).

**2b. The obstacles against every recorded pose and route** (no motion).

```bash
laundry scene check
```

- **Paste back:** the whole output.
- **Expect:** every pose `ok` except `retrieve_3`, which collides with the modelled bucket even though it was recorded on the real arm. Every route `ok` except `retrieve_3`, which has none.

**2c. Where is the bucket really?** The modelled bucket may be ~3 cm off. `laundry scene fit` measures the bucket from the baseline scans. It puts the bucket axis **3.1 cm further along +x** (away from the arm's centre line, sideways) and 0.5 cm lower at the closed end than `config.OBSTACLES`. At that pose, every recorded pose, RETRIEVE_3 included, is collision-free. The fit depends on the ToF mounting offsets, so a tape measure has the final say:

- With the arm at HOME, measure (in link_base: origin at the centre of the arm's base where it sits on the table, +z up):
  - **x of the bucket axis:** the sideways distance from the base's centre to the bucket's centre line. Config says 14.0 cm; the scans say 17.1 cm.
  - **Height of the bucket axis above the table at the closed end.** Config says 42.0 cm; the scans say 41.5 cm.
- **Paste back:** the two numbers, and which side of the base centre (+x) the bucket axis is on.


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

**5. Baked transfers INTER ↔ HOME and INTER ↔ DROP, slowly.** Hand on the e-stop; watch the wiring.

```bash
laundry move inter && laundry move home --speed 0.3 && laundry move inter --speed 0.3 && laundry move drop --speed 0.3 && laundry move inter --speed 0.3
```

- **Paste back:**
  - the output, which should say "Baked transfer to …" for each of the four moves;
  - whether the gripper backed straight out of the bucket before swinging away;
  - whether any cable pulled.
- **Verifies:** these moves now replay fixed, collision-checked paths (`scan_plans/transfers.yaml`) instead of the planner. On the fake controller the planner turned J7 up to 536° on DROP → INTER; the baked path turns it the direct 14°, identically every time.
- **If it's fine at 0.3:** repeat without `--speed`.
- **If something comes too close:** re-bake on the rig with `laundry plan bake transfers`. It moves the arm; watch it. Then paste back the per-joint summary and commit the file.

**6. Gripper through `gripper_node`, then a raw servo angle.**

```bash
laundry gripper close && laundry gripper open && laundry gripper 80
```

- **Paste back:** the output, and whether the claw physically moved each time.
- **Verifies:**
  - open/close go through `gripper_node`'s services.
  - `ANGLE` drives the servo directly. The GPIO library is now loaded on first use rather than at import, so this is the first real check of that change.

**6b. The claw holds when closed.** `gripper_node` now keeps driving the servo after `close`. Close the claw on a rolled towel, then try to pull the towel out by hand:

```bash
laundry gripper close
```

Wait 30 s, then feel whether the servo is warm, and run `laundry gripper open`.

- **Paste back:** whether the claw resisted the pull (before this change it went limp 0.7 s after closing), whether the servo buzzes or gets hot while holding, and whether it opens afterwards.

## C. End scan and scanning

The closed end is now covered by a **baked precession ("coning") sweep** instead of the BOTTOM detour. The tool tilts in a cone about the deepest flange position, so the beam sweeps arcs over the lower closed end (`scan/endcap.py`). Simulated coverage of the closed end's lower half goes from 57% to 81%.

- **It's a fixed joint trajectory** (`scan_plans/endcap.yaml`, baked on the fake controller against the URDF bucket), replayed identically every scan.
- **The old detour** is still there, but only as `--end-scan bottom`.

**7. Replay the end scan alone, slowly.** Hand on the e-stop. Watch the elbow (link3) against the bucket rim and the gripper against the walls, especially in the last ring (the largest tilt).

```bash
laundry plan replay --speed 0.3
```

- **Paste back:** the output, and whether anything came closer than ~2 cm to the bucket.
- **Verifies:** the baked motion is collision-free on the real rig. It was checked against the URDF bucket, whose pose the scans put about 1 cm off.
- **If it comes too close:** re-bake on the rig with `laundry plan bake`. That moves the arm: INTER, in 0.42 m, back out. Then paste back its ring summary, and commit the new `scan_plans/endcap.yaml`.
- **If it's fine at 0.3:** run `laundry plan replay --speed 1.0` once more before scanning.


**8. Empty-bucket scan through the new CLI.** Empty the bucket and open the gripper first.

```bash
laundry scan --save scan_records/hw_empty_01.csv 2>&1 | tail -5 && ros2 param get /scan_recorder_node csv_path
```

- **Paste back:** the output, plus the CSV.
- **Verifies:**
  - `--save` writes exactly that file.
  - The recorder's `csv_path` is handed back afterwards (the second command must print an empty string).
  - The recorder's `TF: ... exact, ... fallback` tally is in the log.

**9. The new scan detects nothing.**

```bash
laundry detect scan_records/hw_empty_01.csv
```

- **Paste back:** everything from "Loaded candidate" down.
- **Verifies:** on real data, the new detector (hysteresis + low-confidence gate) reports 0 clusters in an empty bucket.

**10. Observed-start-state planning on the real arm.** On the fake controller this was required to finish a scan at all; on the rig it's untested.

```bash
laundry scan --observed-start-state --save scan_records/hw_empty_obs_01.csv 2>&1 | grep -cE "failed|deviates"; laundry detect scan_records/hw_empty_obs_01.csv | grep "Found"
```

- **Paste back:** both lines, plus the CSV.
- **Verifies:** the arm can plan each stroke from `/joint_states` instead of MoveIt's own state. If it scans cleanly and the result matches test 9, I'll make it the default everywhere.

**11. Scan with the ray columns** (a check on test 8's CSV).

```bash
head -1 scan_records/hw_empty_01.csv
```

- **Paste back:** the header line.
- **Verifies:** new scans include `raw_range,ox,oy,oz,j7`. The 8 committed baselines predate these columns, so the synthetic evaluation currently has to rebuild each ray from the scan geometry. Real ray columns replace that approximation.

## D. Slower, denser scan

**Why 0.03:**
- **Density:** the ToF samples at a fixed 20 Hz, so a slower stroke gets more readings. On the fake controller, a 3 cm stroke takes 0.95 s at 0.1 and 2.68 s at 0.03, giving ~2.1× the readings for a ~94 s scan.
- **J7 speed:** the J7 twist bypasses MoveIt's velocity limits. At 0.1 it ran at ~157°/s, above J7's 123°/s limit; at 0.03 it's ~56°/s.
- **Localisation:** in simulation, the grasp point's median error dropped from ~1.0 to ~0.65 cm.
- **Coverage is unchanged:** `laundry evaluate --coverage` shows the same area covered, because the cone footprint already bridges the gaps between readings. The upper wall and ceiling stay out of scope by design.

**12. One slow empty scan, compared with a fast one.**

```bash
time laundry scan --save scan_records/hw_empty_slow_01.csv 2>&1 | grep -cE "twist runs|failed"; wc -l scan_records/hw_empty_01.csv scan_records/hw_empty_slow_01.csv
```

- **Paste back:** the time, the count (should be 0), the two line counts, plus the CSV.
- **Note:** test 8 already ran at 0.03 as well. If you want the fast comparison, rerun test 8 with `--velocity 0.1 --acceleration 0.1`.
- **Verifies:** the slow scan completes without J7-rate warnings, and gives roughly twice the points.

**13. Baselines at the new speed.** Empty bucket, open gripper; takes about 13 minutes.

```bash
laundry baseline collect --count 8 --archive && laundry scan --end-scan bottom --save scan_records/hw_empty_bottom.csv && laundry evaluate --sweep --synthetic --coverage --coverage-scan $(ls baseline_scans/*.csv | head -1) --coverage-scan scan_records/hw_empty_bottom.csv && laundry baseline list && laundry scene fit
```

- **Paste back:** the evaluate, `list` and `scene fit` output, plus the 8 CSVs.
- **`--archive`:** the new scans land straight in `baseline_scans/` once all 8 succeed, and the old 0.1-speed set moves to `baseline_scans/archive/20260924_180836/`. If a scan fails, the old set stays active.
- **Verifies:**
  - No false clusters in the leave-one-out.
  - How recall, sigma and occupancy compare with the 0.1 baselines (`laundry evaluate --synthetic --coverage` on `baseline_scans/`, which I can run here).
  - These scans carry the real ray columns, so the synthetic evaluation stops depending on the rebuilt rays.
  - The two `--coverage-scan` lines give the measured closed-end coverage of the precession vs the BOTTOM detour, on real data.
- **If this looks wrong:** `laundry baseline restore 20260924_180836` brings the old set back (the new one is archived, not lost).
- **`scene fit`:** a second measurement of where the bucket is, from the new scans (compare with test 2c).

**14. Optional: 2 cm step.**

```bash
laundry scan --step 0.02 --save scan_records/hw_empty_step2.csv
```

- **Paste back:** the CSV and the time.
- **Why it's optional:** in simulation it adds 43% more readings for no extra area coverage (~127 s per scan). Worth it only if test 13 shows cells still thinly sampled.

## E. Grasping

**15. Grasp offset (measure by hand).** `config.GRIPPER_OFFSET_Z = 0.15` is reported, not verified. It's the distance along link7 +Z (at INTER, horizontally into the bucket) from the flange face to the point where the claw closes.

- **Paste back:** the distance in cm.
- **Also measure** the ToF sensor's offset from the flange axis (7.75 cm expected) and its distance forward of the flange face (2.8 cm expected). The old README had these two swapped relative to the code; the code's values are the ones in use.

**16. Dry run with one item.** Put a small towel on the floor, mid-depth.

```bash
laundry run --dry-run --save scan_records/hw_towel_dry.csv 2>&1 | grep -A12 "Found"
```

- **Paste back:** the output, the CSV, and a photo.
- **Verifies:** detection on real cloth. The printed `grasp=(...)` point should sit on the towel; say how far off it looks.

**17. Full retrieval.**

```bash
laundry run --save scan_records/hw_towel_run.csv 2>&1 | grep -vE "joint[0-9]:|Joint-space target"
```

- **Paste back:** the output, plus whether the towel was picked up and dropped.
- **Verifies:** the whole pipeline end to end on the rig.

---

**17b. The generated grab grid, slowly, empty bucket first.** Hand on the e-stop. It visits 12 spots on the floor, from the mouth inward: centre, then −20° and +20° across the floor. At each spot it descends 8 cm, closes the claw, lifts back up, and drops at DROP.

```bash
laundry scene check && laundry preplanned --limit 3 --speed 0.3
```

- **Watch:** the claw's height above the floor at each grab (the target is 2–3 cm), and the side grabs against the walls. Those two depend most on where the bucket really is (tests 2c and 13).
- **Paste back:** the `scene check` output, and the claw-to-floor gap at each of the 3 grabs (a rough ruler estimate is fine).
- **Then:** the full sweep, `laundry preplanned --speed 0.5`, with some laundry in the bucket. Say how many items came out, and which grabs came up empty. `laundry preplanned --recorded` runs the old four poses, for comparison.
- **If the bucket pose changes** (`config.OBSTACLES`): run `laundry plan bake retrieve` and commit `scan_plans/`.

**17c. `laundry clear`, slowly.** Hand on the e-stop. Put 2–3 items in the bucket. The command runs 3 grid grabs, then scans and grasps until a scan finds nothing:

```bash
laundry clear --grab-limit 3 --speed 0.3 --max-rounds 5
```

- **Watch:** detected items anywhere in the lower half (floor and lower walls) are grabbed like the grid: "Floor grab via grab_NN" in the log, with a straight descent onto the item. Put one item partway up a lower wall to check that case.
- **Paste back:** the log from "ROUND 1" on, how many items came out, and whether it stopped with "The bucket is clear".

## F. Real-cloth validation scans I need

Everything detection-related is currently tuned on synthetic items injected into real empty scans. These scans check it against real cloth. Take them **after test 13**, at the default speed (0.03), so they match the new baselines.

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
| 10 | `val_towel_closed_end_mid` | small towel | against the closed end at axis height (3 or 9 o'clock) | only ~50% of the closed end's lower half is covered, and the gap is this middle band; checks whether it matters in practice |
| 11 | `val_two_items` | towel + sock | towel floor 20 cm, sock lower wall 35 cm | two clusters, correctly ranked |
| 12 | `val_towel_floor_mid_b` | as 2 | as 2, item re-placed | repeatability of detection and grasp point |

**Paste back:** the 12 CSVs, `notes.csv`, and the photos. I'll then run the following, and compare each reported grasp point with the noted position:

```bash
laundry evaluate --laundry validation_scans/val_*.csv
laundry detect validation_scans/<each>.csv
```
