# Operating the rig

Bringing it up, the `laundry` commands, and the routine jobs: baselines, the full-drum sweep, and moving the bucket. How the pieces work is in [how-it-works.md](how-it-works.md).

## Bring-up and commands

### Bring-up

The real rig: arm driver + MoveIt, `tof_sensor`, `scan_recorder_node`, `gripper_node` and RViz.

```bash
ros2 launch laundry_control laundry_bringup.launch.py robot_ip:=192.168.1.207
```

The arm's control loop (ros2_control) runs at 100 Hz, not the stock 150 Hz: each tick waits for the arm twice, and at 150 Hz the ticks kept overrunning. It's set by `config.ARM_CONTROL_RATE_HZ`, and `control_rate_hz:=150` brings back the stock rate for comparison. `xarm_ros2` itself is unmodified: the bring-up passes the rate into the manufacturer's launch.

The ToF sensor and the servo can be on an ESP32 that talks to this Pi over MQTT (the default) or on this Pi (I2C and GPIO). Pick with `tof_source:=i2c|mqtt` and `gripper_backend:=gpio|mqtt`, or change the defaults in `config/hardware.py` (`TOF_SOURCE`, `GRIPPER_BACKEND`). The default needs the broker (mosquitto, a systemd service that starts at boot; check with `mqtt_broker/broker.sh`) and the flashed firmware: set up the broker first ([mqtt_broker/README.md](../mqtt_broker/README.md)) and flash the firmware ([esp32/README.md](../esp32/README.md)).

No hardware: the MoveIt fake controller only. Pair it with `--fake-hardware`.

```bash
ros2 launch laundry_control laundry_bringup.launch.py fake:=true rviz:=false
```

### Commands

| Command | Needs |
|---|---|
| `laundry move inter` / `home` / `bottom` / `drop` / `grab_NN` `[--speed 0.3]` | MoveIt |
| `laundry move joints J1 .. J7 [--degrees]`, `joint1` … `joint7 DEG [--to] [--speed F]`, `linear M`, `twist M DEG` | MoveIt |
| `laundry check-flange` — insertion axis vs bucket axis (run at INTER) | MoveIt |
| `laundry scene apply` / `check` — put the obstacles into MoveIt / also check every recorded pose and baked route against them (no motion) | MoveIt |
| `laundry scene fit` — the bucket pose the baseline scans measure, as a `config.OBSTACLES` entry | nothing |
| `laundry plan bake [poses\|endcap\|transfers\|retrieve\|timing\|all]` — solve, check and save INTER/BOTTOM from the bucket, the grab grid, the routes from INTER to every named pose, and/or the end scan (the end scan **moves the arm**) | MoveIt |
| `laundry plan replay [--speed 0.3]` — the end scan alone, INTER to INTER | MoveIt |
| `laundry scan [--save scan.csv] [--full \| --end-scan precession\|bottom] [--velocity 0.03 ...]` — the quick scan unless `--full` | rig, or `--fake-hardware --scan-from X.csv` |
| `laundry detect scan.csv [-o targets.json] [--publish]` — quick or full is read from the CSV | nothing — plain files |
| `laundry grasp targets.json [--drop] [--dry-run]` | rig, or `--fake-hardware` |
| `laundry run [--dry-run]` — scan → detect → grasp → drop, one item | rig, or `--fake-hardware --scan-from X.csv` |
| `laundry clear [--no-grabs] [--grab-limit N] [--max-rounds 15]` — empty the bucket: the sweep, then scan → grasp → drop until a scan finds nothing | rig, or `--fake-hardware --scan-from A.csv B.csv ...` |
| `laundry gripper open` / `close` / `ANGLE` | `gripper_node` (open/close); the servo on this Pi's GPIO (ANGLE) |
| `laundry baseline collect [--count 8] [--archive] [-- <scan options>]` — straight into `baseline_scans/`; `--archive` replaces the set | rig, empty bucket |
| `laundry baseline promote X.csv\|dir ... [--move] [--archive]`, `archive`, `list`, `restore LABEL` | nothing |
| `laundry replay scan.csv` | a ROS graph (for RViz) |
| `laundry evaluate [--sweep] [--synthetic] [--coverage] [--laundry X.csv ...]` | nothing |
| `laundry preplanned [--limit N] [--speed 0.3]` — sensorless sweep into the pile, layer by layer (refuses if it isn't baked) | rig, or `--fake-hardware` |

`laundry <command> --help` documents every option.

**Scan options:** baselines and detection scans must use the same values. The detector's learned noise field is only valid for the path it was learned on.

**`--fake-hardware`** replaces the gripper and the ToF recorder with stand-ins (`hardware/fake.py`). Every arm motion still goes through MoveIt, so planning failures and collisions still show up. For example, this runs the whole pipeline with no hardware:

```bash
laundry run --fake-hardware --scan-from baseline_scans/baseline_20260924_180836_07.csv
```

On the fake controller, Cartesian strokes are planned from the observed joint state (`--observed-start-state`, implied by `--fake-hardware`). Otherwise MoveIt rejects them with "start point deviates from current robot state". On the rig this is opt-in until it's been tested there.

### Viewing scans in RViz

The live ToF cloud ("ToF Scan Points") only fills during a scan. The recorder records only while a scan has the sensor inside the bucket, so moves, grabs and drops add nothing. The finished scan stays on screen until the next scan starts, or until you clear it with `ros2 service call /clear_scan std_srvs/srv/Trigger`. The cloud is republished only when it changes, with `TRANSIENT_LOCAL` durability, so RViz opened late still shows it:

```bash
rviz2 -d install/laundry_control/share/laundry_control/rviz/scan_visualization.rviz
laundry replay scan_records/<scan>.csv                    # a saved scan
laundry detect scan_records/<scan>.csv --publish          # detected clusters, coloured by index
```

The bucket and table in RViz (the "Obstacles" display) are only as accurate as `config.OBSTACLES`. The detector fits its own bucket model from data; `laundry scene fit` prints how far that fit sits from the configured pose.

## Baseline scans

The detector models the empty bucket from every `*.csv` directly in `baseline_scans/`, full scans only (quick scans use their strokes, see above), and ignores anything in subfolders.

- **New set** (e.g. after moving the bucket): `laundry baseline collect --archive`. The scans are named `baseline_<session>_NN.csv` and go to `baseline_scans/incoming_<session>/`. Only when all of them succeed does the old set move to `baseline_scans/archive/<its session>/` and the new one take its place. A failed run leaves the old set active, so scans of two different scenes are never mixed.
- **Top up** the current set: `laundry baseline collect` (no `--archive`).
- **Adopt scans taken earlier:** `laundry baseline promote scan_records/scan_X.csv ... [--move] [--archive]` names them `baseline_<when taken>_NN.csv`.
- `laundry baseline list` shows the current set and the archive. `laundry baseline restore <label>` brings an archived set back and archives the current one. Nothing is ever deleted.

## Sensorless first pass: grabs for a full drum

The sweep (`laundry preplanned`, and the start of `laundry clear`) runs on a drum that is already full, with nothing taken out and no scan. So it does not grab at the floor: it aims into the pile, and **where depends on how full the drum is**. `config.PILE` (`config/grabs.py`) sets that:

| Setting | Default | |
|---|---|---|
| `top_height_m` | 0.25 | where the pile's top is, above the floor (2/3 of the drum). **Change this for a different fill.** `LAUNDRY_PILE_TOP_M=0.20` overrides it for one run |
| `sink_m` | 0.06 | how far below the pile's surface the first layer grabs |
| `layer_step_m` | 0.06 | how much lower each further layer goes than the one before actually reached |
| `min_height_m` | 0.10 | the lowest sweep grab: laundry below it is for the scan-then-grasp stage, so a pile this low gets no sweep |
| `depths_m`, `angles_deg` | 4 × 3 | the spots (40, 30, 20, 10 cm from the closed end, mouth first; centre, then ±20°) |
| `tilts_deg` | 0 to 60° | tried smallest first |

`laundry plan bake retrieve` generates the grab list from it (`grasp/pile_layers.py`, saved to `scan_plans/grab_targets.yaml`) and solves each grab exactly as generated. For each spot it takes the **highest height that some tilt reaches**: the 15 cm gripper and its approach meet the drum's ceiling, so a grab near the pile top costs tilt (on the fake controller, at 25 cm fill: 45° near the mouth, 60° deeper). Layer 1 aims at the pile's top, `sink_m` below it. Each further layer aims `layer_step_m` below what that spot's last layer actually reached, until it would go below `min_height_m`. The sweep runs layer 1 over every spot, mouth first, then layer 2, because taking laundry lowers the pile. At the defaults: **24 grabs in 2 layers, about 19 and 13 cm above the floor**, about 6 min 30 s on the rig.

**Another fill level means another plan.** Change `PILE['top_height_m']` (or set `LAUNDRY_PILE_TOP_M`) and re-bake with `./rebake.sh retrieve` (about 2.5 minutes); a generated list regenerates by itself when `PILE` changes. `laundry preplanned` warns when the plan was baked for another `PILE`. Bake and run with the same value, since the override is read each time.

**Placing grabs by hand** still works: drag them in `laundry plan edit-grabs` and SAVE. A hand-placed list is kept on re-bake (it stops following `PILE`) until you pass `--regenerate`. The editor draws the pile level from `PILE`.

Each grab is saved to `scan_plans/retrieve.yaml` as two poses:
- **`grab_NN`**: an approach pose 5 cm up the gripper axis, reached from INTER on a baked route like any named pose (`laundry move grab_03` works).
- **The grab itself**: a straight descent onto the laundry, then a straight lift back up.

The sweep runs: route in → descend → close → lift → DROP → open, for each grab. The four grab poses once jogged by hand (RETRIEVE_0..3) are no longer poses. They remain only as `config.GRAB_IK_SEEDS`: starting postures for the IK solver.

**Detected items are grabbed the same way.** When the grid is baked, a detected item anywhere in the lower half of the drum (within `config.DETECTED_GRAB_MAX_ANGLE_DEG` = 90° of the floor's lowest line, so the lower walls too) gets a grab placed over it. The grab sinks halfway into the pile (at most 5 cm) and is solved with the same IK search. The arm takes the baked route to the nearest `grab_NN`, makes a short straight collision-checked move from there, then descends, closes and lifts. Items beyond that angle use the older Cartesian reach from INTER. On the fake controller, that reach couldn't get to a towel in the middle of the floor at any depth; the floor grab could.

**`laundry clear`** chains it all. It fits the bucket model once and runs the sweep (`--no-grabs` skips it). Then it repeats open → scan → detect → grasp the best reachable item → DROP until a scan finds nothing. It stops early if nothing detected is reachable (a rescan would see the same pile), after 2 failed grasps in a row, or after `--max-rounds`.

## After moving the bucket (or table)

1. Edit its `xyz` / `rpy` in `config.OBSTACLES`: metres and radians in link_base, with the same convention as a URDF `<origin>`. Measure it with a tape measure. To refine it, collect baselines (`laundry baseline collect`), then run `laundry scene fit`. It prints the pose that the scans put the bucket at, and that pose depends on the ToF extrinsics.
2. `laundry scene check` (bring-up running, fake or real) lists every named pose and baked route that now collides, and what it hits. Nothing moves.
3. Run `./rebake.sh` (about 10 minutes on the Pi). It starts an isolated fake controller (its own ROS domain, localhost only, so it can't reach the real arm) and runs `laundry plan bake`. That derives INTER and BOTTOM from the new bucket pose, then re-bakes the grab grid, the transfers and the end scan from that INTER. The script then runs `laundry scene check` and shuts the fake controller down. Check what it printed for the move from the jogged poses. `./rebake.sh poses` (or any other `plan bake` target) runs just that part.
4. Re-record HOME or DROP only if the bucket is now in their way.
5. Collect new baselines (`laundry baseline collect --archive`), because the scan now starts from the new INTER. Commit `config/`, `scan_plans/` and `baseline_scans/` together.
