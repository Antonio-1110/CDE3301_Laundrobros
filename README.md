# CDE3301_Laundrobros

ROS 2 Jazzy package (`laundry_control`) that drives a UFactory xArm7 to pull laundry out of a bucket lying on its side (a stand-in for a washer drum). A wrist-mounted VL53L0X time-of-flight sensor is swept through the bucket, the readings become a point cloud, laundry is found by comparing that cloud against a fitted model of the empty bucket, and a servo gripper retrieves it.

It uses our fork of the manufacturer's `xarm_ros2`, which adds the gripper to the URDF/SRDF. The bucket and table are **not** in the URDF: they're MoveIt world objects whose poses are set in `config.OBSTACLES` (see [Obstacles](#obstacles-the-bucket-and-table)).

- **Everything runs through one command, `laundry`** — see [Using it](#using-it).
- **Commands that need the real rig** are collected in [HARDWARE_TESTS.md](HARDWARE_TESTS.md), with what to paste back.

---

## Workspace layout

This repo is one package inside a colcon workspace, next to the `xarm_ros2` fork:

```text
<workspace>/            # any name; nothing depends on it being ros2_ws
├── .venv/              # Python virtual environment
├── src/
│   ├── CDE3301_Laundrobros/   # this repo
│   └── xarm_ros2/             # our fork (tracked via workspace.repos, not a submodule)
├── build/  install/  log/
```

`xarm_ros2` is a sibling rather than a submodule because colcon wants every package collection directly under `src/`.

## Package layout

```text
laundry_control/
  config.py         every measured number: recorded poses, frames, ToF/gripper
                    offsets, servo angles, data paths
  cli.py            the `laundry` command
  pipeline.py       scan -> detect -> grasp -> drop, and the sensorless preplanned sweep
  arm/              controller.py (XArm7Controller: MoveIt joint/Cartesian/twist moves),
                    geometry.py (shared orientation math), flange_check.py
  hardware/         tof_sensor.py, servo.py, gripper_node.py, gripper_client.py,
                    fake.py (stand-ins for --fake-hardware)
  scan/             pattern.py (the helical scan), recorder_node.py / recorder_client.py,
                    cloud_io.py (CSV + PointCloud2), replay.py, baselines.py, coverage.py
  perception/       bucket_model.py (fitted empty bucket + noise field), detect.py,
                    report.py, evaluate.py, synthetic.py / synthetic_eval.py
  grasp/            plan.py (grasp target + reachability), execute.py, targets_io.py
launch/laundry_bringup.launch.py
baseline_scans/     empty-bucket scans the detector models the bucket from
scan_records/       everything else you scan (git-ignored)
```

Stages hand off through files — a scan CSV, then a targets JSON — so each one runs alone and can be rerun offline. `laundry run` chains them in one process.

---

## First-time setup

1. **System tools** (ROS 2 Jazzy itself is assumed installed at `/opt/ros/jazzy`):

   ```bash
   sudo apt-get install -y python3-vcstool python3-venv
   ```

2. **Workspace and sources:**

   ```bash
   mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
   git clone https://github.com/Antonio-1110/CDE3301_Laundrobros.git
   vcs import . < CDE3301_Laundrobros/workspace.repos
   ```

3. **Python venv** at the workspace root. `--system-site-packages` is what lets it see `rclpy` and the other ROS packages:

   ```bash
   cd ~/ros2_ws
   python3 -m venv --system-site-packages .venv
   source .venv/bin/activate
   pip install -r src/CDE3301_Laundrobros/requirements.txt
   ```

   `requirements.txt` holds only non-ROS dependencies (GPIO, the VL53L0X driver, scipy). ROS dependencies are declared in `package.xml`. On a machine with no GPIO or I2C hardware, the GPIO packages still install; they're only imported when the hardware is actually used.

4. **ROS dependencies and build.** Build with colcon running under the **venv's Python**:

   ```bash
   source /opt/ros/jazzy/setup.bash
   rosdep install --from-paths src --ignore-src -r -y
   python3 -m colcon build --symlink-install
   ```

   Plain `colcon` is `/usr/bin/colcon`, which always runs under `/usr/bin/python3` even with the venv active. The interpreter colcon runs under becomes the shebang of every installed executable. Built with plain `colcon`, `tof_sensor` and `gripper_node` start under system Python, can't import the pip-installed `gpiozero`/`adafruit_vl53l0x`, and fail. After you source `env.sh`, plain `colcon` is routed through `python3 -m colcon` for you, and `env.sh` warns if an existing build has the wrong interpreter. To check a build:

   ```bash
   head -1 install/laundry_control/lib/laundry_control/tof_sensor    # must be .../.venv/bin/python3
   ```

   `--symlink-install` matters: Python edits take effect without a rebuild, and the package finds its data directories (`baseline_scans/`, `scan_records/`) through the symlink back to this checkout. Without it, set `LAUNDRY_DATA_DIR` to the checkout's path.

## Every new terminal

```bash
source ~/ros2_ws/src/CDE3301_Laundrobros/env.sh
```

This sources ROS, activates `.venv`, sources `install/setup.bash`, puts `laundry` on `PATH`, and makes `colcon` run under the venv's Python. `ros2 run laundry_control laundry ...` also works.

## Rebuilding

From a shell that has sourced `env.sh` (so `colcon` runs under the venv):

```bash
colcon build --symlink-install --packages-select laundry_control
```

**After pulling a change that adds, renames or removes files** (including this restructure), colcon's symlink install can be left pointing at old paths, and the build fails with `can't copy ... doesn't exist`. Clean just this package:

```bash
rm -rf build/laundry_control install/laundry_control
colcon build --symlink-install --packages-select laundry_control
```

After moving or renaming the workspace, delete `build install log` and recreate `.venv`, since both contain absolute paths.

---

## Using it

### Bring-up

The real rig: arm driver + MoveIt, `tof_sensor`, `scan_recorder_node`, `gripper_node` and RViz.

```bash
ros2 launch laundry_control laundry_bringup.launch.py robot_ip:=192.168.1.207
```

No hardware: the MoveIt fake controller only. Pair it with `--fake-hardware`.

```bash
ros2 launch laundry_control laundry_bringup.launch.py fake:=true rviz:=false
```

### Commands

| Command | Needs |
|---|---|
| `laundry move inter` / `home` / `bottom` / `drop` / `retrieve_0..3` `[--speed 0.3]` | MoveIt |
| `laundry move joints J1 .. J7 [--degrees]`, `joint6 DEG`, `joint7 DEG`, `linear M`, `twist M DEG` | MoveIt |
| `laundry check-flange` — insertion axis vs bucket axis (run at INTER) | MoveIt |
| `laundry scene apply` / `check` — put the obstacles into MoveIt / also check every recorded pose and baked route against them (no motion) | MoveIt |
| `laundry scene fit` — the bucket pose the baseline scans measure, as a `config.OBSTACLES` entry | nothing |
| `laundry plan bake [endcap\|transfers\|retrieve\|all]` — solve, check and save the end scan, the routes from INTER to every named pose, and/or the grab grid (**moves the arm**) | MoveIt |
| `laundry plan replay [--speed 0.3]` — the end scan alone, INTER to INTER | MoveIt |
| `laundry scan [--save scan.csv] [--end-scan precession\|bottom] [--velocity 0.03 ...]` | rig, or `--fake-hardware --scan-from X.csv` |
| `laundry detect scan.csv [-o targets.json] [--publish]` | nothing — plain files |
| `laundry grasp targets.json [--drop] [--dry-run]` | rig, or `--fake-hardware` |
| `laundry run [--dry-run]` — scan → detect → grasp → drop, one item | rig, or `--fake-hardware --scan-from X.csv` |
| `laundry clear [--no-grabs] [--grab-limit N] [--max-rounds 15]` — empty the bucket: the grab grid, then scan → grasp → drop until a scan finds nothing | rig, or `--fake-hardware --scan-from A.csv B.csv ...` |
| `laundry gripper open` / `close` / `ANGLE` | `gripper_node` (open/close); the servo on this Pi's GPIO (ANGLE) |
| `laundry baseline collect [--count 8] [--archive] [-- <scan options>]` — straight into `baseline_scans/`; `--archive` replaces the set | rig, empty bucket |
| `laundry baseline promote X.csv\|dir ... [--move] [--archive]`, `archive`, `list`, `restore LABEL` | nothing |
| `laundry replay scan.csv` | a ROS graph (for RViz) |
| `laundry evaluate [--sweep] [--synthetic] [--coverage] [--laundry X.csv ...]` | nothing |
| `laundry preplanned [--limit N] [--speed 0.3] [--recorded]` — sensorless sweep over the grab grid (else the recorded RETRIEVE poses) | rig, or `--fake-hardware` |

`laundry <command> --help` documents every option.

**Scan options:** baselines and detection scans must use the same values. The detector's learned noise field is only valid for the path it was learned on.

**`--fake-hardware`** replaces the gripper and the ToF recorder with stand-ins (`hardware/fake.py`). Every arm motion still goes through MoveIt, so planning failures and collisions still show up. For example, this runs the whole pipeline with no hardware:

```bash
laundry run --fake-hardware --scan-from baseline_scans/baseline_20260924_180836_07.csv
```

On the fake controller, Cartesian strokes are planned from the observed joint state (`--observed-start-state`, implied by `--fake-hardware`). Otherwise MoveIt rejects them with "start point deviates from current robot state". On the rig this is opt-in until it's been tested there.

### Viewing scans in RViz

The point cloud is published with `TRANSIENT_LOCAL` durability, so RViz opened late still shows it:

```bash
rviz2 -d install/laundry_control/share/laundry_control/rviz/scan_visualization.rviz
laundry replay scan_records/<scan>.csv                    # a saved scan
laundry detect scan_records/<scan>.csv --publish          # detected clusters, coloured by index
```

The bucket and table in RViz (the "Obstacles" display) are only as accurate as `config.OBSTACLES`. The detector fits its own bucket model from data; `laundry scene fit` prints how far that fit sits from the configured pose.

---

## How detection works (short version)

1. **Model the empty bucket.** `perception/bucket_model.py` fits a cone (the wall) plus a flat cap (the closed end) to 8 empty-bucket scans in `baseline_scans/`. It then learns a per-cell offset and noise sigma on the bucket surface, in the bucket's own cylindrical coordinates, so the test is valid on the floor, the walls and the ceiling alike.
2. **Flag intrusions.** Each reading's intrusion inside the modelled wall is compared with the local sigma. Seed points must clear 4σ and 8 mm. Clusters are then grown through connected points clearing 2.5σ and 4 mm (hysteresis).
3. **Filter clusters.** Clusters must pass extent and volume gates. Clusters in thinly-sampled cells must also peak at ≥ 7σ. Each cluster reports its peak σ, and a grasp point: the mean of its top-quartile-intrusion points.
4. **Plan the grasp.** `grasp/plan.py` sinks the grasp point into the pile by as much room as the bucket model says exists underneath, then checks reachability with a plan-only probe.

The scan deliberately covers the **bottom** of the bucket: the floor, the lower walls and the lower half of the closed end. The 150° J7 sweep is centred on the floor. It runs at velocity 0.03 (was 0.1) for denser data and a J7 speed within its limit.

**The closed end** is out of the strokes' reach: the beam is fixed at 90° to the tool axis, and the gripper stops the arm going deeper. At maximum depth, the scan replays a **baked precession sweep** (`scan/endcap.py`, `scan_plans/endcap.yaml`). The tool tilts in a cone about the deepest flange position, so the beam traces arcs across the lower closed end. Simulated coverage of that area goes from 57% (the old BOTTOM detour) to 81%.

The sweep is solved once by `laundry plan bake` and replayed as a fixed joint trajectory:
- one continuous IK branch;
- out-and-back legs that retrace the same joint states;
- every step collision-checked.

The arm gets on and off it with collision-checked straight joint moves, so nothing is planned at scan time and the motion is identical every run. **Re-bake after changing INTER, `config.OBSTACLES`, the gripper, or `--depth`**; the scan refuses a plan baked for another depth. The old detour is available with `--end-scan bottom`.

`laundry evaluate` measures all of this:
- **Leave-one-out** over the empty baselines: every reported cluster is a false positive.
- **`--synthetic`** injects known items into real empty scans, respecting the ToF's ~25° cone. It reports recall by size and region, and localisation error.
- **`--synthetic`** places items on the bottom regions by default; `--all-regions` adds the upper wall and ceiling.
- **`--coverage`** shows how much of the bucket the scan path reaches at all, measured and simulated, and compares scan speeds.

Current numbers and their provenance are in the constants' comments in `perception/detect.py` and in the git log.

## Baseline scans

The detector models the empty bucket from every `*.csv` directly in `baseline_scans/`, and ignores anything in subfolders.

- **New set** (e.g. after moving the bucket): `laundry baseline collect --archive`. The scans are named `baseline_<session>_NN.csv` and go to `baseline_scans/incoming_<session>/`. Only when all of them succeed does the old set move to `baseline_scans/archive/<its session>/` and the new one take its place. A failed run leaves the old set active, so scans of two different scenes are never mixed.
- **Top up** the current set: `laundry baseline collect` (no `--archive`).
- **Adopt scans taken earlier:** `laundry baseline promote scan_records/scan_X.csv ... [--move] [--archive]` names them `baseline_<when taken>_NN.csv`.
- `laundry baseline list` shows the current set and the archive. `laundry baseline restore <label>` brings an archived set back and archives the current one. Nothing is ever deleted.

## Sensorless first pass: the grab grid

Laundry is expected at the start, so `laundry preplanned` grabs at fixed spots without scanning. `laundry plan bake retrieve` generates those spots from `config.RETRIEVE_GRID` in the configured bucket. The default is 4 depths (40, 30, 20, 10 cm from the closed end, mouth first) × 3 positions across the floor (centre, then ±20°).

For each spot, IK finds the lowest gripper height (2 cm above the floor upwards), then the most vertical approach that is collision-free with 1 cm of extra gripper clearance. Each spot is saved to `scan_plans/retrieve.yaml` as two poses:
- **`grab_NN`**: an approach pose 8 cm up the gripper axis, reached from INTER on a baked route like any named pose (`laundry move grab_03` works).
- **The grab itself**: a straight descent onto the laundry, then a straight lift back up.

The sweep runs: route in → descend → close → lift → DROP → open, for each grab. Edit the grid (depths, angles, heights, tilts) in `config.py` and re-bake. `--recorded` still runs the hand-recorded RETRIEVE_3..0.

**Detected items are grabbed the same way.** When the grid is baked, a detected item on the floor (within 50° of its lowest line) gets a grab placed over it. The grab sinks halfway into the pile (at most 5 cm) and is solved with the same IK search. The arm takes the baked route to the nearest `grab_NN`, makes a short straight collision-checked move from there, then descends, closes and lifts. Items off the floor use the older Cartesian reach from INTER. On the fake controller, that reach couldn't get to a towel in the middle of the floor at any depth; the floor grab could.

**`laundry clear`** chains it all. It fits the bucket model once and runs the grab grid (`--no-grabs` skips it). Then it repeats open → scan → detect → grasp the best reachable item → DROP until a scan finds nothing. It stops early if nothing detected is reachable (a rescan would see the same pile), after 2 failed grasps in a row, or after `--max-rounds`.

## Repeatable moves between poses

Named-pose moves (`laundry move <pose>`, and the scan, grasp, drop and preplanned stages) go through `arm/transfers.go_to`, which tries three routes in order:

1. **A baked route** from `scan_plans/transfers.yaml`. INTER is the hub: there is one route from INTER to each of HOME, DROP, BOTTOM, RETRIEVE_0–3 and the grab_NN poses (`laundry scene check` lists which exist).
   - INTER → pose replays the route; pose → INTER replays it in reverse.
   - Pose → another pose (e.g. RETRIEVE_2 → DROP) goes back to INTER along one route and out along the other, so the arm always leaves the bucket through its mouth.
   - Each route is straight joint-space segments through zero to two intermediate poses (for HOME/DROP, after first backing the gripper straight out of the bucket). Every 1° is collision-checked, and the route with the least joint travel wins, weighted toward J1 and J4–J7, which twist the cables.
   - Before every replay the whole route is re-checked against the current planning scene. A route that now collides is refused, with a message to re-bake. A route baked against different obstacle poses still runs if it's clear, with a warning to re-bake.
2. **Otherwise:** a straight, collision-checked joint move, which is the minimum twist.
3. **Only if that collides:** the planner, with a warning.

Re-bake (`laundry plan bake transfers`) after changing a recorded pose, the padding or `config.OBSTACLES`.

### Obstacles: the bucket and table

The bucket and table are MoveIt **world objects**, not robot links. `arm/scene.py` adds `meshes/bucket.obj` and `meshes/table.obj` at the poses in `config.OBSTACLES`, which is the only place those poses are set. Bring-up adds them at start-up (`laundry scene apply`), and every `laundry` command re-checks on connect. (They used to be URDF links in the `xarm_ros2` fork, and MoveIt checks robot-vs-robot contact without padding, so padding never kept the arm away from them.)

**After moving the bucket (or table):**

1. Edit its `xyz` / `rpy` in `config.OBSTACLES`: metres and radians in link_base, with the same convention as a URDF `<origin>`. To measure it, collect fresh baselines (`laundry baseline collect`), then run `laundry scene fit`. It prints the pose that the scans put the bucket at. That pose depends on the ToF extrinsics, so check it with a tape measure.
2. `laundry scene check` (bring-up running, fake or real) lists every recorded pose and baked route that now collides, and what it hits. Nothing moves.
3. Re-record any pose the move invalidated, then run `laundry plan bake` and commit `scan_plans/`. Baked files record the scene they were baked against.

**Padding:**

- **Arm links (link1–link7): 3 cm** (`config.OBSTACLE_PADDING_M`), for everything: the planner, Cartesian strokes, and the straight and baked moves. The exceptions are the end scan (1 cm, `config.ENDCAP_PADDING_M`) and the route to BOTTOM (2 cm, `config.ROUTE_ARM_PADDING_M`), whose tilted tool brings the elbow to the rim. Each is baked and replayed under its own padding.
- **Gripper: 0 cm** (`config.GRIPPER_PADDING_M`), because it works inside the bucket on purpose: the RETRIEVE poses put it within 1–2 cm of the floor.
- **Routes that leave the bucket** (to HOME and DROP) are baked with an extra 1 cm on the gripper (`config.BAKE_GRIPPER_CLEARANCE_M`), so it clears the bucket mouth on the way out.

## Frames and offsets

All of these live in `config.py`, with comments on how each was measured.

- **At INTER**, link7's local +Z points horizontally along −Y, straight into the bucket along its axis. Its local +X is the ToF boresight and points straight down, so the scan's J7 sweep is centred on the floor.
- **ToF sensor:** 7.75 cm along link7 +X (its boresight) and 2.8 cm along +Z. It's published as the static transform `link7 -> tof_sensor_link`, so readings stay correct as J7 rotates.
- **Gripper contact point:** 15 cm along link7 +Z. This is still to be verified on the hardware ([HARDWARE_TESTS.md](HARDWARE_TESTS.md)).

---

## `xarm_ros2` (our fork)

Useful launches outside the bring-up:

```bash
ros2 launch xarm_description xarm7_rviz_display.launch.py                  # model only
ros2 launch xarm_moveit_config xarm7_moveit_fake.launch.py                 # MoveIt, fake controller
ros2 launch xarm_moveit_config xarm7_moveit_gazebo.launch.py               # MoveIt + Gazebo
ros2 launch xarm_moveit_config xarm7_moveit_realmove.launch.py robot_ip:=<ARM_IP>
```

Our changes (the gripper link, J7's ±2π limit, J7's initial state) live on branch `world-obstacles` of https://github.com/Antonio-1110/xarm_ros2-cde3301.git (`workspace.repos` pins it). The modified files are `xarm_description/urdf/xarm7/*.xacro` and `xarm_moveit_config/srdf/_xarm7_macro.srdf.xacro`. The older `my-obstacle-changes` branch also has the bucket and table as URDF links. `arm/scene.py` still works with it, and disables those links in favour of the world objects.

In a checkout made with `vcs import`, `origin` is our fork. Add the manufacturer's repo as `upstream` to pull their fixes:

```bash
cd ~/ros2_ws/src/xarm_ros2
git remote add upstream https://github.com/xArm-Developer/xarm_ros2.git   # once
git fetch upstream
git rebase upstream/jazzy
git push origin world-obstacles --force-with-lease   # only after an intentional rebase
```
