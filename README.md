# CDE3301_Laundrobros

ROS 2 Jazzy package (`laundry_control`) that drives a UFactory xArm7 to pull laundry out of a bucket lying on its side (a stand-in for a washer drum). A wrist-mounted VL53L0X time-of-flight sensor is swept through the bucket, the readings become a point cloud, laundry is found by comparing that cloud against a fitted model of the empty bucket, and a servo gripper retrieves it.

It uses our fork of the manufacturer's `xarm_ros2`, which adds the bucket/table as collision geometry in the URDF/SRDF.

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
| `laundry move inter` / `home` / `bottom` / `drop` / `retrieve_0..3` | MoveIt |
| `laundry move joints J1 .. J7 [--degrees]`, `joint6 DEG`, `joint7 DEG`, `linear M`, `twist M DEG` | MoveIt |
| `laundry check-flange` — insertion axis vs bucket axis (run at INTER) | MoveIt |
| `laundry plan bake` — solve, check and save the end-scan trajectory (**moves the arm**) | MoveIt |
| `laundry plan replay [--speed 0.3]` — the end scan alone, INTER to INTER | MoveIt |
| `laundry scan [--save scan.csv] [--end-scan precession\|bottom] [--velocity 0.03 ...]` | rig, or `--fake-hardware --scan-from X.csv` |
| `laundry detect scan.csv [-o targets.json] [--publish]` | nothing — plain files |
| `laundry grasp targets.json [--drop] [--dry-run]` | rig, or `--fake-hardware` |
| `laundry run [--dry-run]` — scan → detect → grasp → drop | rig, or `--fake-hardware --scan-from X.csv` |
| `laundry gripper open` / `close` / `ANGLE` | `gripper_node` (open/close); the servo on this Pi's GPIO (ANGLE) |
| `laundry baseline collect [--count 8] [-- <scan options>]` | rig, empty bucket |
| `laundry baseline promote scan.csv` | nothing |
| `laundry replay scan.csv` | a ROS graph (for RViz) |
| `laundry evaluate [--sweep] [--synthetic] [--coverage] [--laundry X.csv ...]` | nothing |
| `laundry preplanned` — sensorless sweep over the RETRIEVE poses | rig, or `--fake-hardware` |

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

The bucket and table in RViz are only as accurate as our URDF edits. The detector fits its own bucket model from data, and `laundry detect` prints how far that fit sits from the URDF.

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

The arm gets on and off it with collision-checked straight joint moves, so nothing is planned at scan time and the motion is identical every run. **Re-bake after changing INTER, the URDF bucket/gripper, or `--depth`**; the scan refuses a plan baked for another depth. The old detour is available with `--end-scan bottom`.

`laundry evaluate` measures all of this:
- **Leave-one-out** over the empty baselines: every reported cluster is a false positive.
- **`--synthetic`** injects known items into real empty scans, respecting the ToF's ~25° cone. It reports recall by size and region, and localisation error.
- **`--synthetic`** places items on the bottom regions by default; `--all-regions` adds the upper wall and ceiling.
- **`--coverage`** shows how much of the bucket the scan path reaches at all, measured and simulated, and compares scan speeds.

Current numbers and their provenance are in the constants' comments in `perception/detect.py` and in the git log.

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

Our obstacle changes live on branch `my-obstacle-changes` of https://github.com/Antonio-1110/xarm_ros2-cde3301.git. The modified files are `xarm_description/urdf/xarm7/*.xacro` and `xarm_moveit_config/srdf/_xarm7_macro.srdf.xacro`.

In a checkout made with `vcs import`, `origin` is our fork. Add the manufacturer's repo as `upstream` to pull their fixes:

```bash
cd ~/ros2_ws/src/xarm_ros2
git remote add upstream https://github.com/xArm-Developer/xarm_ros2.git   # once
git fetch upstream
git rebase upstream/jazzy
git push origin my-obstacle-changes --force-with-lease   # only after an intentional rebase
```
