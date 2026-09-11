# CDE3301_Laundrobros

ROS2 package (`laundry_control`) that drives a UFactory xArm7 to sort/handle laundry, using a modified fork of the manufacturer's `xarm_ros2` for custom obstacle geometry (bucket/table meshes) in the URDF/SRDF.

## Repo layout

This package expects to live inside a colcon workspace alongside a checkout of `xarm_ros2` as a **sibling** directory:

```
ros2_ws/
└── src/
    ├── CDE3301_Laundrobros/   (this repo)
    └── xarm_ros2/             (fork, see below — not tracked in this repo)
```

`xarm_ros2` is the manufacturer's repo. We've modified some of its URDF/SRDF files to add our custom obstacle (bucket/table) geometry, so we run our own fork/branch instead of stock upstream. That fork isn't nested inside this repo (colcon needs it as a sibling under `src/`), so it's tracked via a `workspace.repos` file instead of a submodule.

## First-time setup (new machine)

1. Install `vcstool` (used to pull the linked `xarm_ros2` fork):
   ```
   sudo apt-get install -y python3-vcstool
   ```
2. Clone this repo into your workspace's `src/`:
   ```
   cd ~/ros2_ws/src
   git clone https://github.com/Antonio-1110/CDE3301_Laundrobros.git
   ```
3. Pull in the `xarm_ros2` fork (creates `ros2_ws/src/xarm_ros2` at the right branch):
   ```
   vcs import . < CDE3301_Laundrobros/workspace.repos
   ```
4. Install ROS dependencies and build:
   ```
   cd ~/ros2_ws
   rosdep install --from-paths src --ignore-src -r -y
   colcon build
   source install/setup.bash
   ```

## Everyday commands — `laundry_control` (this repo)

```
# Build just this package
colcon build --packages-select laundry_control

# Source the workspace after building
source install/setup.bash

# Run the main entry point
ros2 run laundry_control run_probe

# Run an individual script directly (useful while iterating)
python3 src/CDE3301_Laundrobros/laundry_control/move.py
```

## Everyday commands — `xarm_ros2` (manufacturer side, our fork)

```
# View the arm in RViz only (no MoveIt, no hardware)
ros2 launch xarm_description xarm7_rviz_display.launch.py

# MoveIt with a fake/simulated controller (no hardware needed)
ros2 launch xarm_moveit_config xarm7_moveit_fake.launch.py

# MoveIt + Gazebo simulation (includes our custom obstacle geometry)
ros2 launch xarm_moveit_config xarm7_moveit_gazebo.launch.py

# MoveIt against the real arm (make sure the IP/robot_ip arg is correct!)
ros2 launch xarm_moveit_config xarm7_moveit_realmove.launch.py robot_ip:=<ARM_IP>
```

## Editing the xArm obstacle/URDF files

Our changes live in `xarm_ros2` on branch `my-obstacle-changes`, pushed to our fork (not the manufacturer's repo):

- Fork: https://github.com/Antonio-1110/xarm_ros2-cde3301.git
- Files we've modified: `xarm_description/urdf/xarm7/*.xacro`, `xarm_moveit_config/srdf/_xarm7_macro.srdf.xacro`

Workflow when editing those files:

```
cd ~/ros2_ws/src/xarm_ros2
git checkout my-obstacle-changes   # make sure you're on our branch, not upstream's

# ... edit files ...

git add -A
git commit -m "describe your change"
git push myfork my-obstacle-changes
```

Then, on any other machine, sync the change:

```
cd ~/ros2_ws/src/xarm_ros2
git pull myfork my-obstacle-changes
```

If `myfork` isn't set up as a remote yet on a machine:

```
git remote add myfork https://github.com/Antonio-1110/xarm_ros2-cde3301.git
```

### Pulling upstream fixes from the manufacturer

The fork's `origin` remote still points at the real `xArm-Developer/xarm_ros2`, so you can grab upstream fixes and rebase our changes on top:

```
git fetch origin
git rebase origin/jazzy
git push myfork my-obstacle-changes --force-with-lease   # only after rebasing
```

## ToF sensor scanning + visualization

`tof_sensor.py` is a ROS2 node publishing `sensor_msgs/Range` on `tof_sensor/range`. `scan_move.py` drives the arm through a bucket-scanning sweep and (via `TofScanRecorder` in `scan_record.py`) records each reading as a 3D point, transformed through TF into both the TCP frame (`link7`) and the base frame (`link_base`).

The sensor's mounting offset from the TCP (`TOF_SENSOR_OFFSET_*` in `move.py`) is published as a static transform `link7 -> tof_sensor_link`, so it's automatically joint7-correct without extra math. Current values (measured at the `INTER` pose): 2.8 cm along local +X (the sensor's own boresight/forward direction) and 7.75 cm along local +Z (which points straight down at `INTER`, per `check_flange.py`). If a different physical mounting axis turns out to be "front," these are the two constants to flip/adjust.

### Live visualization while scanning

```
# Terminal 1: bring up the robot model + obstacles (fake controller, no hardware)
ros2 launch xarm_moveit_config xarm7_moveit_fake.launch.py

# Terminal 2: run the scan (records + live-publishes points)
ros2 run laundry_control run_probe   # or: python3 laundry_control/scan_move.py --save-points scan1.csv

# Terminal 3: RViz with a pre-built Displays config (robot model + TF + point cloud)
rviz2 -d install/laundry_control/share/laundry_control/rviz/scan_visualization.rviz
```

The point cloud publishes with `TRANSIENT_LOCAL` durability, so opening RViz after the scan has already started still shows everything recorded so far.

**Caveat:** the obstacle geometry (bucket/table) shown in RViz is only as accurate as our URDF edits — it is *not* guaranteed to match the real bucket/table's exact size, shape, or pose. Treat the RViz scene as an approximate reference for context, not ground truth, when reasoning about where points fall relative to the model.

### Revisiting a previous scan

`--save-points <file>.csv` (on `scan_move.py`) saves the base-frame points recorded during a run. To view a saved scan again later — with no arm and no hardware required — replay it onto the same topic the live recorder uses:

```
ros2 run laundry_control scan_replay scan1.csv
# or: python3 laundry_control/scan_replay.py scan1.csv --topic scan_record/points --frame link_base
```

Then open RViz with the same `scan_visualization.rviz` config as above (optionally alongside `xarm7_rviz_display.launch.py` if you just want the model/obstacles without MoveIt) to see the replayed points in place.

For deeper offline analysis (e.g. clustering points to locate individual laundry items rather than just "something is closer than the wall"), load the CSV in a plotting/analysis tool of your choice — a `numpy`/`matplotlib` 3D scatter or Open3D + DBSCAN both work well against this file format (`x,y,z` columns, base frame).
