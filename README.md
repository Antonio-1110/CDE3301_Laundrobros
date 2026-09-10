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
