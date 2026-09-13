# CDE3301_Laundrobros

ROS 2 package (`laundry_control`) that drives a UFactory xArm7 to sort/handle laundry, using a modified fork of the manufacturer's `xarm_ros2` for custom obstacle geometry (bucket/table meshes) in the URDF/SRDF.

## Repo layout

This repository contains the `laundry_control` ROS 2 package. It is expected to live inside the `src/` directory of a colcon workspace alongside a checkout of `xarm_ros2` as a **sibling** directory:

```text
<workspace>/
├── .venv/                       # Python virtual environment
├── src/
│   ├── CDE3301_Laundrobros/     # this repo
│   └── xarm_ros2/               # fork, see below — not tracked in this repo
├── build/
├── install/
└── log/
```

The workspace itself can have any name. The scripts in this repository determine the workspace location relative to the repository, so there is no requirement to call it `ros2_ws`.

`xarm_ros2` is the manufacturer's repository. We have modified some of its URDF/SRDF files to add our custom obstacle (bucket/table) geometry, so we use our own fork/branch instead of stock upstream.

The fork is not nested inside this repository because colcon expects it as a sibling package collection under `src/`. It is therefore tracked through `workspace.repos` instead of as a Git submodule.

---

## Python environment

The project uses a Python virtual environment located at the **workspace root**:

```text
<workspace>/.venv/
```

The venv is created with:

```bash
python3 -m venv --system-site-packages .venv
```

`--system-site-packages` is important because ROS 2 packages such as `rclpy` are normally installed through the system/ROS installation rather than through `pip`.

Project-specific Python dependencies are listed in:

```text
CDE3301_Laundrobros/requirements.txt
```

ROS dependencies remain declared through `package.xml` and should **not** be duplicated in `requirements.txt`.

The virtual environment itself is not committed to Git.

---

## First-time setup (new machine)

### 1. Install required system tools

Install `vcstool`, the Python venv module, and other required ROS tooling if they are not already installed:

```bash
sudo apt-get update
sudo apt-get install -y python3-vcstool python3-venv
```

### 2. Create a ROS 2 workspace

The workspace can have any name. For example:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### 3. Clone this repository

```bash
git clone https://github.com/Antonio-1110/CDE3301_Laundrobros.git
```

The resulting structure should be:

```text
ros2_ws/
└── src/
    └── CDE3301_Laundrobros/
```

### 4. Pull in the `xarm_ros2` fork

From the workspace `src/` directory:

```bash
vcs import . < CDE3301_Laundrobros/workspace.repos
```

This creates:

```text
ros2_ws/
└── src/
    ├── CDE3301_Laundrobros/
    └── xarm_ros2/
```

at the branch specified by `workspace.repos`.

### 5. Create the Python virtual environment

Go to the workspace root:

```bash
cd ..
```

Create the venv:

```bash
python3 -m venv --system-site-packages .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Verify that the active Python interpreter comes from the workspace:

```bash
which python3
```

It should look similar to:

```text
/home/<user>/ros2_ws/.venv/bin/python3
```

### 6. Install project-specific Python dependencies

With the venv active:

```bash
pip install -r src/CDE3301_Laundrobros/requirements.txt
```

Do not install ROS packages such as `rclpy` using `pip`. They are provided by the ROS installation and exposed to the venv through `--system-site-packages`.

### 7. Source ROS 2

For ROS 2 Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
```

Verify that ROS Python packages are visible from the venv:

```bash
python3 -c "import rclpy; print(rclpy.__file__)"
```

### 8. Install ROS dependencies

From the workspace root:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

### 9. Build the workspace

Make sure the venv is still active before building:

```bash
which python3
```

Then build:

```bash
colcon build --symlink-install
```

Finally:

```bash
source install/setup.bash
```

Using `--symlink-install` is recommended for development because changes to Python source files can often be tested without rebuilding the entire package.

---

## Environment helper

This repository includes:

```text
env.sh
```

It automatically:

1. determines the workspace containing this repository,
2. sources the base ROS 2 installation,
3. activates `<workspace>/.venv`, and
4. sources `<workspace>/install/setup.bash` if the workspace has already been built.

After the first-time setup, open a new terminal and run:

```bash
source ~/ros2_ws/src/CDE3301_Laundrobros/env.sh
```

Replace `~/ros2_ws` with the location of your workspace.

Because `env.sh` determines the workspace from its own location, the workspace does **not** need to be named `ros2_ws`.

You should see information similar to:

```text
ROS workspace : /home/<user>/ros2_ws
ROS distro    : jazzy
Python        : /home/<user>/ros2_ws/.venv/bin/python3
```

After that, ROS commands can be used normally.

---

## Clean rebuild

A clean rebuild is recommended after changing Python environments, moving/renaming the workspace, or encountering stale colcon configuration.

From the workspace root:

```bash
rm -rf build install log
```

Then activate the environment:

```bash
source src/CDE3301_Laundrobros/env.sh
```

Since `install/setup.bash` does not exist yet, `env.sh` will simply skip sourcing the workspace overlay.

Build again:

```bash
colcon build --symlink-install
source install/setup.bash
```

### Renaming or moving the workspace

`build/`, `install/`, and `log/` may contain absolute paths, so delete them after moving the workspace.

Python virtual environments are also not reliably relocatable. If the workspace itself has been moved or renamed, recreate `.venv`:

```bash
rm -rf build install log .venv

python3 -m venv --system-site-packages .venv
source .venv/bin/activate

pip install -r src/CDE3301_Laundrobros/requirements.txt

source /opt/ros/jazzy/setup.bash

colcon build --symlink-install
source install/setup.bash
```

---

## Everyday commands — `laundry_control`

For a normal development session:

```bash
cd ~/ros2_ws
source src/CDE3301_Laundrobros/env.sh
```

### Build just this package

```bash
colcon build --symlink-install --packages-select laundry_control
source install/setup.bash
```

### Run the main entry point

```bash
ros2 run laundry_control run_probe
```

### Run an individual script directly

Useful while iterating:

```bash
python3 src/CDE3301_Laundrobros/laundry_control/move.py
```

Because `env.sh` activates the venv, both direct Python execution and newly built ROS 2 Python executables should use the project's Python environment.

If a Python package works with:

```bash
python3 some_script.py
```

but produces `ModuleNotFoundError` with:

```bash
ros2 run laundry_control ...
```

check which interpreter was used when the ROS package was built. A clean rebuild with the venv active may be required:

```bash
rm -rf build install log
source src/CDE3301_Laundrobros/env.sh
colcon build --symlink-install
source install/setup.bash
```

---

## Everyday commands — `xarm_ros2`

### View the arm in RViz only

No MoveIt and no hardware:

```bash
ros2 launch xarm_description xarm7_rviz_display.launch.py
```

### MoveIt with a fake/simulated controller

No hardware required:

```bash
ros2 launch xarm_moveit_config xarm7_moveit_fake.launch.py
```

### MoveIt + Gazebo simulation

Includes our custom obstacle geometry:

```bash
ros2 launch xarm_moveit_config xarm7_moveit_gazebo.launch.py
```

### MoveIt against the real arm

Make sure the IP/`robot_ip` argument is correct:

```bash
ros2 launch xarm_moveit_config xarm7_moveit_realmove.launch.py robot_ip:=<ARM_IP>
```

---

## Editing the xArm obstacle/URDF files

Our changes live in `xarm_ros2` on branch:

```text
my-obstacle-changes
```

They are pushed to our fork rather than the manufacturer's repository.

Fork:

```text
https://github.com/Antonio-1110/xarm_ros2-cde3301.git
```

Files we have modified include:

```text
xarm_description/urdf/xarm7/*.xacro
xarm_moveit_config/srdf/_xarm7_macro.srdf.xacro
```

### Editing workflow

```bash
cd ~/ros2_ws/src/xarm_ros2

git checkout my-obstacle-changes

# ... edit files ...

git add -A
git commit -m "describe your change"
git push myfork my-obstacle-changes
```

On another machine, sync the changes with:

```bash
cd ~/ros2_ws/src/xarm_ros2
git pull myfork my-obstacle-changes
```

If `myfork` has not yet been configured as a remote:

```bash
git remote add myfork https://github.com/Antonio-1110/xarm_ros2-cde3301.git
```

### Pulling upstream fixes from the manufacturer

The fork's `origin` remote still points to the official `xArm-Developer/xarm_ros2` repository.

To incorporate upstream fixes:

```bash
git fetch origin
git rebase origin/jazzy
```

Then update our fork:

```bash
git push myfork my-obstacle-changes --force-with-lease
```

Only force-push after intentionally rebasing the branch.

---

## ToF sensor scanning + visualization

`tof_sensor.py` is a ROS 2 node publishing `sensor_msgs/Range` on:

```text
tof_sensor/range
```

`scan_move.py` drives the arm through a bucket-scanning sweep and, via `TofScanRecorder` in `scan_record.py`, records each reading as a 3D point.

Each point is transformed through TF into both:

- the TCP frame (`link7`), and
- the base frame (`link_base`).

The sensor's mounting offset from the TCP (`TOF_SENSOR_OFFSET_*` in `move.py`) is published as a static transform:

```text
link7 -> tof_sensor_link
```

This means the transform automatically accounts for joint 7 rotation without requiring additional rotation calculations in the scanning code.

Current values, measured at the `INTER` pose:

- **2.8 cm** along local `+X` — the sensor's boresight/forward direction
- **7.75 cm** along local `+Z` — pointing straight down at `INTER`, according to `check_flange.py`

If a different physical mounting axis turns out to represent the sensor's actual forward direction, these are the constants that should be adjusted.

---

## Live visualization while scanning

First initialize the development environment in each terminal:

```bash
cd ~/ros2_ws
source src/CDE3301_Laundrobros/env.sh
```

### Terminal 1 — robot model + obstacles

Fake controller, no hardware:

```bash
ros2 launch xarm_moveit_config xarm7_moveit_fake.launch.py
```

### Terminal 2 — run the scan

Records and live-publishes points:

```bash
ros2 run laundry_control run_probe
```

or:

```bash
python3 src/CDE3301_Laundrobros/laundry_control/scan_move.py --save-points scan1.csv
```

### Terminal 3 — RViz

Open the pre-built display configuration containing the robot model, TF, and point cloud:

```bash
rviz2 -d install/laundry_control/share/laundry_control/rviz/scan_visualization.rviz
```

The point cloud publishes with `TRANSIENT_LOCAL` durability, so opening RViz after the scan has already started still shows everything recorded so far.

> **Caveat:** The obstacle geometry (bucket/table) shown in RViz is only as accurate as our URDF edits. It is **not** guaranteed to match the real bucket/table's exact size, shape, or pose. Treat the RViz scene as an approximate contextual reference rather than ground truth when reasoning about where points fall relative to the model.

---

## Revisiting a previous scan

Using:

```bash
--save-points <file>.csv
```

with `scan_move.py` saves the base-frame points recorded during a run.

To view a saved scan again later, with no arm or hardware required, replay it onto the same topic used by the live recorder:

```bash
ros2 run laundry_control scan_replay scan1.csv
```

or:

```bash
python3 src/CDE3301_Laundrobros/laundry_control/scan_replay.py \
    scan1.csv \
    --topic scan_record/points \
    --frame link_base
```

Then open RViz with:

```bash
rviz2 -d install/laundry_control/share/laundry_control/rviz/scan_visualization.rviz
```

Optionally, launch:

```bash
ros2 launch xarm_description xarm7_rviz_display.launch.py
```

if you want the robot model/obstacles without MoveIt.

For deeper offline analysis — for example, clustering points to locate individual laundry items rather than simply detecting that something is closer than the bucket wall — load the CSV into a plotting or point-cloud analysis tool.

The saved file contains base-frame:

```text
x,y,z
```

coordinates.

Possible analysis approaches include:

- NumPy + Matplotlib 3D scatter plots
- Open3D point-cloud processing
- DBSCAN clustering