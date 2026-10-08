# Setup and building

Getting the workspace onto a machine, and keeping the build healthy.

## Workspace layout

This repo is one package inside a colcon workspace, next to the manufacturer's `xarm_ros2`:

```text
<workspace>/            # any name; nothing depends on it being ros2_ws
├── .venv/              # Python virtual environment
├── src/
│   ├── CDE3301_Laundrobros/   # this repo
│   └── xarm_ros2/             # the manufacturer's, unmodified (pinned in workspace.repos)
├── build/  install/  log/
```

`xarm_ros2` is a sibling rather than a submodule because colcon wants every package collection directly under `src/`.

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

**After pulling a change that adds, renames or removes files** (such as `config.py` and `cli.py` becoming the `config/` and `cli/` packages), colcon's symlink install can be left pointing at old paths, and the build fails with `can't copy ... doesn't exist`. Clean just this package:

```bash
rm -rf build/laundry_control install/laundry_control
colcon build --symlink-install --packages-select laundry_control
```

After moving or renaming the workspace, delete `build install log` and recreate `.venv`, since both contain absolute paths.

## The manufacturer's `xarm_ros2` (unmodified)

Useful launches outside the bring-up:

```bash
ros2 launch xarm_description xarm7_rviz_display.launch.py                  # model only
ros2 launch xarm_moveit_config xarm7_moveit_fake.launch.py                 # MoveIt, fake controller
ros2 launch xarm_moveit_config xarm7_moveit_gazebo.launch.py               # MoveIt + Gazebo
ros2 launch xarm_moveit_config xarm7_moveit_realmove.launch.py robot_ip:=<ARM_IP>
```

We use the manufacturer's `jazzy` branch as it is, pinned in `workspace.repos` to the commit it was tested with. Everything the rig needs from the robot description is passed through the package's own launch arguments, built by `config.xarm_description_arguments()`:

- **The gripper:** `add_other_geometry:=true` with our STL (`config.GRIPPER_MESH`), turned `config.GRIPPER_MESH_RPY` about link7. Its link is `other_geometry_link` (`config.GRIPPER_LINK`), and the stock SRDF already exempts it from colliding with link3, link6 and link7.
- **Joint limits:** `limited:=false` gives the xArm7's true hardware ranges, J7 ±360° included. The scan and the grabs turn J7 past −180°. The manufacturer's default narrows J7 to ±178°.
- **Fake controller only:** the mock hardware starts at all-zeros, where the modelled gripper is in the table. `laundry scene apply --fake-start-home`, run by bring-up with `fake:=true`, moves the fake arm to HOME.

Our old fork (https://github.com/Antonio-1110/xarm_ros2-cde3301, branches `my-obstacle-changes` and `world-obstacles`) is no longer needed. `arm/scene.py` still copes with a build of it: the bucket and table links are disabled in favour of the world objects. But the gripper link name differs, so switch back to the stock repo (HARDWARE_TESTS 1b).
