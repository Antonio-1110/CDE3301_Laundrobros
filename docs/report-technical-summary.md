# Technical summary for the design report (Sections 3 and 4)

Source material for **Section 3 (Development of Functional Prototype)** and **Section 4 (Testing & Evaluation)** of the CDE3301 report. Everything here is taken from the code on branch `full-drum-sweep-60dps` (including uncommitted work in its working tree) as of 2026-10-27. File paths are relative to the repository root, and line references point at the definitions so they can be cited.

> **Read this first: corrections to the original brief.** The brief that asked for this summary assumed some things this system does not have. Do not put these into the report:
>
> | The brief assumed | What the system actually does |
> |---|---|
> | A finite state machine with state enums (`NAVIGATE_INTER`, `LOOSE_SCAN`, `DENSE_SCAN`, ...) | No FSM class or enums. Each task is a fixed sequence of Python function calls that stops at the first failed step (§3). |
> | The drum hard-coded into `xarm_device.urdf.xacro` with a fixed joint | The bucket and table are **MoveIt world collision objects**, added at run time from `config.OBSTACLES` (§2.1). An earlier fork did put them in the URDF; that was removed on purpose. |
> | A custom `link_eef` | `link_eef` is the manufacturer's (fixed to `link7`, zero offset). The claw is attached to it through the stock `add_other_geometry` option (§2.2). |
> | `LC_NUMERIC="en_US.UTF-8"` or similar locale exports | None in the repo. The environment script is `env.sh` (§1.3). |
> | Calls to `/xarm/clear_err` | The code never calls the xArm's error services. Recovery is by stopping the controller and refusing unsafe motions (§5.2). |
> | Loose scan vs dense scan | There is a **quick** scan and a **full** scan, and the stroke spacing is configurable (§3.5). |
> | A range threshold that decides if laundry is present | Detection compares each reading against a statistical model of the **empty** bucket built from baseline scans (§4.2). |
> | `/sensor/tof_distance`, micro-ROS, WebSocket | MQTT between the ESP32 and the Pi, then `sensor_msgs/Range` on `tof_sensor/range` inside ROS 2 (§4). |

---

## 1. Repository structure and launch pipeline

### 1.1 Packages, nodes and executables

The project is one **ament_python** package, `laundry_control` (`package.xml`, `setup.py`). The arm driver and MoveIt configuration are the manufacturer's **unmodified** `xarm_ros2` (separate repository, `src/xarm_ros2`), launched with our arguments.

Executables (`setup.py`, `entry_points['console_scripts']`):

| Executable | Entry point | Role |
|---|---|---|
| `laundry` | `laundry_control.cli:main` | The single user command: `laundry run`, `clear`, `scan`, `detect`, `grasp`, `preplanned`, `move`, `plan bake`, ... Each run is a short-lived ROS 2 node (`xarm7_controller`) that talks to MoveIt. |
| `tof_sensor` | `laundry_control.hardware.tof_sensor:main` | Publishes ToF readings as `sensor_msgs/Range`, plus the static `link7 -> tof_sensor_link` transform. |
| `gripper_node` | `laundry_control.hardware.gripper_node:main` | Owns the claw servo. Serves `open_gripper` / `close_gripper` (`std_srvs/Trigger`). |
| `scan_recorder_node` | `laundry_control.scan.recorder_node:main` | Turns each Range reading plus TF into a 3D point. Publishes a `PointCloud2`, saves CSVs, serves `save_scan` / `clear_scan`. |
| `scan_replay` | `laundry_control.scan.replay:main` | Republishes a saved scan to RViz. |

Main Python modules (package `laundry_control/`):

| Folder | Contents |
|---|---|
| `arm/` | `controller.py` (`XArm7Controller`, the MoveIt client), `kinematics.py` (FK/IK/state-validity queries), `transfers.py` (baked routes between named poses), `pass_through.py` (time-optimal timing of baked routes), `direct.py` (grab <-> DROP routes that skip INTER), `scene.py` (planning-scene obstacles), `joint_path.py`, `trajectory.py`, `joint_ranges.py`, `bucket_poses.py` |
| `scan/` | `pattern.py` (the scan motion), `strokes.py`, `endcap.py` (end-of-drum scan), `recorder_node.py`, `segments.py`, `baselines.py` |
| `perception/` | `bucket_model.py` (empty-bucket model), `detect.py` (laundry detection), `coverage.py`, `synthetic*.py` (evaluation) |
| `grasp/` | `execute.py` (detected-item grasp), `retrieve_grid.py` (sensorless grab sweep), `pile_layers.py` (sweep generated from the fill level), `plan.py` |
| `hardware/` | `tof_sensor.py`, `gripper_node.py`, `gripper_client.py`, `esp32_protocol.py`, `mqtt_link.py`, `mqtt_servo.py`, `servo.py`, `fake.py` |
| `config/` | Every measured constant: `poses.py`, `scene.py`, `robot.py`, `hardware.py`, `grabs.py` |
| `cli/` | The `laundry` subcommands |
| `pipeline.py` | The stages chained together (`run_full`, `run_preplanned`, `run_clear`) |

Other top-level items: `launch/laundry_bringup.launch.py`; `meshes/` (`bucket.obj`, `table.obj`, the claw mesh `xArm7_ArmPlate_Gripper_CoACD.obj`); `scan_plans/` (baked motions, YAML); `baseline_scans/` (empty-bucket scans); `esp32/tof_gripper_bridge/` (ESP32 firmware); `mqtt_broker/` (mosquitto config); `test/` (pytest, 33 files); `benchmarks/` (simulation timing scripts).

### 1.2 Headless launch of the arm and MoveIt

One launch file brings up everything: `launch/laundry_bringup.launch.py`.

```bash
ros2 launch laundry_control laundry_bringup.launch.py                 # real arm
ros2 launch laundry_control laundry_bringup.launch.py rviz:=false     # fully headless
ros2 launch laundry_control laundry_bringup.launch.py fake:=true rviz:=false   # simulation
```

**How MoveIt is started.** The launch includes the manufacturer's own `xarm_moveit_config/launch/_robot_moveit_realmove.launch.py` (real arm) or `_robot_moveit_fake.launch.py` (simulation). Their files are loaded **unmodified** from where they are installed. Two things are injected into a private copy of the module (`_private_moveit_launch`, `_builder_with_task_ranges`, `_realmove_with_control_rate`):

- The task joint ranges go into MoveIt's joint limits (`arm/joint_ranges.add_to_moveit_limits`).
- The ros2_control update rate is set to 100 Hz instead of the stock 150 Hz (`config.ARM_CONTROL_RATE_HZ`).

**How MoveIt's RViz is disabled.** `_moveit_include` always passes `show_rviz='false'` to the manufacturer launch. Our own RViz node is started separately, after a delay, only if `rviz:=true`. It isn't the manufacturer's because closing that window shut down the whole launch, arm driver included.

```python
# launch/laundry_bringup.launch.py, _moveit_include
return IncludeLaunchDescription(
    source,
    launch_arguments=dict(
        config.xarm_description_arguments(),
        show_rviz='false',
        **extra_arguments,          # robot_ip, control_rate_hz
    ).items(),
    condition=condition,
)
```

**Arguments passed to the xArm description and MoveIt** (`config/robot.py`, `xarm_description_arguments()`):

```python
{
    'dof': '7', 'robot_type': 'xarm', 'hw_ns': 'xarm',
    'no_gui_ctrl': 'false',
    'limited': 'false',                       # XARM_LIMITED = False
    'add_other_geometry': 'true',             # the claw
    'geometry_type': 'mesh',
    'geometry_mesh_filename':
        'package://laundry_control/meshes/xArm7_ArmPlate_Gripper_CoACD.obj',
    'geometry_mesh_origin_xyz': '"0 0 0"',
    'geometry_mesh_origin_rpy': '"0 0 1.57"',
}
```

**Launch arguments** (all declared in `generate_launch_description`): `fake` (false), `rviz` (true), `robot_ip` (192.168.1.207), `control_rate_hz` (100), `rviz_delay` (3.0 s), `csv_path`, `records_dir`, `offset_cm` (−8.5, ToF calibration), `publish_rate_hz` (5.0, point-cloud republish), `tof_source` (`mqtt` | `i2c`), `gripper_backend` (`mqtt` | `gpio`), `mqtt_host` (`localhost`), `mqtt_port` (1883).

**Nodes started on the real rig:** the xArm driver and `move_group` (manufacturer launch); `tof_sensor`; `scan_recorder_node`; `gripper_node`; a one-shot `laundry scene apply` that adds the obstacles and exits; RViz (optional). With `fake:=true`, only the MoveIt fake controller and the obstacles start, and `laundry ... --fake-hardware` replaces the ToF sensor and claw with in-process stand-ins (`hardware/fake.py`).

**MoveIt execution parameters** (set by the manufacturer's `uf_ros_lib/moveit_configs_builder.py`, not changed by us): `allowed_execution_duration_scaling` 1.2, `allowed_goal_duration_margin` 0.5 s, `allowed_start_tolerance` 0.01 rad, `execution_duration_monitoring` False.

### 1.3 Environment and setup

- **`env.sh`** (source it in every shell): sources `/opt/ros/jazzy/setup.bash`, activates the Python venv at `<workspace>/.venv` (needed for `gpiozero`/Adafruit drivers), sources `<workspace>/install/setup.bash`, and puts `install/laundry_control/lib/laundry_control` on `PATH`. It warns if the package was built with a Python other than the venv's. ROS distribution: **Jazzy**. **There is no locale (`LC_NUMERIC`) setting.**
- **Build:** `colcon build --symlink-install --packages-select laundry_control`. With symlink install, the package in `install/` points back to the source tree, so edits take effect without rebuilding.
- **`setup.py` `data_files`** (what `package://laundry_control/...` resolves to):

```python
data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
    (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
],
```

- **Environment variables the code reads:** `LAUNDRY_DATA_DIR` (where scans and plans live; default: the source checkout, `config.repo_root()`). Simulation scripts use `ROS_DOMAIN_ID` 73/77 with `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` so they can never reach the real arm (`rebake.sh`, `benchmarks/sim_timing.sh`).

---

## 2. Kinematic and scene modelling

### 2.1 The drum (bucket) and table: planning-scene objects, not URDF

The bucket and table are added to MoveIt as **world collision objects** by `arm/scene.py` (`apply()`), through the `/apply_planning_scene` service. Their poses live in one place, `config/scene.py`:

```python
# laundry_control/config/scene.py
OBSTACLES = {
    'bucket': {
        'mesh': 'bucket.obj',
        'xyz': [0.14, -0.72, 0.42],        # metres, in link_base
        'rpy': [1.68, 3.14, -3.14],        # radians, URDF convention
        'allowed_links': [],
    },
    'table': {
        'mesh': 'table.obj',
        'xyz': [0.19, -0.43, -0.02],
        'rpy': [3.14, 3.14, 1.57],
        'allowed_links': ['link_base'],    # the arm's base rests on it
    },
}
```

```python
# laundry_control/arm/scene.py, _collision_objects
obj = CollisionObject()
obj.id = object_id
obj.header.frame_id = config.BASE_FRAME          # 'link_base'
obj.pose = obstacle_pose(spec)
obj.meshes.append(load_obj_mesh(os.path.join(mesh_dir(), spec['mesh'])))
obj.operation = CollisionObject.ADD
```

**Why not the URDF** (`arm/scene.py` docstring): when the bucket was a URDF link (an earlier `xarm_ros2` fork had `laundry_bucket_link` and `table_link`), MoveIt treated contact with it as self-collision, which it checks **without padding**, so the safety clearance never applied. Moving the bucket also meant editing the manufacturer's package. As world objects, they get padding and are moved by editing one config entry.

**Collision clearance ("padding")** (`config/scene.py`):

| Parameter | Value | Applies to |
|---|---|---|
| `OBSTACLE_PADDING_M` | 0.03 m | every moving arm link, for all motion |
| `GRIPPER_PADDING_M` | 0.0 m | the claw (it works inside the bucket by design) |
| `ENDCAP_PADDING_M` | 0.01 m | arm links during the baked end scan |
| `ROUTE_ARM_PADDING_M` | `bottom`: 0.02 m | the INTER -> BOTTOM route |
| `ROUTE_GRIPPER_CLEARANCE_M` | `drop`: 0.05, `drop_return`: 0.01, `home`: 0.01 m | claw clearance used when baking routes out of the bucket |

Every baked plan records a fingerprint of the obstacle geometry (`scene.signature()`, SHA-1 of poses and mesh files). Replaying a plan baked against a different scene prints a warning, and a plan that now collides is refused.

**Meshes and paths:** `meshes/bucket.obj`, `meshes/table.obj` (loaded by `scene.load_obj_mesh` straight from the package's `meshes/` folder, not through `package://`), and the claw mesh, given to the URDF as `package://laundry_control/meshes/xArm7_ArmPlate_Gripper_CoACD.obj`. **Collision simplification:** the claw is the arm plate (bolted to the flange) with the gripper plate stacked on its top face, each decomposed into convex hulls with CoACD: 189 hulls, about 11,000 triangles, in place of a 403,000-triangle STL (2026-10-22). The full-detail CAD is kept outside the package. The bucket and table are used as full meshes.

### 2.2 End effector and tool frames

The claw is the manufacturer's **"other geometry"** option (`xarm_description/urdf/other/other_geometry.urdf.xacro`), turned on with `add_other_geometry:=true`. The resulting frames, as generated by the stock xacro:

```xml
<!-- xarm_description/urdf/xarm_device_macro.xacro (manufacturer's, unmodified) -->
<link name="link_eef"/>
<joint name="joint_eef" type="fixed">
  <origin xyz="0 0 0" rpy="0 0 0" />
  <parent link="link7" />
  <child link="link_eef" />
</joint>

<!-- xarm_description/urdf/other/other_geometry.urdf.xacro, with our arguments -->
<joint name="other_geometry_fix" type="fixed">
  <parent link="link_eef"/>
  <child link="other_geometry_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>
<link name="other_geometry_link">
  <visual>    <origin xyz="0 0 0" rpy="0 0 1.57"/> <geometry><mesh filename="package://laundry_control/meshes/xArm7_ArmPlate_Gripper_CoACD.obj"/></geometry></visual>
  <collision> <origin xyz="0 0 0" rpy="0 0 1.57"/> <geometry><mesh filename="(same)"/></geometry></collision>
</link>
<link name="link_tcp"/>
<joint name="joint_tcp" type="fixed">
  <origin xyz="0 0 0" rpy="0 0 0"/>        <!-- geometry_mesh_tcp_xyz/rpy left at default -->
  <parent link="other_geometry_link"/>
  <child link="link_tcp"/>
</joint>
```

The stock SRDF already lets `other_geometry_link` touch `link3`, `link6` and `link7` without counting as collision.

**Frames the software uses** (`config/robot.py`, `config/hardware.py`):

| Frame / offset | Value | Meaning |
|---|---|---|
| `BASE_FRAME` | `link_base` | all poses |
| `FLANGE_LINK` | `link7` | MoveIt's Cartesian link; all offsets are in link7's own axes |
| `GRIPPER_LINK` | `other_geometry_link` | the claw, for padding |
| Claw contact point | `GRIPPER_OFFSET_Z` = 0.15 m along link7 +Z | where the claw closes (user-measured; marked "not yet verified" in the code) |
| ToF sensor | `tof_sensor_link`, offset (0.061, 0.0, 0.02) m from link7, no rotation | published by `tof_sensor` as a static transform; the beam points along +X |

Because the ToF sensor is offset sideways (+X) while the claw is on the wrist axis (+Z), turning J7 sweeps the beam around the bucket but does not move the claw's contact point.

### 2.3 Joint limits

`XARM_LIMITED = False` loads the xArm7's full hardware ranges, because J7 must go past −180° (INTER's J7 is about −159°, grabs reach about −197°). Automatic motion is then kept inside **task joint ranges** that are written into MoveIt's own joint limits at launch (`config.JOINT_TASK_RANGES_DEG`, `arm/joint_ranges.py`, issue #7):

| Joint | J1 | J2 | J3 | J4 | J5 | J6 | J7 |
|---|---|---|---|---|---|---|---|
| Task range (deg) | −80..170 | −30..143 | −210..30 | −30..157 | −70..145 | −110..125 | −270..30 |

A 2° margin from the hard limits (`JOINT_LIMIT_MARGIN_RAD`) is checked on top, because MoveIt's state-validity service does not reject joint-limit violations. (On 2026-09-29 a route reached J2 = 125° against a 120° limit and the arm stopped with error C23.)

---

## 3. Task orchestration

### 3.1 Architecture

There is **no state machine framework, no Lifecycle nodes and no action server of our own**. Each task is a Python function in `pipeline.py` (or `grasp/`) that calls the stages in order and stops at the first step that returns `False`. The `laundry` CLI is the client. It holds one ROS 2 node (`XArm7Controller`, `arm/controller.py`) that calls MoveIt's actions and services directly and blocks until each finishes.

The three top-level tasks:

```text
laundry run     (pipeline.run_full)
  open claw -> SCAN, coarse first (12 cm pass, then 3 cm, then the end scan
               alone, detecting after each and stopping at the first find;
               the arm stays inside the bucket between passes)
            -> [nothing found: done]
            -> go to INTER -> PLAN GRASP (best reachable cluster)
            -> approach + close claw -> carry to DROP with the claw turned
               +90 deg -> open claw -> INTER

laundry preplanned   (pipeline.run_preplanned -> grasp/retrieve_grid.run)
  go to INTER -> open claw
  for each baked grab_NN:  route in + descend (one trajectory) -> close
                           -> lift + route to DROP (one trajectory, claw
                              turned on the way and back) -> open
  -> INTER

laundry clear   (pipeline.run_clear)
  fit bucket model -> [preplanned sweep] ->
  repeat up to 15 rounds:  open -> SCAN -> DETECT
       no clusters  -> "The bucket is clear", stop (success)
       none reachable -> stop
       grasp + DROP; stop after 2 failed grasps in a row
```

From `pipeline.run_clear`:

```python
DEFAULT_CLEAR_MAX_ROUNDS = 15
DEFAULT_CLEAR_MAX_FAILED = 2
...
for index in range(1, max_rounds + 1):
    ...
    result = scan_and_detect(arm, recorder, csv_path, baseline,
                             scan_kwargs, detect_params, model)
    if result is None:
        print('Scan failed; stopping.')
        return False
    clusters, surface, csv_path = result
    if not clusters:
        print(f'The bucket is clear: {retrieved} item(s) retrieved ...')
        return True
    ...
```

**Coarse-first scanning** (`pipeline.scan_and_detect`, `scan.pattern.DEFAULT_COARSE_STEPS_M = (0.12,)`): a quick 12 cm pass first, then the 3 cm one only if that found nothing, then the end scan alone. Each pass is detected while the arm is still in the bucket, at the entry pose, and the arm goes for the laundry as soon as one pass finds it. The 3 cm pass is also checked at the deepest point of its inward strokes, and laundry found there ends the scan with a straight exit. The bucket is only called empty after the 3 cm pass and the end scan: a coarse pass that finds nothing proves nothing. `laundry run --coarse none --no-bottom-check` gives a single 3 cm quick scan.

**Overlapping checks and claw commands** (`grasp/retrieve_grid.run`, `grasp/carry.py`): each move is collision-checked while the claw is still opening or closing, and the claw is commanded as the arm arrives (`move(before_end=...)`), so the arm doesn't stand still for either.

**Layers** (`config.PILE['layer_heights_m']` 4 / 9 / 14 / 19 cm above the floor): the sweep is baked once, every layer at rows 32 / 21 / 10 cm from the closed end and 0 / ±12 cm sideways (the claw down; side spots step toward the middle where they do not fit). `laundry preplanned --fill N` (default 4) or `laundry clear --fill N` (default 1) starts at layer N and sweeps every layer under it, top down. The 2026-10-28 bake: 29 grabs, 9 / 9 / 9 / 2 from the top layer down.

If the report needs a state diagram, these are descriptive names (they do not exist in the code): `OPEN_GRIPPER -> SCAN_QUICK -> DETECT -> (EMPTY? END_SCAN -> DETECT_FULL) -> GO_INTER -> PLAN_GRASP -> APPROACH -> CLOSE -> RETRACT_INTER -> GO_DROP -> RELEASE -> GO_INTER`.

### 3.2 MoveIt 2 calls

`XArm7Controller.__init__` (`arm/controller.py`) creates:

| Interface | Type | Used for |
|---|---|---|
| `/move_action` | action, `moveit_msgs/action/MoveGroup` | planned joint-space moves (planner fallback) |
| `/compute_cartesian_path` | service, `moveit_msgs/srv/GetCartesianPath` | straight tool-axis strokes, grasp approach, reachability probing |
| `/execute_trajectory` | action, `moveit_msgs/action/ExecuteTrajectory` | executing every trajectory, including our own pre-computed ones |
| `/check_state_validity` | service, `GetStateValidity` | collision-checking every state of every non-planned motion |
| `/compute_ik`, `/compute_fk` | services | IK/FK |
| `/apply_planning_scene`, `/get_planning_scene` | services | obstacles and padding |
| `/xarm7_traj_controller/follow_joint_trajectory/_action/cancel_goal` | service | stopping the arm on Ctrl+C |

**Planned joint-space moves** (`move_joints` -> `_move_joints_once`). Pilz PTP first (deterministic straight-line interpolation), OMPL RRTConnect only if Pilz fails *to plan*:

```python
# laundry_control/arm/controller.py, _move_joints_once
constraints = Constraints()
for joint_name, angle in zip(self.JOINT_NAMES, joint_angles):
    joint_constraint = JointConstraint()
    joint_constraint.joint_name = joint_name
    joint_constraint.position = float(angle)
    joint_constraint.tolerance_above = tolerance      # 0.01 rad
    joint_constraint.tolerance_below = tolerance
    joint_constraint.weight = 1.0
    constraints.joint_constraints.append(joint_constraint)

goal = MoveGroup.Goal()
goal.request.group_name = self.group_name             # 'xarm7'
goal.request.pipeline_id = pipeline_id                # 'pilz_industrial_motion_planner', then 'ompl'
goal.request.planner_id = planner_id                  # 'PTP', then 'RRTConnect'
goal.request.num_planning_attempts = planning_attempts    # 10
goal.request.allowed_planning_time = planning_time        # 5.0 s
goal.request.max_velocity_scaling_factor = velocity       # 0.3
goal.request.max_acceleration_scaling_factor = acceleration   # 0.3
goal.request.goal_constraints = [constraints]
goal.planning_options.plan_only = False
goal.planning_options.replan = replan                     # True
goal.planning_options.replan_attempts = replan_attempts   # 5
```

**In normal operation the planner is rarely used.** Moves between named poses replay **baked routes** (`arm/transfers.go_to`). These are straight joint-space segments through a few via poses, found once on the simulator (`laundry plan bake`), saved to `scan_plans/transfers.yaml`, re-checked against the live scene, and sent as a fixed trajectory through `/execute_trajectory`. The fallback order is: baked route -> straight collision-checked joint move -> planner (with a warning). This exists because OMPL's random routes once turned J7 through 536° for a 12° change (`transfers.py` docstring).

**How baked routes are timed and shortened** (`arm/pass_through.py`, `arm/direct.py`, `arm/fast_route.py`):
- each route is timed by MoveIt's time-optimal parameterisation (TOTG) to pass through its vias without stopping, within the transfer limits (§5.1), checked on the cubic the controller actually draws;
- the sweep's grab <-> DROP moves are timed in one piece through INTER (`chains`), or replaced by **direct routes** that skip INTER where that is faster;
- HOME's route was re-searched for replay time rather than least joint travel: 5.2 s -> 3.1 s (2026-10-26), at the cost of more wrist travel;
- every timing keeps **every point of the claw under 0.9 m/s** (`TRANSFER_MAX_TIP_SPEED_M_S`): forward kinematics along the path, at the corners of a prism around link7's axis that holds the claw at any J7 angle. A route that would exceed it when stopping at its vias is slowed by a stored factor. This cap exists because the xArm stopped with C24 "Speed Exceeds Limit" on 2026-10-22 when the acceleration was raised without it.

**Cartesian paths** (`_plan_to_pose`), used for scan strokes and grasp approaches:

```python
# laundry_control/arm/controller.py, _plan_to_pose
request = GetCartesianPath.Request()
request.header.frame_id = self.base_frame      # 'link_base'
request.group_name = self.group_name           # 'xarm7'
request.link_name = self.flange_link           # 'link7'
request.waypoints = [target_pose]
request.max_step = max_step                    # 0.005 m (scan: DEFAULT_CARTESIAN_STEP_M)
request.jump_threshold = 0.0                   # joint-jump check disabled
request.avoid_collisions = True
request.max_velocity_scaling_factor = velocity
request.max_acceleration_scaling_factor = acceleration
...
if response.fraction < 0.999:                  # anything short of the full path is refused
    self.get_logger().error('Complete Cartesian path could not be generated.')
    return None, response.fraction
```

There is no explicit SLERP code. The orientation is held fixed along each stroke (`_build_pose` reuses the flange's current orientation), and MoveIt's Cartesian interpolator handles the path. Reachability probing for detected items reads the returned `fraction` rather than a yes/no answer.

### 3.3 The INTER stand-off pose

INTER is the reference pose at the bucket mouth: every scan starts and ends there, and the claw and ToF offsets were measured there. It is **derived from the bucket's modelled pose** by IK (`laundry plan bake poses`, `arm/bucket_poses.py`): flange on the bucket axis, `standoff_m` = 0.054 m outside the mouth plane, tool +Z pointing down the axis into the bucket, ToF beam (+X) pointing at the floor. The joint angles recorded by hand (`config.INTER_RECORDED`) only seed that IK.

Values in use (`scan_plans/bucket_poses.yaml`, baked 2026-09-28):

| | J1 | J2 | J3 | J4 | J5 | J6 | J7 |
|---|---|---|---|---|---|---|---|
| INTER (rad) | 1.549696 | 1.542439 | −2.519840 | 0.191257 | 0.264254 | 0.252419 | −2.779362 |
| INTER (deg) | 88.8 | 88.4 | −144.4 | 11.0 | 15.1 | 14.5 | −159.2 |

Flange position at INTER: (0.1392, −0.1395, 0.4836) m in `link_base`. Tool +Z: (0.0014, −0.994, −0.109), so it points about 6° below horizontal, along the bucket axis.

Other named poses (deg): HOME (90, 0, 0, 0, 0, 0, 0); DROP (5.8, 85.9, −177.2, 17.0, −6.8, 41.2, −166.9); BOTTOM, derived, 45° tilt at 0.42 m depth (−33.0, 64.2, −74.5, 87.8, 38.5, −59.8, −148.0).

### 3.4 Grasp planning for detected items

`grasp/execute.plan_grasp` walks the detected clusters best-first (`rank_clusters`: confident clusters nearest the mouth first, then the rest by volume) and takes the first one with a reachable grasp:

1. **Floor grab** (`retrieve_grid.plan_floor_grab`): IK over the item, a straight descent, approached from the nearest baked `grab_NN` pose. Used for items near the floor's lowest line.
2. **Cartesian reach** (`grasp/plan.compute_grasp_target`): a straight tool-axis reach from INTER, checked with `/compute_cartesian_path`. Used for items away from the floor.

The aim point is the mean of the cluster's top-quartile-intrusion points (`GRASP_POINT_QUANTILE = 75.0`, `perception/detect.py`), which is closer to the item's apex than the centroid (median 1.5 cm vs 2.6 cm in simulation).

**How deep the claw digs** (`retrieve_grid.deepest_grab`, `config.DETECTED_GRAB_DIG`). The ToF sees only the pile's top, so the grab is sunk into it as deep as is **collision-free**. For each claw tilt, heights are tried from the deepest allowed upward: at least `wall_clearance_m` (2 cm) off the bucket wall and at most `max_sink_m` (15 cm) below the sensed top. The first height whose grab, approach and descent pass IK and MoveIt's collision checks wins. The approach starts 3 cm above the pile where it fits, so the digging is a straight descent. On the fake controller (36 spots) the claw went 9.5 cm into the pile on average, against 4.3 cm with the earlier rule (half the pile's height, at most 5 cm).

**Carrying with the claw turned** (`grasp/carry.py`, `config.CARRY_J7_TURN_DEG` = 90): after closing, J7 turns +90° during the move to DROP and back before opening, so the claw hooks the item instead of relying on grip force alone (issue #21). The turn is superimposed on the baked move without stopping (S-curve ramps of 1.3–1.7 s, J7 within 0.9 × its limits). The turned states are collision-checked, and the claw is carried unturned if they collide.

### 3.5 The scan trajectory (quick and full scans)

The ToF beam is fixed at 90° to the wrist axis, so the scan inserts the tool **straight along its own axis** in short strokes while turning J7, which sweeps the beam around the wall: a helix (`scan/pattern.py`, `scan/strokes.py`, `arm/trajectory.add_joint7_twist`).

Sequence (`scan.pattern.scan`):

```text
1. enter          straight in from INTER by entry_depth (0.06 m), not recorded
2. initial_offset J7 to -sweep/2 (-75 deg), on the spot
3. strokes in     N strokes of `step` along tool +Z, each with a +/-150 deg J7 twist (alternating)
4. end scan       quick: J7 turnaround only;  full: baked precession sweep of the closed end
5. strokes out    the same strokes outward
6. save CSV, 7. back to INTER
```

Stroke count and spacing (`strokes.stroke_plan`):

```python
count = max(1, math.ceil(span / max_step - _EPSILON))   # span = depth - entry_depth
step  = span / count
```

J7 twist on each stroke (`trajectory.add_joint7_twist`). MoveIt plans the straight stroke, then J7 gets an extra rotation proportional to progress along the path, so it turns a fixed angle per centimetre: a true helix. Velocities and accelerations get the matching derivatives, and the whole stroke is slowed if J7 would exceed 2.14 rad/s or 10 rad/s².

```text
q7(t) = q7_moveit(t) + twist * p(t),   p(t) in [0, 1] = normalised joint-space arc length
```

| Parameter (`scan/pattern.py`) | Default |
|---|---|
| `DEFAULT_DEPTH_M` (deepest stroke) | 0.42 m |
| `DEFAULT_ENTRY_DEPTH_M` | 0.06 m |
| `DEFAULT_STEP_M` (largest stroke spacing) | 0.03 m -> 12 strokes in, 12 out |
| `MAX_STEP_M` (coarsest allowed, `strokes.py`) | 0.12 m |
| `DEFAULT_SWEEP_DEG` (J7 per stroke) | 150° |
| `DEFAULT_VELOCITY` / `DEFAULT_ACCELERATION` (MoveIt scaling) | 0.03 / 0.03, scaled with the step: `min(1, 0.03 * step / 0.03)` |
| `DEFAULT_CARTESIAN_STEP_M` | 0.005 m |
| `DEFAULT_END_SCAN` | `none` (quick scan) |

**One trajectory per pass** (`arm/controller.move_tool_z_strokes`, `arm/trajectory.join_trajectories`): the 12 strokes of a pass are still planned and collision-checked one by one, then sent as a single trajectory. The arm therefore doesn't wait ~0.45 s at each end of the J7 sweep for each stroke to be confirmed. `--pause 0.01` restores stroke-at-a-time.

**Baselines without leaving the bucket** (`laundry scan --stay-inside` / `--start-inside`, used by `laundry baseline collect`): consecutive scans hand over at the entry pose, 6 cm inside, instead of going out to INTER: about 3.4 s per handoff instead of ~17 s in simulation.

**Quick vs full, and coarse vs fine.** The closest match to "loose vs dense":

- **Quick scan** (default, `laundry scan`): strokes only. 97 s in simulation.
- **Full scan** (`--full`): adds the end-of-drum precession sweep. 140 s.
- **Coarse spacing** (`--step 0.06` or `0.12`): 57 s / 37 s, but a 12 cm spacing finds only about half the socks (simulation). Saying the bucket is empty needs the default 3 cm.

**Full scan: precession ("coning") of the closed end** (`scan/endcap.py`). With the flange held at the deepest stroke, the tool axis is tilted by α in direction φ, and the roll is chosen so the beam leans toward the closed end:

```text
u      = cos(phi) * up + sin(phi) * side      (tilt direction)
tool Z = cos(alpha) * z0 + sin(alpha) * u
beam X = sin(alpha) * z0 - cos(alpha) * u
```

Rings: α = 10°, 25°, 40° over φ ±90°, and α = 55° over φ 0..70°. The whole sweep is solved once by IK on the simulator (each pose seeded from the previous one, so the elbow stays on one branch), collision-checked at 1° spacing and replayed as a fixed trajectory. Simulated coverage of the closed end's lower half went from 57% (the old single tilted pose) to 82%.

---

## 4. ESP32 wireless link and end effector

### 4.1 Communication

The ToF sensor (VL53L0X) and the claw servo are on an **ESP32-C3** on the arm (`esp32/tof_gripper_bridge/tof_gripper_bridge.ino`). It talks to the Pi over **MQTT** through a mosquitto broker on the Pi (`mqtt_broker/`, port 1883). The Pi-side protocol is `hardware/esp32_protocol.py`; the MQTT client is `hardware/mqtt_link.py`. Topics (prefix `laundrobros`):

| Topic | Direction | QoS | Payload |
|---|---|---|---|
| `laundrobros/tof/range` | ESP32 -> Pi | 0 | `{"seq": 812, "t_us": 48211034, "mm": 143, "boot": "3fa2c1d0"}` |
| `laundrobros/time/ping` | Pi -> ESP32 | 0 | `"<id>"` |
| `laundrobros/time/pong` | ESP32 -> Pi | 0 | `{"id": 17, "t_us": 48211102, "boot": "3fa2c1d0"}` |
| `laundrobros/gripper/cmd` | Pi -> ESP32 | 1, never retained | `{"id": 4, "angle": 100.0, "hold": true, "settle": 0.5}` |
| `laundrobros/gripper/ack` | ESP32 -> Pi | 0 | `{"id": 4, "ok": true, "msg": "..."}` |
| `laundrobros/esp32/status` | ESP32 -> Pi | retained | `"online"` / `"offline"` (MQTT last will) |

**Timestamping.** The ESP32 stamps each reading with its own clock at the middle of the measurement window. The Pi maps that clock to its own NTP-style, keeping the ping with the shortest round trip:

```text
offset = (t0 + t1) / 2 - t_esp,   error <= (t1 - t0) / 2
```

This matters because J7 turns about 2° per reading (about 7 mm at the wall), and WiFi latency varies between ~3 ms and 100+ ms.

The Pi's own I2C sensor and GPIO servo remain as alternatives (`tof_source:=i2c gripper_backend:=gpio`).

### 4.2 ToF data in ROS 2, and how laundry is detected

**Message and rate.** `tof_sensor` publishes `sensor_msgs/msg/Range` on **`tof_sensor/range`**, with `header.frame_id = tof_sensor_link`, `radiation_type = INFRARED`, `field_of_view = 0.436 rad`, `min_range = 0.03 m`, `max_range = 0.25 m`, and `range` in metres after the calibration offset (`offset_cm` = −8.5 on the rig). Rate:
- from the ESP32: one reading per 33 ms timing budget, about **30 Hz** (`TOF_TIMING_BUDGET_US 33000`, continuous mode);
- from the Pi's I2C sensor: **20 Hz** (`TOF_PUBLISH_PERIOD_SEC` = 0.05).

`scan_recorder_node` subscribes to it (sensor-data QoS), looks up the sensor's transform to `link_base` (and to `link7`) in TF at each reading's timestamp, turns the range into a 3D point, republishes the cloud as `PointCloud2` on `scan_record/points` at 5 Hz, and saves CSVs (services `save_scan`, `clear_scan`).

**Detection is not a range threshold.** It compares each scan with a model of the **empty** bucket (`perception/bucket_model.py`, `perception/detect.py`):

1. **Model:** 8 empty-bucket scans (`baseline_scans/`) are fitted with the bucket's shape (a cone with a flat closed end). For each cell of the surface (2 cm along the bucket's profile × 10° around, `DEFAULT_ARC_BIN_M`, `DEFAULT_ANGULAR_BIN_RAD`; 19 × 36 cells for this bucket), the model stores the expected wall offset and its spread σ. A cell needs at least 5 samples to count as confident (`DEFAULT_MIN_CELL_SAMPLES`).
2. **Intrusion test:** for each new reading, intrusion = how far it stands in front of the modelled wall. A point is flagged if:

```python
# laundry_control/perception/detect.py
threshold = np.maximum(k_sigma * sigma, abs_floor_m)   # k_sigma = 4.0, abs_floor_m = 0.008 m
```

3. **Clustering and gates** (defaults in `perception/detect.py`):

| Parameter | Default | Purpose |
|---|---|---|
| `DEFAULT_K_SIGMA` | 4.0 | seed: intrusion > 4 local σ |
| `DEFAULT_ABS_FLOOR_M` | 0.008 m | ... and > 8 mm |
| `DEFAULT_GROW_K_SIGMA` / `DEFAULT_GROW_FLOOR_M` | 2.5 / 0.004 m | grow clusters through weaker neighbours |
| `DEFAULT_BEHIND_WALL_MARGIN_M` | 0.03 m | readings this far behind the wall are discarded |
| `DEFAULT_CLUSTER_RADIUS_M` | 0.04 m | 3D linking radius |
| `DEFAULT_MIN_CLUSTER_SIZE` | 3 points | pre-filter |
| `DEFAULT_MIN_EXTENT_M` | 0.025 m | footprint on the wall |
| `DEFAULT_MIN_VOLUME_M3` | 1.0e-5 m³ (10 cm³) | integrated intrusion volume |
| `DEFAULT_LOW_CONFIDENCE_MIN_PEAK_SIGMA` | 7.0 | clusters in thinly sampled cells must peak this high |
| `DEFAULT_MIN_PEAK_SIGMA` / `DEFAULT_LARGE_VOLUME_M3` | 5.5 σ / 70 cm³ | **strong or large**: every cluster must peak at 5.5 σ OR reach 70 cm³. Real laundry is one or the other (bulky items stand proud of the wall; a flat cloth covers a wide area); noise is neither (issue #22) |

4. **Decision:** "laundry present" means at least one cluster survives the gates. `laundry clear` stops when a scan returns **no** clusters.

### 4.3 Claw actuation

Interfaces: ROS 2 services **`open_gripper`** and **`close_gripper`** (`std_srvs/srv/Trigger`), served by `gripper_node`. The caller is `hardware/gripper_client.GripperClient`:

```python
# laundry_control/hardware/gripper_client.py
def start_close(self, timeout_sec=5.0):
    """Ask the gripper to close; wait() on the result for confirmation."""
    return self._start(self._close_client, CLOSE_SERVICE, timeout_sec)

def _start(self, client, name, timeout_sec):
    if not client.wait_for_service(timeout_sec=timeout_sec):
        ...
        return Pending(lambda: False)
    future = client.call_async(Trigger.Request())
    return Pending(lambda: self._finish(future, name, timeout_sec))
```

`gripper_node` then publishes `laundrobros/gripper/cmd` and blocks until the matching ack arrives (`hardware/mqtt_servo.MqttServoDriver.move`; timeout = settle + 2 s). A missing or failed ack makes the service call fail.

| Parameter (`config/hardware.py`) | Value |
|---|---|
| `GRIPPER_OPEN_ANGLE_DEG` / `GRIPPER_CLOSE_ANGLE_DEG` | 55° / 100° |
| `ESP32_GRIPPER_SETTLE_SEC` | 0.5 s per move (0.75 s until 2026-10-21, 1.5 s before) |
| `GRIPPER_HOLD_PULSES` | True: the ESP32 keeps the PWM on after a move (without it the claw stopped short of its angle) |
| Servo PWM (ESP32) | 50 Hz, 500–2500 µs for 0–180°, `SERVO_PIN 20` |

In the sweep, the arm's next motion is collision-checked while the claw is still moving (`start_close()` ... `wait()`, `grasp/retrieve_grid.run`).

---

## 5. Speed, safety and recovery

### 5.1 Speed and acceleration limits

| Where | Setting | Value |
|---|---|---|
| Planned joint moves (`move_joints`) | `max_velocity_scaling_factor` / `max_acceleration_scaling_factor` | 0.3 / 0.3 |
| Scan strokes (`/compute_cartesian_path`) | velocity / acceleration scaling | 0.03 / 0.03 at the default 3 cm step |
| Stationary J7 turns (`rotate_joint7`) | velocity / acceleration scaling | 0.5 / 0.5 |
| Grasp approaches (`move_to_pose`) | velocity / acceleration scaling | 0.1 / 0.1 |
| Baked transfers (`config/robot.py`) | `TRANSFER_MAX_VELOCITY_RAD_S` | 60°/s (1.047 rad/s) |
| | `TRANSFER_MAX_ACCELERATION_RAD_S2` | 7.0 rad/s² (70% of MoveIt's xArm7 limit; 5.0 until 2026-10-26) |
| | `TRANSFER_MAX_JERK_RAD_S3` | 59 rad/s³ (42 until 2026-10-26) |
| | `TRANSFER_MAX_TIP_SPEED_M_S` | 0.9 m/s: no point of the claw faster (checked by forward kinematics at bake time; the xArm stops with C24 over 1 m/s at the tool, seen on 2026-10-22 at 7.0 without this cap) |
| Straight joint moves and the end scan | `LINEAR_JOINT_MOVE_MAX_VELOCITY_RAD_S` | 45°/s |
| J7 during scan twists | `JOINT7_MAX_VELOCITY_RAD_S` / `..._ACCELERATION_RAD_S2` | 2.14 rad/s / 10 rad/s² |
| Any command | `--speed` (`time_scale`) | e.g. 0.3 for cautious first runs |

Baked transfers are timed with MoveIt's time-optimal trajectory generation (TOTG) to pass through their vias. The limits are checked on the cubic the trajectory controller actually draws, not only on TOTG's points (`arm/pass_through.py`). The ros2_control loop runs at 100 Hz (`ARM_CONTROL_RATE_HZ`), because at the stock 150 Hz the controller's ticks overran and the arm jerked.

### 5.2 Error handling and recovery

**Before moving: nothing is executed unchecked.**
- Every state of every non-planned motion (baked routes, straight moves, the end scan, the sweep) is collision-checked at 1° spacing against the **live** planning scene immediately before it runs (`kinematics.first_invalid_state`, `transfers._checked_plan`). If any state collides, nothing moves.
- Joint targets outside the task ranges are refused before planning (`move_joints`).
- Cartesian paths short of 99.9% are refused (`_plan_to_pose`).
- A state-validity request MoveIt doesn't answer within 1 s is re-sent up to 3 times, then counts as a collision (`kinematics.VALIDITY_TIMEOUT_SEC`, `VALIDITY_ATTEMPTS`).

**Planning failures.** Pilz falls back to OMPL only for failures that happen before anything moves (`arm/moveit_errors.PLANNING_STAGE_ERROR_CODES`: −2 PLANNING_FAILED, −3, −11..−17, −31 NO_IK_SOLUTION). A failure during or after execution is never retried, because the arm may have been stopped part-way:

```python
# laundry_control/arm/controller.py, move_joints
if not retryable:
    self.get_logger().error(
        f'Joint movement failed with {attempt_pipeline}/{attempt_planner} '
        f'after planning: {failure_reason}. '
        'Not retrying - the arm may have been stopped part-way.')
    return False
```

**Stopping (Ctrl+C).** `move_group` does not act on a cancel while it is executing, so `_stop_after_interrupt` cancels the goal **at the trajectory controller itself** (`/xarm7_traj_controller/follow_joint_trajectory/_action/cancel_goal`, an all-zero goal id = cancel all) and repeats until MoveIt releases the goal. If the controller can't be reached, it tells the operator to use the e-stop.

**Task-level recovery.** On a failed step the tasks bring the arm back to a known pose where they can. The sweep always lifts the claw back out of the pile, then returns to INTER. `laundry clear` stops after 2 failed grasps in a row or when nothing detected is reachable. Gripper failures are detected from the missing ack. If the claw doesn't confirm it opened at DROP, the arm returns by the route for a loaded claw.

**Execution timeouts** are MoveIt's: `allowed_execution_duration_scaling` 1.2 and `allowed_goal_duration_margin` 0.5 s, with `execution_duration_monitoring` False (manufacturer defaults).

**xArm hardware errors.** The code does **not** call `/xarm/clean_error`, `set_mode` or `set_state`, although the driver offers them (`xarm_api/config/xarm_params.yaml`). A controller error (for example C23, joint limit exceeded) stops the run, and the operator clears it. The prevention is upstream: the 2° joint-limit margin and the task ranges exist because of such an error.

---

## 6. Testing and evaluation (for Section 4)

### 6.1 Test layers

| Layer | What | Where |
|---|---|---|
| Unit tests | 556 pytest tests (geometry, scan pattern, detection, baked-plan sanity, routing, gripper protocol, ESP32 clock sync, flake8/pep257) | `test/` (`python3 -m pytest test`); also run in GitHub Actions on every PR |
| Simulation | Real-time MoveIt fake controller, isolated ROS domain: motion times, collision-free plans | `benchmarks/sim_timing.sh`, `rebake.sh`, `laundry scene check` |
| Offline detection | Leave-one-out on empty baselines, synthetic items injected into real empty scans with a ToF model | `laundry evaluate`, `benchmarks/sim_metrics.sh` |
| Rig | Step-by-step hardware checklist, hand on the e-stop | `HARDWARE_TESTS.md` |

### 6.2 Results (simulation and offline, `docs/performance.md`)

**Scan time** (from INTER to INTER): quick 97 s, full 140 s, quick at 6 cm spacing 57 s, at 12 cm 37 s. The J7 sweep dominates (about 3 s per stroke).

**Detection** (8 empty baselines from 2026-09-30; the false-cluster row is updated below for the 2026-10-22 baselines):

| Measure | Quick scan | Full scan |
|---|---|---|
| False clusters, leave-one-out over 8 empty scans | 1 (0.12 per scan) | 1 (0.12 per scan) |
| Synthetic items found, floor and lower wall (5 item types, 480 trials) | 480/480 | |
| Dark fabric (0.3 reflectivity) | 477/480 (99.4%) | |
| Synthetic items found, all regions, reachable ones | | 703/782 (89.9%) |
| Grasp point error to item apex, median / 90th percentile | 0.7 / 1.6 cm | 1.0 / 6.3 cm |
| Time to build the model and detect | 0.4 s | |

Surface coverage (8 scans pooled): floor 100%, lower wall 99%, lower closed end 74%, upper wall 6%, ceiling 0% (the upper wall and ceiling are outside the scan's design).

**False clusters against the new baselines** (2026-10-22, new ToF mount; leave-one-out): **6 over 8 folds** without a further gate, all shallow (4.0–5.0 σ, 8.5–12 mm, 12–61 cm³) → **0** with the strong-or-large gate, quick and full. Synthetic quick scan, floor and lower wall: 476/480 → 474/480 found (the two lost are flat cloths on the lower wall); stray clusters 0.19 → 0.00 per trial. A real rig scan with laundry (2026-10-21) still returns its one cluster at 44.8 σ.

**Retrieve time:**

| | Simulation | Rig |
|---|---|---|
| `laundry run`, one item | 125 s (scan 93 s, detect 0.4 s, grasp 28 s) | about 130 s |
| Sensorless sweep, 24 grabs, 45°/s (2026-10-19) | 346 s | about 7 min |
| Same, 60°/s with the 2026-10-21 optimisations | **213 s** (8.9 s per grab) | not yet run |
| Same plan, 60°/s and 7 rad/s², claw cap (2026-10-27) | 178 s of motion + ~65–75 s of checks (CPU-loaded sim) | not yet run |
| One sweep grab, rig log before the optimisations (2026-10-21) | | about 14 s |

**Where the 2026-10-21 speed-up came from** (measured on the rig log, then verified in simulation):
- The arm stood still for 1.2–2.3 s before every transfer. Plan files were re-parsed on each move (about 0.6 s), and about 340 collision-check calls were made serially (2.3 ms each). Fixed by caching the files and running the checks while the claw moves.
- Each grab's descent and lift were separate goals. Each is now one trajectory with the route.
- Direct grab <-> DROP routes that skip INTER save 0.3 s per move on average. The straight line collides for all 24 grabs.
- The claw wait went from 0.75 s to 0.5 s.

**Sweep time as a function of the transfer limits** (2026-10-27, the same 24-grab plan, re-timed at each setting with the claw cap, motion time from the planned durations of every trajectory the sweep sends; 2 settings also run in real time, matching move for move):

| deg/s \ rad/s² | 3 | 5 | 7 |
|---|---|---|---|
| 45 | 242 s | 226 s | 226 s |
| 60 | 217 s | 193 s | 178 s (current) |
| 90 | 217 s | 168 s | 149 s |

Fit: **M(v, a) ≈ 15.8 + 140.5 / v + 202.6 · v / a** s (v in rad/s, a in rad/s²; each move ≈ distance / v + v / a), rms 4.2 s. At 45°/s the top speed limits it (more acceleration buys nothing); at 90°/s and 3 rad/s² the DROP routes lose their rounded timing. Per grab: 7.3 s of motion on average at 60 / 7 (6.4 s near the mouth, 8.9 s deepest), plus ~3 s of checks, so adding or removing a grab changes the sweep by about 10 s.

**Safety finding on the rig (2026-10-22):** raising the transfer acceleration to 7.0 rad/s² without a claw-speed limit stopped the arm with C24 "Speed Exceeds Limit" on INTER → HOME. No joint was over its limit, but J1, J3 and J7 near 60°/s together took the claw to ~1050 mm/s against the xArm's 1000 mm/s (975 mm/s at 5.0). That led to the 0.9 m/s claw cap (§3.2).

### 6.3 Known limitations and open issues (for the evaluation discussion)

- **Grasp success is not sensed.** Nothing detects whether the claw closed on laundry; simulation counts a grasp as done when the motions finish.
- **Mouth false positives (issue #20, resolved).** On the rig (2026-10-21), shallow anomalies at the bucket mouth (16–104 points, 6–7 mm intrusion) were ranked ahead of real laundry. Cause: the baselines predated the new ToF mount. Against the re-collected baselines, only the laundry cluster remains. Example scan in `validation_scans/mouth_false_positive_20261021/`.
- **Most of the last week's work is untested on the rig:** the claw cap at 7.0 rad/s², the fast HOME route, one-trajectory scans, coarse-first `run`/`clear`, stay-inside baselines, deeper detected grabs and the new gripper mesh. The claw turn, the layered sweep and coarse-first `run`/`clear` have run on the rig (2026-10-27/28). The staged tests are in `HARDWARE_TESTS.md` section G.
- **The detector gate was tuned on the 6 false clusters it now rejects,** with thin margins (5.0 vs 5.5 σ, 61 vs 70 cm³). New empty scans with the current mount are needed to test it independently.
- **The detection figures are an upper bound:** synthetic items, a simple ToF model, and baselines from the old mount.
- **The claw contact offset** (`GRIPPER_OFFSET_Z` = 0.15 m) is user-measured and marked unverified in the code.
- **The layered sweep** (issue #18) and **shorter scans** (issue #10) have run on the rig (2026-10-27/28); the scan times in §6.2 predate the one-trajectory and coarse-first changes and need re-measuring.
