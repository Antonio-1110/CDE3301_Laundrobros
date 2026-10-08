# CDE3301_Laundrobros

ROS 2 Jazzy package (`laundry_control`) that drives a UFactory xArm7 to pull laundry out of a bucket lying on its side (a stand-in for a washer drum). A wrist-mounted VL53L0X time-of-flight sensor is swept through the bucket, the readings become a point cloud, laundry is found by comparing that cloud against a fitted model of the empty bucket, and a servo gripper retrieves it.

It uses the manufacturer's `xarm_ros2` **unmodified**. The gripper and the joint limits are passed in as that package's own launch arguments. The bucket and table are MoveIt world objects whose poses are set in `config.OBSTACLES` (`laundry_control/config/scene.py`).

Everything runs through one command, `laundry`.

## Documentation

| Read | For |
|---|---|
| [docs/setup.md](docs/setup.md) | Getting the workspace onto a machine, building, `xarm_ros2` |
| [docs/operating.md](docs/operating.md) | Bring-up, every `laundry` command, baselines, the grab grid, moving the bucket |
| [docs/how-it-works.md](docs/how-it-works.md) | Detection, the scan path, baked motions, obstacles and padding, frames |
| [HARDWARE_TESTS.md](HARDWARE_TESTS.md) | Commands that need the real rig, with what to paste back |
| [esp32/README.md](esp32/README.md), [setup/](setup/) | The ESP32 link, the MQTT broker, hardware PWM |

Quick start, once set up: `source ~/ros2_ws/src/CDE3301_Laundrobros/env.sh`, then

```bash
ros2 launch laundry_control laundry_bringup.launch.py fake:=true rviz:=false   # no hardware
laundry run --fake-hardware --scan-from baseline_scans/<one of them>.csv
```

---

## Package layout

The package is in layers. A module imports only from its own layer and the ones above it in this list, never from one below it (`test/test_layers.py` checks this):

```text
laundry_control/
  config/            1. every hand-measured number; config.X reads any of them
    poses.py           recorded joint poses (HOME, INTER, DROP, ...), BUCKET_POSES
    grabs.py           the grab search for detected items
    scene.py           bucket and table poses, collision padding
    robot.py           MoveIt/xarm_ros2 names, joint limits, planners, speed limits
    hardware.py        ToF sensor, gripper servo, ESP32/MQTT link
  bucket.py          2. the bucket's cone, axis and (s, theta, r) coordinates

  hardware/          3. the devices, each owned by its own node
    tof_sensor.py      VL53L0X node (I2C, or readings from the ESP32)
    gripper_node.py    servo node: open_gripper / close_gripper services
    servo.py           the servo on this Pi's GPIO
    gripper_client.py  the services, as the pipeline calls them
    esp32_protocol.py, mqtt_link.py, mqtt_servo.py   the ESP32 over MQTT
    fake.py            stand-ins for --fake-hardware
  arm/               3. moving the arm
    controller.py      XArm7Controller: every motion goes through it
    kinematics.py      FK, IK, collision checks (part of XArm7Controller)
    trajectory.py      the scan's J7 twist, trajectory timing
    moveit_errors.py   what a MoveIt failure means; whether to retry
    transfers.py       baked routes between named poses (go_to)
    joint_path.py      planner-free joint paths and their timing
    scene.py           the bucket and table in MoveIt
    bucket_poses.py    INTER and BOTTOM, derived from the bucket
    geometry.py        tool-orientation math

  scan/              4. sweeping the sensor through the bucket
    pattern.py         scan(): the sequence of strokes, end scan, way out
    strokes.py         how the strokes divide the insertion
    endcap.py          the baked precession end scan
    recorder_node.py   records ToF points into a CSV (its own node)
    recorder_client.py the recorder, as scan() calls it
    segments.py        which readings are strokes, which end scan
    cloud_io.py        scan CSVs and PointCloud2
    baselines.py       collecting and managing the empty-bucket set
    replay.py          republish a saved scan for RViz

  perception/        5. finding laundry in a scan
    bucket_model.py    the empty bucket fitted to the baselines, and its noise
    detect.py          intrusions -> clusters -> ranked targets
    report.py          printing and publishing detections
    evaluate.py        `laundry evaluate`: false positives, recall
    synthetic.py, synthetic_eval.py   injected test laundry
    coverage.py        how much of the bucket a scan path sees
    flange_check.py    `laundry check-flange`

  grasp/             6. getting a detected item out
    execute.py         pick a reachable target and grab it
    plan.py            where the gripper goes for a cluster
    retrieve_grid.py   solving grabs: the sweep's, and detected items'
    grab_targets.py    the sweep's grabs as placed by hand
    grab_editor.py     placing them in RViz (`laundry plan edit-grabs`)
    targets_io.py      the targets JSON between detect and grasp

  pipeline.py        7. the stages chained: run, clear, preplanned
  cli/               8. the `laundry` command: one module per command group
                       (jobs, stages, arm, scene, plan, baseline), each
                       handler next to its parser

launch/laundry_bringup.launch.py   arm driver + MoveIt, the three device nodes, RViz
scan_plans/          baked motions (bake with ./rebake.sh, commit them)
baseline_scans/      the empty-bucket scans the detector models the bucket from
scan_records/        everything else you scan (git-ignored)
esp32/               firmware for the ESP32 that can own the ToF sensor and servo
test/                pytest; `python3 -m pytest test/` (no ROS graph needed)
```

Stages hand off through files (a scan CSV, then a targets JSON), so each runs alone and can be rerun offline. `laundry run` and `laundry clear` chain them in one process.

## How one run flows

What `laundry run` does, and where to look for each part:

1. **`cli/jobs.py` `cmd_run`** connects to MoveIt (`cli/common.RosSession` creates the `XArm7Controller`, which puts the padded bucket and table into the scene) and picks the recorder and gripper: the real nodes' clients, or the `--fake-hardware` stand-ins.
2. **`pipeline.run_full`** opens the gripper, then runs `scan_and_detect`.
3. **`scan/pattern.scan`** moves the arm: INTER (`arm/transfers.go_to`, a baked route) → straight in → J7 offset → inward strokes (`XArm7Controller.move_tool_z_with_twist`) → end scan or turnaround → outward strokes → save → straight out → INTER. It only moves. Meanwhile `scan_recorder_node` turns ToF readings into points through TF and writes the CSV.
4. **`perception/detect`** compares the scan with the empty-bucket model fitted to `baseline_scans/` (`perception/bucket_model`, once per run) and returns ranked clusters. A quick scan that finds nothing gets the end scan on its own and is judged again (`pipeline.scan_and_detect`).
5. **`grasp/execute.grasp_best`** picks the best reachable cluster. In the lower half of the drum, it is grabbed like the sweep's grabs (`grasp/retrieve_grid.plan_floor_grab`); otherwise by a Cartesian reach (`grasp/plan.compute_grasp_target`). Then: close, back to INTER, DROP, open, back to INTER.

`laundry clear` repeats steps 2–5 until a scan finds nothing, after the sensorless grab sweep (`pipeline.run_preplanned`).
