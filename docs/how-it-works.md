# How it works

The ideas behind detection, the scan path, the baked motions and the obstacles. For the commands, see [operating.md](operating.md); for where each piece lives in the code, the package map in the [README](../README.md#package-layout).

## How detection works (short version)

1. **Model the empty bucket.** `perception/bucket_model.py` fits a cone (the wall) plus a flat cap (the closed end) to 8 empty-bucket scans in `baseline_scans/`. It then learns a per-cell offset and noise sigma on the bucket surface, in the bucket's own cylindrical coordinates, so the test is valid on the floor, the walls and the ceiling alike.
2. **Flag intrusions.** Each reading's intrusion inside the modelled wall is compared with the local sigma. Seed points must clear 4σ and 8 mm. Clusters are then grown through connected points clearing 2.5σ and 4 mm (hysteresis).
3. **Filter clusters.** Clusters must pass extent and volume gates. Clusters in thinly-sampled cells must also peak at ≥ 7σ. Each cluster reports its peak σ, and a grasp point: the mean of its top-quartile-intrusion points.
4. **Plan the grasp.** `grasp/plan.py` sinks the grasp point into the pile by as much room as the bucket model says exists underneath, then checks reachability with a plan-only probe.

The scan deliberately covers the **bottom** of the bucket: the floor, the lower walls and the lower half of the closed end. The 150° J7 sweep is centred on the floor. It runs at velocity 0.03 (was 0.1) for denser data and a J7 speed within its limit.

**The scan path.** From INTER the arm moves straight in by `--entry-depth` (default 6 cm) without recording. The sensor starts just outside the mouth, so readings taken there caught the lip and things past it. Recording starts once the sensor is inside. The strokes then swing J7 150° each while moving in to `--depth` (0.42 m). They're spread evenly and at most `--step` apart (`scan/strokes.py`: 12 strokes of 3 cm with the defaults). `--step` is capped at 3 cm, and a larger value is rejected. After the end scan and the outward strokes back up to the entry depth, recording stops, J7 turns back, and the arm moves straight out. Changing `--entry-depth` or `--step` needs no re-bake, because the strokes are planned live and the end scan pivots at `--depth`. It does change the scan path, so re-collect the baselines with the same values.

**Full and quick scans.** The full scan (`--full`, `--end-scan precession`) ends with the tilting end scan, the only part that sees the closed end. It swings the wrist and elbow low near the lower walls. The quick scan (`--quick`, `--end-scan none`) is the same strokes without it. Both use **one baseline set**, `baseline_scans/` (full scans): every reading is labelled strokes or end scan (`scan/segments.py`; older files are labelled from the beam's tilt), and a quick scan is judged against the baselines' strokes alone. `laundry scan`, `laundry run` and `laundry clear` scan quick by default (`--full` for the end scan too), and `laundry detect` tells the two apart from the CSV. When a quick scan finds nothing, they run the end scan on its own (straight in, end scan, straight out) and judge quick scan + end scan against the full baselines. So the bucket is emptied with quick scans, and the tilting end scan runs only near the end. Grabs go front (mouth) first among confident detections. Baselines are always collected as full scans.

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

## Repeatable moves between poses

Named-pose moves (`laundry move <pose>`, and the scan, grasp, drop and preplanned stages) go through `arm/transfers.go_to`, which tries three routes in order:

1. **A baked route** from `scan_plans/transfers.yaml`. INTER is the hub: there is one route from INTER to each of HOME, DROP, BOTTOM and the grab_NN poses (`laundry scene check` lists which exist).
   - INTER → pose replays the route; pose → INTER replays it in reverse.
   - Pose → another pose (e.g. grab_04 → DROP) goes back to INTER along one route and out along the other, so the arm always leaves the bucket through its mouth.
   - Each route is straight joint-space segments through zero to two intermediate poses (for HOME/DROP, after first backing the gripper straight out of the bucket). Every 1° is collision-checked, and the route with the least joint travel wins, weighted toward J1 and J4–J7, which twist the cables.
   - Before every replay the whole route is re-checked against the current planning scene. A route that now collides is refused, with a message to re-bake. A route baked against different obstacle poses still runs if it's clear, with a warning to re-bake. A route baked from a different INTER than the current one (after INTER is re-derived or re-recorded) is not used at all.
2. **Otherwise:** a straight, collision-checked joint move, which is the minimum twist.
3. **Only if that collides:** the planner, with a warning.

Re-bake (`laundry plan bake transfers`) after changing a recorded pose, the padding or `config.OBSTACLES`.

### INTER and BOTTOM, from the bucket

INTER (on the bucket axis, just outside the mouth, where the scan starts) and BOTTOM (the scan depth in, tilted up) are defined relative to the bucket in `config.BUCKET_POSES`, so they move with it:

- **INTER:** the flange on the bucket axis, `standoff_m` outside the mouth plane, with the tool pointing straight down the axis and the ToF beam at the floor.
- **BOTTOM:** the flange `depth_m` in from INTER along the axis, with the tool tilted `tilt_deg` up from the axis (the end scan's tilt-up pose).

`laundry plan bake poses` solves their joint angles by IK in the bucket of `config.OBSTACLES` and saves them to `scan_plans/bucket_poses.yaml`. No motion. The IK is seeded from the hand-jogged `INTER_RECORDED` / `BOTTOM_RECORDED`, so the arm keeps the same elbow posture. The command prints how far each derived pose is from the jogged one, and calls out anything over 3 cm or 5°: that means `config.OBSTACLES` and the real bucket disagree. Until the file exists, the jogged angles are used. `laundry scene check` says which one is in use.

Everything else is baked from INTER, so `laundry plan bake` (all) derives the poses first, then bakes the grab grid, the transfers and the end scan. The derived poses are only as accurate as `config.OBSTACLES`. The scan also runs from INTER, so changing INTER invalidates the baselines.

### Obstacles: the bucket and table

The bucket and table are MoveIt **world objects**, not robot links. `arm/scene.py` adds `meshes/bucket.obj` and `meshes/table.obj` at the poses in `config.OBSTACLES`, which is the only place those poses are set. Bring-up adds them at start-up (`laundry scene apply`), and every `laundry` command re-checks on connect. (They used to be URDF links in the `xarm_ros2` fork, and MoveIt checks robot-vs-robot contact without padding, so padding never kept the arm away from them.)

To move them, see [After moving the bucket](operating.md#after-moving-the-bucket-or-table).

**Padding:**

- **Arm links (link1–link7): 3 cm** (`config.OBSTACLE_PADDING_M`), for everything: the planner, Cartesian strokes, and the straight and baked moves. The exceptions are the end scan (1 cm, `config.ENDCAP_PADDING_M`) and the route to BOTTOM (2 cm, `config.ROUTE_ARM_PADDING_M`), whose tilted tool brings the elbow to the rim. Each is baked and replayed under its own padding.
- **Gripper: 0 cm** (`config.GRIPPER_PADDING_M`), because it works inside the bucket on purpose: the grabs put it within 2–3 cm of the floor.
- **Routes that leave the bucket** (to HOME and DROP) are baked with extra gripper padding set per route (`config.ROUTE_GRIPPER_CLEARANCE_M`: 1 cm for HOME, 5 cm for DROP, where laundry hangs from the gripper), so it clears the bucket mouth on the way out.
- **The way back from DROP** has its own route, `drop_return`, baked at 1 cm: the 5 cm is for laundry hanging from the gripper, and after DROP nothing does. It is only used once the gripper has confirmed it opened; otherwise the arm goes back the 5 cm way.
- **Timing:** each leg of a baked route is a jerk-limited S-curve that cruises at the top speed, stopping at each via so the arm stays on the checked straight lines. Speed, acceleration and jerk are `config.TRANSFER_MAX_VELOCITY_RAD_S`, `TRANSFER_MAX_ACCELERATION_RAD_S2` and `TRANSFER_MAX_JERK_RAD_S3` (45 °/s, 2 rad/s², 17 rad/s³). INTER → DROP is 4.2 s and INTER ↔ grab_NN 2.2 s each way. To go faster, raise them one step at a time on the cable; `config/robot.py` has the steps and what to check. Blending through the vias was looked at and isn't worth it: most vias are where a joint reverses, so it has to stop there anyway.

## Frames and offsets

All of these live in `config/` (`config/hardware.py` for the sensor and gripper, `config/poses.py` for INTER), with comments on how each was measured.

- **At INTER**, link7's local +Z points horizontally along −Y, straight into the bucket along its axis. Its local +X is the ToF boresight and points straight down, so the scan's J7 sweep is centred on the floor.
- **ToF sensor:** 7.75 cm along link7 +X (its boresight) and 2.8 cm along +Z. It's published as the static transform `link7 -> tof_sensor_link`, so readings stay correct as J7 rotates.
- **Gripper contact point:** 15 cm along link7 +Z. This is still to be verified on the hardware ([HARDWARE_TESTS.md](../HARDWARE_TESTS.md)).
