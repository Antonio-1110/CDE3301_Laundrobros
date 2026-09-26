#!/usr/bin/env python3

"""
The end-of-drum scan: a baked precession ("coning") sweep of the closed end.

WHY
---
The ToF beam is fixed at 90 deg to link7's axis (boresight = +X, the
strokes travel along +Z), so the strokes can only look radially at
the walls, never at the closed end - and the gripper, 15cm ahead of
the flange, stops the arm going deep enough to look past it. The old
BOTTOM detour tilted the tool up 42 deg to one recorded pose and swept
J7: the beam leans toward the end only where J7 points it downward,
and at the +/-75 deg ends of the sweep it is back to looking mostly
sideways (66-77 deg off the bucket axis). Simulated coverage of the
closed end's lower half: 57%.

HOW
---
Keep the flange at the deepest stroke position (the pivot). Tilt the
tool axis by alpha, in a DIRECTION phi around the insertion axis, and
choose the roll so the beam lies in the tilt plane on the far side:

    u        = cos(phi) * up + sin(phi) * side       (tilt direction)
    tool Z   = cos(alpha) * z0 + sin(alpha) * u
    beam (X) = sin(alpha) * z0 - cos(alpha) * u

so every beam leans sin(alpha) toward the closed end, and sweeping phi
at fixed alpha traces an arc across it; a few alphas are concentric
rings. phi = 0 tilts the tool UP and points the beam DOWN - the
bottom, which is the part that matters. alpha = phi = 0 is exactly
INTER's orientation.

Rings (DEFAULT_RINGS) are what the arm can reach without collisions
on the MoveIt fake controller (modelled bucket): alpha 10/25/40 over
phi +/-90, alpha 55 over phi 0..70 (the other side puts link3, the
elbow, into the bucket; alpha 65 is unreachable). Simulated: closed
end lower half 57% -> 82%, floor 97% -> 100%, lower wall 94% -> 95%.

REPEATABLE BY CONSTRUCTION
--------------------------
Nothing is planned at scan time. `laundry plan bake` solves the whole
sweep once, against a running MoveIt:

  - IK for every pose is seeded from the previous solution, so the
    elbow (the xArm7 is redundant) stays on one continuous branch.
    Solving poses independently jumps between branches, and the
    motion between them collides.
  - Each half-ring is solved OUTWARD from phi = 0 and returned along
    the same joint states reversed, and the climbs between rings are
    retraced on the way down, so the plan ends at exactly the state
    it started from. Solving returns separately lands on a different
    branch (same pose, different elbow) - one that hit the bucket.
  - Every state along the straight joint-space steps between poses
    is collision-checked at 1 deg spacing.
  - The result is densified, timed (minimum-jerk per out-and-back
    leg, arm/joint_path.py) and saved to a plan file.

At scan time the plan is replayed as a fixed joint trajectory, and
the arm gets on and off it with collision-checked straight joint
moves - so the commanded motion is identical every run.

A plan is only valid for the scan depth it was baked at (that sets
the pivot); the scan refuses a mismatch rather than guessing.
Re-bake after changing INTER, config.OBSTACLES, the gripper, or the depth.
"""

from dataclasses import dataclass, field
import datetime
import os

from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .. import config
from ..arm.joint_path import densify, time_path

PLAN_VERSION = 1

# (alpha_deg, phi_min_deg, phi_max_deg), in the order they are swept.
DEFAULT_RINGS = (
    (10.0, -90.0, 90.0),
    (25.0, -90.0, 90.0),
    (40.0, -90.0, 90.0),
    (55.0, 0.0, 70.0),
)

# phi step along a ring, and alpha step between rings, when baking.
# 5 deg keeps consecutive IK solutions within a few degrees of each
# other, so a branch jump is detectable (MAX_JOINT_STEP_DEG).
PHI_STEP_DEG = 5.0
ALPHA_STEP_DEG = 2.5
MAX_JOINT_STEP_DEG = 20.0


def default_plan_path():
    """Return <repo>/scan_plans/endcap.yaml."""
    return os.path.join(config.repo_root(), 'scan_plans', 'endcap.yaml')


def reference_axes(tool_z):
    """Return (z0, up, side): the insertion axis and its perpendicular basis."""
    z0 = np.asarray(tool_z, dtype=np.float64)
    z0 = z0 / np.linalg.norm(z0)

    up = np.array([0.0, 0.0, 1.0])
    up = up - (up @ z0) * z0
    up = up / np.linalg.norm(up)

    return z0, up, np.cross(z0, up)


def precession_axes(alpha_deg, phi_deg, z0, up, side):
    """Return (tool_z, boresight) for a tilt alpha in direction phi."""
    alpha = np.deg2rad(alpha_deg)
    phi = np.deg2rad(phi_deg)

    u = np.cos(phi) * up + np.sin(phi) * side

    tool_z = np.cos(alpha) * z0 + np.sin(alpha) * u
    boresight = np.sin(alpha) * z0 - np.cos(alpha) * u

    return tool_z, boresight


def precession_pose(pivot, alpha_deg, phi_deg, z0, up, side):
    """Return the flange Pose (base_frame) for a precession sample."""
    tool_z, boresight = precession_axes(alpha_deg, phi_deg, z0, up, side)

    matrix = np.stack([boresight, np.cross(tool_z, boresight), tool_z], axis=1)
    qx, qy, qz, qw = Rotation.from_matrix(matrix).as_quat()

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, pivot)
    pose.orientation.x = float(qx)
    pose.orientation.y = float(qy)
    pose.orientation.z = float(qz)
    pose.orientation.w = float(qw)

    return pose


def sample_beams(pivot, z0, up, side, alpha_phi):
    """Return (origins, directions) of the ToF beam at (alpha, phi) samples."""
    origins = []
    directions = []

    for alpha, phi in alpha_phi:
        tool_z, boresight = precession_axes(alpha, phi, z0, up, side)
        origins.append(
            np.asarray(pivot)
            + config.TOF_SENSOR_OFFSET_Z * tool_z
            + config.TOF_SENSOR_OFFSET_X * boresight
        )
        directions.append(boresight)

    return np.array(origins), np.array(directions)


@dataclass
class EndcapPlan:
    """A baked end scan: a fixed joint trajectory plus its provenance."""

    depth_m: float
    pivot: np.ndarray
    tool_z: np.ndarray
    waypoints: np.ndarray
    times: np.ndarray
    velocities: np.ndarray
    alpha_phi: np.ndarray
    rings: list
    max_velocity_rad_s: float
    joint_names: list = field(default_factory=lambda: list(config.JOINT_NAMES))
    created: str = ''
    baked_on: str = ''
    # Arm-link padding the plan was solved with (see
    # config.ENDCAP_PADDING_M); run_plan() replays it under the same.
    # Plans from before padding existed were solved against the bare
    # URDF: 0.
    padding_m: float = 0.0
    # The obstacle geometry the plan was checked against
    # (arm/scene.signature); run_plan() warns when it has changed.
    scene: str = ''
    obstacles: dict = field(default_factory=dict)

    @property
    def duration_s(self):
        """Return the replay duration at full speed."""
        return float(self.times[-1])

    @property
    def start(self):
        """Return the joint state the plan starts (and ends) at."""
        return list(self.waypoints[0])

    def beams(self):
        """Return (origins, directions) of the beam along the plan."""
        z0, up, side = reference_axes(self.tool_z)
        return sample_beams(self.pivot, z0, up, side, self.alpha_phi)

    def save(self, path):
        """Write the plan as YAML."""
        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        document = {
            'version': PLAN_VERSION,
            'created': self.created,
            'baked_on': self.baked_on,
            'depth_m': float(self.depth_m),
            'padding_m': float(self.padding_m),
            'scene': self.scene,
            'obstacles': self.obstacles,
            'pivot': [float(v) for v in self.pivot],
            'tool_z': [float(v) for v in self.tool_z],
            'max_velocity_rad_s': float(self.max_velocity_rad_s),
            'rings': [
                {
                    'alpha_deg': float(a),
                    'phi_min_deg': float(lo),
                    'phi_max_deg': float(hi),
                }
                for a, lo, hi in self.rings
            ],
            'joint_names': list(self.joint_names),
            'duration_s': self.duration_s,
            'points': [
                {
                    't': round(float(t), 6),
                    'q': [round(float(v), 6) for v in q],
                    'qd': [round(float(v), 6) for v in qd],
                    'alpha_phi': [round(float(v), 3) for v in ap],
                }
                for t, q, qd, ap in zip(
                    self.times, self.waypoints, self.velocities, self.alpha_phi
                )
            ],
        }

        with open(path, 'w') as handle:
            handle.write(
                '# Baked end-of-drum scan (laundry_control/scan/endcap.py).\n'
                '# Generated by `laundry plan bake` - do not edit by hand;\n'
                '# re-bake after changing INTER, config.OBSTACLES, or '
                '--depth.\n'
            )
            yaml.safe_dump(document, handle, sort_keys=False, width=100)

    @classmethod
    def load(cls, path):
        """Read a plan written by save()."""
        with open(path) as handle:
            document = yaml.safe_load(handle)

        if document.get('version') != PLAN_VERSION:
            raise ValueError(
                f'{path!r} is plan version {document.get("version")!r}; '
                f'this code reads version {PLAN_VERSION}. Re-bake it.'
            )

        if list(document['joint_names']) != list(config.JOINT_NAMES):
            raise ValueError(f'{path!r} was baked for different joints.')

        points = document['points']

        return cls(
            depth_m=float(document['depth_m']),
            pivot=np.array(document['pivot']),
            tool_z=np.array(document['tool_z']),
            waypoints=np.array([p['q'] for p in points]),
            times=np.array([p['t'] for p in points]),
            velocities=np.array([p['qd'] for p in points]),
            alpha_phi=np.array([p['alpha_phi'] for p in points]),
            rings=[
                (r['alpha_deg'], r['phi_min_deg'], r['phi_max_deg'])
                for r in document['rings']
            ],
            max_velocity_rad_s=float(document['max_velocity_rad_s']),
            joint_names=list(document['joint_names']),
            created=document.get('created', ''),
            baked_on=document.get('baked_on', ''),
            padding_m=float(document.get('padding_m', 0.0)),
            scene=str(document.get('scene', '')),
            obstacles=document.get('obstacles', {}),
        )


def _steps(start, stop, step):
    """Return values from start (exclusive) to stop (inclusive) by |step|."""
    if np.isclose(start, stop):
        return []

    step = abs(step) if stop > start else -abs(step)
    values = list(np.arange(start + step, stop, step))

    return values + [stop]


class BakeError(RuntimeError):
    """The requested sweep has no continuous collision-free solution."""


def bake(
    arm,
    depth_m,
    rings=DEFAULT_RINGS,
    max_velocity_rad_s=config.LINEAR_JOINT_MOVE_MAX_VELOCITY_RAD_S,
    padding_m=config.ENDCAP_PADDING_M,
    log=print,
):
    """
    Bake the end scan under its own arm-link padding (see _bake).

    padding_m (default config.ENDCAP_PADDING_M) is recorded in the plan,
    and the usual config.OBSTACLE_PADDING_M is restored afterwards.
    """
    arm.set_arm_padding(padding_m)

    try:
        plan = _bake(arm, depth_m, rings, max_velocity_rad_s, log)
    finally:
        arm.set_arm_padding(config.OBSTACLE_PADDING_M)

    from ..arm import scene

    plan.padding_m = float(padding_m)
    plan.scene = scene.signature()
    plan.obstacles = scene.describe()

    return plan


def _bake(arm, depth_m, rings, max_velocity_rad_s, log):
    """
    Solve, check and time the end scan against a running MoveIt.

    MOVES THE ARM: to INTER, then straight in by depth_m (to read the
    real pivot pose and seed the IK), and back out to INTER at the
    end. Run it on the fake controller, or on the rig with the bucket
    in place and an eye on the e-stop.

    A spoke that becomes infeasible part-way is truncated at its last
    good pose (and logged) rather than failing the bake, since rings
    near the reach limit are expected to be partial; the plan records
    the phi range each ring actually reached. A ring whose very first
    step fails raises BakeError.
    """
    log('Baking the end scan: moving to INTER and in to the pivot...')

    from ..arm.transfers import go_to

    if not go_to(arm, 'inter'):
        raise BakeError('Could not reach INTER.')

    if not arm.move_tool_z(depth_m):
        raise BakeError(f'Could not insert {depth_m:.3f} m.')

    tf = arm.get_flange_transform()
    t = tf.transform.translation
    q = tf.transform.rotation

    pivot = np.array([t.x, t.y, t.z])
    tool_z = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()[:, 2]

    z0, up, side = reference_axes(tool_z)

    def solve(seed, alpha, phi):
        pose = precession_pose(pivot, alpha, phi, z0, up, side)
        solution = arm.compute_ik(pose, seed)

        if solution is None:
            return None

        solution = np.array(solution)

        if np.degrees(np.abs(solution - seed).max()) >= MAX_JOINT_STEP_DEG:
            return None

        if arm.first_invalid_state([seed, solution]) is not None:
            return None

        return solution

    def chain(seed, targets):
        out = []

        for alpha, phi in targets:
            solution = solve(seed, alpha, phi)

            if solution is None:
                break

            out.append((solution, (alpha, phi)))
            seed = solution

        return out

    current = np.array(arm.get_current_joints())

    hub = solve(current, 0.0, 0.0)

    if hub is None:
        raise BakeError('No collision-free IK for the pivot pose itself.')

    path = [(hub, (0.0, 0.0))]
    reached_rings = []
    alpha_prev = 0.0

    # Ring centres visited on the way up, retraced on the way down, so
    # the plan ends at exactly the joint state it started from.
    ascent = [path[0]]

    for alpha, phi_min, phi_max in rings:
        climb = chain(
            path[-1][0],
            [(a, 0.0) for a in _steps(alpha_prev, alpha, ALPHA_STEP_DEG)],
        )

        if len(climb) < len(_steps(alpha_prev, alpha, ALPHA_STEP_DEG)):
            log(f'  ring alpha={alpha:g}: cannot tilt that far at phi=0; stopping.')
            break

        path += climb
        ascent += climb
        centre = path[-1]
        reached = [0.0, 0.0]

        for index, edge in enumerate((phi_min, phi_max)):
            targets = [(alpha, phi) for phi in _steps(0.0, edge, PHI_STEP_DEG)]

            if not targets:
                continue

            spoke = chain(centre[0], targets)

            if len(spoke) < len(targets):
                log(
                    f'  ring alpha={alpha:g}: phi toward {edge:+g} stops at '
                    f'{spoke[-1][1][1] if spoke else 0.0:+g} (collision/IK).'
                )

            if spoke:
                reached[index] = spoke[-1][1][1]
                # Out along the spoke, back along the same states.
                path += spoke + spoke[::-1][1:] + [centre]

        reached_rings.append((alpha, reached[0], reached[1]))
        alpha_prev = alpha

    # Back down along the same states the climbs went up through.
    path += ascent[::-1][1:]

    joints = np.array([solution for solution, _ in path])
    angles = np.array([ap for _, ap in path])

    # Densify so the controller's interpolation between points can
    # never stray from the collision-checked straight segments; carry
    # (alpha, phi) along for the coverage model.
    dense = densify(joints)
    arc = np.concatenate(
        [[0.0], np.cumsum(np.abs(np.diff(joints, axis=0)).max(axis=1))]
    )
    dense_arc = np.concatenate(
        [[0.0], np.cumsum(np.abs(np.diff(dense, axis=0)).max(axis=1))]
    )
    dense_angles = np.stack(
        [np.interp(dense_arc, arc, angles[:, k]) for k in range(2)], axis=1
    )

    times, velocities = time_path(dense, max_velocity_rad_s)

    log(
        f'  {len(path)} poses, {dense.shape[0]} trajectory points, '
        f'{times[-1]:.1f} s at {np.degrees(max_velocity_rad_s):.0f} deg/s peak.'
    )

    log('Returning to INTER...')
    arm.move_tool_z(-depth_m)
    go_to(arm, 'inter')

    return EndcapPlan(
        depth_m=float(depth_m),
        pivot=pivot,
        tool_z=tool_z,
        waypoints=dense,
        times=times,
        velocities=velocities,
        alpha_phi=dense_angles,
        rings=reached_rings,
        max_velocity_rad_s=float(max_velocity_rad_s),
        created=datetime.datetime.now().isoformat(timespec='seconds'),
    )


def run_plan(arm, plan, depth_m, time_scale=1.0):
    """
    Replay a baked end scan from wherever the strokes left the arm.

    Refuses a plan baked for a different depth, or one that collides
    with the current planning scene. Runs under the arm-link padding
    the plan was baked with (plan.padding_m), then restores the usual
    config.OBSTACLE_PADDING_M. Gets onto the plan's start with a
    collision-checked straight joint move, replays the
    fixed trajectory, and leaves the arm at the plan's start (the
    pivot, INTER's orientation) - the caller turns J7 for the outward
    pass from there.
    """
    if abs(plan.depth_m - depth_m) > 1e-6:
        raise ValueError(
            f'End-scan plan was baked for depth {plan.depth_m:.3f} m but '
            f'the scan uses {depth_m:.3f} m. Re-bake: '
            f'laundry plan bake --depth {depth_m:.3f}'
        )

    from ..arm import scene

    stale = scene.stale_plan_message(
        plan.scene, 'The end scan (scan_plans/endcap.yaml)',
        f'laundry plan bake endcap --depth {plan.depth_m:.3f}',
    )

    if stale:
        arm.get_logger().warning(stale)

    if abs(plan.padding_m - config.ENDCAP_PADDING_M) > 1e-9:
        arm.get_logger().warning(
            f'The end scan was baked with {plan.padding_m * 100:g} cm arm '
            f'padding; config.ENDCAP_PADDING_M is now '
            f'{config.ENDCAP_PADDING_M * 100:g} cm. It is checked under the '
            f'new value. Re-bake: laundry plan bake endcap --depth '
            f'{plan.depth_m:.3f}'
        )

    # The plan was collision-checked when it was baked - possibly on
    # another machine, against an older bucket pose. Re-check
    # every state against the planning scene loaded NOW, under the
    # padding it was baked with, before the arm moves at all.
    arm.set_arm_padding(config.ENDCAP_PADDING_M)

    try:
        bad = arm.first_invalid_state(plan.waypoints)

        if bad is not None:
            arm.get_logger().error(
                f'The baked end scan collides with the current planning '
                f'scene (at checked state {bad}); the obstacles or INTER '
                f'changed since it was baked. Not moving. Re-bake: laundry plan '
                f'bake endcap --depth {plan.depth_m:.3f}'
            )
            return False

        if not arm.move_joints_linear(plan.start, time_scale=time_scale):
            arm.get_logger().error(
                'Could not reach the end-scan start state.'
            )
            return False

        return arm.execute_joint_path(
            plan.waypoints, plan.times, plan.velocities, time_scale=time_scale
        )

    finally:
        arm.set_arm_padding(config.OBSTACLE_PADDING_M)
