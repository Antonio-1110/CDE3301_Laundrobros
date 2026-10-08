#!/usr/bin/env python3

"""
Baked transfers that pass through their vias instead of stopping.

time_stop_at_each() brings the arm to rest at every via of a baked
route, so it follows exactly the straight lines that were checked.
Here the route is timed by MoveIt's time-optimal trajectory generation
(TOTG) instead: it rounds each corner by at most a path tolerance and
moves as fast as each joint's speed and acceleration limits allow, so
joints that keep going through a via keep moving while one that
reverses there slows down and turns round.

The timing is computed ONCE, at bake time (`laundry plan bake
timing`), and stored with the route in transfers.yaml; a replay sends
exactly what was checked. moveit_py is needed only for baking.

WHAT IS CHECKED
---------------
The trajectory controller draws a cubic through each point's position
and velocity. TOTG's acceleration switches on and off abruptly, and
that cubic overshoots TOTG's limits at every switch, so the limits are
checked on the cubic itself (controller_curve), not on TOTG's points:
speed, acceleration and jerk must all stay within the transfer limits
(config.TRANSFER_MAX_*). TOTG is run with lower limits until they do,
and the fastest passing setting is kept. Points are 0.2 s apart: the
cubic's jerk grows as the spacing shrinks (about 30 rad/s^3 at 0.1 s).

The rounded corners leave the baked straight lines (by ~1.5 deg), so
the cubic itself is collision-checked: when baking, with the route's
own gripper clearance; and again before every replay, against the
scene as it is then. A route whose timing fails either check falls
back to stopping at its vias.
"""

import numpy as np

# Spacing of the timed points, seconds (see WHAT IS CHECKED).
SAMPLE_DT = 0.2

# TOTG settings tried, fastest first: corner tolerance (rad), then the
# fraction of each transfer limit TOTG is given.
PATH_TOLERANCES = (0.05, 0.02)
ACCELERATION_FRACTIONS = (0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)
VELOCITY_FRACTIONS = (1.0, 0.987, 0.975)

# Collision checks need states at most this far apart.
CHECK_STEP_RAD = np.deg2rad(1.0)

# Relative slack on the limits, for the cubic's sampling.
_SLACK = 1e-3


def merge_short_tail(times, positions, velocities, sample_dt=SAMPLE_DT):
    """
    Fold a final interval shorter than half a step into the one before.

    TOTG samples every sample_dt, so the end of the path leaves a
    sliver of an interval - a few ms - and the controller's cubic over
    it has a jerk in the hundreds of rad/s^3.
    """
    if len(times) > 2 and times[-1] - times[-2] < 0.5 * sample_dt:
        keep = np.r_[np.arange(len(times) - 2), len(times) - 1]
        return times[keep], positions[keep], velocities[keep]

    return times, positions, velocities


def at_rest_at_both_ends(velocities):
    """
    Return velocities with the first and last exactly zero.

    TOTG leaves a residue (~1e-3 rad/s) there, and the trajectory
    controller rejects a trajectory whose last point is not at rest.
    """
    velocities = np.array(velocities, dtype=np.float64)
    velocities[0] = 0.0
    velocities[-1] = 0.0
    return velocities


def _cubics(times, positions, velocities):
    """Yield (t0, h, p0, v0, c2, c3) of the controller's cubic per interval."""
    for k in range(len(times) - 1):
        h = times[k + 1] - times[k]
        p0, p1 = positions[k], positions[k + 1]
        v0, v1 = velocities[k], velocities[k + 1]
        c2 = (3.0 * (p1 - p0) - h * (2.0 * v0 + v1)) / h ** 2
        c3 = (2.0 * (p0 - p1) + h * (v0 + v1)) / h ** 3
        yield times[k], h, p0, v0, c2, c3


def curve_peaks(times, positions, velocities, step=0.002):
    """Return the controller cubic's peak |speed|, |acceleration|, |jerk|."""
    v_peak = a_peak = j_peak = 0.0

    for _t0, h, _p0, v0, c2, c3 in _cubics(times, positions, velocities):
        s = np.append(np.arange(0.0, h, step), h)[:, None]
        v_peak = max(v_peak, np.abs(v0 + 2 * c2 * s + 3 * c3 * s ** 2).max())
        # Acceleration is linear in s: its peak is at an end.
        a_peak = max(a_peak, np.abs(2 * c2).max(), np.abs(2 * c2 + 6 * c3 * h).max())
        j_peak = max(j_peak, np.abs(6 * c3).max())

    return float(v_peak), float(a_peak), float(j_peak)


def within_limits(peaks, v_max, a_max, j_max):
    """Return True if (speed, acceleration, jerk) peaks keep all three limits."""
    slack = 1.0 + _SLACK
    v, a, j = peaks

    return v <= v_max * slack and a <= a_max * slack and j <= j_max * slack


def dense_states(times, positions, velocities, max_step_rad=CHECK_STEP_RAD):
    """
    Return states along the controller's cubic, at most max_step_rad apart.

    For collision checking the path the arm will actually follow.
    """
    states = [np.asarray(positions[0], dtype=np.float64)]

    for _t0, h, p0, v0, c2, c3 in _cubics(times, positions, velocities):
        # Sample finely, then keep a state every max_step_rad of travel.
        s = np.linspace(0.0, h, 50)[1:, None]
        for q in p0 + v0 * s + c2 * s ** 2 + c3 * s ** 3:
            if np.abs(q - states[-1]).max() >= max_step_rad:
                states.append(q)

    if np.abs(np.asarray(positions[-1]) - states[-1]).max() > 1e-9:
        states.append(np.asarray(positions[-1], dtype=np.float64))

    return np.array(states)


def reverse(times, positions, velocities):
    """Return the same motion run backwards (end to start)."""
    times = np.asarray(times, dtype=np.float64)

    return (
        times[-1] - times[::-1],
        np.asarray(positions)[::-1],
        -np.asarray(velocities)[::-1],
    )


def to_document(times, positions, velocities, limits, settings):
    """Return a timed route as plain lists, for transfers.yaml."""
    def rows(values):
        return [[round(float(x), 6) for x in row] for row in values]

    return {
        # The transfer limits it was made for; replays of other limits
        # stop at the vias instead.
        'limits': {
            key: round(float(value), 6) for key, value in limits.items()
        },
        'totg': settings,
        'times': [round(float(t), 6) for t in times],
        'positions': rows(positions),
        'velocities': rows(velocities),
    }


def from_document(document):
    """Return (times, positions, velocities) arrays from to_document()."""
    return (
        np.asarray(document['times'], dtype=np.float64),
        np.asarray(document['positions'], dtype=np.float64),
        at_rest_at_both_ends(document['velocities']),
    )


def current_limits():
    """Return the transfer limits config sets now, as to_document stores them."""
    from .. import config

    return {
        'velocity_rad_s': round(float(config.TRANSFER_MAX_VELOCITY_RAD_S), 6),
        'acceleration_rad_s2': round(
            float(config.TRANSFER_MAX_ACCELERATION_RAD_S2), 6
        ),
        'jerk_rad_s3': round(float(config.TRANSFER_MAX_JERK_RAD_S3), 6),
    }


class Retimer:
    """
    TOTG through moveit_py, for baking only.

    The robot model gets the transfer acceleration limit (moveit_py's
    plain RobotModel has none, and its bounds are read-only), which is
    why this starts a MoveItPy instance (~25 s, ~430 MB) - run it on
    the fake controller's domain, never in a replay.
    """

    def __init__(self, v_max, a_max, j_max, node_name='laundry_retimer'):
        from launch import LaunchContext
        from moveit.planning import MoveItPy
        from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder

        from .. import config

        self.v_max, self.a_max, self.j_max = v_max, a_max, j_max

        params = MoveItConfigsBuilder(
            context=LaunchContext(), **config.xarm_description_arguments()
        ).to_moveit_configs().to_dict()

        # Every joint gets the transfer limits, so TOTG's scaling
        # factors below are fractions of them.
        for limit in params['robot_description_planning']['joint_limits'].values():
            limit.update(
                has_velocity_limits=True, max_velocity=v_max,
                has_acceleration_limits=True, max_acceleration=a_max,
            )

        params['planning_pipelines'] = {'pipeline_names': ['ompl']}
        params['planning_scene_monitor_options'] = {
            'name': 'planning_scene_monitor',
            'robot_description': 'robot_description',
            'joint_state_topic': '/joint_states',
            'attached_collision_object_topic': '/attached_collision_object',
            'publish_planning_scene_topic': '/publish_planning_scene',
            'monitored_planning_scene_topic': '/monitored_planning_scene',
            'wait_for_initial_state_timeout': 0.0,
        }

        self._moveit = MoveItPy(node_name=node_name, config_dict=params)
        self._model = self._moveit.get_robot_model()
        self._joint_names = list(config.JOINT_NAMES)

    def close(self):
        self._moveit.shutdown()

    def _totg(self, waypoints, v, a, tolerance):
        from moveit.core.robot_state import RobotState
        from moveit.core.robot_trajectory import RobotTrajectory
        from moveit_msgs.msg import RobotTrajectory as TrajectoryMsg
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        trajectory = RobotTrajectory(self._model)
        trajectory.joint_model_group_name = 'xarm7'

        message = TrajectoryMsg()
        message.joint_trajectory = JointTrajectory(
            joint_names=self._joint_names
        )

        for joints in waypoints:
            message.joint_trajectory.points.append(
                JointTrajectoryPoint(positions=[float(q) for q in joints])
            )

        trajectory.set_robot_trajectory_msg(RobotState(self._model), message)

        if not trajectory.apply_totg_time_parameterization(
            v / self.v_max, a / self.a_max,
            path_tolerance=tolerance, resample_dt=SAMPLE_DT,
        ):
            return None

        points = trajectory.get_robot_trajectory_msg().joint_trajectory.points

        if len(points) < 2:
            return None

        times = np.array([
            p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
            for p in points
        ])

        times, positions, velocities = merge_short_tail(
            times,
            np.array([p.positions for p in points]),
            np.array([p.velocities for p in points]),
        )

        return times, positions, at_rest_at_both_ends(velocities)

    def time_route(self, waypoints):
        """
        Return the fastest timing within limits as (t, q, qd, settings), or None.

        Tries PATH_TOLERANCES x ACCELERATION_FRACTIONS x
        VELOCITY_FRACTIONS and keeps the fastest whose controller cubic
        keeps all three limits. Not collision-checked: the caller does.
        """
        best = None

        for tolerance in PATH_TOLERANCES:
            for a_fraction in ACCELERATION_FRACTIONS:
                for v_fraction in VELOCITY_FRACTIONS:
                    timed = self._totg(
                        waypoints,
                        self.v_max * v_fraction,
                        self.a_max * a_fraction,
                        tolerance,
                    )

                    if timed is None:
                        continue

                    peaks = curve_peaks(*timed)

                    if not within_limits(
                        peaks, self.v_max, self.a_max, self.j_max
                    ):
                        continue

                    if best is None or timed[0][-1] < best[0][-1]:
                        best = timed + ({
                            'path_tolerance_rad': tolerance,
                            'acceleration_fraction': a_fraction,
                            'velocity_fraction': v_fraction,
                            'peaks': [round(x, 4) for x in peaks],
                        },)

        return best


def time_routes(routes, v_max, a_max, j_max, timeout_sec=900):
    """
    Return {route: (times, positions, velocities, settings) or None}.

    Runs the Retimer in a child process (main() below): MoveItPy then
    never shares a process with the caller's rclpy session, and its
    ~430 MB are freed before the caller's collision checks start.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory(prefix='laundry_timing_') as folder:
        request = os.path.join(folder, 'routes.json')
        answer = os.path.join(folder, 'timings.json')

        with open(request, 'w') as handle:
            json.dump({
                'limits': [v_max, a_max, j_max],
                'routes': {
                    name: np.asarray(waypoints).tolist()
                    for name, waypoints in routes.items()
                },
            }, handle)

        subprocess.run(
            [sys.executable, '-m', 'laundry_control.arm.pass_through',
             request, answer],
            timeout=timeout_sec, check=False,
        )

        if not os.path.isfile(answer):
            raise RuntimeError('Timing transfers with TOTG failed (no result).')

        with open(answer) as handle:
            result = json.load(handle)

    return {
        name: None if entry is None else (
            np.asarray(entry['times']),
            np.asarray(entry['positions']),
            np.asarray(entry['velocities']),
            entry['settings'],
        )
        for name, entry in result.items()
    }


def main(argv=None):
    """Child process of time_routes(): REQUEST.json -> ANSWER.json."""
    import json
    import os
    import sys

    request, answer = (argv or sys.argv)[1:3]

    with open(request) as handle:
        job = json.load(handle)

    retimer = Retimer(*job['limits'])
    result = {}

    for name, waypoints in job['routes'].items():
        timed = retimer.time_route(np.asarray(waypoints))
        result[name] = None if timed is None else {
            'times': timed[0].tolist(),
            'positions': timed[1].tolist(),
            'velocities': timed[2].tolist(),
            'settings': timed[3],
        }

    with open(answer + '.part', 'w') as handle:
        json.dump(result, handle)

    os.replace(answer + '.part', answer)

    # MoveItPy can crash while tearing down; the answer is written, so
    # skip the teardown entirely.
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
