from dataclasses import dataclass

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from laundry_control.grasp_plan import (
    DEFAULT_MAX_SINK_M,
    compute_grasp_target,
    estimate_baseline_depth,
    look_at_quaternion,
)
from laundry_control.gripper import GRIPPER_OFFSET_Z
from laundry_control.laundry_detect import ClusterSummary


def _make_grid(n_side=10, spacing=0.05, z=0.0):
    xs = np.arange(n_side) * spacing
    ys = np.arange(n_side) * spacing
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, z)
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def _make_cluster(x, y, z):
    points = np.array([[x, y, z]])
    return ClusterSummary(
        points=points,
        centroid=np.array([x, y, z]),
        size=1,
        bbox_min=points[0],
        bbox_max=points[0],
        extent=np.zeros(3),
        highest_point=points[0],
        mean_deviation_m=0.05,
    )


@dataclass
class _Vec3:
    x: float
    y: float
    z: float


@dataclass
class _Quat:
    x: float
    y: float
    z: float
    w: float


@dataclass
class _Transform:
    translation: _Vec3
    rotation: _Quat


@dataclass
class _TransformStamped:
    transform: _Transform


class _StubArm:
    """
    current_position/current_quat_xyzw describe where get_flange_
    transform() reports the arm as currently being. reachable_at_
    or_above simulates an obstruction: only reachable once tcp_z
    has risen to or above that threshold.
    """

    def __init__(
        self,
        reachable_at_or_above,
        current_position=(0.0, 0.0, 0.5),
        current_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
    ):
        self.reachable_at_or_above = reachable_at_or_above
        self.current_position = current_position
        self.current_quat_xyzw = current_quat_xyzw
        self.probed = []

    def get_flange_transform(self):
        x, y, z = self.current_position
        qx, qy, qz, qw = self.current_quat_xyzw
        return _TransformStamped(
            transform=_Transform(
                translation=_Vec3(x, y, z),
                rotation=_Quat(qx, qy, qz, qw),
            )
        )

    def check_pose_reachable(self, x, y, z, **kwargs):
        self.probed.append((x, y, z, kwargs.get("orientation")))
        return 1.0 if z >= self.reachable_at_or_above else 0.3


# ---------------------------------------------------------------
# estimate_baseline_depth
# ---------------------------------------------------------------

def test_estimate_baseline_depth_flat_grid():
    baseline = _make_grid(z=0.0)

    depth = estimate_baseline_depth(baseline, (0.1, 0.1))

    assert depth == pytest.approx(0.0, abs=1e-9)


def test_estimate_baseline_depth_picks_local_region():
    left = _make_grid(n_side=5, spacing=0.02, z=0.0)
    right = _make_grid(n_side=5, spacing=0.02, z=-0.05)
    right[:, 0] += 1.0  # shift the right region far away in x

    baseline = np.concatenate([left, right], axis=0)

    depth_left = estimate_baseline_depth(baseline, (0.02, 0.02), k=3)
    depth_right = estimate_baseline_depth(baseline, (1.02, 0.02), k=3)

    assert depth_left == pytest.approx(0.0, abs=1e-9)
    assert depth_right == pytest.approx(-0.05, abs=1e-9)


def test_estimate_baseline_depth_empty_baseline_raises():
    with pytest.raises(ValueError):
        estimate_baseline_depth(np.empty((0, 3)), (0.0, 0.0))


def test_estimate_baseline_depth_ignores_wall_points_above_the_item():
    """
    The multivalued-z case this function exists to survive: a floor
    and a wall standing over it at nearly the same (x, y). Averaging
    both - what this used to do - returns a height describing
    neither surface.
    """

    floor = np.array([[0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.0, 0.004, 0.0]])
    wall = np.array([[0.002, 0.002, 0.10], [0.006, 0.002, 0.10]])

    baseline = np.concatenate([floor, wall], axis=0)

    # Item sensed at 5cm, i.e. above the floor but below the wall.
    depth = estimate_baseline_depth(baseline, (0.002, 0.002), k=5, below_z=0.05)

    assert depth == pytest.approx(0.0, abs=1e-9)

    # Averaging all five would have landed between the two surfaces.
    assert depth != pytest.approx(baseline[:, 2].mean(), abs=1e-9)


def test_estimate_baseline_depth_returns_highest_surface_below():
    # A ledge partway up is what the gripper meets first on the way
    # down, so it - not the floor beneath it - bounds the sink.
    baseline = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.004, 0.0, 0.03],
            [0.0, 0.004, 0.01],
        ]
    )

    depth = estimate_baseline_depth(baseline, (0.0, 0.0), k=3, below_z=0.05)

    assert depth == pytest.approx(0.03, abs=1e-9)


def test_estimate_baseline_depth_none_when_nothing_below():
    baseline = np.array(
        [
            [0.0, 0.0, 0.20],
            [0.004, 0.0, 0.22],
            [0.0, 0.004, 0.25],
        ]
    )

    depth = estimate_baseline_depth(baseline, (0.0, 0.0), k=3, below_z=0.05)

    assert depth is None


# ---------------------------------------------------------------
# look_at_quaternion
# ---------------------------------------------------------------

def test_look_at_quaternion_local_z_points_along_direction():
    direction = np.array([1.0, 2.0, -3.0])
    direction = direction / np.linalg.norm(direction)

    q = look_at_quaternion(direction, reference_x_axis=[1.0, 0.0, 0.0])

    rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
    resulting_z = rotation.apply([0.0, 0.0, 1.0])

    assert resulting_z == pytest.approx(direction, abs=1e-9)


def test_look_at_quaternion_straight_down_matches_identity_like_case():
    # direction = -Z, reference_x = +X -> local X should stay +X
    # (no roll introduced), same as an unrotated frame's -Z look.
    q = look_at_quaternion([0.0, 0.0, -1.0], reference_x_axis=[1.0, 0.0, 0.0])

    rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
    resulting_x = rotation.apply([1.0, 0.0, 0.0])

    assert resulting_x == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_look_at_quaternion_handles_parallel_reference_x():
    # reference_x_axis parallel to direction -- must not raise/NaN.
    q = look_at_quaternion([0.0, 0.0, 1.0], reference_x_axis=[0.0, 0.0, 5.0])

    rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
    resulting_z = rotation.apply([0.0, 0.0, 1.0])

    assert resulting_z == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    assert np.all(np.isfinite([q.x, q.y, q.z, q.w]))


# ---------------------------------------------------------------
# compute_grasp_target
# ---------------------------------------------------------------

def test_compute_grasp_target_reachable_at_first_fraction():
    baseline = _make_grid(z=-0.10)
    cluster = _make_cluster(0.1, 0.1, 0.0)

    # Arm sits directly above the cluster's (x, y) -> approach
    # direction is purely -Z, reproducing the old pure-Z-offset
    # behaviour as a special case.
    arm = _StubArm(
        reachable_at_or_above=-1.0,
        current_position=(0.1, 0.1, 0.5),
    )

    result = compute_grasp_target(cluster, baseline, arm)

    assert result is not None
    assert result.sink_fraction_used == 0.5
    assert len(arm.probed) == 1
    assert result.sink_amount_m == pytest.approx(0.05)
    assert result.grasp_point[2] == pytest.approx(-0.05)
    assert result.tcp_position[2] == pytest.approx(-0.05 + GRIPPER_OFFSET_Z)
    assert result.tcp_position[:2] == pytest.approx(result.grasp_point[:2])


def test_compute_grasp_target_falls_through_to_floor():
    baseline = _make_grid(z=-0.10)
    cluster = _make_cluster(0.1, 0.1, 0.0)

    arm = _StubArm(
        reachable_at_or_above=GRIPPER_OFFSET_Z - 1e-9,
        current_position=(0.1, 0.1, 0.5),
    )

    result = compute_grasp_target(cluster, baseline, arm)

    assert result is not None
    assert result.sink_fraction_used == 0.0
    assert result.sink_amount_m == pytest.approx(0.0)
    assert result.grasp_point[2] == pytest.approx(0.0)


def test_compute_grasp_target_never_reachable_returns_none():
    baseline = _make_grid(z=-0.10)
    cluster = _make_cluster(0.1, 0.1, 0.0)

    arm = _StubArm(
        reachable_at_or_above=1000.0,
        current_position=(0.1, 0.1, 0.5),
    )

    result = compute_grasp_target(cluster, baseline, arm)

    assert result is None


def test_compute_grasp_target_gap_clamped_when_baseline_above_cluster():
    baseline = _make_grid(z=0.10)
    cluster = _make_cluster(0.1, 0.1, 0.0)

    arm = _StubArm(
        reachable_at_or_above=-1.0,
        current_position=(0.1, 0.1, 0.5),
    )

    result = compute_grasp_target(cluster, baseline, arm)

    assert result is not None
    assert result.gap_m == pytest.approx(0.0)
    assert result.sink_amount_m == pytest.approx(0.0)


def test_compute_grasp_target_max_sink_cap():
    baseline = _make_grid(z=-1.0)
    cluster = _make_cluster(0.1, 0.1, 0.0)

    arm = _StubArm(
        reachable_at_or_above=-1.0,
        current_position=(0.1, 0.1, 0.5),
    )

    result = compute_grasp_target(cluster, baseline, arm)

    assert result is not None
    assert result.sink_fraction_used == 0.5
    assert result.sink_amount_m == pytest.approx(DEFAULT_MAX_SINK_M)


def test_compute_grasp_target_offset_arithmetic_directly_above():
    baseline = _make_grid(z=-0.10)
    cluster = _make_cluster(0.2, -0.3, 0.05)

    arm = _StubArm(
        reachable_at_or_above=-1.0,
        current_position=(0.2, -0.3, 0.5),
    )

    result = compute_grasp_target(cluster, baseline, arm)

    assert result is not None
    assert result.tcp_position[2] - result.grasp_point[2] == pytest.approx(
        GRIPPER_OFFSET_Z
    )
    assert result.tcp_position[:2] == pytest.approx(result.grasp_point[:2])
    assert result.tcp_position[:2] == pytest.approx([0.2, -0.3])


def test_compute_grasp_target_tilts_when_arm_offset_laterally():
    # Arm is NOT above the cluster -- offset well to the side. The
    # reach from tcp_position to grasp_point must still be exactly
    # gripper_offset_z long, but no longer purely vertical.
    baseline = _make_grid(z=-0.10)
    cluster = _make_cluster(0.0, 0.0, 0.0)

    arm = _StubArm(
        reachable_at_or_above=-1000.0,  # always reachable
        current_position=(0.3, 0.0, 0.2),
    )

    result = compute_grasp_target(cluster, baseline, arm)

    assert result is not None

    reach_vector = result.grasp_point - result.tcp_position
    reach_length = np.linalg.norm(reach_vector)

    assert reach_length == pytest.approx(GRIPPER_OFFSET_Z)
    # Not purely vertical this time: has a nonzero X component.
    assert abs(reach_vector[0]) > 1e-6

    # The probed orientation's local Z must match the reach direction.
    _x, _y, _z, orientation = arm.probed[0]
    rotation = Rotation.from_quat(
        [orientation.x, orientation.y, orientation.z, orientation.w]
    )
    resulting_z = rotation.apply([0.0, 0.0, 1.0])
    assert resulting_z == pytest.approx(
        reach_vector / reach_length, abs=1e-6
    )
