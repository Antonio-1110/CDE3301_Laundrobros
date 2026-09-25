from dataclasses import dataclass

from laundry_control.config import GRIPPER_OFFSET_Z
from laundry_control.grasp.plan import (
    compute_grasp_target,
    DEFAULT_MAX_SINK_M,
    estimate_surface_depth_below,
    look_at_quaternion,
)
from laundry_control.perception.bucket_model import (
    _axis_basis,
    build_baseline_surface,
    seed_cone,
)
from laundry_control.perception.detect import ClusterSummary
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

CONE = seed_cone()
E1, E2 = _axis_basis(CONE.axis_dir)


def _sample_bucket(n=5000, seed=0, noise_m=0.002, cap_fraction=0.1):
    """Sample an empty bucket: lateral wall plus the flat closed end."""
    rng = np.random.default_rng(seed)

    n_cap = int(n * cap_fraction)
    n_wall = n - n_cap

    s = rng.uniform(0.02, 0.42, n_wall)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_wall)
    r = CONE.radius_at(s) + rng.normal(0.0, noise_m, n_wall)

    r_cap = CONE.radius_at(0.02) * np.sqrt(rng.uniform(0.0, 1.0, n_cap))
    theta_cap = rng.uniform(0.0, 2.0 * np.pi, n_cap)
    s_cap = np.full(n_cap, 0.02) + rng.normal(0.0, noise_m, n_cap)

    s = np.concatenate([s, s_cap])
    theta = np.concatenate([theta, theta_cap])
    r = np.concatenate([r, r_cap])

    return (
        CONE.axis_point
        + s[:, None] * CONE.axis_dir
        + (r * np.cos(theta))[:, None] * E1
        + (r * np.sin(theta))[:, None] * E2
    )


@pytest.fixture(scope='module')
def surface():
    return build_baseline_surface([_sample_bucket(seed=i) for i in range(6)])


def _axis_point_at(s_along=0.25):
    """Return a point on the bucket axis, well inside the scanned extent."""
    return CONE.axis_point + s_along * CONE.axis_dir


def _cluster_above_floor(surface, gap_m, s_along=0.25):
    """
    Build a one-point cluster exactly gap_m above the bucket wall.

    Place a single-point cluster exactly gap_m above the bucket
    wall, on the vertical line through the axis - so the gap the
    grasp planner derives is known by construction.
    """
    x, y, z_axis = _axis_point_at(s_along)

    floor_z = estimate_surface_depth_below(surface, (x, y), z_axis)
    assert floor_z is not None

    return _make_cluster(float(x), float(y), floor_z + gap_m), floor_z


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
    Stand in for XArm7Controller in grasp planning tests.

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
        self.probed.append((x, y, z, kwargs.get('orientation')))
        return 1.0 if z >= self.reachable_at_or_above else 0.3


# ---------------------------------------------------------------
# estimate_surface_depth_below
# ---------------------------------------------------------------

def test_depth_below_finds_the_bucket_wall(surface):
    """
    Straight down from the axis, the wall is one radius below.

    Straight down from the axis, the wall sits one model radius
    below - that is the room a gripper has before it hits bucket.
    """
    x, y, z_axis = _axis_point_at(0.25)

    floor_z = estimate_surface_depth_below(surface, (x, y), z_axis)

    assert floor_z is not None
    assert floor_z < z_axis

    drop = z_axis - floor_z
    expected = CONE.radius_at(0.25)

    assert drop == pytest.approx(expected, abs=0.01)


def test_depth_below_returns_none_outside_the_bucket(surface):
    """
    A point already outside the wall has no depth below it.

    A point already below the wall has nothing under it to measure,
    and must say so rather than inventing a number.
    """
    x, y, z_axis = _axis_point_at(0.25)

    below_the_wall = z_axis - CONE.radius_at(0.25) - 0.05

    assert estimate_surface_depth_below(
        surface, (x, y), below_the_wall
    ) is None


def test_depth_below_returns_none_past_the_mouth(surface):
    """
    A point beyond the mouth is outside the bucket.

    Beyond the mouth the cone is extrapolation, so a point out
    there is outside the bucket however wide the extended cone
    would be.
    """
    beyond = CONE.axis_point + (surface.cone.s_max + 0.2) * CONE.axis_dir

    assert estimate_surface_depth_below(
        surface, (beyond[0], beyond[1]), beyond[2]
    ) is None


def test_depth_below_is_answerable_without_baseline_coverage(surface):
    """
    Depth must be answered where no baseline point landed.

    The point of using the model rather than the baseline points:
    the real scan path reaches only ~30% of the bucket's surface
    cells, and a depth query over an unsampled patch still has to
    get a real answer.
    """
    x, y, z_axis = _axis_point_at(0.25)

    sparse = build_baseline_surface([_sample_bucket(n=400, seed=99)])

    assert (sparse.count == 0).sum() > 0.5 * sparse.count.size

    floor_z = estimate_surface_depth_below(sparse, (x, y), z_axis)

    assert floor_z is not None
    assert z_axis - floor_z == pytest.approx(CONE.radius_at(0.25), abs=0.02)


def test_depth_below_resolves_more_finely_than_sensor_noise(surface):
    """
    Depth estimates must resolve millimetre differences.

    Two queries a millimetre apart must not land in the same
    bisection bucket, or the gap would quantise coarser than the
    2mm the sensor itself resolves.
    """
    x, y, z_axis = _axis_point_at(0.25)

    a = estimate_surface_depth_below(surface, (x, y), z_axis)
    b = estimate_surface_depth_below(surface, (x + 0.02, y), z_axis)

    assert a is not None and b is not None
    assert abs(a - b) < 0.02


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

def test_compute_grasp_target_reachable_at_first_fraction(surface):
    cluster, floor_z = _cluster_above_floor(surface, gap_m=0.06)
    top_z = float(cluster.centroid[2])

    # Arm sits directly above the cluster's (x, y) -> approach
    # direction is purely -Z, reproducing the old pure-Z-offset
    # behaviour as a special case.
    arm = _StubArm(
        reachable_at_or_above=-1.0,
        current_position=(cluster.centroid[0], cluster.centroid[1], 0.9),
    )

    result = compute_grasp_target(cluster, surface, arm)

    assert result is not None
    assert result.sink_fraction_used == 0.5
    assert len(arm.probed) == 1
    # Within the depth search's own 0.5mm bisection tolerance: the
    # planner re-queries from the cluster's top rather than from the
    # axis, so its bracket lands slightly differently. Still an
    # order below the sensor's 2mm noise.
    assert result.gap_m == pytest.approx(0.06, abs=1e-3)
    assert result.sink_amount_m == pytest.approx(0.03, abs=1e-3)
    assert result.grasp_point[2] == pytest.approx(top_z - 0.03, abs=1e-3)
    assert result.tcp_position[2] == pytest.approx(
        top_z - 0.03 + GRIPPER_OFFSET_Z, abs=1e-3
    )
    assert result.tcp_position[:2] == pytest.approx(result.grasp_point[:2])
    assert floor_z < result.grasp_point[2]


def test_compute_grasp_target_falls_through_to_floor(surface):
    cluster, _floor = _cluster_above_floor(surface, gap_m=0.06)
    top_z = float(cluster.centroid[2])

    # Only the un-sunk grasp clears the obstruction.
    arm = _StubArm(
        reachable_at_or_above=top_z + GRIPPER_OFFSET_Z - 1e-9,
        current_position=(cluster.centroid[0], cluster.centroid[1], 0.9),
    )

    result = compute_grasp_target(cluster, surface, arm)

    assert result is not None
    assert result.sink_fraction_used == 0.0
    assert result.sink_amount_m == pytest.approx(0.0)
    assert result.grasp_point[2] == pytest.approx(top_z)


def test_compute_grasp_target_never_reachable_returns_none(surface):
    cluster, _floor = _cluster_above_floor(surface, gap_m=0.06)

    arm = _StubArm(
        reachable_at_or_above=1000.0,
        current_position=(cluster.centroid[0], cluster.centroid[1], 0.9),
    )

    assert compute_grasp_target(cluster, surface, arm) is None


def test_compute_grasp_target_no_sink_when_nothing_below(surface):
    """
    A cluster outside the bucket must be grasped without sinking.

    A cluster outside the modelled bucket has no measurable room
    underneath, so the planner must grasp at the sensed surface
    rather than sink on a guess.
    """
    x, y, z_axis = _axis_point_at(0.25)
    outside_z = z_axis - CONE.radius_at(0.25) - 0.05

    cluster = _make_cluster(float(x), float(y), float(outside_z))

    arm = _StubArm(reachable_at_or_above=-1.0, current_position=(x, y, 0.9))

    result = compute_grasp_target(cluster, surface, arm)

    assert result is not None
    assert result.gap_m == pytest.approx(0.0)
    assert result.sink_amount_m == pytest.approx(0.0)


def test_compute_grasp_target_max_sink_cap(surface):
    """
    The sink must be capped at DEFAULT_MAX_SINK_M.

    A cluster sitting high in the bucket has most of a bucket
    radius underneath it - far more than the planner should dig in
    one go.
    """
    x, y, z_axis = _axis_point_at(0.25)
    cluster = _make_cluster(float(x), float(y), float(z_axis))

    arm = _StubArm(reachable_at_or_above=-1.0, current_position=(x, y, 0.9))

    result = compute_grasp_target(cluster, surface, arm)

    assert result is not None
    assert result.gap_m > 2 * DEFAULT_MAX_SINK_M
    assert result.sink_fraction_used == 0.5
    assert result.sink_amount_m == pytest.approx(DEFAULT_MAX_SINK_M)


def test_compute_grasp_target_offset_arithmetic_directly_above(surface):
    cluster, _floor = _cluster_above_floor(surface, gap_m=0.06)
    cx, cy = float(cluster.centroid[0]), float(cluster.centroid[1])

    arm = _StubArm(reachable_at_or_above=-1.0, current_position=(cx, cy, 0.9))

    result = compute_grasp_target(cluster, surface, arm)

    assert result is not None
    assert result.tcp_position[2] - result.grasp_point[2] == pytest.approx(
        GRIPPER_OFFSET_Z
    )
    assert result.tcp_position[:2] == pytest.approx(result.grasp_point[:2])
    assert result.tcp_position[:2] == pytest.approx([cx, cy])


def test_compute_grasp_target_tilts_when_arm_offset_laterally(surface):
    # Arm is NOT above the cluster -- offset well to the side. The
    # reach from tcp_position to grasp_point must still be exactly
    # gripper_offset_z long, but no longer purely vertical.
    cluster, _floor = _cluster_above_floor(surface, gap_m=0.06)
    cx, cy = float(cluster.centroid[0]), float(cluster.centroid[1])

    arm = _StubArm(
        reachable_at_or_above=-1000.0,  # always reachable
        current_position=(cx + 0.3, cy + 0.2, 0.7),
    )

    result = compute_grasp_target(cluster, surface, arm)

    assert result is not None

    reach_vector = result.grasp_point - result.tcp_position
    reach_length = np.linalg.norm(reach_vector)

    assert reach_length == pytest.approx(GRIPPER_OFFSET_Z)
    # Not purely vertical this time: has a nonzero lateral component.
    assert np.linalg.norm(reach_vector[:2]) > 1e-6

    # The probed orientation's local Z must match the reach direction.
    _x, _y, _z, orientation = arm.probed[0]
    rotation = Rotation.from_quat(
        [orientation.x, orientation.y, orientation.z, orientation.w]
    )
    resulting_z = rotation.apply([0.0, 0.0, 1.0])
    assert resulting_z == pytest.approx(
        reach_vector / reach_length, abs=1e-6
    )
