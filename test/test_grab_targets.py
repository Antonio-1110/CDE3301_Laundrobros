"""The sweep's hand-placed grabs (grasp/grab_targets.py)."""

from laundry_control import config
from laundry_control.bucket import seed_cone
from laundry_control.grasp import grab_targets
import numpy as np
import pytest


def _target(**changes):
    target = {
        'depth_m': 0.25, 'floor_angle_deg': -20.0, 'height_m': 0.11,
        'tilt_deg': 15.0, 'approach_m': 0.05,
    }
    target.update(changes)
    return target


def test_targets_survive_a_save_and_load(tmp_path):
    path = str(tmp_path / 'grab_targets.yaml')
    targets = [_target(), _target(depth_m=0.4, tilt_deg=0.0)]

    grab_targets.save(targets, path)

    assert grab_targets.load(path) == targets


def test_a_missing_file_loads_as_no_targets(tmp_path):
    assert grab_targets.load(str(tmp_path / 'absent.yaml')) == []


def test_targets_are_clamped_and_rounded():
    cleaned = grab_targets.clean(_target(
        height_m=0.9, floor_angle_deg=123.0, tilt_deg=14.7, approach_m=-0.2,
        depth_m=0.25049,
    ))

    assert cleaned['height_m'] == grab_targets.LIMITS['height_m'][1]
    assert cleaned['floor_angle_deg'] == 90.0
    assert cleaned['tilt_deg'] == 14.5
    assert cleaned['approach_m'] == 0.0
    assert cleaned['depth_m'] == 0.25


def test_the_starting_layout_is_the_old_twelve_spots_up_in_the_pile():
    targets = grab_targets.default_targets()

    assert len(targets) == 12
    assert all(t['height_m'] >= 0.08 for t in targets)


@pytest.mark.parametrize('angle', [-60.0, -20.0, 0.0, 35.0])
@pytest.mark.parametrize('tilt', [-10.0, 0.0, 25.0, 50.0])
def test_a_pose_converts_back_to_its_target(angle, tilt):
    target = grab_targets.clean(_target(floor_angle_deg=angle, tilt_deg=tilt))
    contact, tool_z, _ = grab_targets.geometry(target)

    back = grab_targets.from_pose(contact, tool_z, target['approach_m'])

    assert back == target


def test_the_approach_handle_sets_the_approach_distance():
    target = _target()
    points = grab_targets.stack(target)

    assert grab_targets.approach_from_handle(
        target, points['approach']
    ) == pytest.approx(target['approach_m'])

    further = points['approach'] - 0.03 * points['tool_z']
    assert grab_targets.approach_from_handle(
        target, further
    ) == pytest.approx(target['approach_m'] + 0.03)


def test_the_stack_runs_back_along_the_tool_axis():
    points = grab_targets.stack(_target())

    assert np.linalg.norm(points['contact'] - points['flange']) == (
        pytest.approx(config.GRIPPER_OFFSET_Z)
    )
    assert grab_targets.wall_margin(points['contact']) == pytest.approx(0.11)


def test_the_fill_surface_is_level_at_the_fraction():
    cone = seed_cone()
    triangles = grab_targets.fill_surface(2.0 / 3.0, cone)

    assert len(triangles) and len(triangles) % 3 == 0

    # Level, and 2/3 of the drum's height up at mid-depth.
    assert np.ptp(triangles[:, 2]) < 1e-9
    mid = 0.5 * cone.s_max
    floor = grab_targets.stack(_target(
        depth_m=mid, floor_angle_deg=0.0, height_m=0.0, tilt_deg=0.0,
    ), cone)['contact']
    height = triangles[0, 2] - floor[2]
    assert height == pytest.approx(
        (2.0 / 3.0) * 2 * cone.radius_at(mid) * abs(
            np.dot(grab_targets._frame(0.0, cone)[1], [0, 0, 1])
        ), rel=0.02,
    )


def test_the_fill_outline_is_level_and_matches_the_surface():
    cone = seed_cone()
    outline = grab_targets.fill_outline(2.0 / 3.0, cone)
    surface = grab_targets.fill_surface(2.0 / 3.0, cone)

    assert len(outline) and len(outline) % 2 == 0
    assert np.ptp(outline[:, 2]) < 1e-9
    assert outline[0, 2] == pytest.approx(surface[0, 2])


def test_the_bucket_wireframe_lies_on_the_bucket():
    cone = seed_cone()
    lines = grab_targets.bucket_wireframe(cone)

    assert len(lines) and len(lines) % 2 == 0

    # Rings and wall lines are on the wall; only the closed end's
    # cross runs inside it, across s = 0.
    from laundry_control.bucket import to_cylindrical

    s, _, r = to_cylindrical(lines, cone)
    on_wall = np.abs(cone.radius_at(s) - r) < 1e-6
    on_closed_end = np.abs(s) < 1e-6

    assert np.all(on_wall | on_closed_end)
    assert s.max() == pytest.approx(cone.s_max)
