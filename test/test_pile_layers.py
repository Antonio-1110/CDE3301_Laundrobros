"""The sweep's grabs generated from how full the drum is (grasp/pile_layers.py)."""

import contextlib

from laundry_control import config
from laundry_control.bucket import axis_basis, seed_cone
from laundry_control.grasp import grab_targets, pile_layers, retrieve_grid
import numpy as np
import pytest


def _quiet(*_args):
    return None


@pytest.fixture
def reach(monkeypatch):
    """
    Stand in for the arm: a grab is reachable up to 7 cm + 0.15 cm per degree of tilt.

    The 15 cm gripper meets the drum's ceiling, so higher needs tilt.
    """
    tries = []

    cone = seed_cone()
    up, _side = axis_basis(cone.axis_dir)

    def try_pose(arm, contact, tool_z, claw_x, approach_m, seeds, inter):
        relative = np.asarray(contact) - cone.axis_point
        depth = relative @ cone.axis_dir
        vertical = relative @ up + cone.radius_at(depth)
        tilt = np.degrees(np.arcsin(-np.asarray(tool_z) @ cone.axis_dir))
        tries.append((round(vertical, 3), round(tilt, 1)))

        if vertical <= 0.07 + 0.0015 * tilt + 1e-6:
            return {'contact': [0.0, 0.0, 0.0]}, None

        return None, 'out of reach'

    monkeypatch.setattr(retrieve_grid, '_try_pose', try_pose)
    # The tests' own layers, so they do not follow the default.
    monkeypatch.setitem(
        config.PILE, 'layer_heights_m', (0.04, 0.09, 0.14, 0.19)
    )
    monkeypatch.setitem(config.PILE, 'depths_m', (0.32, 0.21, 0.10))
    monkeypatch.setitem(config.PILE, 'lateral_m', (0.0, -0.12, 0.12))
    monkeypatch.setattr(
        retrieve_grid, 'gripper_padding',
        lambda *a, **k: contextlib.nullcontext(),
    )

    return tries


@pytest.mark.parametrize('depth', [0.1, 0.4])
@pytest.mark.parametrize('angle', [0.0, -20.0, 20.0])
def test_vertical_and_radial_heights_convert_both_ways(depth, angle):
    cone = seed_cone()

    height = pile_layers.radial_height(depth, angle, 0.14, cone)

    assert pile_layers.vertical_height(depth, angle, height, cone) == (
        pytest.approx(0.14)
    )


def test_height_at_the_floor_line_is_the_same_both_ways():
    assert pile_layers.vertical_height(0.3, 0.0, 0.11) == pytest.approx(0.11)


def test_the_editor_draws_the_pile_at_its_fraction_of_the_drum():
    cone = seed_cone()
    drum = 2.0 * cone.radius_at(0.5 * cone.s_max)

    assert pile_layers.fill_fraction(0.5 * drum, cone) == pytest.approx(0.5)
    # By default, as full as the top layer reaches into.
    top = max(config.PILE['layer_heights_m']) + 0.05
    assert pile_layers.fill_fraction() == pytest.approx(top / drum)


def test_a_spot_takes_the_highest_height_some_tilt_reaches(reach):
    target, reason = pile_layers.find_grab(None, 0.30, 0.0, 0.19, 0.10)

    assert reason is None
    # 60 deg reaches 16 cm; nothing reaches 17-19 cm.
    assert target['tilt_deg'] == 60.0
    assert pile_layers.target_vertical(target) == pytest.approx(0.16, abs=2e-3)


def test_a_spot_takes_the_least_tilt_that_reaches_the_height(reach):
    target, _ = pile_layers.find_grab(None, 0.30, 0.0, 0.10, 0.08)

    # Vertical reaches 7 cm, 15 deg 9.25, 30 deg 11.5 cm.
    assert target['tilt_deg'] == 30.0
    assert pile_layers.target_vertical(target) == pytest.approx(0.10, abs=2e-3)


def test_a_spot_is_not_tried_below_its_layers_window(reach):
    target, reason = pile_layers.find_grab(None, 0.30, 0.0, 0.19, 0.17)

    # Reachable only up to 16 cm: nothing in 17-19 cm.
    assert target is None and reason
    assert all(height >= 0.17 - 1e-6 for height, _tilt in reach)


def test_every_layer_is_generated_top_first_each_grab_tagged(reach):
    targets, misses, layers = pile_layers.generate(None, log=_quiet)

    spots = len(config.PILE['depths_m']) * len(config.PILE['lateral_m'])

    # Layer 4 (19 cm, down to 17) is out of the stand-in's reach.
    assert {m[0] for m in misses} == {4} and len(misses) == spots
    assert [t['layer'] for t in targets] == [3] * spots + [2] * spots + [1] * spots
    # Each layer at its own height, within its 2 cm window.
    assert all(0.14 - 0.02 - 1e-3 <= v <= 0.14 + 1e-3 for v in layers[1])
    assert all(0.09 - 0.02 - 1e-3 <= v <= 0.09 + 1e-3 for v in layers[2])
    assert all(0.04 - 0.01 - 1e-3 <= v <= 0.04 + 1e-3 for v in layers[3])
    assert layers[0] == []


def test_within_a_layer_the_mouth_row_goes_first_middle_then_sides(reach):
    targets, _, _ = pile_layers.generate(None, log=_quiet)

    first = targets[:9]
    assert [t['depth_m'] for t in first[::3]] == [0.32, 0.21, 0.10]
    assert [t['lateral_m'] for t in first[:3]] == [0.0, -0.12, 0.12]


def test_a_side_spot_that_does_not_fit_moves_toward_the_middle(
    reach, monkeypatch
):
    real = retrieve_grid._try_pose
    side = axis_basis(seed_cone().axis_dir)[1]
    side = side if side[0] >= 0 else -side

    def try_pose(arm, contact, tool_z, claw_x, approach_m, seeds, inter):
        offset = (np.asarray(contact) - seed_cone().axis_point) @ side

        if abs(offset) > 0.09 + 1e-3:
            return None, 'the gripper collides'

        return real(arm, contact, tool_z, claw_x, approach_m, seeds, inter)

    monkeypatch.setattr(retrieve_grid, '_try_pose', try_pose)

    targets, misses, _ = pile_layers.generate(None, log=_quiet)

    assert {m[0] for m in misses} == {4}
    assert sorted({t['lateral_m'] for t in targets}) == [-0.09, 0.0, 0.09]


def test_a_spot_out_of_reach_is_left_out_of_each_layer_alone(
    reach, monkeypatch
):
    real = retrieve_grid._try_pose
    side = axis_basis(seed_cone().axis_dir)[1]
    side = side if side[0] >= 0 else -side

    def try_pose(arm, contact, tool_z, claw_x, approach_m, seeds, inter):
        depth, _angle, _ = retrieve_grid.floor_position(contact, seed_cone())
        offset = (np.asarray(contact) - seed_cone().axis_point) @ side

        # Nowhere on the -x side at 10 cm deep, however near the middle.
        if abs(depth - 0.10) < 1e-3 and offset < -1e-3:
            return None, 'the gripper collides'

        return real(arm, contact, tool_z, claw_x, approach_m, seeds, inter)

    monkeypatch.setattr(retrieve_grid, '_try_pose', try_pose)

    targets, misses, _ = pile_layers.generate(None, log=_quiet)

    spots = len(config.PILE['depths_m']) * len(config.PILE['lateral_m'])
    # Layers 1-3 each miss that one spot; every other spot is there.
    assert sorted(
        (m[0], m[1], m[2]) for m in misses if m[0] != 4
    ) == [(1, 0.10, -0.12), (2, 0.10, -0.12), (3, 0.10, -0.12)]
    assert 'collides' in misses[-1][3]
    assert len(targets) == 3 * (spots - 1)


def test_fill_sweeps_its_layer_and_every_one_under_it_in_order():
    grabs = [{'name': f'g{n}', 'layer': layer}
             for n, layer in enumerate([4, 4, 3, 2, 2, 1])]

    assert [g['layer'] for g in pile_layers.select_layers(grabs, 2)] == [
        2, 2, 1
    ]
    assert pile_layers.select_layers(grabs, 4) == grabs
    assert pile_layers.select_layers(grabs, None) == grabs
    # A hand-placed list has no layers: swept whole.
    placed = [{'name': 'g0'}, {'name': 'g1'}]
    assert pile_layers.select_layers(placed, 1) == placed


def test_generated_targets_keep_what_they_were_generated_from(tmp_path):
    path = str(tmp_path / 'grab_targets.yaml')
    target = grab_targets.clean({
        'depth_m': 0.3, 'floor_angle_deg': 0.0, 'height_m': 0.11,
        'tilt_deg': 15.0, 'approach_m': 0.05,
    })

    grab_targets.save([target], path)
    assert grab_targets.generated_for(path) is None

    generated = {'pile': pile_layers.pile_signature(), 'layers': 2}
    grab_targets.save([target], path, generated=generated)

    assert grab_targets.generated_for(path) == generated
    assert grab_targets.load(path) == [target]
    assert not pile_layers.pile_changed(generated)
    assert not pile_layers.pile_changed(None)


def test_a_changed_pile_is_reported_as_a_stale_plan(tmp_path, monkeypatch):
    path = str(tmp_path / 'retrieve.yaml')
    grab = {
        'name': 'grab_01', 'depth_m': 0.3, 'floor_angle_deg': 0.0,
        'height_m': 0.11, 'tilt_deg': 15.0, 'approach_m': 0.05,
        'contact': [0.1, -0.4, 0.3], 'approach': [0.5] * 7, 'grab': [0.5] * 7,
    }
    targets = [grab_targets.clean(grab)]
    generated = {'pile': pile_layers.pile_signature(), 'layers': 1}
    retrieve_grid.save([grab], path=path, targets=targets)

    assert retrieve_grid.grid_mismatch(
        path, targets=targets, generated=generated
    ) is None

    monkeypatch.setitem(config.PILE, 'layer_window_m', 0.03)

    assert 'config.PILE' in retrieve_grid.grid_mismatch(
        path, targets=targets, generated=generated
    )
    # Hand-placed targets do not follow PILE.
    assert retrieve_grid.grid_mismatch(
        path, targets=targets, generated=None
    ) is None
