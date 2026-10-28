"""Fill levels: one baked sweep plan set each (config.FILL_LEVELS)."""

import os

from laundry_control import config
from laundry_control.arm import transfers
from laundry_control.bucket import seed_cone
import pytest


@pytest.fixture(autouse=True)
def no_fill_after():
    yield
    config.set_fill(None)


def test_the_drum_height_is_the_fitted_buckets():
    cone = seed_cone()

    assert config.DRUM_HEIGHT_M == pytest.approx(
        2.0 * cone.radius_at(0.5 * cone.s_max), abs=0.005
    )


def test_no_fill_level_uses_scan_plans_itself():
    base = os.path.join(config.repo_root(), 'scan_plans')

    assert config.fill_level() is None
    assert config.retrieve_plan_path() == os.path.join(base, 'retrieve.yaml')
    assert transfers.default_plan_path() == os.path.join(base, 'transfers.yaml')


def test_a_fill_level_has_its_own_plan_set_and_pile_top():
    top = config.PILE['top_height_m']
    level = config.FILL_LEVELS[-1]

    config.set_fill(level)

    folder = os.path.join(
        config.repo_root(), 'scan_plans', f'fill_{round(level * 100)}'
    )
    assert config.retrieve_plan_path() == os.path.join(folder, 'retrieve.yaml')
    assert config.grab_targets_path() == os.path.join(
        folder, 'grab_targets.yaml'
    )
    assert transfers.default_plan_path() == os.path.join(
        folder, 'transfers.yaml'
    )
    assert config.PILE['top_height_m'] == pytest.approx(
        level * config.DRUM_HEIGHT_M, abs=1e-3
    )

    config.set_fill(None)
    assert config.PILE['top_height_m'] == top


def test_an_unbaked_fill_level_is_refused():
    with pytest.raises(ValueError, match='baked ones'):
        config.set_fill(0.33)


def test_every_fill_level_leaves_room_for_a_sweep_layer():
    # A pile whose first layer would aim below min_height_m gets no
    # sweep at all (grasp/pile_layers.py).
    for level in config.FILL_LEVELS:
        config.set_fill(level)
        aim = config.PILE['top_height_m'] - config.PILE['sink_m']
        assert aim >= config.PILE['min_height_m'], level
        assert config.PILE['min_height_m'] >= config.LOW_FILL_MIN_HEIGHT_M


def test_only_low_fill_levels_go_under_the_piles_own_minimum():
    own = config.PILE['min_height_m']

    config.set_fill(0.25)
    assert config.PILE['min_height_m'] == pytest.approx(0.03)
    config.set_fill(0.3)
    assert config.PILE['min_height_m'] == pytest.approx(0.0375, abs=1e-3)
    config.set_fill(0.5)
    assert config.PILE['min_height_m'] == own

    config.set_fill(None)
    assert config.PILE['min_height_m'] == own


def test_clear_sweeps_at_its_default_fill_unless_told_otherwise(monkeypatch):
    from laundry_control import cli

    monkeypatch.delenv('LAUNDRY_FILL', raising=False)
    parser = cli.build_parser()

    def chosen(*argv):
        from laundry_control.cli import jobs

        jobs._choose_fill(parser.parse_args(['clear', *argv]))
        return config.fill_level()

    assert chosen() == 0.25
    assert chosen('--fill', '0.6') == 0.6
    assert chosen('--fill', 'none') is None

    monkeypatch.setenv('LAUNDRY_FILL', '0.75')
    config.set_fill(0.75)
    assert chosen() == 0.75
