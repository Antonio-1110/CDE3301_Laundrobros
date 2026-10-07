"""Tests for how the scan's strokes divide the insertion (scan/strokes.py)."""

import argparse

from laundry_control.scan import coverage, pattern
from laundry_control.scan.strokes import MAX_STEP_M, stroke_plan
import numpy as np
import pytest


@pytest.mark.parametrize('entry, count, step', [
    (0.00, 14, 0.03),
    (0.05, 13, 0.37 / 13),
    (0.06, 12, 0.03),
    (0.08, 12, 0.34 / 12),
])
def test_strokes_are_even_and_at_most_the_spacing_apart(entry, count, step):
    assert stroke_plan(0.42, entry, 0.03) == (count, pytest.approx(step))


def test_a_denser_spacing_means_more_strokes():
    count, step = stroke_plan(0.42, 0.06, 0.02)

    assert (count, step) == (18, pytest.approx(0.02))


@pytest.mark.parametrize('depth, entry, spacing, why', [
    (0.42, 0.0, 0.121, 'maximum stroke spacing'),
    (0.42, 0.0, 0.0, 'greater than zero'),
    (0.42, 0.42, 0.03, 'entry depth'),
    (0.42, -0.01, 0.03, 'entry depth'),
    (0.0, 0.0, 0.03, 'depth must'),
])
def test_bad_plans_are_refused(depth, entry, spacing, why):
    with pytest.raises(ValueError, match=why):
        stroke_plan(depth, entry, spacing)


def _parse(argv):
    parser = argparse.ArgumentParser()
    pattern.add_scan_arguments(parser)
    return parser.parse_args(argv)


def test_step_over_the_cap_is_rejected_on_the_command_line(capsys):
    with pytest.raises(SystemExit):
        _parse(['--step', '0.13'])

    assert f'at most {MAX_STEP_M}' in capsys.readouterr().err


def test_coarse_steps_scale_the_stroke_speed():
    default = pattern.scan_kwargs_from_args(_parse(['--quick']))
    coarse = pattern.scan_kwargs_from_args(_parse(['--quick', '--step', '0.12']))
    given = pattern.scan_kwargs_from_args(
        _parse(['--quick', '--step', '0.12', '--velocity', '0.05'])
    )

    # The default scan is unchanged; 4x the spacing, 4x the speed, so
    # each stroke (and its J7 sweep) takes as long as a default one.
    assert default['velocity'] == pytest.approx(pattern.DEFAULT_VELOCITY)
    assert default['acceleration'] == pytest.approx(pattern.DEFAULT_ACCELERATION)
    assert stroke_plan(0.42, 0.06, 0.12) == (3, pytest.approx(0.12))
    assert coarse['velocity'] == pytest.approx(0.12)
    assert coarse['acceleration'] == pytest.approx(0.12)
    assert given['velocity'] == pytest.approx(0.05)
    assert given['acceleration'] == pytest.approx(0.12)


def test_baselines_are_never_coarse(capsys):
    from argparse import Namespace

    from laundry_control.scan.baselines import forwarded_step, run_collect

    assert forwarded_step(['--depth', '0.4', '--step', '0.06']) == 0.06
    assert forwarded_step(['--step=0.02']) == 0.02
    assert forwarded_step([]) is None

    assert run_collect(Namespace(dest=None), ['--step', '0.12']) == 2
    assert 'at most' in capsys.readouterr().err


def test_scan_options_carry_the_entry_depth():
    args = _parse(['--entry-depth', '0.05', '--step', '0.025',
                   '--end-scan', 'bottom'])

    kwargs = pattern.scan_kwargs_from_args(args)

    assert kwargs['entry_depth'] == pytest.approx(0.05)
    assert kwargs['step'] == pytest.approx(0.025)


def test_default_scan_starts_inside_the_bucket():
    assert pattern.DEFAULT_ENTRY_DEPTH_M > 0.0
    assert pattern.DEFAULT_STEP_M <= MAX_STEP_M


def test_simulated_strokes_start_at_the_entry_depth():
    path = coverage.ScanPath(entry_m=0.06, bottom_detour=False)
    origins, _ = coverage._stroke_beams(path)

    tool_z = coverage._unit(coverage.INTER_TOOL_Z)
    flange = origins - coverage.TOF_SENSOR_OFFSET_Z * tool_z
    depth = (flange - coverage.INTER_FLANGE_POSITION) @ tool_z

    # The sensor's radial offset turns with J7, not along the axis.
    assert depth.min() >= 0.06 - 1e-9
    assert depth.max() <= 0.42 + 1e-9
    assert np.isclose(
        len(origins),
        2 * 12 * coverage.DEFAULT_SAMPLES_PER_STROKE,
    )


def test_the_quick_scan_is_the_full_baselines_strokes():
    from laundry_control import config
    from laundry_control.scan import segments

    kwargs = pattern.scan_kwargs_from_args(_parse(['--quick']))

    # No plan needed to scan, but loaded for run/clear's end-scan pass.
    assert kwargs['end_scan'] == 'none'
    assert segments.for_end_scan('none') == (segments.STROKES,)
    assert segments.for_end_scan('precession') is None
    assert config.baseline_dir().endswith('baseline_scans')


def test_baselines_are_full_scans_only(capsys):
    from argparse import Namespace

    from laundry_control.scan.baselines import forwarded_end_scan, run_collect

    assert forwarded_end_scan(['--end-scan', 'none']) == 'none'
    assert forwarded_end_scan(['--depth', '0.4', '--end-scan=none']) == 'none'
    assert forwarded_end_scan(['--quick']) == 'none'
    assert forwarded_end_scan(['--depth', '0.4']) is None

    assert run_collect(Namespace(dest=None), ['--quick']) == 2
    assert 'full scans' in capsys.readouterr().err


def test_baseline_scans_ask_for_the_full_scan(monkeypatch):
    from laundry_control.cli import build_parser
    from laundry_control.scan import baselines

    commands = []

    class Done:
        returncode = 0

    monkeypatch.setattr(
        baselines.subprocess, 'run',
        lambda command, timeout: commands.append(command) or Done(),
    )

    # `laundry scan` itself defaults to the quick scan.
    assert baselines.run_one_scan('b.csv', [], 1.0)
    assert baselines.run_one_scan('b.csv', ['--end-scan', 'bottom'], 1.0)

    scan_args = [build_parser().parse_args(c[3:]) for c in commands]
    assert scan_args[0].end_scan == 'precession'
    assert scan_args[1].end_scan == 'bottom'


def test_end_scan_of_reads_the_scan_kind_from_the_csv(tmp_path):
    from laundry_control.scan import segments

    def write(name, header, rows):
        path = tmp_path / name
        path.write_text(
            header + '\n' + '\n'.join(','.join(map(str, r)) for r in rows) + '\n'
        )
        return str(path)

    base = 'x,y,z,stamp_sec,stamp_nanosec'
    full = write('full.csv', base + ',segment', [
        (0, 0, 0, 0, 0, segments.STROKES), (0, 0, 0, 0, 0, segments.END),
    ])
    quick = write('quick.csv', base + ',segment', [
        (0, 0, 0, 0, 0, segments.STROKES), (0, 0, 0, 0, 0, segments.STROKES),
    ])
    old = write('old.csv', base, [(0, 0, 0, 0, 0)])

    assert segments.end_scan_of(full) == 'precession'
    assert segments.end_scan_of(quick) == 'none'
    assert segments.end_scan_of(old) == 'precession'
