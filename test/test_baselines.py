"""Baseline-set management: naming, archiving, restoring (scan/baselines.py)."""

import os

from laundry_control.scan import baselines
import pytest


def _scan(path, rows=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('x,y,z\n' + '0,0,0\n' * rows)
    return str(path)


def test_archive_moves_the_set_under_its_session(tmp_path):
    for i in (1, 2):
        _scan(tmp_path / f'baseline_20260924_180836_{i:02d}.csv')
    (tmp_path / 'baseline_20260924_180836_03.csv.rejected').write_text('')

    target, moved = baselines.archive(str(tmp_path))

    assert target == str(tmp_path / 'archive' / '20260924_180836')
    assert len(moved) == 3
    assert baselines.active_scans(str(tmp_path)) == []
    assert baselines.archived_sets(str(tmp_path)) == [('20260924_180836', 2)]


def test_archive_never_overwrites_an_earlier_archive(tmp_path):
    for _ in range(2):
        _scan(tmp_path / 'baseline_20260924_180836_01.csv')
        baselines.archive(str(tmp_path))

    labels = [label for label, _n in baselines.archived_sets(str(tmp_path))]
    assert labels == ['20260924_180836', '20260924_180836_2']


def test_archive_of_an_empty_directory_does_nothing(tmp_path):
    assert baselines.archive(str(tmp_path)) == (None, [])


def test_mixed_sessions_get_a_range_label():
    paths = ['baseline_20260924_180836_01.csv', 'baseline_20260926_101500_01.csv']

    assert baselines.set_label(paths) == '20260924_180836..20260926_101500'


def test_restore_swaps_sets_without_losing_either(tmp_path):
    _scan(tmp_path / 'baseline_20260924_180836_01.csv')
    baselines.archive(str(tmp_path))
    _scan(tmp_path / 'baseline_20260926_101500_01.csv')

    restored, shelved = baselines.restore(str(tmp_path), '20260924_180836')

    assert [os.path.basename(p) for p in restored] == [
        'baseline_20260924_180836_01.csv'
    ]
    assert shelved == str(tmp_path / 'archive' / '20260926_101500')
    assert baselines.archived_sets(str(tmp_path)) == [('20260926_101500', 1)]


def test_restore_of_an_unknown_label_says_what_exists(tmp_path):
    _scan(tmp_path / 'baseline_20260924_180836_01.csv')
    baselines.archive(str(tmp_path))

    with pytest.raises(FileNotFoundError, match='20260924_180836'):
        baselines.restore(str(tmp_path), 'nope')


def test_promote_names_scans_by_when_they_were_taken(tmp_path):
    src = _scan(tmp_path / 'records' / 'scan_20260926_093000.csv')
    dest = str(tmp_path / 'baselines')

    first, _ = baselines.promote(src, dest=dest)
    second, _ = baselines.promote(src, dest=dest)

    assert os.path.basename(first[0]) == 'baseline_20260926_093000_01.csv'
    assert os.path.basename(second[0]) == 'baseline_20260926_093000_02.csv'
    assert os.path.isfile(src)


def test_promote_a_directory_with_move_and_archive_replaces_the_set(tmp_path):
    dest = tmp_path / 'baselines'
    _scan(dest / 'baseline_20260924_180836_01.csv')
    staged = [
        _scan(tmp_path / 'incoming' / f'baseline_20260926_101500_{i:02d}.csv')
        for i in (1, 2)
    ]

    written, shelved = baselines.promote(
        [str(tmp_path / 'incoming')], dest=str(dest), move=True,
        archive_first=True,
    )

    assert shelved == str(dest / 'archive' / '20260924_180836')
    assert [os.path.basename(p) for p in written] == [
        'baseline_20260926_101500_01.csv', 'baseline_20260926_101500_02.csv'
    ]
    assert not any(os.path.exists(p) for p in staged)


class _Args:
    count = 2
    dest = None
    min_points = 1
    keep_going = False
    timeout = 10.0
    settle = 0.0
    archive = True
    yes = True


def _fake_scans(monkeypatch, succeed):
    """Replace the real `laundry scan` subprocess with file writes."""
    results = iter(succeed)
    calls = []

    def run_one_scan(csv_path, _scan_args, _timeout, **flags):
        calls.append(flags)
        ok = next(results)
        if ok:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, 'w') as handle:
                handle.write('x,y,z\n0,0,0\n')
        return ok

    monkeypatch.setattr(baselines, 'run_one_scan', run_one_scan)
    left = []
    monkeypatch.setattr(baselines, 'leave_bucket', lambda: left.append(1) or True)
    return calls, left


def test_collect_archive_swaps_in_a_complete_set(tmp_path, monkeypatch):
    _scan(tmp_path / 'baseline_20260924_180836_01.csv')
    _fake_scans(monkeypatch, [True, True])
    args = _Args()
    args.dest = str(tmp_path)

    assert baselines.run_collect(args, []) == 0

    current = baselines.active_scans(str(tmp_path))
    assert len(current) == 2
    assert all('20260924_180836' not in p for p in current)
    assert baselines.archived_sets(str(tmp_path)) == [('20260924_180836', 1)]
    assert not [d for d in os.listdir(tmp_path) if d.startswith('incoming_')]


def test_collect_archive_keeps_the_old_set_after_a_failure(tmp_path, monkeypatch):
    old = _scan(tmp_path / 'baseline_20260924_180836_01.csv')
    _fake_scans(monkeypatch, [True, False])
    args = _Args()
    args.dest = str(tmp_path)

    assert baselines.run_collect(args, []) == 1

    assert baselines.active_scans(str(tmp_path)) == [old]
    staging = [d for d in os.listdir(tmp_path) if d.startswith('incoming_')]
    assert len(staging) == 1
    assert len(baselines.active_scans(str(tmp_path / staging[0]))) == 1


def test_collect_stays_inside_between_scans(tmp_path, monkeypatch):
    calls, left = _fake_scans(monkeypatch, [True, True, True])
    args = _Args()
    args.count = 3
    args.dest = str(tmp_path)

    assert baselines.run_collect(args, []) == 0

    assert calls == [
        {'start_inside': False, 'stay_inside': True},
        {'start_inside': True, 'stay_inside': True},
        {'start_inside': True, 'stay_inside': False},
    ]
    assert left == []  # the last scan goes out to INTER itself


def test_collect_after_a_failed_scan_starts_from_inter_again(
    tmp_path, monkeypatch,
):
    calls, left = _fake_scans(monkeypatch, [False, True])
    args = _Args()
    args.keep_going = True
    args.archive = False
    args.dest = str(tmp_path)

    baselines.run_collect(args, [])

    assert calls[1]['start_inside'] is False
    assert left == []


def test_collect_moves_out_when_it_stops_with_the_arm_inside(
    tmp_path, monkeypatch,
):
    # The scan succeeds but has too few points: rejected, stop, and
    # the arm is still at the entry pose (it stayed inside).
    calls, left = _fake_scans(monkeypatch, [True, True])
    args = _Args()
    args.min_points = 5
    args.dest = str(tmp_path)

    assert baselines.run_collect(args, []) == 1

    assert len(calls) == 1
    assert left == [1]
