"""`laundry clear`: grab sweep, then scan -> grasp until empty (pipeline.run_clear)."""

from laundry_control import pipeline
from laundry_control.hardware.fake import FakeRecorder
import pytest


@pytest.fixture
def stages(monkeypatch):
    """Stub every stage; `script` sets what each scan detects."""
    log = []
    state = {
        'detections': [], 'grasp_results': [], 'sweep_ok': True,
        'scans': [],
    }

    monkeypatch.setattr(
        pipeline, 'build_model',
        lambda baseline, segments=None: log.append(
            'model' if segments is None else 'strokes model'
        ) or ('S' if segments is None else 'Q'),
    )

    def preplanned(arm, gripper, limit=None, time_scale=1.0):
        log.append('sweep')
        return state['sweep_ok']

    def scan(arm, recorder, csv_path, scan_kwargs, check=None):
        log.append('scan')
        state['scans'].append(dict(scan_kwargs, csv_path=csv_path))
        # A check at the bottom that finds laundry ends the pass there.
        return True if check is None else (check() or True)

    def detect(csv_path, baseline, params, surface=None, segments=None):
        assert surface in ('S', 'Q')  # fitted once, reused
        log.append(f'detect {surface}')
        clusters = state['detections'].pop(0)
        return clusters, surface

    def end_only(arm, recorder, csv_path, scan_kwargs, start_inside=False):
        log.append('end scan' + (' (inside)' if start_inside else ''))
        return True

    def plan(arm, clusters, surface):
        return None if clusters == ['unreachable'] else 'PLAN'

    def execute(arm, gripper, plan, drop=False):
        assert drop and plan == 'PLAN'
        log.append('grasp')
        return state['grasp_results'].pop(0) if state['grasp_results'] else True

    monkeypatch.setattr(pipeline, 'run_preplanned', preplanned)
    monkeypatch.setattr(pipeline, 'run_scan', scan)
    monkeypatch.setattr(pipeline, 'run_end_scan_only', end_only)
    monkeypatch.setattr(pipeline, 'detect_scan', detect)
    monkeypatch.setattr(pipeline, 'plan_grasp', plan)
    monkeypatch.setattr(pipeline, 'execute_plan', execute)
    monkeypatch.setattr(pipeline, 'describe_plan', lambda plan: plan)
    monkeypatch.setattr(
        pipeline, 'go_to',
        lambda arm, name, **k: log.append(f'go_to {name}') or True,
    )
    monkeypatch.setattr(pipeline, 'open_for_scan', lambda g, dry_run=False: True)

    return log, state


def _moves(log):
    """Drop the detections and go_to from the log, as older tests expect."""
    return [
        entry for entry in log
        if not entry.startswith(('detect', 'go_to'))
    ]


def _clear(**kwargs):
    return pipeline.run_clear(
        None, None, None, 'baselines', {}, {},
        scan_path=lambda i: f'clear_{i}.csv', **kwargs
    )


def test_sweeps_then_grasps_until_a_scan_finds_nothing(stages):
    log, state = stages
    state['detections'] = [['a', 'b'], ['b'], []]

    assert _clear()
    assert _moves(log) == [
        'model', 'sweep', 'scan', 'grasp', 'scan', 'grasp', 'scan',
    ]


def test_no_grabs_starts_with_a_scan(stages):
    log, state = stages
    state['detections'] = [[]]

    assert _clear(sweep=False)
    assert _moves(log) == ['model', 'scan']


def test_a_failed_sweep_stops_before_scanning(stages):
    log, state = stages
    state['sweep_ok'] = False

    assert not _clear()
    assert _moves(log) == ['model', 'sweep']


def test_one_failed_grasp_is_retried_two_in_a_row_stop(stages):
    log, state = stages
    state['detections'] = [['a']] * 5
    state['grasp_results'] = [False, True, False, False]

    assert not _clear(sweep=False, max_failed=2)
    assert log.count('grasp') == 4


def test_stops_at_once_when_nothing_detected_is_reachable(stages):
    log, state = stages
    state['detections'] = [['unreachable']] * 3

    assert not _clear(sweep=False)
    # One scan: rescanning an untouched pile cannot help.
    assert _moves(log) == ['model', 'scan']


def test_gives_up_after_max_rounds(stages):
    log, state = stages
    state['detections'] = [['a']] * 10

    assert not _clear(sweep=False, max_rounds=3)
    assert log.count('scan') == 3


class _Node:

    class _Log:

        def info(self, *_a, **_k):
            pass

    def get_logger(self):
        return self._Log()


def test_fake_recorder_plays_scans_in_order_then_repeats_the_last(tmp_path):
    first, second = tmp_path / 'laundry.csv', tmp_path / 'empty.csv'
    first.write_text('laundry')
    second.write_text('empty')

    recorder = FakeRecorder(_Node(), [str(first), str(second)])
    saved = []

    for index in range(3):
        out = tmp_path / f'scan_{index}.csv'
        recorder.set_csv_path(str(out))
        recorder.save_blocking()
        saved.append(out.read_text())

    assert saved == ['laundry', 'empty', 'empty']


def test_fake_recorder_saves_of_one_scan_use_one_file(tmp_path):
    first, second = tmp_path / 'laundry.csv', tmp_path / 'empty.csv'
    first.write_text('empty')
    second.write_text('laundry')

    recorder = FakeRecorder(_Node(), [str(first), str(second)])
    out = tmp_path / 'coarse.csv'

    # A check at the bottom of a pass, its final save, then the
    # recorder's auto-naming handed back: still the first scan.
    recorder.set_csv_path(str(out))
    recorder.save_blocking()
    recorder.save_blocking()
    recorder.set_csv_path('')
    assert out.read_text() == 'empty'

    recorder.set_csv_path(str(tmp_path / 'dense.csv'))
    recorder.save_blocking()
    assert (tmp_path / 'dense.csv').read_text() == 'laundry'


def _quick_clear(**kwargs):
    return pipeline.run_clear(
        None, None, None, 'baselines', {'end_scan': 'none', 'end_plan': 'P'},
        {}, scan_path=lambda i: f'clear_{i}.csv', **kwargs
    )


def test_quick_rounds_then_the_end_scan_alone_once_they_find_nothing(stages):
    log, state = stages
    # Quick scan: a -> grab. Quick scan: nothing -> end scan alone: c
    # -> grab. Quick: nothing -> end scan alone: nothing -> clear.
    state['detections'] = [['a'], [], ['c'], [], []]

    assert _quick_clear(sweep=False, coarse_steps=(), bottom_check=False)
    assert _moves(log) == [
        'strokes model',
        'scan', 'grasp',
        'scan', 'end scan (inside)', 'model', 'grasp',
        'scan', 'end scan (inside)',
    ]


def test_each_pass_is_judged_inside_the_bucket(stages):
    log, state = stages
    state['detections'] = [['a']]

    assert _quick_clear(sweep=False, coarse_steps=(), bottom_check=False,
                        max_rounds=1) is False
    # Detected at the entry pose, then out to INTER, then the grasp.
    assert log[1:5] == ['scan', 'detect Q', 'go_to inter', 'go_to inter']
    assert state['scans'][0]['stay_inside']
    assert not state['scans'][0]['start_inside']


def test_coarse_pass_first_then_the_dense_one_from_inside(stages):
    log, state = stages
    # 12 cm: nothing. 3 cm: b -> grab. Round 2: 12 cm finds c at once.
    state['detections'] = [[], ['b'], ['c'], [], [], []]

    assert not _quick_clear(sweep=False, bottom_check=False, max_rounds=2)

    steps = [
        (scan.get('step', 0.03), scan['start_inside'])
        for scan in state['scans']
    ]
    assert steps == [(0.12, False), (0.03, True), (0.12, False)]
    assert state['scans'][0]['csv_path'] == 'clear_1_12cm.csv'
    assert state['scans'][1]['csv_path'] == 'clear_1.csv'
    assert log.count('grasp') == 2


def test_coarse_pass_scales_the_stroke_speed_with_its_step():
    kwargs = pipeline.coarse_pass_kwargs(
        {'step': 0.03, 'velocity': 0.03, 'acceleration': 0.06}, 0.12
    )

    assert kwargs['step'] == 0.12
    assert kwargs['velocity'] == pytest.approx(0.12)
    assert kwargs['acceleration'] == pytest.approx(0.24)
    assert pipeline.coarse_pass_kwargs(
        {'step': 0.03, 'velocity': 0.5}, 0.12
    )['velocity'] == 1.0


def test_found_at_the_bottom_goes_for_it_without_finishing_the_pass(stages):
    log, state = stages
    # 12 cm: nothing (judged whole). The 3 cm pass's inward strokes
    # already show a.
    state['detections'] = [[], ['a']]

    assert not _quick_clear(sweep=False, max_rounds=1)
    # At the bottom scan() left the bucket itself: no detection at the
    # entry pose, no move out of it.
    assert log[1:6] == ['scan', 'detect Q', 'scan', 'detect Q', 'go_to inter']
    assert log[6] == 'grasp'


def test_coarse_passes_run_whole_without_a_check_at_the_bottom(stages):
    log, state = stages
    state['detections'] = [['a']]

    assert not _quick_clear(sweep=False, max_rounds=1)
    # Judged once complete, at the entry pose, then out to INTER.
    assert log[1:4] == ['scan', 'detect Q', 'go_to inter']
    assert len(state['scans']) == 1


def test_nothing_anywhere_ends_with_the_end_scan_from_inside(stages):
    log, state = stages
    # 12 cm whole, 3 cm at the bottom and whole, then the end scan.
    state['detections'] = [[], [], [], []]

    assert _quick_clear(sweep=False)
    assert _moves(log) == [
        'strokes model', 'scan', 'scan', 'end scan (inside)', 'model',
    ]
    assert log.count('detect Q') == 3
    assert log.count('detect S') == 1


def test_without_a_baked_end_scan_a_quick_miss_is_final(stages):
    log, state = stages
    # 12 cm whole, 3 cm at the bottom and whole.
    state['detections'] = [[]] * 3

    assert pipeline.run_clear(
        None, None, None, 'baselines', {'end_scan': 'none', 'end_plan': None},
        {}, sweep=False, scan_path=lambda i: f'clear_{i}.csv',
    )
    assert 'end scan' not in log
    # Still out of the bucket.
    assert log[-1] == 'go_to inter'
