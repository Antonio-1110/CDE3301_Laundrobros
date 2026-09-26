"""`laundry clear`: grab sweep, then scan -> grasp until empty (pipeline.run_clear)."""

from laundry_control import pipeline
from laundry_control.hardware.fake import FakeRecorder
import pytest


@pytest.fixture
def stages(monkeypatch):
    """Stub every stage; `script` sets what each scan detects."""
    log = []
    state = {'detections': [], 'grasp_results': [], 'sweep_ok': True}

    monkeypatch.setattr(
        pipeline, 'build_model', lambda baseline: log.append('model') or 'S'
    )

    def preplanned(arm, gripper, limit=None, time_scale=1.0):
        log.append('sweep')
        return state['sweep_ok']

    def scan(arm, recorder, csv_path, scan_kwargs):
        log.append('scan')
        return True

    def detect(csv_path, baseline, params, surface=None):
        assert surface == 'S'  # fitted once, reused
        clusters = state['detections'].pop(0)
        return clusters, surface

    def plan(arm, clusters, surface):
        return None if clusters == ['unreachable'] else 'PLAN'

    def execute(arm, gripper, plan, drop=False):
        assert drop and plan == 'PLAN'
        log.append('grasp')
        return state['grasp_results'].pop(0) if state['grasp_results'] else True

    monkeypatch.setattr(pipeline, 'run_preplanned', preplanned)
    monkeypatch.setattr(pipeline, 'run_scan', scan)
    monkeypatch.setattr(pipeline, 'detect_scan', detect)
    monkeypatch.setattr(pipeline, 'plan_grasp', plan)
    monkeypatch.setattr(pipeline, 'execute_plan', execute)
    monkeypatch.setattr(pipeline, 'describe_plan', lambda plan: plan)
    monkeypatch.setattr(pipeline, 'go_to', lambda arm, name, **k: True)
    monkeypatch.setattr(pipeline, 'open_for_scan', lambda g, dry_run=False: True)

    return log, state


def _clear(**kwargs):
    return pipeline.run_clear(
        None, None, None, 'baselines', {}, {},
        scan_path=lambda i: f'clear_{i}.csv', **kwargs
    )


def test_sweeps_then_grasps_until_a_scan_finds_nothing(stages):
    log, state = stages
    state['detections'] = [['a', 'b'], ['b'], []]

    assert _clear()
    assert log == [
        'model', 'sweep', 'scan', 'grasp', 'scan', 'grasp', 'scan',
    ]


def test_no_grabs_starts_with_a_scan(stages):
    log, state = stages
    state['detections'] = [[]]

    assert _clear(sweep=False)
    assert log == ['model', 'scan']


def test_a_failed_sweep_stops_before_scanning(stages):
    log, state = stages
    state['sweep_ok'] = False

    assert not _clear()
    assert log == ['model', 'sweep']


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
    assert log == ['model', 'scan']


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
