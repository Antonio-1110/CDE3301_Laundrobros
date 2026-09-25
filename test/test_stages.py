"""Tests for the stage plumbing: targets JSON, fakes, CLI, grasp sequence."""

from dataclasses import dataclass

from laundry_control import cli, config
from laundry_control.grasp import execute
from laundry_control.grasp.targets_io import load_targets, save_targets
from laundry_control.hardware.fake import FakeRecorder
from laundry_control.perception.detect import ClusterSummary
from laundry_control.pipeline import run_scan
import numpy as np
import pytest


def _cluster(volume, centroid=(0.1, -0.4, 0.3), size=10):
    points = np.array([centroid, np.add(centroid, 0.01)])
    return ClusterSummary(
        points=points,
        centroid=np.array(centroid),
        size=size,
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        extent=points.max(axis=0) - points.min(axis=0),
        highest_point=points[1],
        mean_deviation_m=0.02,
        volume_m3=volume,
        max_intrusion_m=float('nan'),
        surface_extent_m=0.05,
        confident=False,
    )


class _Logger:

    def info(self, *_args, **_kwargs):
        pass


class _Node:

    def get_logger(self):
        return _Logger()


# ------------------------------------------------------------ targets


def test_targets_round_trip(tmp_path):
    clusters = [_cluster(2e-4), _cluster(1e-4, centroid=(0.2, -0.5, 0.25))]
    path = tmp_path / 'targets.json'

    save_targets(str(path), clusters, 'scan.csv', 'baselines', {'k_sigma': 4.0})

    loaded, document = load_targets(str(path))

    assert document['scan'] == 'scan.csv'
    assert document['baseline'] == 'baselines'
    assert len(loaded) == 2
    assert np.allclose(loaded[1].centroid, clusters[1].centroid)
    assert loaded[0].volume_m3 == pytest.approx(2e-4)
    # NaN survives as NaN (stored as JSON null).
    assert np.isnan(loaded[0].max_intrusion_m)
    assert loaded[0].confident is False


def test_targets_reject_unknown_version(tmp_path):
    path = tmp_path / 'targets.json'
    path.write_text('{"version": 99, "clusters": []}')

    with pytest.raises(ValueError, match='format'):
        load_targets(str(path))


# ------------------------------------------------------------ fakes


def test_fake_recorder_copies_source_on_final_save(tmp_path):
    source = tmp_path / 'saved.csv'
    source.write_text('x,y,z\n1,2,3\n')
    dest = tmp_path / 'out' / 'scan.csv'

    recorder = FakeRecorder(_Node(), str(source))

    assert recorder.set_csv_path(str(dest))
    recorder.save_async()
    assert not dest.exists()

    assert recorder.save_blocking()
    assert dest.read_text() == source.read_text()


def test_fake_recorder_requires_existing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        FakeRecorder(_Node(), str(tmp_path / 'missing.csv'))


# ------------------------------------------------------------ run_scan


class _StubRecorder:

    def __init__(self):
        self.paths = []
        self.cleared = False

    def set_csv_path(self, path):
        self.paths.append(path)
        return True

    def clear_blocking(self):
        self.cleared = True
        return True


def test_run_scan_restores_auto_naming_even_when_scan_fails(
    monkeypatch, tmp_path
):
    import laundry_control.pipeline as pipeline

    def exploding_scan(**_kwargs):
        raise RuntimeError('arm fault')

    monkeypatch.setattr(pipeline, 'scan', exploding_scan)

    recorder = _StubRecorder()
    target = tmp_path / 'scan.csv'

    with pytest.raises(RuntimeError):
        run_scan(object(), recorder, str(target), {})

    assert recorder.cleared
    assert recorder.paths == [str(target), '']


# ------------------------------------------------------------ grasp


@dataclass
class _Grasp:
    tcp_position: np.ndarray
    orientation: object = None
    grasp_point: np.ndarray = None
    sink_fraction_used: float = 0.0
    sink_amount_m: float = 0.0
    gap_m: float = 0.0
    reachability_fraction: float = 1.0


class _StubArm:

    def __init__(self, fail_pose=False):
        self.calls = []
        self.fail_pose = fail_pose

    def move_joints(self, joints):
        self.calls.append(('joints', tuple(joints)))
        return True

    def move_to_pose(self, x, y, z, orientation=None):
        self.calls.append(('pose', (x, y, z)))
        return not self.fail_pose


class _StubGripper:

    def __init__(self, arm, fail_open=False, fail_close=False):
        self.arm = arm
        self.fail_open = fail_open
        self.fail_close = fail_close

    def open_blocking(self):
        self.arm.calls.append(('gripper', 'open'))
        return not self.fail_open

    def close_blocking(self):
        self.arm.calls.append(('gripper', 'close'))
        return not self.fail_close


@pytest.fixture(autouse=True)
def _named_moves_through_stub(monkeypatch):
    """Route go_to() through the stub arm's move_joints, as before."""
    def fake_go_to(arm, name, **_kwargs):
        return arm.move_joints(config.get_named_pose(name))

    monkeypatch.setattr(execute, 'go_to', fake_go_to)


def _sequence(arm):
    names = {tuple(config.INTER): 'INTER', tuple(config.DROP): 'DROP'}
    return [
        names.get(value, 'POSE') if kind == 'joints'
        else ('POSE' if kind == 'pose' else value)
        for kind, value in arm.calls
    ]


def test_execute_grasp_with_drop_retracts_through_inter():
    arm = _StubArm()
    grasp = _Grasp(tcp_position=np.array([0.1, -0.3, 0.4]))

    assert execute.execute_grasp(arm, _StubGripper(arm), grasp, drop=True)

    assert _sequence(arm) == [
        'open', 'POSE', 'close', 'INTER', 'DROP', 'open', 'INTER',
    ]


def test_execute_grasp_without_drop_matches_old_retrieve():
    arm = _StubArm()
    grasp = _Grasp(tcp_position=np.array([0.1, -0.3, 0.4]))

    assert execute.execute_grasp(arm, _StubGripper(arm), grasp)

    assert _sequence(arm) == ['open', 'POSE', 'close', 'INTER']


def test_execute_grasp_stops_if_target_unreachable():
    arm = _StubArm(fail_pose=True)
    grasp = _Grasp(tcp_position=np.array([0.1, -0.3, 0.4]))

    assert not execute.execute_grasp(arm, _StubGripper(arm), grasp, drop=True)
    assert _sequence(arm) == ['open', 'POSE']


def test_execute_grasp_never_approaches_if_the_gripper_did_not_open():
    arm = _StubArm()
    grasp = _Grasp(tcp_position=np.array([0.1, -0.3, 0.4]))

    assert not execute.execute_grasp(
        arm, _StubGripper(arm, fail_open=True), grasp, drop=True
    )
    assert _sequence(arm) == ['open']


def test_execute_grasp_retracts_but_never_drops_if_not_closed():
    arm = _StubArm()
    grasp = _Grasp(tcp_position=np.array([0.1, -0.3, 0.4]))

    assert not execute.execute_grasp(
        arm, _StubGripper(arm, fail_close=True), grasp, drop=True
    )
    assert _sequence(arm) == ['open', 'POSE', 'close', 'INTER']


def test_preplanned_stops_when_the_gripper_fails(monkeypatch):
    from laundry_control import pipeline

    def fake_go_to(arm, name, **_kwargs):
        return arm.move_joints(config.get_named_pose(name))

    monkeypatch.setattr(pipeline, 'go_to', fake_go_to)

    arm = _StubArm()

    assert not pipeline.run_preplanned(arm, _StubGripper(arm, fail_close=True))
    # INTER, the first RETRIEVE pose, close (fails), back to INTER - no DROP.
    assert _sequence(arm) == ['INTER', 'POSE', 'close', 'INTER']


def test_grasp_best_dry_run_never_approaches(monkeypatch):
    arm = _StubArm()

    monkeypatch.setattr(
        execute,
        'compute_grasp_target',
        lambda cluster, surface, arm: _Grasp(tcp_position=np.zeros(3)),
    )

    assert execute.grasp_best(
        arm, _StubGripper(arm), [_cluster(1e-4)], None, dry_run=True
    )
    assert _sequence(arm) == ['INTER']


def test_plan_first_reachable_skips_unreachable_clusters():
    big, small = _cluster(3e-4), _cluster(1e-4)

    def planner(cluster, surface, arm):
        return None if cluster is big else _Grasp(tcp_position=np.zeros(3))

    chosen, grasp = execute.plan_first_reachable(
        [small, big], None, None, planner
    )

    assert chosen is small
    assert grasp is not None


# ------------------------------------------------------------ CLI


def _parse(argv):
    own, forwarded = cli._split_forwarded(argv)
    args = cli.build_parser().parse_args(own)
    args.forwarded_scan_args = forwarded
    return args


def test_cli_move_named_pose_and_joints():
    assert _parse(['move', 'inter']).move_kind == 'inter'

    args = _parse(['move', '--velocity', '0.2', 'joints'] + ['0'] * 7)
    assert args.move_kind == 'joints'
    assert args.velocity == 0.2


def test_cli_scan_options_keep_scan_move_names():
    args = _parse(['scan', '--save', 'a.csv', '--velocity', '0.05',
                   '--sweep', '180'])

    from laundry_control.scan.pattern import scan_kwargs_from_args

    kwargs = scan_kwargs_from_args(args)

    assert kwargs['velocity'] == 0.05
    assert kwargs['sweep_deg'] == 180.0
    assert args.save == 'a.csv'


def test_cli_baseline_collect_forwards_scan_args():
    args = _parse(['baseline', 'collect', '--count', '3', '--', '--depth',
                   '0.40'])

    assert args.count == 3
    assert args.forwarded_scan_args == ['--depth', '0.40']


def test_cli_detect_params_reach_detector_names():
    args = _parse(['detect', 'scan.csv', '--k-sigma', '5', '-o', 't.json'])

    params = cli._detect_params(args)

    assert params['k_sigma'] == 5.0
    assert set(params) == {
        'k_sigma', 'abs_floor_m', 'cluster_radius_m', 'min_cluster_size',
        'min_extent_m', 'min_volume_m3',
    }


def test_cli_gripper_rejects_out_of_range_angle():
    with pytest.raises(ValueError):
        cli.cmd_gripper(_parse(['gripper', '400', '--fake-hardware']))


def test_cli_gripper_fake_angle_does_not_touch_gpio(capsys):
    assert cli.cmd_gripper(_parse(['gripper', '90', '--fake-hardware'])) == 0
    assert 'fake gripper' in capsys.readouterr().out
