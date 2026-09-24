"""
Tests for the scan CSV schema.

The extended schema records the ToF RAY (raw range plus the beam's
origin) and the sweep joint angle, not just the endpoint. Endpoints
alone cannot be turned back into rays after the fact, so getting
this written correctly is a prerequisite for range-space residuals
and for extrinsic calibration.
"""

from builtin_interfaces.msg import Time
from laundry_control.scan.cloud_io import (
    EXTENDED_COLUMNS,
    load_scan_csv,
    load_xyz_csv,
    save_xyz_csv,
)
import pytest


def test_base_schema_round_trips(tmp_path):
    stamp = Time()
    stamp.sec = 7
    stamp.nanosec = 8

    path = tmp_path / 'base.csv'
    save_xyz_csv(str(path), [(0.1, 0.2, 0.3, stamp)])

    points = load_xyz_csv(str(path))

    assert len(points) == 1
    x, y, z, loaded_stamp = points[0]
    assert (x, y, z) == pytest.approx((0.1, 0.2, 0.3))
    assert (loaded_stamp.sec, loaded_stamp.nanosec) == (7, 8)


def test_extended_schema_round_trips(tmp_path):
    """
    The extended schema must round-trip through save and load.

    Regression: the extended-schema check compared the TUPLE length
    against 5 + len(EXTENDED_COLUMNS), but a base point is a
    4-tuple whose stamp expands to two columns. Nothing ever met
    the condition, so the ray columns were silently never written.
    """
    stamp = Time()
    path = tmp_path / 'extended.csv'

    save_xyz_csv(
        str(path),
        [(0.1, 0.2, 0.3, stamp, 0.12, 1.0, 2.0, 3.0, 1.23)],
    )

    columns = load_scan_csv(str(path))

    for name in EXTENDED_COLUMNS:
        assert name in columns, f'{name} missing from extended CSV'

    assert columns['raw_range'][0] == pytest.approx(0.12)
    assert columns['ox'][0] == pytest.approx(1.0)
    assert columns['j7'][0] == pytest.approx(1.23)


def test_extended_file_still_loads_as_plain_points(tmp_path):
    """
    Extended-schema files must still load as plain 4-tuples.

    scan_replay.py unpacks load_xyz_csv() results positionally as
    4-tuples, so the extra columns must not change that contract.
    """
    stamp = Time()
    path = tmp_path / 'extended.csv'

    save_xyz_csv(
        str(path),
        [(0.1, 0.2, 0.3, stamp, 0.12, 1.0, 2.0, 3.0, 1.23)],
    )

    points = load_xyz_csv(str(path))

    assert len(points[0]) == 4


def test_old_schema_file_reports_no_ray_columns(tmp_path):
    """
    Old-schema scans must report that they lack ray columns.

    A scan recorded before ray capture must be distinguishable from
    one with genuinely zero-valued rays, so callers can say "this
    scan predates ray recording" instead of analysing zeros.
    """
    stamp = Time()
    path = tmp_path / 'base.csv'
    save_xyz_csv(str(path), [(0.1, 0.2, 0.3, stamp)])

    columns = load_scan_csv(str(path))

    for name in EXTENDED_COLUMNS:
        assert name not in columns
