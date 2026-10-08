#!/usr/bin/env python3

"""
Which part of a scan each reading came from: the strokes or the end scan.

A full scan is strokes in -> the tilting end scan -> strokes out; a
quick scan is the same strokes with the end scan left out (nothing is
recorded at the deepest point). So a full scan's STROKE readings are
exactly what a quick scan records, and one set of full-scan baselines
serves both: all of it for full scans, its strokes for quick ones.
The end scan can also run on its own after a quick scan (scan.pattern
end_scan_only); the quick scan's strokes plus that pass are then a
full scan's readings.

scan_recorder_node writes each reading's segment (STROKES or END) to
the scan CSV. Older files have no such column: for those, the end
scan is found from the beam - during the strokes it points square to
the bucket axis (tilt within ~1 deg on 2026-09-30's baselines), during
the end scan it leans 10-55 deg toward the closed end - as the one
contiguous stretch of readings tilted by more than TILT_THRESHOLD_DEG
(38.8-38.9 s in each of those baselines, nothing tilted outside it).
Files too old to have the ray columns are taken as all strokes.
"""

import numpy as np

STROKES = 0
END = 1

# Beam tilt from square-to-the-axis above which a reading belongs to
# the end scan (its rings are at 10, 25, 40 and 55 deg).
TILT_THRESHOLD_DEG = 6.0


def labels(columns, axis_dir=None):
    """
    Return each reading's segment (STROKES or END) as an int array.

    columns: scan.cloud_io.load_scan_csv() output. axis_dir: the
    bucket axis (default: the configured bucket's), only needed for
    files without a segment column.
    """
    count = len(columns['x'])

    recorded = np.asarray(columns.get('segment', []), dtype=float)

    if recorded.size == count and np.all(np.isfinite(recorded)):
        return recorded.astype(int)

    if not all(name in columns for name in ('ox', 'oy', 'oz')):
        return np.full(count, STROKES, dtype=int)

    if axis_dir is None:
        from ..bucket import seed_cone

        axis_dir = seed_cone().axis_dir

    ends = np.column_stack([columns[k] for k in ('x', 'y', 'z')])
    origins = np.column_stack([columns[k] for k in ('ox', 'oy', 'oz')])

    return labels_from_tilt(ends, origins, axis_dir)


def labels_from_tilt(ends, origins, axis_dir):
    """Label the one contiguous stretch of tilted beams END, the rest STROKES."""
    beams = np.asarray(ends, dtype=float) - np.asarray(origins, dtype=float)
    lengths = np.linalg.norm(beams, axis=1)
    lengths[lengths == 0.0] = 1.0

    cosine = np.clip((beams / lengths[:, None]) @ np.asarray(axis_dir), -1, 1)
    tilt = np.abs(90.0 - np.degrees(np.arccos(cosine)))

    result = np.full(len(beams), STROKES, dtype=int)
    tilted = np.flatnonzero(np.isfinite(tilt) & (tilt > TILT_THRESHOLD_DEG))

    if tilted.size:
        result[tilted[0]:tilted[-1] + 1] = END

    return result


def load_points(path, segments=None):
    """
    Return a scan CSV's (N, 3) points, only those of `segments` if given.

    segments: an iterable of STROKES/END, or None for every reading.
    """
    from .cloud_io import load_scan_csv

    columns = load_scan_csv(path)
    points = np.column_stack([columns[k] for k in ('x', 'y', 'z')])

    if segments is None:
        return points

    keep = np.isin(labels(columns), list(segments))

    return points[keep]


def end_scan_of(path):
    """
    Return the --end-scan kind a scan CSV was recorded as.

    'precession' (full) if any reading is labelled END, else 'none'
    (quick). Files too old to label (no segment or ray columns) are
    taken as full, which is how they were always judged.
    """
    from .cloud_io import load_scan_csv

    columns = load_scan_csv(path)

    if 'segment' not in columns and not all(
        name in columns for name in ('ox', 'oy', 'oz')
    ):
        return 'precession'

    return 'precession' if np.any(labels(columns) == END) else 'none'


def for_end_scan(end_scan):
    """Return the segments a scan of this --end-scan kind records."""
    return (STROKES,) if end_scan == 'none' else None
