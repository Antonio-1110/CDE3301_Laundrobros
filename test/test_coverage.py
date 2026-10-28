"""The simulated scan path (perception/coverage.py) against scan.pattern."""

from laundry_control.perception import coverage
import numpy as np
import pytest


def _angles_by_depth(step):
    """Return (depths, inward angles, outward angles), sorted by depth."""
    path = coverage.ScanPath(
        depth_m=0.42, entry_m=0.06, step_m=step, bottom_detour=False,
        samples_per_stroke=4,
    )
    origins, directions = coverage._stroke_beams(path)

    tool_z = coverage._unit(coverage.INTER_TOOL_Z)
    reference = coverage._unit(coverage.INTER_BORESIGHT)
    reference = reference - (reference @ tool_z) * tool_z
    across = directions - np.outer(directions @ tool_z, tool_z)
    angles = np.degrees(np.arctan2(
        np.cross(reference, across) @ tool_z, across @ reference
    ))
    depths = origins @ tool_z

    half = len(origins) // 2
    inward, outward = np.argsort(depths[:half]), half + np.argsort(depths[half:])

    return depths[inward], angles[inward], angles[outward]


@pytest.mark.parametrize('step', [0.03, 0.06, 0.12])
def test_outward_strokes_cross_the_inward_ones_at_every_depth(step):
    # scan.pattern's turnaround puts the outward pass on the other side
    # at each depth, whether the stroke count is even (3, 6 cm) or odd
    # (12 cm: 3 strokes), so the two passes cross.
    depths_in, inward, outward = _angles_by_depth(step)

    assert np.allclose(outward, -inward, atol=0.5)
    assert not np.allclose(outward, inward, atol=0.5)
    assert depths_in[0] < depths_in[-1]
