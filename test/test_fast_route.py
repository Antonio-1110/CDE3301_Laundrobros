"""Re-baking a transfer for replay time (arm/fast_route.py)."""

from laundry_control.arm import fast_route
import numpy as np

INTER = np.zeros(7)
TARGET = np.array([1.0, 0.5, -0.5, 0.2, 0.0, 0.0, 0.8])


class _Arm:
    """Collides wherever joint 0 goes below -0.2 (a wall on one side)."""

    def first_invalid_state(self, waypoints, limits=None):
        for k, q in enumerate(np.asarray(waypoints)):
            if q[0] < -0.2:
                return k
        return None


def test_screening_keeps_the_fastest_clear_routes_fastest_first(monkeypatch):
    monkeypatch.setattr(fast_route, 'SAMPLES_PER_CONFIG', 60)
    monkeypatch.setattr(fast_route, 'RETREAT_DISTANCES_M', (0.0,))

    routes = fast_route.screen(
        _Arm(), INTER, TARGET, lambda d: None, log=lambda *_a: None
    )

    seconds = [fast_route._stop_time(path) for path in routes]

    assert 1 <= len(routes) <= fast_route.KEEP
    assert seconds == sorted(seconds)
    # The straight line is clear here and nothing beats it.
    assert len(routes[0]) == 2
    assert all(_Arm().first_invalid_state(path) is None for path in routes)
