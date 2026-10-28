"""`--fill N`: the sweep's starting layer (config.PILE['layer_heights_m'])."""

from laundry_control import cli, config
import pytest


def _parsed(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_clear_starts_at_the_lowest_layer_unless_told_otherwise():
    assert _parsed('clear').fill == 1
    assert _parsed('clear', '--fill', '3').fill == 3


def test_preplanned_sweeps_every_layer_by_default():
    assert _parsed('preplanned').fill == len(config.PILE['layer_heights_m'])


def test_fill_is_one_of_the_layers():
    layers = len(config.PILE['layer_heights_m'])

    for bad in ('0', str(layers + 1), '0.5'):
        with pytest.raises(SystemExit):
            _parsed('preplanned', '--fill', bad)


def test_the_layers_climb_from_the_floor():
    heights = config.PILE['layer_heights_m']

    assert list(heights) == sorted(heights)
    assert heights[0] >= config.PILE['floor_clearance_m']
