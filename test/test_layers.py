"""The package's layering: no module imports from a layer above its own."""

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / 'laundry_control'

# Lower numbers sit lower: a module may import from its own layer and
# below. hardware and arm share a layer but must not import each other.
LAYERS = {
    'config': 0,
    'bucket': 1,
    'hardware': 2,
    'arm': 2,
    'scan': 3,
    'perception': 4,
    'grasp': 5,
    'pipeline': 6,
    'cli': 7,
}


def _imports(path):
    """Yield (line, top-level part imported) for each relative import."""
    parts = path.relative_to(PACKAGE).with_suffix('').parts
    package = list(parts[:-1])

    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue

        base = package[:len(package) - (node.level - 1)]
        module = base + (node.module.split('.') if node.module else [])

        if module:
            yield node.lineno, module[0]
        else:
            for alias in node.names:
                yield node.lineno, alias.name


def _modules():
    return sorted(
        path for path in PACKAGE.rglob('*.py') if path.name != '__init__.py'
        or path.parent != PACKAGE
    )


def test_every_top_level_part_has_a_layer():
    top = {
        path.relative_to(PACKAGE).with_suffix('').parts[0]
        for path in _modules()
    }

    assert top <= set(LAYERS), f'add these to LAYERS: {top - set(LAYERS)}'


@pytest.mark.parametrize(
    'path', _modules(), ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_no_upward_imports(path):
    own = path.relative_to(PACKAGE).with_suffix('').parts[0]

    upward = [
        f'line {line}: imports {target}'
        for line, target in _imports(path)
        if target in LAYERS and target != own
        and LAYERS[target] >= LAYERS[own]
    ]

    assert not upward, (
        f'{own} (layer {LAYERS[own]}) may only import from lower layers '
        f'(see LAYERS): ' + '; '.join(upward)
    )
