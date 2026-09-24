#!/usr/bin/env python3

"""
Save and load detected laundry clusters as a targets JSON file.

This is the hand-off between `laundry detect` (offline, no arm
needed) and `laundry grasp` (needs the arm): detection can be rerun
and inspected on any machine, and the grasp stage can be rerun
against the same targets without rescanning.

The file records which scan and which baseline set produced it,
because the grasp stage rebuilds the bucket model from that same
baseline set to work out how far it may sink into the pile. Planning
the grasp against a different model than the one detection used
would be a silent inconsistency, so load_targets() hands the
baseline path back and the grasp stage warns if it differs.

Format (version 1):

    {
      "version": 1,
      "scan": "<candidate scan CSV>",
      "baseline": "<baseline dir or CSV>",
      "detector": {<detect_on_points keyword arguments>},
      "clusters": [ {<ClusterSummary fields>}, ... ]   # best first
    }
"""

import json
import math

import numpy as np

from ..perception.detect import ClusterSummary

FORMAT_VERSION = 1

_ARRAY_FIELDS = (
    'points',
    'centroid',
    'bbox_min',
    'bbox_max',
    'extent',
    'highest_point',
)

_SCALAR_FIELDS = (
    'size',
    'mean_deviation_m',
    'volume_m3',
    'max_intrusion_m',
    'surface_extent_m',
    'confident',
)


def _json_float(value):
    """Return value as a float, or None for NaN (not valid JSON)."""
    value = float(value)
    return None if math.isnan(value) else value


def cluster_to_dict(cluster):
    """Serialise one ClusterSummary to plain JSON types."""
    data = {
        name: np.asarray(getattr(cluster, name), dtype=float).tolist()
        for name in _ARRAY_FIELDS
    }

    data['size'] = int(cluster.size)
    data['confident'] = bool(cluster.confident)

    for name in ('mean_deviation_m', 'volume_m3', 'max_intrusion_m',
                 'surface_extent_m'):
        data[name] = _json_float(getattr(cluster, name))

    # Optional extras added by newer detectors (confidence score,
    # grasp point) are carried through if present.
    for name in ('score', 'grasp_point'):
        value = getattr(cluster, name, None)

        if value is not None:
            data[name] = np.asarray(value, dtype=float).tolist()

    return data


def cluster_from_dict(data):
    """Rebuild a ClusterSummary from cluster_to_dict() output."""
    kwargs = {
        name: np.asarray(data[name], dtype=np.float64)
        for name in _ARRAY_FIELDS
    }

    for name in _SCALAR_FIELDS:
        value = data.get(name)

        if name == 'size':
            kwargs[name] = int(value)
        elif name == 'confident':
            kwargs[name] = bool(value if value is not None else True)
        else:
            kwargs[name] = float('nan') if value is None else float(value)

    cluster = ClusterSummary(**kwargs)

    for name in ('score', 'grasp_point'):
        if name in data and hasattr(cluster, name):
            value = data[name]
            setattr(
                cluster,
                name,
                np.asarray(value, dtype=np.float64)
                if isinstance(value, list)
                else float(value),
            )

    return cluster


def save_targets(path, clusters, scan, baseline, detector_params):
    """Write clusters (best first) and their provenance to a JSON file."""
    document = {
        'version': FORMAT_VERSION,
        'scan': scan,
        'baseline': baseline,
        'detector': detector_params,
        'clusters': [cluster_to_dict(cluster) for cluster in clusters],
    }

    with open(path, 'w') as handle:
        json.dump(document, handle, indent=2)


def load_targets(path):
    """
    Read a targets file.

    Returns (clusters, document): the ClusterSummary list in stored
    order, and the whole decoded document for its provenance fields.
    """
    with open(path) as handle:
        document = json.load(handle)

    version = document.get('version')

    if version != FORMAT_VERSION:
        raise ValueError(
            f'{path!r} is targets format {version!r}; this code reads '
            f'format {FORMAT_VERSION}.'
        )

    clusters = [cluster_from_dict(data) for data in document['clusters']]

    return clusters, document
