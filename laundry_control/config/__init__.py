#!/usr/bin/env python3

"""
Every hand-measured number the package depends on, in one place.

They are the values most likely to need changing when the rig is
touched (a remounted sensor, a re-recorded pose, a moved bucket, a
different machine), so they live here and nowhere else. One module
per topic:

    poses.py     recorded joint poses (HOME, INTER, DROP, ...) and
                 where INTER/BOTTOM sit relative to the bucket
    grabs.py     the grab search for detected items
    scene.py     the bucket and table poses, and collision padding
    robot.py     MoveIt/xarm_ros2 names, joint limits, planners,
                 speed limits
    hardware.py  the ToF sensor, the gripper servo, the ESP32 link

Everything is re-exported here, so code reads it as config.NAME
and never needs to know which file a value is in. This module adds
the lookups that combine those values with the baked plan files
(named poses) and the data locations.

No ROS dependency and no I/O at import time, so anything - tests
included - can import it freely.
"""

import os

from .grabs import *  # noqa: F401,F403
from .hardware import *  # noqa: F401,F403
from .poses import *  # noqa: F401,F403
from .robot import *  # noqa: F401,F403
from .scene import *  # noqa: F401,F403

# =============================================================
# NAMED POSES
#
# `laundry move <name>` and every stage look poses up here: the
# recorded ones (config/poses.py), with INTER and BOTTOM replaced by
# the ones derived from the bucket once baked, plus the generated
# grab_NN poses.
# =============================================================


def recorded_poses():
    """Return the hand-recorded poses as {lower-case name: joints}."""
    return {
        name.lower().removesuffix('_recorded'): list(globals()[name])
        for name in RECORDED_POSE_NAMES  # noqa: F405
    }


def named_poses():
    """
    Return every named pose as {lower-case name: joint list}.

    The recorded poses, with INTER and BOTTOM replaced by the ones
    derived from the bucket (scan_plans/bucket_poses.yaml) if they
    have been baked, plus the generated grab_NN poses (the approach
    pose above each grab point) from scan_plans/retrieve.yaml, if it
    has been baked.
    """
    poses = recorded_poses()
    poses.update(derived_bucket_poses())
    poses.update(generated_grab_poses())

    return poses


def get_named_pose(name):
    """
    Look up a recorded pose by name, case-insensitively.

    Returns a fresh list of 7 joint angles in radians. Raises
    KeyError listing what IS available, so a typo is reported before
    anything waits on MoveIt.
    """
    poses = named_poses()

    key = name.strip().lower()

    if key not in poses:
        raise KeyError(
            f"Unknown arm pose '{name}'. "
            f"Available: {', '.join(sorted(poses))}"
        )

    return poses[key]


def _read_plan(path):
    """Return a baked plan file's document, or None if it is missing."""
    if not os.path.isfile(path):
        return None

    import yaml

    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def generated_grab_poses():
    """Return {grab_NN: approach joints} from retrieve.yaml, or {}."""
    document = _read_plan(retrieve_plan_path()) or {}

    return {
        grab['name']: list(grab['approach'])
        for grab in document.get('grabs', [])
    }


def derived_bucket_poses():
    """Return {inter/bottom: joints} derived from the bucket, or {}."""
    document = _read_plan(bucket_poses_path()) or {}

    return {
        name: list(pose['joints'])
        for name, pose in document.get('poses', {}).items()
    }


# =============================================================
# DATA LOCATIONS
#
# Saved scans must live in the SOURCE tree, never under install/ or
# build/: both get wiped by `rm -rf build install log` before a
# clean rebuild, which would silently delete every scan.
#
# The source tree is found from this file's own real path. With
# `colcon build --symlink-install` (what the README prescribes)
# the installed module resolves back to the source checkout, so this
# works on any machine and any workspace name. Set LAUNDRY_DATA_DIR
# to override it (e.g. for a non-symlink install).
# =============================================================

DATA_DIR_ENV = 'LAUNDRY_DATA_DIR'


def repo_root():
    """
    Return the CDE3301_Laundrobros checkout that data lives under.

    Order: $LAUNDRY_DATA_DIR, then the source tree this module was
    loaded from (if it looks like one), then the current directory.
    """
    override = os.environ.get(DATA_DIR_ENV)

    if override:
        return os.path.abspath(os.path.expanduser(override))

    # <repo>/laundry_control/config/__init__.py
    here = os.path.dirname(os.path.realpath(__file__))
    candidate = os.path.dirname(os.path.dirname(here))

    if os.path.isfile(os.path.join(candidate, 'package.xml')):
        return candidate

    return os.getcwd()


def baseline_dir():
    """
    Directory of the empty-bucket baseline scans the detector models.

    Full scans, one set for both kinds of scan: a quick scan is modelled
    from their strokes alone (scan/segments.py).
    """
    return os.path.join(repo_root(), 'baseline_scans')


def scan_records_dir():
    """Directory ordinary scans are saved under."""
    return os.path.join(repo_root(), 'scan_records')


def bucket_poses_path():
    """Return <repo>/scan_plans/bucket_poses.yaml (see arm/bucket_poses.py)."""
    return os.path.join(repo_root(), 'scan_plans', 'bucket_poses.yaml')


def retrieve_plan_path():
    """Return <repo>/scan_plans/retrieve.yaml (see grasp/retrieve_grid.py)."""
    return os.path.join(repo_root(), 'scan_plans', 'retrieve.yaml')


def grab_targets_path():
    """Return <repo>/scan_plans/grab_targets.yaml (see grasp/grab_targets.py)."""
    return os.path.join(repo_root(), 'scan_plans', 'grab_targets.yaml')
