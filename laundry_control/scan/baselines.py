#!/usr/bin/env python3

"""
Build the empty-bucket baseline set the detector models the bucket from.

    laundry baseline collect [--archive]  - run N empty-bucket scans
                                           in a row, straight into
                                           baseline_scans/
    laundry baseline promote X [--move]   - add already-saved scans
                                           (files or a directory)
    laundry baseline archive              - shelve the current set
    laundry baseline list                 - the current set and archives
    laundry baseline restore LABEL        - bring an archived set back

THE SET AND ITS ARCHIVE
-----------------------
The detector uses every *.csv directly in baseline_scans/ - nothing
else, subdirectories included. So an old set is shelved by moving it
into baseline_scans/archive/<label>/, where the label is the
collection session(s) it came from (baseline_<YYYYmmdd_HHMMSS>_NN.csv
-> 20260924_180836). Nothing is ever deleted.

`collect --archive` is the one-step way to REPLACE the set, e.g.
after the bucket moved: the new scans go to a staging directory
(baseline_scans/incoming_<session>/) first, and only when the whole
collection succeeded is the old set archived and the new one moved
in. An interrupted or failed collection therefore never leaves a mix
of old and new scans active - the two describe different scenes.
Without --archive, new scans are added to the current set.

WHY A COMMAND AND NOT JUST RUNNING `laundry scan` EIGHT TIMES
-------------------------------------------------------------
Two things have to be right, and both are easy to get wrong by hand:

  - Each run must land in its own file, named for which baseline it
    is. Each run is a `laundry scan --save <path>`, which pins
    scan_recorder_node's csv_path for that one scan and puts it
    back afterwards (in a finally, so it survives Ctrl-C too). If it
    were left pinned at a baseline path, the next ordinary scan
    would silently overwrite a baseline with a scan that has laundry
    in it - poisoning the model with no visible symptom.

  - A scan that produced nothing must not pass silently. A ToF node
    that died, or a sensor that returned nothing but out-of-range
    readings, yields a valid-looking CSV with no points in it. Each
    run is checked for a plausible point count before it counts
    toward the total, and a failed one is moved aside as .rejected.

Each scan is a SUBPROCESS rather than an in-process call: a scan
that wedges can be killed without taking the collection with it,
and every run starts from a fresh rclpy context.

BEFORE COLLECTING
-----------------
  1. The bucket must be EMPTY, and stay empty for every run.
  2. The gripper must be OPEN, on every run. A closed gripper
     changes the geometry the sensor sees and shows up later as
     laundry that is not there.
  3. Nothing may move - not the bucket, not the robot base. Every
     baseline has to describe the same physical scene.
  4. `ros2 launch laundry_control laundry_bringup.launch.py` must
     already be running.

Anything after -- is forwarded to `laundry scan`, so the baselines
can be captured with the same non-default scan settings the real
detection scans will use:

    laundry baseline collect --count 10 -- --depth 0.40 --sweep 150

HARDWARE ONLY (collect). promote is a plain file copy.
"""

import datetime
import glob
import os
import re
import shutil
import subprocess
import sys
import time

from .. import config

DEFAULT_COUNT = 8

ARCHIVE_DIR = 'archive'
STAGING_PREFIX = 'incoming_'

# baseline_<session>_NN.csv; the session is also the archive label.
_TIMESTAMP = re.compile(r'(\d{8}_\d{6})')

# A scan that comes back with fewer points than this did not really
# happen - the ToF node is down, the sensor is out of range, or the
# arm never moved. Far below a healthy scan (several thousand), so
# it only catches genuine failures.
DEFAULT_MIN_POINTS = 200


def count_points(csv_path):
    """Rows in a scan CSV, not counting the header."""
    if not os.path.isfile(csv_path):
        return 0

    with open(csv_path) as handle:
        return max(0, sum(1 for line in handle if line.strip()) - 1)


def run_one_scan(csv_path, scan_args, timeout_sec):
    """Run one `laundry scan --save csv_path`; True if it reported success."""
    command = [
        sys.executable,
        '-m',
        'laundry_control.cli',
        'scan',
        '--save',
        csv_path,
    ] + list(scan_args)

    try:
        result = subprocess.run(command, timeout=timeout_sec)

    except subprocess.TimeoutExpired:
        print(f'  ! scan exceeded {timeout_sec:.0f}s and was killed')
        return False

    return result.returncode == 0


def confirm_setup(count, dest):

    print('=' * 62)
    print(f'About to record {count} EMPTY-bucket baseline scans into')
    print(f'  {dest}')
    print('=' * 62)
    print()
    print('Check all of these before continuing:')
    print('  [ ] the bucket is EMPTY')
    print('  [ ] the gripper is OPEN')
    print('  [ ] the bucket and robot base will not be moved')
    print('  [ ] laundry_bringup.launch.py is running')
    print()

    answer = input("Type 'yes' to start: ").strip().lower()

    return answer == 'yes'


def _now():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def active_scans(directory):
    """Return the scans the detector would use: *.csv directly in directory."""
    return sorted(glob.glob(os.path.join(directory, '*.csv')))


def session_of(path):
    """Return the YYYYmmdd_HHMMSS stamp in a scan's file name, or None."""
    match = _TIMESTAMP.search(os.path.basename(path))

    return match.group(1) if match else None


def set_label(paths):
    """Name a set of scans after the collection session(s) it came from."""
    sessions = sorted({session_of(p) for p in paths} - {None})

    if not sessions:
        return f'archived_{_now()}'

    if len(sessions) == 1:
        return sessions[0]

    return f'{sessions[0]}..{sessions[-1]}'


def _free_directory(path):
    candidate, n = path, 2

    while os.path.exists(candidate):
        candidate, n = f'{path}_{n}', n + 1

    return candidate


def archive(directory, label=None):
    """
    Move the current set (and rejected scans) into archive/<label>/.

    Returns (archive directory, moved paths); (None, []) if there was
    nothing to move. label defaults to set_label() of the scans.
    """
    scans = active_scans(directory)
    rejected = sorted(glob.glob(os.path.join(directory, '*.csv.rejected')))

    if not scans and not rejected:
        return None, []

    target = _free_directory(
        os.path.join(directory, ARCHIVE_DIR, label or set_label(scans))
    )
    os.makedirs(target)

    for path in scans + rejected:
        os.replace(path, os.path.join(target, os.path.basename(path)))

    return target, scans + rejected


def archived_sets(directory):
    """Return [(label, number of scans)] under archive/, oldest first."""
    root = os.path.join(directory, ARCHIVE_DIR)

    if not os.path.isdir(root):
        return []

    return [
        (label, len(active_scans(os.path.join(root, label))))
        for label in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, label))
    ]


def restore(directory, label):
    """
    Make archive/<label>/ the current set again.

    The current set is archived first, so nothing is lost. Returns
    (restored paths, where the replaced set went or None).
    """
    source = os.path.join(directory, ARCHIVE_DIR, label)

    if not os.path.isdir(source):
        known = ', '.join(name for name, _n in archived_sets(directory))
        raise FileNotFoundError(
            f'No archived baseline set {label!r}. Archived: {known or "none"}.'
        )

    shelved, _moved = archive(directory)

    restored = []

    for name in sorted(os.listdir(source)):
        path = os.path.join(directory, name)
        os.replace(os.path.join(source, name), path)
        restored.append(path)

    os.rmdir(source)

    return restored, shelved


def collect(
    count,
    dest,
    scan_args,
    min_points,
    keep_going,
    timeout_sec,
    settle_sec,
    session=None,
):

    os.makedirs(dest, exist_ok=True)

    session = session or _now()

    written = []
    failed = []

    started = time.time()

    try:

        for index in range(1, count + 1):

            csv_path = os.path.join(
                dest, f'baseline_{session}_{index:02d}.csv'
            )

            elapsed = time.time() - started

            if written:
                per_scan = elapsed / len(written)
                remaining = per_scan * (count - index + 1)
                eta = f', ~{remaining / 60:.0f} min left'
            else:
                eta = ''

            print()
            print('-' * 62)
            print(
                f'Scan {index}/{count} -> {os.path.basename(csv_path)}'
                f'{eta}'
            )
            print('-' * 62)

            scan_started = time.time()
            ok = run_one_scan(csv_path, scan_args, timeout_sec)
            took = time.time() - scan_started

            points = count_points(csv_path)

            if not ok:
                reason = 'laundry scan reported failure'

            elif points < min_points:
                # A CSV with almost nothing in it is worse than no
                # CSV: it would quietly drag the model toward
                # whatever handful of points it does contain.
                reason = f'only {points} points (expected >= {min_points})'

            else:
                reason = None

            if reason is None:
                written.append(csv_path)
                print(
                    f'  ok - {points} points in {took / 60:.1f} min'
                )

            else:
                failed.append((index, reason))
                print(f'  FAILED - {reason}')

                if os.path.isfile(csv_path):
                    os.replace(csv_path, csv_path + '.rejected')
                    print(
                        '  moved aside as '
                        f'{os.path.basename(csv_path)}.rejected so it '
                        'cannot be picked up as a baseline'
                    )

                if not keep_going:
                    print('\nStopping after a failed scan.')
                    break

            if index < count and settle_sec > 0:
                print(f'  settling {settle_sec:.0f}s...')
                time.sleep(settle_sec)

    except KeyboardInterrupt:
        print('\n\nInterrupted.')

    return written, failed


def report(written, failed, dest):

    print()
    print('=' * 62)
    print(f'Collected {len(written)} baseline scan(s) into {dest}')
    print('=' * 62)

    for path in written:
        print(f'  {os.path.basename(path)}  ({count_points(path)} pts)')

    if failed:
        print()
        print(f'{len(failed)} scan(s) failed:')

        for index, reason in failed:
            print(f'  scan {index}: {reason}')

    total = len(
        [
            name
            for name in os.listdir(dest)
            if name.endswith('.csv')
        ]
    )

    print()
    print(f'{dest} now holds {total} baseline CSV(s) in total.')

    if total < 5:
        print(
            '\nThat is thin. The per-cell noise map needs about 8-10 '
            'empty scans before the detection threshold means much; '
            'below 5 most cells fall back to a pooled sigma. Run '
            'this again to top up - it adds to the set rather than '
            'replacing it.'
        )

    print()
    print('Next, check the model the detector builds from these:')
    print()
    print(f'  laundry evaluate --baseline {dest}')
    print()
    print('Look for all of:')
    print("  - 'closed end : modelled as a flat cap'")
    print('  - the fitted cone within a few mm of the configured bucket (laundry scene fit)')
    print('  - median cell count >= 5, empty cells near 0%')
    print('  - 0 false clusters across the leave-one-out folds')


def baseline_name(src_path, dest_dir):
    """
    Return a free baseline_<stamp>_NN.csv name in dest_dir for a scan.

    The stamp is the one in the scan's own file name (when it was
    taken), else its modification time; NN is the next free number.
    """
    stamp = session_of(src_path) or datetime.datetime.fromtimestamp(
        os.path.getmtime(src_path)
    ).strftime('%Y%m%d_%H%M%S')

    index = 1

    while True:
        name = f'baseline_{stamp}_{index:02d}.csv'

        if not os.path.exists(os.path.join(dest_dir, name)):
            return name

        index += 1


def _expand_sources(sources):
    paths = []

    for source in sources:
        if os.path.isdir(source):
            found = active_scans(source)

            if not found:
                raise FileNotFoundError(f'No .csv files in {source!r}.')

            paths.extend(found)

        elif os.path.isfile(source):
            paths.append(source)

        else:
            raise FileNotFoundError(f'No such scan CSV or directory: {source!r}')

    return paths


def promote(sources, dest=None, move=False, archive_first=False):
    """
    Add already-saved empty-bucket scans to the baseline set.

    For when a scan is only decided to be a good empty-bucket
    reference after the fact, or to adopt a staged collection. A
    baseline file is a completely ordinary scan CSV - nothing tags
    it as special on disk, so it stays viewable with `laundry
    replay`; being "a baseline" is purely a matter of sitting
    directly in the baseline directory.

    sources: scan CSVs and/or directories of them. Each is named
    baseline_<stamp>_NN.csv (baseline_name), copied - or moved, with
    move=True - into dest (default: config.baseline_dir()). With
    archive_first=True the current set is archived first, so these
    REPLACE it. Existing files are never overwritten.

    Returns (written paths, archive directory or None).
    """
    if isinstance(sources, str):
        sources = [sources]

    dest = dest or config.baseline_dir()
    paths = _expand_sources(sources)

    os.makedirs(dest, exist_ok=True)

    shelved = archive(dest)[0] if archive_first else None

    written = []

    for path in paths:
        target = os.path.join(dest, baseline_name(path, dest))

        if move:
            shutil.move(path, target)
        else:
            shutil.copyfile(path, target)

        written.append(target)

    return written, shelved


def add_collect_arguments(parser):
    """Add `laundry baseline collect` options to a parser."""
    parser.add_argument(
        '--count',
        type=int,
        default=DEFAULT_COUNT,
        help=f'How many empty scans to record (default: {DEFAULT_COUNT}).',
    )

    parser.add_argument(
        '--dest',
        type=str,
        default=None,
        help='Where to put them (default: <repo>/baseline_scans).',
    )

    parser.add_argument(
        '--min-points',
        type=int,
        default=DEFAULT_MIN_POINTS,
        help=(
            'Reject a scan that produced fewer points than this '
            f'(default: {DEFAULT_MIN_POINTS}).'
        ),
    )

    parser.add_argument(
        '--keep-going',
        action='store_true',
        help=(
            'Carry on after a failed scan instead of stopping. Off '
            'by default: a failure usually means something is wrong '
            'with the setup, and seven more scans will not fix it.'
        ),
    )

    parser.add_argument(
        '--settle',
        type=float,
        default=3.0,
        help=(
            'Seconds to wait between scans, so the arm comes fully '
            'to rest (default: 3).'
        ),
    )

    parser.add_argument(
        '--timeout',
        type=float,
        default=900.0,
        help=(
            'Give up on a single scan after this many seconds '
            '(default: 900).'
        ),
    )

    parser.add_argument(
        '--archive',
        action='store_true',
        help=(
            'REPLACE the current set: collect into a staging directory, '
            'and only once every scan succeeded, move the current set to '
            'baseline_scans/archive/<its session>/ and the new scans in. '
            'Without it, new scans are added to the current set.'
        ),
    )

    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip the confirmation prompts.',
    )


def run_collect(args, scan_args):
    """Run `laundry baseline collect`; returns a process exit code."""
    dest = args.dest or config.baseline_dir()

    if args.count < 1:
        print('--count must be at least 1.', file=sys.stderr)
        return 2

    replacing = getattr(args, 'archive', False)
    current = active_scans(dest)

    if replacing and current:
        print(
            f'--archive: the {len(current)} current baseline scan(s) move '
            f'to {os.path.join(dest, ARCHIVE_DIR, set_label(current))}/ once '
            'the new set is complete.'
        )

    if not args.yes and not confirm_setup(args.count, dest):
        print('Aborted.')
        return 1

    if scan_args:
        print(f"\nForwarding to laundry scan: {' '.join(scan_args)}")

    session = _now()
    target = (
        os.path.join(dest, STAGING_PREFIX + session) if replacing else dest
    )

    written, failed = collect(
        count=args.count,
        dest=target,
        scan_args=scan_args,
        min_points=args.min_points,
        keep_going=args.keep_going,
        timeout_sec=args.timeout,
        settle_sec=args.settle,
        session=session,
    )

    if replacing:
        written = _adopt_staged(target, dest, written, failed, args.count, args.yes)

    report(written, failed, dest)

    return 0 if written and not failed else 1


def _adopt_staged(staging, dest, written, failed, count, assume_yes):
    """Swap a staged collection in for the current set; return its paths."""
    complete = written and not failed and len(written) == count

    if not complete:
        if not written:
            print('\nNo usable scans; the current set is unchanged.')
            return []

        print(
            f'\nOnly {len(written)} of {count} scans succeeded; the current '
            'set is still active.'
        )

        answer = 'no' if assume_yes else input(
            f'Replace it with these {len(written)} scan(s) anyway? '
            "Type 'yes': "
        ).strip().lower()

        if answer != 'yes':
            print(
                f'Kept. The new scans are in {staging}; to use them later:\n'
                f'  laundry baseline promote {staging} --archive --move'
            )
            return []

    shelved, moved = archive(dest)

    if shelved:
        print(f'\nArchived the previous {len(moved)} file(s) to {shelved}')

    adopted = []

    for path in written:
        final = os.path.join(dest, os.path.basename(path))
        os.replace(path, final)
        adopted.append(final)

    leftovers = os.listdir(staging)

    if leftovers:
        print(f'Rejected scans stay in {staging}')
    else:
        os.rmdir(staging)

    return adopted
