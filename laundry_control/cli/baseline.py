#!/usr/bin/env python3

"""Baselines and detector tuning: `laundry baseline ...`, `laundry evaluate`."""


def cmd_baseline(args):
    """Run `laundry baseline collect|promote|archive|list|restore`."""
    from .. import config
    from ..scan import baselines

    action = args.baseline_action

    if action == 'collect':
        return baselines.run_collect(args, args.forwarded_scan_args)

    dest = args.dest or config.baseline_dir()

    if action == 'promote':
        written, shelved = baselines.promote(
            args.sources, dest=dest, move=args.move,
            archive_first=args.archive,
        )

        if shelved:
            print(f'Archived the previous set to {shelved}')

        verb = 'Moved' if args.move else 'Copied'

        for path in written:
            print(f'{verb} -> {path}')

    elif action == 'archive':
        shelved, moved = baselines.archive(dest, label=args.label)

        if shelved is None:
            print(f'Nothing to archive in {dest}.')
        else:
            print(f'Archived {len(moved)} file(s) to {shelved}')
            print('The detector now has NO baseline set until you collect '
                  'or restore one.')

    elif action == 'restore':
        restored, shelved = baselines.restore(dest, args.label)

        if shelved:
            print(f'Archived the replaced set to {shelved}')

        print(f'Restored {len(restored)} file(s) from archive/{args.label}')

    if action in ('promote', 'archive', 'restore', 'list'):
        _print_baseline_sets(baselines, dest)

    return 0


def _print_baseline_sets(baselines, dest):
    from collections import Counter

    current = baselines.active_scans(dest)
    sessions = Counter(baselines.session_of(p) or '?' for p in current)

    print(f'\nCurrent set in {dest}: {len(current)} scan(s)')

    for session, count in sorted(sessions.items()):
        print(f'  {session}: {count}')

    archived = baselines.archived_sets(dest)

    if archived:
        print(f'Archived ({baselines.ARCHIVE_DIR}/):')

        for label, count in archived:
            print(f'  {label}: {count}')


def cmd_evaluate(args):
    """Run `laundry evaluate`."""
    from ..perception import evaluate

    return evaluate.run_evaluate(args)


def add_baseline_parser(subparsers):
    """Add `laundry baseline` and its actions."""
    from ..scan.baselines import add_collect_arguments

    baseline = subparsers.add_parser(
        'baseline', help='Collect, add, archive and restore empty-bucket baseline scans.'
    )
    baseline_actions = baseline.add_subparsers(
        dest='baseline_action', required=True
    )
    collect = baseline_actions.add_parser(
        'collect',
        help='Record N empty-bucket scans (options after -- go to `laundry scan`).',
    )
    add_collect_arguments(collect)
    promote = baseline_actions.add_parser(
        'promote',
        help=(
            'Add saved empty-bucket scans to the set, named '
            'baseline_<when taken>_NN.csv.'
        ),
    )
    promote.add_argument(
        'sources', nargs='+', help='Scan CSVs and/or directories of them.'
    )
    promote.add_argument(
        '--move', action='store_true', help='Move instead of copying.'
    )
    promote.add_argument(
        '--archive', action='store_true',
        help='Archive the current set first, so these REPLACE it.',
    )
    archive = baseline_actions.add_parser(
        'archive',
        help='Move the current set to baseline_scans/archive/<label>/.',
    )
    archive.add_argument(
        '--label', type=str, default=None,
        help="Archive folder name (default: the scans' session stamp).",
    )
    baseline_actions.add_parser(
        'list', help='Show the current set and the archived ones.'
    )
    restore = baseline_actions.add_parser(
        'restore',
        help='Make an archived set current again (archiving the current one).',
    )
    restore.add_argument('label', help='Archived set, as `list` names it.')

    for sub in baseline_actions.choices.values():
        if sub is not collect:
            sub.add_argument(
                '--dest', type=str, default=None,
                help='Baseline directory (default: <repo>/baseline_scans).',
            )
    baseline.set_defaults(func=cmd_baseline)


def add_evaluate_parser(subparsers):
    """Add `laundry evaluate`."""
    from ..perception.evaluate import add_evaluate_arguments

    evaluate = subparsers.add_parser(
        'evaluate', help='Measure false positives / recall of the detector.'
    )
    add_evaluate_arguments(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)
