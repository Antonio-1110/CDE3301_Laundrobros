#!/usr/bin/env bash
#
# Bake the sweep's plan set for each fill level (config.FILL_LEVELS),
# on the isolated fake controller, as rebake.sh does - never the real
# arm:
#
#   ./rebake_fills.sh              # every level (~10 min each on the Pi)
#   ./rebake_fills.sh 0.5 0.75     # just these
#
# Each level's set is scan_plans/fill_<percent>/: grab_targets.yaml,
# retrieve.yaml and transfers.yaml. Its transfers.yaml starts as a copy
# of scan_plans/transfers.yaml (HOME, DROP and the other shared routes),
# then the level's grab routes, direct grab <-> DROP routes and timings
# are baked into it - retrieve, direct, timing, each its own rebake.sh
# (a fresh fake bring-up; one move_group re-padding the gripper for all
# three grew to 3.5 GB, 2026-10-21). Re-run it after re-baking the
# shared routes, or changing config.PILE (other than its top).
#
# Use a level with `laundry preplanned --fill 0.75` (or clear/run), or
# LAUNDRY_FILL=0.75 before any command.
#
# Never while the real bring-up is running: a bake can take the Pi's
# memory (2026-10-21, an OOM took the real stack down with it).

set -eo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
source "$REPO/env.sh" > /dev/null

if pgrep -x move_group > /dev/null && [ -z "$LAUNDRY_ALLOW_WITH_BRINGUP" ]; then
    echo "ERROR: a move_group is running (the real bring-up, or another"
    echo "       bake?). Stop it first, or wait."
    exit 1
fi

if [ $# -gt 0 ]; then
    levels=("$@")
else
    mapfile -t levels < <(python3 -c \
        'from laundry_control import config; print(*config.FILL_LEVELS, sep="\n")')
fi

problems=()

for level in "${levels[@]}"; do
    dir="$(LAUNDRY_FILL="$level" python3 -c \
        'from laundry_control import config; print(config.plan_set_dir())')"

    echo
    echo "################ FILL $level -> $dir ################"

    mkdir -p "$dir"
    cp "$REPO/scan_plans/transfers.yaml" "$dir/transfers.yaml"

    # A bake that reports problems (a grab out of reach, left out: the
    # others are saved) still needs its routes timed: carry on, and
    # list it at the end.
    for target in retrieve direct timing; do
        if ! LAUNDRY_FILL="$level" "$REPO/rebake.sh" "$target"; then
            problems+=("fill $level: $target")
        fi
    done
done

if [ ${#problems[@]} -gt 0 ]; then
    echo
    echo "Reported problems (see each bake's output above):"
    printf '  %s\n' "${problems[@]}"
fi

echo
echo "Baked: ${levels[*]}. Check each on the real arm (slowly) before relying on it:"
echo "  laundry preplanned --fill <level> --limit 3 --speed 0.3"
