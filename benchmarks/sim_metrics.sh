#!/usr/bin/env bash
#
# The detection and coverage numbers that need no arm and no ROS graph,
# from the baseline scans in baseline_scans/ (`laundry evaluate`):
#
#   false positives   leave-one-out over the empty baselines
#   recall            synthetic items injected into held-out empty scans
#   dark fabric       the same at 0.3 reflectivity
#   coverage          how much of the bucket a scan path reaches
#
# Usage:
#
#   benchmarks/sim_metrics.sh            # about 6 min on the Pi
#   benchmarks/sim_metrics.sh quick      # only the quick-scan runs
#
# Results: docs/performance.md. The timing of the motions is the other
# script, benchmarks/sim_timing.sh.

set -eo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=/dev/null
source "$REPO/env.sh" > /dev/null

only="${1:-all}"

section() {
    echo
    echo "################ $*"
}

if [ "$only" = all ] || [ "$only" = quick ]; then
    section "quick scan: false positives and recall (floor, lower wall)"
    laundry evaluate --quick --synthetic | sed -n '/^FALSE POSITIVES/,$p'

    section "quick scan: dark fabric (reflectivity 0.3)"
    laundry evaluate --quick --synthetic --reflectivity 0.3 \
        | sed -n '/^SYNTHETIC LAUNDRY/,$p'
fi

if [ "$only" = all ] || [ "$only" = full ]; then
    section "full scan: false positives and recall, all regions"
    laundry evaluate --full --synthetic --all-regions \
        | sed -n '/^FALSE POSITIVES/,$p'
fi

if [ "$only" = all ] || [ "$only" = coverage ]; then
    section "scan coverage, measured and simulated"
    laundry evaluate --coverage | sed -n '/^SCAN COVERAGE/,$p'
fi
