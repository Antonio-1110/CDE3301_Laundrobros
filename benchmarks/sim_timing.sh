#!/usr/bin/env bash
#
# Time the scan and retrieve motions on an isolated MoveIt fake
# controller - never the real arm:
#
#   1. start the fake bring-up on its own ROS domain, localhost only,
#   2. run each scenario with `--fake-hardware`, stamping every output
#      line with the seconds since the command started,
#   3. print the stage times and the wall time of each,
#   4. shut the fake bring-up down again.
#
# Usage:
#
#   benchmarks/sim_timing.sh                 # quick full step06 step12 sweep
#   benchmarks/sim_timing.sh quick full      # only those scenarios
#
# Scenarios:
#
#   quick     laundry scan --quick (the default scan, about 1.6 min)
#   full      laundry scan --full (adds the end scan)
#   step06    quick scan with --step 0.06
#   step12    quick scan with --step 0.12
#   sweep     laundry preplanned: every grab of the generated sweep
#   run       laundry run on a scan with an item in it (needs
#             LAUNDRY_SCAN=<a scan csv that detects something>)
#
# The fake controller runs the planned trajectories in real time, so
# the motion times match the rig's. What it leaves out: the gripper
# (open or close is 1.5 s on the rig, SETTLE_SEC, plus the ESP32 ack)
# and the real arm's tracking. Results: docs/performance.md.
#
# Environment: SIM_DOMAIN_ID (default 77), SIM_OUT (log directory,
# default a new temp directory), BASELINE_SCAN (the csv the fake
# recorder replays; default: the first full baseline).

set -eo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# The fake stack's own graph: nothing it or `laundry` sends here can
# reach the real arm's bring-up (default domain), running or not.
export ROS_DOMAIN_ID="${SIM_DOMAIN_ID:-77}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export PYTHONUNBUFFERED=1

# shellcheck source=/dev/null
source "$REPO/env.sh" > /dev/null

OUT="${SIM_OUT:-$(mktemp -d --tmpdir laundry_sim_timing.XXXXXX)}"
mkdir -p "$OUT"
BASELINE_SCAN="${BASELINE_SCAN:-$(ls "$REPO"/baseline_scans/baseline_*.csv | head -1)}"
READY_TIMEOUT_S=240
LAUNCH_PID=''

stamp() {
    python3 -c '
import sys, time
t0 = time.time()
for line in sys.stdin:
    sys.stdout.write(f"[{time.time() - t0:7.1f}s] {line}")
    sys.stdout.flush()
print(f"[{time.time() - t0:7.1f}s] END", flush=True)
'
}

stop_fake() {
    if [ -n "$LAUNCH_PID" ] && kill -0 "$LAUNCH_PID" 2> /dev/null; then
        echo
        echo "Stopping the fake bring-up..."
        # The launch is its own process group (setsid): stop all of it.
        kill -INT -- "-$LAUNCH_PID" 2> /dev/null || true

        for _ in $(seq 20); do
            kill -0 "$LAUNCH_PID" 2> /dev/null || break
            sleep 1
        done

        kill -TERM -- "-$LAUNCH_PID" 2> /dev/null || true
    fi
}

trap stop_fake EXIT
trap 'exit 130' INT TERM

if timeout 10 ros2 node list 2> /dev/null | grep -q .; then
    echo "ERROR: ROS domain $ROS_DOMAIN_ID already has nodes running (another"
    echo "       fake bring-up?). Stop them, or pick another domain:"
    echo "         SIM_DOMAIN_ID=78 $0 $*"
    exit 1
fi

echo "Starting the fake bring-up (ROS domain $ROS_DOMAIN_ID, localhost only)."
echo "Logs: $OUT"

setsid ros2 launch laundry_control laundry_bringup.launch.py \
    fake:=true rviz:=false > "$OUT/bringup.log" 2>&1 &
LAUNCH_PID=$!

echo "Waiting for MoveIt and the fake arm at HOME (about 40 s)..."

for _ in $(seq "$READY_TIMEOUT_S"); do
    if grep -q 'Fake arm moved to HOME' "$OUT/bringup.log"; then
        break
    fi

    if ! kill -0 "$LAUNCH_PID" 2> /dev/null \
        || grep -qE 'laundry-[0-9]+\]: process has died|Traceback' "$OUT/bringup.log"; then
        echo "ERROR: the fake bring-up failed; the end of its log:"
        tail -20 "$OUT/bringup.log"
        exit 1
    fi

    sleep 1
done

if ! grep -q 'Fake arm moved to HOME' "$OUT/bringup.log"; then
    echo "ERROR: the fake bring-up was not ready after ${READY_TIMEOUT_S} s."
    exit 1
fi

# Every scan starts and ends at INTER; get there first so the first
# scenario is not charged the trip from HOME.
laundry move inter --speed 1 > /dev/null 2>&1 || true

# run_scenario NAME STAGE_REGEX COMMAND...
run_scenario() {
    local name="$1" stages="$2"
    shift 2
    local start end
    start=$(date +%s.%N)
    "$@" 2>&1 | stamp > "$OUT/$name.log" || true
    end=$(date +%s.%N)

    echo
    echo "=== $name ($*)"
    grep -E "$stages" "$OUT/$name.log" | cut -c1-140 || true
    printf 'wall time: %.1f s\n' "$(echo "$end - $start" | bc)"
}

SCAN_STAGES='MOVE TO INTER|\[IN 1/|END SCAN|\[OUT 1/|RETURN TO INTER|SCAN COMPLETE'
SWEEP_STAGES='Sensorless sweep|grab_[0-9]+:|Returning to INTER'
RUN_STAGES='==========|Trying cluster|Floor grab via'

scenarios=("$@")
[ "${#scenarios[@]}" -eq 0 ] && scenarios=(quick full step06 step12 sweep)

for scenario in "${scenarios[@]}"; do
    scan_args=(--fake-hardware --scan-from "$BASELINE_SCAN" --save "$OUT/$scenario.csv")

    case "$scenario" in
        quick)  run_scenario quick  "$SCAN_STAGES" laundry scan --quick "${scan_args[@]}" ;;
        full)   run_scenario full   "$SCAN_STAGES" laundry scan --full "${scan_args[@]}" ;;
        step06) run_scenario step06 "$SCAN_STAGES" laundry scan --quick --step 0.06 "${scan_args[@]}" ;;
        step12) run_scenario step12 "$SCAN_STAGES" laundry scan --quick --step 0.12 "${scan_args[@]}" ;;
        sweep)  run_scenario sweep  "$SWEEP_STAGES" laundry preplanned --fake-hardware ;;
        run)
            if [ -z "${LAUNDRY_SCAN:-}" ]; then
                echo "Skipping 'run': set LAUNDRY_SCAN to a scan csv with an item in it."
                continue
            fi

            run_scenario run "$RUN_STAGES" \
                laundry run --fake-hardware --scan-from "$LAUNDRY_SCAN"
            ;;
        *)
            echo "Unknown scenario '$scenario' (quick full step06 step12 sweep run)."
            exit 2
            ;;
    esac
done

echo
echo "Full logs: $OUT"
