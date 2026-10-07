#!/usr/bin/env bash
#
# Re-bake every stored motion for the bucket in config.OBSTACLES, on
# an isolated MoveIt fake controller - never the real arm:
#
#   1. start the fake bring-up on its own ROS domain, localhost only,
#   2. `laundry plan bake` - INTER/BOTTOM from the bucket, then the
#      grab grid, the transfers and the end scan from that INTER,
#   3. `laundry scene check`,
#   4. shut the fake bring-up down again.
#
# Usage:
#
#   ./rebake.sh            # everything (~10 min on the Pi)
#   ./rebake.sh poses      # any `laundry plan bake` target: poses,
#                          # retrieve, transfers, endcap, all
#
# It writes scan_plans/ in this checkout; `git checkout scan_plans/`
# undoes it. Run it after moving the bucket (config.OBSTACLES) or
# changing config.BUCKET_POSES, the padding, config.RETRIEVE_GRID or a
# recorded pose - not before every run: replaying baked motions is
# instant, and nothing here changes unless one of those does.
#
# What it cannot do, and you then must (docs/operating.md, "After moving the
# bucket"): check the new INTER on the real arm by eye
# (`laundry move inter --speed 0.3`), collect new baselines, commit.

set -eo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# The fake stack's own graph: nothing it or `laundry` sends here can
# reach the real arm's bring-up (default domain), running or not.
export ROS_DOMAIN_ID="${LAUNDRY_BAKE_DOMAIN_ID:-73}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

# shellcheck source=/dev/null
source "$REPO/env.sh" > /dev/null

LOG="$(mktemp --tmpdir laundry_rebake_fake.XXXXXX.log)"
READY_TIMEOUT_S=240
LAUNCH_PID=''

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
    echo "         LAUNDRY_BAKE_DOMAIN_ID=74 $0 $*"
    exit 1
fi

echo "Starting the fake bring-up (ROS domain $ROS_DOMAIN_ID, localhost only)."
echo "Its log: $LOG"

setsid ros2 launch laundry_control laundry_bringup.launch.py \
    fake:=true rviz:=false > "$LOG" 2>&1 &
LAUNCH_PID=$!

echo "Waiting for MoveIt and the fake arm at HOME (about 40 s)..."

for _ in $(seq "$READY_TIMEOUT_S"); do
    if grep -q 'Fake arm moved to HOME' "$LOG"; then
        break
    fi

    if ! kill -0 "$LAUNCH_PID" 2> /dev/null \
        || grep -qE 'laundry-[0-9]+\]: process has died|Traceback' "$LOG"; then
        echo "ERROR: the fake bring-up failed; the end of its log:"
        tail -20 "$LOG"
        exit 1
    fi

    sleep 1
done

if ! grep -q 'Fake arm moved to HOME' "$LOG"; then
    echo "ERROR: the fake bring-up was not ready after ${READY_TIMEOUT_S} s."
    exit 1
fi

echo
echo "=== laundry plan bake ${*:-all} ==="

bake_status=0
laundry plan bake "$@" || bake_status=$?

echo
echo "=== laundry scene check ==="

laundry scene check || true

echo
echo "=== Changed plan files ==="
git -C "$REPO" status --short -- scan_plans/

echo
if [ "$bake_status" -ne 0 ]; then
    echo "The bake reported problems (see above): a pose or route that could"
    echo "not be found. Everything that could be was still saved."
fi

cat << 'EOF'
Next, on the real rig (normal bring-up, default ROS domain):
  1. laundry move inter --speed 0.3   (hand on the e-stop; check the
     gripper points down the middle of the bucket), laundry check-flange
  2. laundry baseline collect --archive   (empty bucket)
  3. commit config/, scan_plans/ and baseline_scans/ together
EOF

exit "$bake_status"
