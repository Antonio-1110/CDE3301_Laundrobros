#!/usr/bin/env bash

# Location of this repository/package
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Assumes package is cloned into:
#
#   <workspace>/src/CDE3301_Laundrobros/
#
WS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ROS distribution
ROS_DISTRO="jazzy"

# -------------------------------------------------------
# 1. Source base ROS 2
# -------------------------------------------------------

ROS_SETUP="/opt/ros/$ROS_DISTRO/setup.bash"

if [ ! -f "$ROS_SETUP" ]; then
    echo "ERROR: ROS 2 $ROS_DISTRO not found at $ROS_SETUP"
    return 1 2>/dev/null || exit 1
fi

source "$ROS_SETUP"


# -------------------------------------------------------
# 2. Activate workspace virtual environment
# -------------------------------------------------------

VENV="$WS_DIR/.venv"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: Virtual environment not found:"
    echo "  $VENV"
    echo
    echo "Create it with:"
    echo "  cd $WS_DIR"
    echo "  python3 -m venv --system-site-packages .venv"
    return 1 2>/dev/null || exit 1
fi

source "$VENV/bin/activate"


# -------------------------------------------------------
# 3. Source built workspace, if available
# -------------------------------------------------------

if [ -f "$WS_DIR/install/setup.bash" ]; then
    source "$WS_DIR/install/setup.bash"
fi


# -------------------------------------------------------
# 4. Put `laundry` (and the node executables) on PATH
#
# ament_python installs console scripts under lib/<package>/, where
# only `ros2 run` looks. Adding it here means `laundry scan` works
# as well as `ros2 run laundry_control laundry scan`.
# -------------------------------------------------------

LAUNDRY_BIN="$WS_DIR/install/laundry_control/lib/laundry_control"

if [ -d "$LAUNDRY_BIN" ]; then
    case ":$PATH:" in
        *":$LAUNDRY_BIN:"*) ;;
        *) export PATH="$LAUNDRY_BIN:$PATH" ;;
    esac
fi


# -------------------------------------------------------
# 5. Build with the venv's Python, not /usr/bin/colcon's
#
# /usr/bin/colcon always runs under /usr/bin/python3, whatever venv is
# active, and ament_python writes the interpreter colcon ran under
# into every executable's shebang. Built that way, `ros2 run
# laundry_control tof_sensor` starts under system Python, which does
# NOT have the pip-installed hardware libraries (gpiozero, lgpio,
# adafruit_vl53l0x) - tof_sensor crashes and every gripper_node
# service call fails. Running colcon as `python3 -m colcon` from the
# venv writes the venv's interpreter instead.
# -------------------------------------------------------

colcon() {
    python3 -m colcon "$@"
}

LAUNDRY_EXE="$LAUNDRY_BIN/laundry"

if [ -f "$LAUNDRY_EXE" ] && ! head -1 "$LAUNDRY_EXE" | grep -q "$VENV/"; then
    echo "WARNING: laundry_control was built with $(head -1 "$LAUNDRY_EXE" | cut -c3-),"
    echo "         not the venv's Python, so the hardware nodes cannot import"
    echo "         gpiozero/adafruit. Rebuild from this shell:"
    echo "           cd $WS_DIR && rm -rf build/laundry_control install/laundry_control"
    echo "           colcon build --symlink-install --packages-select laundry_control"
fi


# -------------------------------------------------------
# Information
# -------------------------------------------------------

echo "ROS workspace : $WS_DIR"
echo "ROS distro    : $ROS_DISTRO"
echo "Python        : $(which python3)"