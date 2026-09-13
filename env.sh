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
# Information
# -------------------------------------------------------

echo "ROS workspace : $WS_DIR"
echo "ROS distro    : $ROS_DISTRO"
echo "Python        : $(which python3)"