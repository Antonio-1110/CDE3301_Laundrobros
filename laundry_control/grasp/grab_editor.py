#!/usr/bin/env python3

"""
Place the sweep's grabs by dragging them in RViz (`laundry plan edit-grabs`).

Each grab in scan_plans/grab_targets.yaml (grasp/grab_targets.py) is
shown in the bucket as the gripper it would be: a ball where the claw
closes, the 15 cm gripper back along the tool axis, and a thin rod
for the approach above it. Drag:

    the arrows        deeper / shallower along the bucket axis
    the ball          around the floor and up from it (the drag stays
                      in the bucket's cross-section)
    the ring          tilt toward the closed end or the mouth
    the cyan cube     the approach distance, along the tool axis

Only the grabs ticked under the SHOW button are drawn in full (at
start, #1); the rest are small numbered dots - click one to show it.
Right-click a grab to insert one after it, delete or hide it, show
only it, or change the visiting order. The CHECK button solves every grab exactly as the
bake would (IK, approach pose, straight descent, gripper padded) and
colours it green, or red with the reason in its label - it needs
MoveIt running (fake or real; nothing moves), and on the real arm's
bring-up it pads the gripper in its planning scene for the few
seconds it takes, as the bake does. SAVE writes the file; then
`./rebake.sh retrieve`.

Colours: grey not checked, orange not checked but the wrist or the
approach is within WRIST_MARGIN_M of the wall (it probably will not
fit), green reachable, red not.

The bucket is drawn as a green wireframe, and the laundry's level top
at --fill (default 2/3 of the drum's height) as a tan outline: what
the sweep aims into. Both are lines, so they do not catch clicks;
switch off RViz's solid Obstacles (planning scene) display while
dragging, or the bucket mesh takes the clicks meant for the grabs.

Needs only a ROS graph and RViz with the InteractiveMarkers display on
namespace /grab_editor and a MarkerArray display on
/grab_editor/scene (rviz/scan_visualization.rviz has both).
"""

import argparse
import threading

from geometry_msgs.msg import Point, Pose, Quaternion
from interactive_markers import InteractiveMarkerServer, MenuHandler
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
    MarkerArray,
)

from . import grab_targets
from .. import config
from ..arm.geometry import look_at_quaternion
from ..perception.bucket_model import _axis_basis, seed_cone

NAMESPACE = 'grab_editor'
SCENE_TOPIC = 'grab_editor/scene'

DEFAULT_FILL_FRACTION = 2.0 / 3.0

# Under this much room between the wall and the wrist (at the grab or
# at the approach pose), an unchecked grab is shown orange: the
# flange is ~4 cm in radius and the arm links are padded on top.
WRIST_MARGIN_M = 0.05

_COLOURS = {
    'unchecked': (0.65, 0.65, 0.72, 0.85),
    'tight': (1.0, 0.55, 0.1, 0.9),
    'ok': (0.2, 0.8, 0.3, 0.9),
    'unreachable': (0.9, 0.15, 0.15, 0.9),
}

_HALF = np.sqrt(0.5)

# Control orientations in the grab's own frame (z = tool axis): a
# MOVE/ROTATE_AXIS control acts along/about its local x.
_X_TO_Y = Quaternion(w=_HALF, x=0.0, y=0.0, z=_HALF)
_X_TO_Z = Quaternion(w=_HALF, x=0.0, y=-_HALF, z=0.0)


def _quat_x_along(direction):
    """Return the quaternion turning +x onto `direction` (shortest arc)."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    x = np.array([1.0, 0.0, 0.0])
    w = 1.0 + x @ d
    v = np.cross(x, d)
    q = np.array([w, *v]) / np.linalg.norm([w, *v])

    return Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])


def _matrix(q):
    """Return the rotation matrix of a geometry_msgs Quaternion."""
    w, x, y, z = q.w, q.x, q.y, q.z

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _point(xyz):
    return Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))


def _colour(name, alpha=None):
    r, g, b, a = _COLOURS[name]

    return ColorRGBA(r=r, g=g, b=b, a=a if alpha is None else alpha)


def _visual(shape, position, scale, colour):
    marker = Marker()
    marker.type = shape
    marker.pose.position = _point(position)
    marker.pose.orientation.w = 1.0
    marker.scale.x, marker.scale.y, marker.scale.z = map(float, scale)
    marker.color = colour

    return marker


class GrabEditor(Node):
    """The interactive markers, the scene and the CHECK worker."""

    def __init__(self, fill_fraction=DEFAULT_FILL_FRACTION, file_path=None):
        super().__init__('grab_editor')

        self.file_path = file_path or grab_targets.path()
        self.fill_fraction = fill_fraction
        self.cone = seed_cone()

        self.targets = grab_targets.load(self.file_path)
        self.dirty = not self.targets

        if not self.targets:
            self.targets = grab_targets.default_targets()

        # Per grab: None (not checked) or (ok, reason).
        self.results = [None] * len(self.targets)

        # Per grab: drawn in full (True) or as a dot (False).
        self.shown = [index == 0 for index in range(len(self.targets))]

        self.lock = threading.Lock()
        self.pending = None
        self.checking = False
        self.arm = None

        self.message = (
            f'{len(self.targets)} grabs from {self.file_path}'
            if not self.dirty else
            'No grab_targets.yaml yet: starting layout - SAVE to keep it'
        )

        self.server = InteractiveMarkerServer(self, NAMESPACE)
        self.scene_pub = self.create_publisher(
            MarkerArray, SCENE_TOPIC,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        self.menu = MenuHandler()
        self.menu.insert('Insert a grab after this one', callback=self._on_insert)
        self.menu.insert('Delete this grab', callback=self._on_delete)
        self.menu.insert('Hide this grab', callback=self._on_hide)
        self.menu.insert('Show only this grab', callback=self._on_show_only)
        self.menu.insert('Visit earlier', callback=self._on_earlier)
        self.menu.insert('Visit later', callback=self._on_later)
        self.menu.insert('Check reachability (all)', callback=self._on_check)
        self.menu.insert('Save', callback=self._on_save)

        self.create_timer(0.2, self._apply_pending)
        self.create_timer(1.0, self._publish_scene)

        self._rebuild()

    # -- markers -------------------------------------------------------

    def _status(self, index):
        result = self.results[index]

        if result is not None:
            return 'ok' if result[0] else 'unreachable'

        points = grab_targets.stack(self.targets[index], self.cone)

        if min(
            grab_targets.wall_margin(points['flange'], self.cone),
            grab_targets.wall_margin(points['approach'], self.cone),
        ) < WRIST_MARGIN_M:
            return 'tight'

        return 'unchecked'

    def _label(self, index):
        target = self.targets[index]
        result = self.results[index]
        text = (
            f'#{index + 1}  {target["height_m"] * 100:.1f} cm up  '
            f'tilt {target["tilt_deg"]:.0f}  '
            f'appr {target["approach_m"] * 100:.0f} cm'
        )

        if result is None:
            status = self._status(index)
            return text + ('  (wrist near wall)' if status == 'tight' else '')

        return text + ('  OK' if result[0] else f'  NO: {result[1]}')

    def _orientation(self, index):
        tool_z = grab_targets.stack(self.targets[index], self.cone)['tool_z']

        return look_at_quaternion(tool_z, self.cone.axis_dir)

    def _grab_marker(self, index):
        target = self.targets[index]
        points = grab_targets.stack(target, self.cone)
        colour = _colour(self._status(index))
        gripper = config.GRIPPER_OFFSET_Z
        approach = target['approach_m']

        marker = InteractiveMarker()
        marker.header.frame_id = config.BASE_FRAME
        marker.name = f'grab_{index}'
        marker.description = self._label(index)
        marker.scale = 0.12
        marker.pose = Pose(
            position=_point(points['contact']),
            orientation=self._orientation(index),
        )

        # The gripper, in the grab's frame: contact at the origin, the
        # tool axis +z pointing at the wall, so the body is along -z.
        body = InteractiveMarkerControl()
        body.name = 'body'
        body.orientation_mode = InteractiveMarkerControl.INHERIT
        body.interaction_mode = InteractiveMarkerControl.MENU
        body.always_visible = True
        body.markers.append(_visual(
            Marker.CYLINDER, (0, 0, -gripper / 2), (0.05, 0.05, gripper),
            colour,
        ))
        body.markers.append(_visual(
            Marker.CYLINDER, (0, 0, -gripper), (0.08, 0.08, 0.01), colour,
        ))

        if approach > 0.0:
            body.markers.append(_visual(
                Marker.CYLINDER, (0, 0, -gripper - approach / 2),
                (0.012, 0.012, approach), _colour(self._status(index), 0.5),
            ))

        marker.controls.append(body)

        along_axis = _quat_x_along(self.cone.axis_dir)

        ball = InteractiveMarkerControl()
        ball.name = 'around_and_up'
        ball.orientation_mode = InteractiveMarkerControl.FIXED
        ball.orientation = along_axis
        ball.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        ball.always_visible = True
        ball.markers.append(_visual(
            Marker.SPHERE, (0, 0, 0), (0.035, 0.035, 0.035), colour,
        ))
        marker.controls.append(ball)

        depth = InteractiveMarkerControl()
        depth.name = 'depth'
        depth.orientation_mode = InteractiveMarkerControl.FIXED
        depth.orientation = along_axis
        depth.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        marker.controls.append(depth)

        tilt = InteractiveMarkerControl()
        tilt.name = 'tilt'
        tilt.orientation_mode = InteractiveMarkerControl.INHERIT
        tilt.orientation = _X_TO_Y
        tilt.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        marker.controls.append(tilt)

        return marker

    def _approach_marker(self, index):
        points = grab_targets.stack(self.targets[index], self.cone)

        marker = InteractiveMarker()
        marker.header.frame_id = config.BASE_FRAME
        marker.name = f'approach_{index}'
        marker.scale = 0.06
        marker.pose = Pose(
            position=_point(points['approach']),
            orientation=self._orientation(index),
        )

        handle = InteractiveMarkerControl()
        handle.name = 'approach'
        handle.orientation_mode = InteractiveMarkerControl.INHERIT
        handle.orientation = _X_TO_Z
        handle.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        handle.always_visible = True
        handle.markers.append(_visual(
            Marker.CUBE, (0, 0, 0), (0.02, 0.02, 0.02),
            ColorRGBA(r=0.1, g=0.85, b=0.95, a=0.9),
        ))
        marker.controls.append(handle)

        return marker

    def _button(self, name, text, offset_m, callback=None, menu=None):
        up, side = _axis_basis(self.cone.axis_dir)
        mouth = self.cone.axis_point + self.cone.s_max * self.cone.axis_dir
        where = (
            mouth + 0.1 * self.cone.axis_dir
            + (self.cone.radius_at(self.cone.s_max) + 0.08) * up
            + offset_m * side
        )

        marker = InteractiveMarker()
        marker.header.frame_id = config.BASE_FRAME
        marker.name = name
        marker.scale = 0.1
        marker.pose.position = _point(where)
        marker.pose.orientation.w = 1.0

        control = InteractiveMarkerControl()
        control.interaction_mode = (
            InteractiveMarkerControl.MENU if menu is not None
            else InteractiveMarkerControl.BUTTON
        )
        control.always_visible = True

        label = _visual(
            Marker.TEXT_VIEW_FACING, (0, 0, 0), (0, 0, 0.04),
            ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
        )
        label.text = text
        control.markers.append(label)
        control.markers.append(_visual(
            Marker.CUBE, (0, 0, -0.005), (0.14, 0.14, 0.05),
            ColorRGBA(r=0.15, g=0.35, b=0.75, a=0.6),
        ))
        marker.controls.append(control)

        if menu is not None:
            self.server.insert(marker)
            menu.apply(self.server, name)
            return

        self.server.insert(
            marker,
            feedback_callback=lambda feedback: (
                callback(feedback)
                if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK
                else None
            ),
        )

    def _insert_grab(self, index):
        self.server.insert(
            self._grab_marker(index), feedback_callback=self._on_grab_feedback
        )
        self.menu.apply(self.server, f'grab_{index}')
        self.server.insert(
            self._approach_marker(index),
            feedback_callback=self._on_approach_feedback,
        )

    def _insert_dot(self, index):
        """Draw a hidden grab as a small numbered dot; a click shows it."""
        points = grab_targets.stack(self.targets[index], self.cone)

        marker = InteractiveMarker()
        marker.header.frame_id = config.BASE_FRAME
        # Same name as the full grab: see _rebuild.
        marker.name = f'grab_{index}'
        marker.scale = 0.05
        marker.pose.position = _point(points['contact'])
        marker.pose.orientation.w = 1.0

        control = InteractiveMarkerControl()
        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.always_visible = True
        control.markers.append(_visual(
            Marker.SPHERE, (0, 0, 0), (0.02, 0.02, 0.02),
            _colour(self._status(index), 0.7),
        ))

        number = _visual(
            Marker.TEXT_VIEW_FACING, (0, 0, 0.025), (0, 0, 0.025),
            ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9),
        )
        number.text = str(index + 1)
        control.markers.append(number)
        marker.controls.append(control)

        def show(feedback):
            if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
                self.shown[index] = True
                self._rebuild()

        self.server.insert(marker, feedback_callback=show)

    def _show_menu(self):
        """Build the SHOW button's menu: Show all, Hide all, a tick per grab."""
        menu = MenuHandler()
        menu.insert('Show all', callback=lambda _f: self._set_shown(True))
        menu.insert('Hide all', callback=lambda _f: self._set_shown(False))

        for index, target in enumerate(self.targets):
            handle = menu.insert(
                f'#{index + 1}  {target["depth_m"] * 100:.0f} cm deep, '
                f'{target["floor_angle_deg"]:+.0f} deg',
                callback=lambda _f, i=index: self._toggle_shown(i),
            )
            menu.setCheckState(
                handle,
                MenuHandler.CHECKED if self.shown[index]
                else MenuHandler.UNCHECKED,
            )

        return menu

    def _set_shown(self, value):
        self.shown = [value] * len(self.targets)
        self._rebuild()

    def _toggle_shown(self, index):
        self.shown[index] = not self.shown[index]
        self._rebuild()

    def _rebuild(self):
        # Replace markers in place and erase only the ones that went:
        # the server drops any click arriving within 1 s of a marker
        # being created (its guard against two clients at once), so
        # clearing and re-creating everything would swallow a quick
        # second click.
        before = set(self.server.marker_contexts) | set(
            self.server.pending_updates
        )

        for index in range(len(self.targets)):
            if self.shown[index]:
                self._insert_grab(index)
            else:
                self._insert_dot(index)

        self._button('button_check', 'CHECK', -0.2, self._on_check)
        self._button('button_show', 'SHOW', 0.0, menu=self._show_menu())
        self._button('button_save', 'SAVE', 0.2, self._on_save)

        wanted = {f'grab_{index}' for index in range(len(self.targets))}
        wanted |= {
            f'approach_{index}'
            for index, shown in enumerate(self.shown) if shown
        }
        wanted |= {'button_check', 'button_show', 'button_save'}

        for name in before - wanted:
            self.server.erase(name)

        self.server.applyChanges()
        self._publish_scene()

    # -- scene ---------------------------------------------------------

    def _publish_scene(self):
        stamp = self.get_clock().now().to_msg()

        # Lines, not surfaces: a solid surface in front of a grab
        # catches the click meant for it.
        def lines(ns, segments, width, colour):
            marker = Marker()
            marker.header.frame_id = config.BASE_FRAME
            marker.header.stamp = stamp
            marker.ns = ns
            marker.type = Marker.LINE_LIST
            marker.pose.orientation.w = 1.0
            marker.scale.x = width
            marker.color = colour
            marker.points = [_point(p) for p in segments]
            return marker

        fill = lines(
            'fill',
            grab_targets.fill_outline(self.fill_fraction, self.cone),
            0.003, ColorRGBA(r=0.95, g=0.8, b=0.5, a=0.9),
        )
        bucket = lines(
            'bucket', grab_targets.bucket_wireframe(self.cone),
            0.002, ColorRGBA(r=0.2, g=0.9, b=0.2, a=0.6),
        )

        up, _ = _axis_basis(self.cone.axis_dir)
        mouth = (
            self.cone.axis_point + self.cone.s_max * self.cone.axis_dir
        )

        status = _visual(
            Marker.TEXT_VIEW_FACING,
            mouth + 0.1 * self.cone.axis_dir
            + (self.cone.radius_at(self.cone.s_max) + 0.2) * up,
            (0, 0, 0.03), ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
        )
        status.header.frame_id = config.BASE_FRAME
        status.header.stamp = stamp
        status.ns = 'status'

        with self.lock:
            status.text = self.message + ('  [unsaved]' if self.dirty else '')

        self.scene_pub.publish(MarkerArray(markers=[bucket, fill, status]))

    def _say(self, message):
        with self.lock:
            self.message = message

        self.get_logger().info(message)

    # -- editing -------------------------------------------------------

    @staticmethod
    def _index(feedback):
        return int(feedback.marker_name.rsplit('_', 1)[1])

    def _changed(self, index):
        self.results[index] = None
        self.dirty = True

    def _on_grab_feedback(self, feedback):
        index = self._index(feedback)

        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            p = feedback.pose.position
            tool_z = _matrix(feedback.pose.orientation)[:, 2]

            self.targets[index] = grab_targets.from_pose(
                [p.x, p.y, p.z], tool_z,
                self.targets[index]['approach_m'], self.cone,
            )
            self._changed(index)

            points = grab_targets.stack(self.targets[index], self.cone)
            self.server.setPose(
                f'approach_{index}',
                Pose(
                    position=_point(points['approach']),
                    orientation=self._orientation(index),
                ),
            )
            self.server.applyChanges()

        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            # Snap to the clamped, rounded grab and refresh its label.
            self._insert_grab(index)
            self.server.applyChanges()
            self._say(f'#{index + 1}: {grab_targets.describe(self.targets[index])}')

    def _on_approach_feedback(self, feedback):
        index = self._index(feedback)
        target = self.targets[index]

        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            p = feedback.pose.position
            approach = grab_targets.approach_from_handle(
                target, [p.x, p.y, p.z], self.cone
            )
            self.targets[index] = grab_targets.clean(
                dict(target, approach_m=approach)
            )
            self._changed(index)

        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            self._insert_grab(index)
            self.server.applyChanges()
            self._say(
                f'#{index + 1}: {grab_targets.describe(self.targets[index])}'
            )

    def _edit_list(self, change):
        """Apply change(targets, results, shown) to the lists, then redraw."""
        change(self.targets, self.results, self.shown)
        self.dirty = True
        self._rebuild()

    def _on_insert(self, feedback):
        index = self._index(feedback)
        base = self.targets[index]
        new = grab_targets.clean(dict(base, depth_m=base['depth_m'] - 0.05))

        def change(targets, results, shown):
            targets.insert(index + 1, new)
            results.insert(index + 1, None)
            shown.insert(index + 1, True)

        self._edit_list(change)
        self._say(f'Inserted #{index + 2}, 5 cm deeper than #{index + 1}')

    def _on_delete(self, feedback):
        index = self._index(feedback)

        def change(targets, results, shown):
            del targets[index]
            del results[index]
            del shown[index]

        self._edit_list(change)
        self._say(f'Deleted #{index + 1}; {len(self.targets)} grabs left')

    def _swap(self, feedback, step):
        index = self._index(feedback)
        other = index + step

        if not 0 <= other < len(self.targets):
            return

        def change(targets, results, shown):
            for items in (targets, results, shown):
                items[index], items[other] = items[other], items[index]

        self._edit_list(change)
        self._say(f'#{index + 1} is now visited as #{other + 1}')

    def _on_hide(self, feedback):
        self.shown[self._index(feedback)] = False
        self._rebuild()

    def _on_show_only(self, feedback):
        index = self._index(feedback)
        self.shown = [i == index for i in range(len(self.targets))]
        self._rebuild()

    def _on_earlier(self, feedback):
        self._swap(feedback, -1)

    def _on_later(self, feedback):
        self._swap(feedback, 1)

    def _on_save(self, _feedback):
        grab_targets.save(self.targets, self.file_path)
        self.dirty = False
        self._say(
            f'Saved {len(self.targets)} grabs to {self.file_path}; '
            'next: ./rebake.sh retrieve'
        )

    # -- CHECK ---------------------------------------------------------

    def _on_check(self, _feedback):
        if self.checking:
            self._say('Already checking...')
            return

        self.checking = True
        snapshot = [dict(target) for target in self.targets]
        threading.Thread(
            target=self._check, args=(snapshot,), daemon=True
        ).start()

    def _check(self, snapshot):
        """Worker thread: solve every grab as the bake would."""
        from moveit_msgs.srv import GetPositionIK

        try:
            client = self.create_client(GetPositionIK, '/compute_ik')
            available = client.wait_for_service(timeout_sec=2.0)
            self.destroy_client(client)

            if not available:
                self._say(
                    'MoveIt is not running (no /compute_ik): start the '
                    'fake or real bring-up, then CHECK again'
                )
                return

            from . import retrieve_grid

            if self.arm is None:
                self._say('Connecting to MoveIt...')

                from ..arm.controller import XArm7Controller

                self.arm = XArm7Controller()

            results = []

            with retrieve_grid.gripper_padding(self.arm):
                for index, target in enumerate(snapshot):
                    self._say(f'Checking #{index + 1} of {len(snapshot)}...')
                    plan, reason = retrieve_grid.solve_target(self.arm, target)
                    results.append((plan is not None, reason))

            bad = sum(not ok for ok, _ in results)
            self._say(
                f'Checked {len(results)}: '
                + ('all reachable' if not bad else f'{bad} NOT reachable (red)')
            )

            with self.lock:
                self.pending = (snapshot, results)

        except Exception as error:  # noqa: B902 - reported, not raised
            self._say(f'CHECK failed: {error}')

        finally:
            self.checking = False

    def _apply_pending(self):
        with self.lock:
            pending, self.pending = self.pending, None

        if pending is None:
            return

        snapshot, results = pending

        # Only grabs left as they were while the check ran.
        for index, target in enumerate(snapshot):
            if index < len(self.targets) and self.targets[index] == target:
                self.results[index] = results[index]

        self._rebuild()

    def destroy_node(self):
        if self.arm is not None:
            self.arm.destroy_node()

        super().destroy_node()


def main(argv=None):
    """Run `laundry plan edit-grabs`."""
    parser = argparse.ArgumentParser(
        prog='laundry plan edit-grabs', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--fill', type=float, default=DEFAULT_FILL_FRACTION,
        help='How full the drum is shown, as a fraction of its height '
             '(default: 2/3).',
    )
    parser.add_argument(
        '--file', default=None,
        help='Grab targets file (default: <repo>/scan_plans/grab_targets.yaml).',
    )
    args = parser.parse_args(argv)

    rclpy.init()
    editor = GrabEditor(args.fill, args.file)

    # Its own executor: CHECK's XArm7Controller spins itself on the
    # global one from the worker thread.
    executor = SingleThreadedExecutor()
    executor.add_node(editor)

    print(
        'Grab editor up. In RViz: InteractiveMarkers on /grab_editor, '
        'MarkerArray on /grab_editor/scene, Interact tool (i) to drag. '
        'Ctrl+C quits, discarding anything not SAVEd.'
    )

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if editor.dirty:
            print(
                f'Changes since the last SAVE discarded; {editor.file_path} '
                'is unchanged.'
            )

        editor.server.shutdown()
        editor.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    return 0
