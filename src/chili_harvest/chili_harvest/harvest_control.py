"""Chili harvest controller (MoveIt MoveGroup action client).

Waits at the scan pose for a stable ripe-chili cut point on /chili/harvest_target,
then runs one harvest cycle:

  open cutter + gripper -> pre-approach (8 cm back) -> cut pose -> grip -> cut
  -> open cutter -> retreat -> drop pose -> release -> close tools -> scan pose

Disabled at start for safety. Enable with:
  ros2 service call /chili/enable std_srvs/srv/SetBool "{data: true}"

With dry_run:=true every motion is planned (shown in RViz) but never executed,
so the real arm does not move.
"""
import csv
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PointStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (Constraints, JointConstraint, OrientationConstraint,
                             PositionConstraint)
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import SetBool

SUCCESS = 1  # moveit_msgs/MoveItErrorCodes.SUCCESS


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class HarvestControl(Node):
    def __init__(self):
        super().__init__('chili_harvest_control')
        p = self.declare_parameter
        p('arm_group', 'arm_group')
        p('gripper_group', 'gripper_group')
        p('cutter_group', 'cutter_group')
        p('base_frame', 'base_link')
        p('ee_link', 'tcp_link')
        p('arm_joints', ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5'])
        p('scan_joints', [0.0, 0.35, -0.55, -0.2, 0.0])   # SRDF "detect"
        p('drop_joints', [0.0, 0.0, 0.0, 0.0, 0.0])       # SRDF "rest"
        # Tool angles (rad) for the controlled joints; left side mimics right.
        p('gripper_joint', 'right_gripper_joint')
        p('cutter_joint', 'right_cutter_joint')
        p('gripper_open', 0.30)
        p('gripper_hold', 0.06)
        p('gripper_closed', 0.0)
        p('cutter_open', 0.69)
        p('cutter_closed', 0.0)
        # Geometry
        p('cutter_offset', [0.0, 0.0, 0.09])  # cutter cut-point minus TCP, tool frame (m)
        p('target_offset', [0.0, 0.0, 0.0])   # extra correction in base frame (m)
        p('approach_distance', 0.08)
        p('position_tolerance', 0.01)
        p('level_tolerance', 0.35)            # roll/pitch tolerance (rad) to keep cutter level
        p('min_reach', 0.15)
        p('max_reach', 0.60)
        p('min_z', -0.10)
        p('max_z', 0.60)
        # Behaviour
        p('stable_count', 5)
        p('stable_radius', 0.015)
        p('settle_time', 2.0)
        p('fast_velocity', 0.3)
        p('slow_velocity', 0.1)
        p('planning_time', 5.0)
        p('planning_attempts', 10)
        p('dry_run', False)
        p('start_enabled', False)
        p('go_to_scan_on_start', True)
        p('log_dir', os.path.expanduser('~/chili_harvest_logs'))

        self.gp = lambda n: self.get_parameter(n).value
        self.enabled = bool(self.gp('start_enabled'))
        self.state = 'STARTING'
        self.buffer = []
        self.first_seen = None
        self.ready_since = None
        self.steps, self.step_i = [], 0
        self.cycle = {}
        self.wait_timer = None

        self.client = ActionClient(self, MoveGroup, 'move_action')
        self.create_subscription(PointStamped, '/chili/harvest_target', self.on_target, 10)
        self.pub_status = self.create_publisher(String, '/chili/harvest_status', 10)
        self.create_service(SetBool, '/chili/enable', self.on_enable)

        os.makedirs(self.gp('log_dir'), exist_ok=True)
        self.log_path = os.path.join(self.gp('log_dir'), 'cycles.csv')
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                csv.writer(f).writerow(['time', 'result', 'failed_step', 'x', 'y', 'z',
                                        'detect_to_start_s', 'cycle_s', 'dry_run'])

        self.get_logger().info('Waiting for MoveIt move_action server...')
        while not self.client.wait_for_server(timeout_sec=2.0):
            self.get_logger().info('  ...still waiting (is MoveIt running?)')
        mode = 'DRY RUN (plan only)' if self.gp('dry_run') else 'LIVE (arm will move)'
        self.get_logger().info(f'Harvest controller online: {mode}. '
                               f"Enabled={self.enabled}")
        if self.gp('go_to_scan_on_start'):
            self.run_steps([('go to scan pose', 'joints', self.gp('scan_joints'),
                             self.gp('fast_velocity'))], on_done=self.at_scan)
        else:
            self.at_scan(True, None)

    # ------------------------------------------------------------ status
    def set_state(self, s):
        self.state = s
        self.pub_status.publish(String(data=s))
        self.get_logger().info(f'[STATE] {s}')

    def on_enable(self, req, res):
        self.enabled = bool(req.data)
        self.buffer.clear()
        res.success, res.message = True, f'enabled={self.enabled}'
        self.get_logger().info(res.message)
        return res

    def at_scan(self, ok, failed):
        self.ready_since = time.time()
        self.buffer.clear()
        self.first_seen = None
        self.set_state('IDLE')

    # ------------------------------------------------------------ targets
    def on_target(self, msg):
        if self.state != 'IDLE' or not self.enabled:
            return
        if time.time() - self.ready_since < self.gp('settle_time'):
            return
        if msg.header.frame_id != self.gp('base_frame'):
            self.get_logger().warn(f'Target in {msg.header.frame_id}, expected base frame',
                                   throttle_duration_sec=5.0)
            return
        pt = np.array([msg.point.x, msg.point.y, msg.point.z])
        if not self.reachable(pt):
            return
        if self.buffer and np.linalg.norm(pt - np.median(self.buffer, axis=0)) > self.gp('stable_radius'):
            self.buffer = []            # jumped to another chili / noise: start again
        if not self.buffer:
            self.first_seen = time.time()
        self.buffer.append(pt)
        if len(self.buffer) >= self.gp('stable_count'):
            target = np.median(self.buffer, axis=0)
            self.buffer = []
            self.start_cycle(target)

    def reachable(self, pt):
        r = math.hypot(pt[0], pt[1])
        ok = (self.gp('min_reach') <= r <= self.gp('max_reach')
              and self.gp('min_z') <= pt[2] <= self.gp('max_z'))
        if not ok:
            self.get_logger().info(f'Target {np.round(pt, 3)} out of reach, ignored',
                                   throttle_duration_sec=3.0)
        return ok

    # ------------------------------------------------------------ cycle
    def start_cycle(self, cut):
        cut = cut + np.array(self.gp('target_offset'))
        yaw = math.atan2(cut[1], cut[0])
        R = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                      [math.sin(yaw), math.cos(yaw), 0.0],
                      [0.0, 0.0, 1.0]])
        tcp_goal = cut - R @ np.array(self.gp('cutter_offset'))
        pre = tcp_goal - R @ np.array([self.gp('approach_distance'), 0.0, 0.0])

        self.cycle = {'t0': time.time(), 'cut': cut,
                      'detect_to_start': time.time() - (self.first_seen or time.time())}
        self.get_logger().info(
            f'\n>>> HARVEST CYCLE  cut point (base) = {np.round(cut, 3)} m'
            f'\n    TCP goal = {np.round(tcp_goal, 3)}, pre-approach = {np.round(pre, 3)}'
            f'\n    yaw = {math.degrees(yaw):.1f} deg')
        fast, slow = self.gp('fast_velocity'), self.gp('slow_velocity')
        g, c = self.gp('gripper_joint'), self.gp('cutter_joint')
        self.run_steps([
            ('open cutter', 'tool', 'cutter_group', c, self.gp('cutter_open')),
            ('open gripper', 'tool', 'gripper_group', g, self.gp('gripper_open')),
            ('move to pre-approach', 'pose', pre, yaw, fast),
            ('move to cut pose', 'pose', tcp_goal, yaw, slow),
            ('grip chili', 'tool', 'gripper_group', g, self.gp('gripper_hold')),
            ('wait', 'wait', 0.5),
            ('CUT', 'tool', 'cutter_group', c, self.gp('cutter_closed')),
            ('wait', 'wait', 0.5),
            ('open cutter', 'tool', 'cutter_group', c, self.gp('cutter_open')),
            ('retreat', 'pose', pre, yaw, slow),
            ('move to drop pose', 'joints', self.gp('drop_joints'), fast),
            ('release chili', 'tool', 'gripper_group', g, self.gp('gripper_open')),
            ('wait', 'wait', 0.5),
            ('close gripper', 'tool', 'gripper_group', g, self.gp('gripper_closed')),
            ('close cutter', 'tool', 'cutter_group', c, self.gp('cutter_closed')),
            ('back to scan pose', 'joints', self.gp('scan_joints'), fast),
        ], on_done=self.end_cycle)

    def end_cycle(self, ok, failed):
        dt = time.time() - self.cycle.get('t0', time.time())
        cut = self.cycle.get('cut', [float('nan')] * 3)
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'success' if ok else 'failed', failed or '',
                                    *[round(float(v), 4) for v in cut],
                                    round(self.cycle.get('detect_to_start', 0.0), 2),
                                    round(dt, 1), self.gp('dry_run')])
        if ok:
            self.get_logger().info(f'[DONE] Cycle complete in {dt:.1f} s')
            self.at_scan(True, None)
        else:
            self.get_logger().error(f'[FAILED] at step "{failed}". Returning to scan pose.')
            self.run_steps([('open gripper', 'tool', 'gripper_group',
                             self.gp('gripper_joint'), self.gp('gripper_open')),
                            ('back to scan pose', 'joints', self.gp('scan_joints'),
                             self.gp('fast_velocity'))],
                           on_done=self.at_scan)

    # ------------------------------------------------------------ step runner
    def run_steps(self, steps, on_done):
        self.steps, self.step_i, self.on_done = steps, 0, on_done
        self.next_step()

    def next_step(self):
        if self.step_i >= len(self.steps):
            return self.on_done(True, None)
        step = self.steps[self.step_i]
        name, kind = step[0], step[1]
        self.set_state(name.upper())
        if kind == 'wait':
            self.wait_timer = self.create_timer(step[2], self._wait_done)
            return
        goal = MoveGroup.Goal()
        req = goal.request
        req.num_planning_attempts = int(self.gp('planning_attempts'))
        req.allowed_planning_time = float(self.gp('planning_time'))
        goal.planning_options.plan_only = bool(self.gp('dry_run'))
        cons = Constraints()
        if kind == 'tool':
            _, _, group, joint, value = step
            req.group_name = self.gp(group)
            req.max_velocity_scaling_factor = 1.0
            cons.joint_constraints.append(self.jc(joint, value, 0.01))
        elif kind == 'joints':
            _, _, values, vel = step
            req.group_name = self.gp('arm_group')
            req.max_velocity_scaling_factor = float(vel)
            req.max_acceleration_scaling_factor = float(vel)
            for j, v in zip(self.gp('arm_joints'), values):
                cons.joint_constraints.append(self.jc(j, v, 0.02))
        elif kind == 'pose':
            _, _, xyz, yaw, vel = step
            req.group_name = self.gp('arm_group')
            req.max_velocity_scaling_factor = float(vel)
            req.max_acceleration_scaling_factor = float(vel)
            cons.position_constraints.append(self.pc(xyz, yaw))
            cons.orientation_constraints.append(self.oc(yaw))
        req.goal_constraints.append(cons)
        self.client.send_goal_async(goal).add_done_callback(self._accepted)

    def _wait_done(self):
        self.wait_timer.cancel()
        self.destroy_timer(self.wait_timer)
        self.step_i += 1
        self.next_step()

    def _accepted(self, fut):
        gh = fut.result()
        if not gh.accepted:
            return self.on_done(False, self.steps[self.step_i][0] + ' (goal rejected)')
        gh.get_result_async().add_done_callback(self._finished)

    def _finished(self, fut):
        code = fut.result().result.error_code.val
        if code != SUCCESS:
            name = self.steps[self.step_i][0]
            return self.on_done(False, f'{name} (MoveIt error {code})')
        self.step_i += 1
        self.next_step()

    # ------------------------------------------------------------ constraints
    @staticmethod
    def jc(name, pos, tol):
        return JointConstraint(joint_name=name, position=float(pos), tolerance_above=tol,
                               tolerance_below=tol, weight=1.0)

    def pc(self, xyz, yaw):
        c = PositionConstraint()
        c.header.frame_id = self.gp('base_frame')
        c.link_name = self.gp('ee_link')
        sphere = SolidPrimitive(type=SolidPrimitive.SPHERE,
                                dimensions=[float(self.gp('position_tolerance'))])
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = [float(v) for v in xyz]
        pose.orientation.w = 1.0
        c.constraint_region.primitives.append(sphere)
        c.constraint_region.primitive_poses.append(pose)
        c.weight = 1.0
        return c

    def oc(self, yaw):
        # Keep the tool level (cutter blades horizontal, cutting a hanging stem).
        # Yaw is left free: joint_1 sets it, and tcp_joint is virtual.
        c = OrientationConstraint()
        c.header.frame_id = self.gp('base_frame')
        c.link_name = self.gp('ee_link')
        q = yaw_quat(yaw)
        c.orientation.x, c.orientation.y, c.orientation.z, c.orientation.w = q
        tol = float(self.gp('level_tolerance'))
        c.absolute_x_axis_tolerance = tol
        c.absolute_y_axis_tolerance = tol
        c.absolute_z_axis_tolerance = math.pi
        c.weight = 1.0
        return c


def main(args=None):
    rclpy.init(args=args)
    node = HarvestControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
