#!/usr/bin/env python3
"""
navigation_brain.py
===================
Main Finite State Machine (FSM) for the UGV Autonomous Navigation Challenge.

Improvements over v1:
  [BP1] Heading aligned to aisle vanishing point (gap_angle → Nav2 orientation),
        not naively pointed at the local waypoint. Eliminates crabbing / weaving.
  [BP2] Lateral wall-centering: samples left vs right LiDAR sectors and nudges
        the Nav2 waypoint sideways to keep the robot in the middle of the aisle.
  [BP3] Longitudinal sign distance: perception node now sends the horizontal angle
        to the sign; the FSM projects that to a forward-axis distance so approach
        thresholds fire at exactly the right moment regardless of sign placement.

FSM States:
  CORRIDOR_FOLLOWING  – LiDAR centroid gap + wall centering
  APPROACH_SIGN       – Straight step toward sign using projected distance
  EXECUTE_MANEUVER    – Kinematic LEFT / RIGHT / INPLACE ROTATION / STOP / FORWARD
  POST_MANEUVER_DRIVE – 2.5 m straight to clear intersection corner
  RECOVERY            – Reverse 3 s → rotate 180° when stuck > 10 s
  MISSION_COMPLETE    – Permanent halt after GOAL sign

Key invariants:
  • One Nav2 goal in-flight at a time (goal_in_progress flag + result callback).
  • Kinematic turns (LEFT/RIGHT/INPLACE) bypass Nav2 and use direct cmd_vel.
  • All goals are in the 'odom' frame – no map, no SLAM.
  • Per-sign cooldown timer prevents re-triggering after a maneuver.
  • Stuck detector tracks both position AND yaw so a spinning robot is not
    considered "stuck" while it is actively turning.
"""

import math
import json

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32MultiArray, String


# ──────────────────────────────────────────────────────────────────────────────
# Tunable constants
# ──────────────────────────────────────────────────────────────────────────────

# Sign detection / approach
SIGN_APPROACH_DISTANCE  = 1.9   # m – enter APPROACH_SIGN when sign closer than this
SIGN_TURN_THRESHOLD     = 1.1  # m – execute maneuver when projected dist < this
SIGN_COOLDOWN           = 12.0  # s – ignore same sign class after a maneuver
STOP_SIGN_DURATION      = 6.0   # s – hold-still duration for STOP sign

# Nav2 step sizes
NAV_STEP_CORRIDOR       = 1.2   # m – goal step in corridor following
NAV_STEP_APPROACH       = 0.7   # m – goal step during approach
NAV_STEP_POST           = 1.0   # m – goal step in post-maneuver straight drive
FORWARD_OVERRIDE_DIST   = 3.0   # m – override distance for FORWARD sign

# Post-maneuver clearing drive
POST_MANEUVER_DISTANCE  = 2.3   # m – straight drive after any turn

# LiDAR corridor following
OPEN_RANGE_THRESHOLD    = 1.2   # m – LiDAR reading counted as "open" space
FOV_CORRIDOR_DEG        = 40    # °  – half-angle of front cone for gap detection
SIDE_SECTOR_INNER_DEG   = 45    # °  – inner boundary of left/right sampling sectors
SIDE_SECTOR_OUTER_DEG   = 90    # °  – outer boundary of left/right sampling sectors
K_CENTER                = 0.3   # lateral centering proportional gain (tune in sim)
MAX_LATERAL_OFFSET      = 0.35  # m  – clamp on the lateral nudge

# Stuck / recovery
STUCK_TIME_THRESHOLD    = 10.0  # s
STUCK_DIST_THRESHOLD    = 0.10  # m
STUCK_YAW_THRESHOLD     = 0.10  # rad

DEAD_END_THRESHOLD = 1.0   # m - If max forward space is less than this, it's a dead end

class NavigationBrainNode(Node):

    STATE_CORRIDOR      = "CORRIDOR_FOLLOWING"
    STATE_APPROACH      = "APPROACH_SIGN"
    STATE_MANEUVER      = "EXECUTE_MANEUVER"
    STATE_POST_MANEUVER = "POST_MANEUVER_DRIVE"
    STATE_RECOVERY      = "RECOVERY"
    STATE_DONE          = "MISSION_COMPLETE"

    def __init__(self):
        super().__init__('navigation_brain_node')

        # ── Pose ──────────────────────────────────────────────────────────
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0

        # ── FSM ───────────────────────────────────────────────────────────
        self.current_state = self.STATE_CORRIDOR

        # ── Nav2 ──────────────────────────────────────────────────────────
        self.goal_in_progress = False
        self.goal_handle      = None

        # ── Sensors ───────────────────────────────────────────────────────
        self.latest_scan = None

        # ── Sign state ────────────────────────────────────────────────────
        self.active_command  = None
        self.active_distance = 99.0   # projected longitudinal distance [BP3]
        self.sign_cooldowns  = {}     # {command: rclpy.Time}

        # ── Maneuver state ────────────────────────────────────────────────
        self.target_yaw             = 0.0
        self.approach_yaw           = 0.0     
        self._rotation_last_yaw     = 0.0
        self._rotation_accumulated  = 0.0
        self._stop_sign_start_time  = None
        self._forward_goal_sent     = False

        # ── Post-maneuver ─────────────────────────────────────────────────
        self.post_start_x = 0.0
        self.post_start_y = 0.0

        # ── Stuck / recovery ──────────────────────────────────────────────
        self.last_progress_x    = 0.0
        self.last_progress_y    = 0.0
        self.last_progress_yaw  = 0.0
        self.last_progress_time = self.get_clock().now()
        self.recovery_phase         = 0
        self.recovery_start_time    = None
        self.recovery_target_yaw    = 0.0

        # ── Sign state ────────────────────────────────────────────────────
        self.active_command  = None
        self.active_distance = 99.0   
        self.sign_cooldowns  = {}     
        
        # --- NEW: ODOMETRY DEAD-RECKONING ---
        self.last_vision_x    = 0.0
        self.last_vision_y    = 0.0
        self.last_vision_dist = 99.0

        # ── ArUco log ─────────────────────────────────────────────────────
        self.visited_aruco = set()

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            Odometry,        '/odometry/filtered', self.odom_callback,  10)
        self.create_subscription(
            LaserScan,       '/r1_mini/lidar',     self.lidar_callback, 10)
        self.create_subscription(
            String,          '/active_sign',        self.sign_callback,  10)
        self.create_subscription(
            Int32MultiArray, '/aruco_ids',          self.aruco_callback, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # ── Timers ────────────────────────────────────────────────────────
        self.create_timer(0.1, self.control_loop)
        self.create_timer(1.0, self.stuck_check)

        self.get_logger().info("Navigation Brain Node started.")

    # ══════════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════════

    def euler_from_quaternion(self, q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, a: float) -> float:
        while a >  math.pi: a -= 2.0 * math.pi
        while a < -math.pi: a += 2.0 * math.pi
        return a

    def euclidean(self, x1, y1, x2, y2) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    def local_to_global(self, lx: float, ly: float):
        c, s = math.cos(self.current_yaw), math.sin(self.current_yaw)
        return (self.current_x + lx * c - ly * s,
                self.current_y + lx * s + ly * c)

    def yaw_to_quat(self, yaw: float):
        """Return (qz, qw) for a pure-yaw quaternion (roll=0, pitch=0)."""
        return math.sin(yaw / 2.0), math.cos(yaw / 2.0)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def is_on_cooldown(self, cmd: str) -> bool:
        if cmd not in self.sign_cooldowns:
            return False
        elapsed = (self.get_clock().now() -
                   self.sign_cooldowns[cmd]).nanoseconds / 1e9
        return elapsed < SIGN_COOLDOWN

    def set_cooldown(self, cmd: str):
        self.sign_cooldowns[cmd] = self.get_clock().now()
        self.get_logger().info(f"[COOLDOWN] {SIGN_COOLDOWN}s on '{cmd}'")

    def cancel_current_goal(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None
        self.goal_in_progress = False

    # ══════════════════════════════════════════════════════════════════════
    # ROS callbacks
    # ══════════════════════════════════════════════════════════════════════

    def odom_callback(self, msg):
        self.current_x   = msg.pose.pose.position.x
        self.current_y   = msg.pose.pose.position.y
        self.current_yaw = self.euler_from_quaternion(msg.pose.pose.orientation)

    def lidar_callback(self, msg):
        self.latest_scan = msg

    def aruco_callback(self, msg: Int32MultiArray):
        for mid in msg.data:
            if mid not in self.visited_aruco:
                self.visited_aruco.add(mid)
                self.get_logger().info(
                    f"[ARUCO] ID={mid}  total={len(self.visited_aruco)}/4")

    def sign_callback(self, msg: String):
        """
        Parse active_sign JSON.  Fields expected:
          command  – normalised uppercase FSM command string
          distance – raw Euclidean distance from camera (m)
          angle    – horizontal angle to sign centre from camera axis (rad)
          plane_angle - the 3D orientation of the sign's wall (rad)
        """
        if self.current_state in (self.STATE_MANEUVER,
                                  self.STATE_POST_MANEUVER,
                                  self.STATE_RECOVERY,
                                  self.STATE_DONE):
            return

        try:
            data        = json.loads(msg.data)
            command     = data['command']
            raw_dist    = float(data['distance'])
            angle       = float(data.get('angle', 0.0))
            plane_angle = float(data.get('plane_angle', 0.0))
        except Exception as e:
            self.get_logger().error(f"sign JSON parse error: {e}")
            return

        # Project Euclidean distance onto the forward axis
        proj_dist = raw_dist * math.cos(angle)

        if proj_dist > SIGN_APPROACH_DISTANCE:
            return

        if command != 'GOAL' and self.is_on_cooldown(command):
            return

        if self.current_state == self.STATE_CORRIDOR:
            self.active_command  = command
            self.active_distance = proj_dist
            self.current_state   = self.STATE_APPROACH
            
            # --- NEW: SET ODOMETRY ANCHOR ---
            self.last_vision_x    = self.current_x
            self.last_vision_y    = self.current_y
            self.last_vision_dist = proj_dist
            
            self.cancel_current_goal()
            self.stop_robot()
            
            self.approach_yaw = self.normalize_angle(self.current_yaw + plane_angle)
            
            self.get_logger().info(
                f"[FSM] CORRIDOR→APPROACH  cmd='{command}'  "
                f"Locked Yaw={math.degrees(self.approach_yaw):.1f}°")

        elif self.current_state == self.STATE_APPROACH:
            if command == self.active_command:
                # --- NEW: UPDATE ANCHOR IF SIGN IS STILL VISIBLE ---
                self.last_vision_x    = self.current_x
                self.last_vision_y    = self.current_y
                self.last_vision_dist = proj_dist
                self.active_distance  = proj_dist
    # ══════════════════════════════════════════════════════════════════════
    # Nav2 goal helpers
    # ══════════════════════════════════════════════════════════════════════

    def send_nav2_goal(self, gx: float, gy: float, target_yaw: float = None):
        """
        Submit a NavigateToPose goal.  One goal in-flight at a time.

        [BP1] If target_yaw is provided (corridor / straight states), set the
        goal orientation to that yaw so Nav2 aligns the robot with the aisle
        rather than pointing it at the waypoint XY.
        If target_yaw is None, fall back to atan2 toward the waypoint.
        """
        if self.goal_in_progress:
            return
        if not self.nav_client.wait_for_server(timeout_sec=0.5):
            return

        if target_yaw is None:
            target_yaw = math.atan2(gy - self.current_y, gx - self.current_x)

        qz, qw = self.yaw_to_quat(target_yaw)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id        = 'odom'
        goal_msg.pose.header.stamp           = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x        = gx
        goal_msg.pose.pose.position.y        = gy
        goal_msg.pose.pose.orientation.x     = 0.0
        goal_msg.pose.pose.orientation.y     = 0.0
        goal_msg.pose.pose.orientation.z     = qz
        goal_msg.pose.pose.orientation.w     = qw

        self.goal_in_progress = True
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("[Nav2] Goal rejected.")
            self.goal_in_progress = False
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self._goal_result_cb)

    def _goal_result_cb(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().debug("[Nav2] Goal reached.")
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().debug("[Nav2] Goal cancelled.")
        else:
            self.get_logger().warn(f"[Nav2] Goal ended status={status}.")
        self.goal_in_progress = False
        self.goal_handle      = None

    # ══════════════════════════════════════════════════════════════════════
    # LiDAR goal computation
    # ══════════════════════════════════════════════════════════════════════

    def _corridor_goal(self):
        """
        [BP1] Centroid gap + aisle-heading orientation.
        [BP2] Lateral centering nudge from side LiDAR sectors.

        Returns (gx, gy, gap_angle_global) or (None, None, None).

        gap_angle_global is the aisle heading in the odom frame.
        Passing it to send_nav2_goal as target_yaw makes Nav2 orient the robot
        parallel to the corridor instead of twisting toward the XY waypoint.
        """
        if self.latest_scan is None:
            return None, None, None

        ranges  = np.array(self.latest_scan.ranges)
        ranges  = np.nan_to_num(ranges, posinf=10.0, neginf=0.0)
        n       = len(ranges)
        amin    = self.latest_scan.angle_min
        ainc    = self.latest_scan.angle_increment
        fov_rad = math.radians(FOV_CORRIDOR_DEG)

        # ── [BP1] Gap angle (forward cone centroid) ───────────────────────
        open_angles = []
        for i in range(n):
            a = amin + i * ainc
            if abs(a) <= fov_rad and ranges[i] > OPEN_RANGE_THRESHOLD:
                open_angles.append(a)

        if open_angles:
            gap_angle = sum(open_angles) / len(open_angles)
        else:
            # Fallback: absolute max-range direction anywhere in the scan
            gap_angle = amin + int(np.argmax(ranges)) * ainc

        # ── [BP2] Lateral centering ───────────────────────────────────────
        inner = math.radians(SIDE_SECTOR_INNER_DEG)
        outer = math.radians(SIDE_SECTOR_OUTER_DEG)
        left_r, right_r = [], []
        for i in range(n):
            a = amin + i * ainc
            if  inner <= a <= outer  and ranges[i] > 0.05:
                left_r.append(ranges[i])
            elif -outer <= a <= -inner and ranges[i] > 0.05:
                right_r.append(ranges[i])

        lateral_offset = 0.0
        if left_r and right_r:
            # positive error → left wall closer → nudge right (negative Y)
            error          = float(np.median(left_r)) - float(np.median(right_r))
            lateral_offset = K_CENTER * error
            lateral_offset = max(-MAX_LATERAL_OFFSET,
                                 min(MAX_LATERAL_OFFSET, lateral_offset))

        # ── Safe step (don't overshoot a wall) ───────────────────────────
        tidx      = int(round((gap_angle - amin) / ainc))
        tidx      = max(0, min(n - 1, tidx))
        step_dist = min(NAV_STEP_CORRIDOR, ranges[tidx] - 0.4)
        if step_dist < 0.2:
            return None, None, None   # wall too close; let stuck detector handle it

        # ── Build goal ────────────────────────────────────────────────────
        local_x = step_dist * math.cos(gap_angle)
        local_y = step_dist * math.sin(gap_angle) + lateral_offset  # [BP2]
        gx, gy  = self.local_to_global(local_x, local_y)

        # [BP1] Translate robot-relative gap_angle to global heading
        gap_angle_global = self.normalize_angle(self.current_yaw + gap_angle)

        return gx, gy, gap_angle_global

    def _straight_ahead_goal(self, step: float):
        """Return a goal {step} metres ahead at the robot's current heading."""
        return self.local_to_global(step, 0.0)

    # ══════════════════════════════════════════════════════════════════════
    # Stuck detection (1 Hz)
    # ══════════════════════════════════════════════════════════════════════

    def stuck_check(self):
        # Always reset baseline during non-driving states to avoid false fires
        if self.current_state in (self.STATE_MANEUVER,
                                  self.STATE_POST_MANEUVER,
                                  self.STATE_RECOVERY,
                                  self.STATE_DONE):
            self.last_progress_x    = self.current_x
            self.last_progress_y    = self.current_y
            self.last_progress_yaw  = self.current_yaw
            self.last_progress_time = self.get_clock().now()
            return

        dist     = self.euclidean(self.last_progress_x, self.last_progress_y,
                                  self.current_x, self.current_y)
        yaw_diff = abs(self.normalize_angle(
            self.current_yaw - self.last_progress_yaw))

        if dist >= STUCK_DIST_THRESHOLD or yaw_diff >= STUCK_YAW_THRESHOLD:
            self.last_progress_x    = self.current_x
            self.last_progress_y    = self.current_y
            self.last_progress_yaw  = self.current_yaw
            self.last_progress_time = self.get_clock().now()
        else:
            elapsed = (self.get_clock().now() -
                       self.last_progress_time).nanoseconds / 1e9
            if elapsed >= STUCK_TIME_THRESHOLD:
                self.get_logger().warn(
                    f"[STUCK] No progress for {elapsed:.1f}s → RECOVERY")
                self.cancel_current_goal()
                self.stop_robot()
                self._enter_recovery()

    # ══════════════════════════════════════════════════════════════════════
    # Main FSM dispatch (10 Hz)
    # ══════════════════════════════════════════════════════════════════════

    def control_loop(self):
        dispatch = {
            self.STATE_CORRIDOR:      self._run_corridor,
            self.STATE_APPROACH:      self._run_approach,
            self.STATE_MANEUVER:      self._run_maneuver,
            self.STATE_POST_MANEUVER: self._run_post_maneuver,
            self.STATE_RECOVERY:      self._run_recovery,
            self.STATE_DONE:          self.stop_robot,
        }
        dispatch.get(self.current_state, lambda: None)()

    # ──────────────────────────────────────────────────────────────────────
    # CORRIDOR_FOLLOWING
    # ──────────────────────────────────────────────────────────────────────

    def _run_corridor(self):
        # (Assuming you added the dead end check here)
        if self.check_dead_end():
            self.get_logger().warn("[FSM] Dead end detected! Triggering immediate U-Turn recovery.")
            self.cancel_current_goal()
            self.stop_robot()
            self._enter_recovery()
            return

        # NEW: Catch all three variables
        gx, gy, target_yaw = self._corridor_goal()
        
        if gx is not None:
            # Pass the target_yaw to the goal sender
            self.send_nav2_goal(gx, gy, target_yaw)
    # ──────────────────────────────────────────────────────────────────────
    # APPROACH_SIGN
    # ──────────────────────────────────────────────────────────────────────
    def _run_approach(self):
        # --- NEW: DEAD-RECKON THE DISTANCE ---
        # Calculate how far we've physically driven since the sign was last seen in the camera
        driven_dist = self.euclidean(self.current_x, self.current_y, self.last_vision_x, self.last_vision_y)
        
        # Continuously update the active distance mathematically
        self.active_distance = self.last_vision_dist - driven_dist

        if self.active_distance <= SIGN_TURN_THRESHOLD:
            self.get_logger().info(
                f"[FSM] APPROACH→MANEUVER  cmd='{self.active_command}'  "
                f"est_dist={self.active_distance:.2f}m")
            self.cancel_current_goal()
            self.stop_robot()
            self._initiate_maneuver()
            return

        if not self.goal_in_progress:
            # Drive along locked yaw
            gx = self.current_x + NAV_STEP_APPROACH * math.cos(self.approach_yaw)
            gy = self.current_y + NAV_STEP_APPROACH * math.sin(self.approach_yaw)
            
            self.send_nav2_goal(gx, gy, target_yaw=self.approach_yaw)
    # ──────────────────────────────────────────────────────────────────────
    # EXECUTE_MANEUVER – entry
    # ──────────────────────────────────────────────────────────────────────

    def _initiate_maneuver(self):
        self.current_state = self.STATE_MANEUVER
        cmd = self.active_command

        if cmd == 'GOAL':
            self.get_logger().info(
                f"[MISSION] GOAL reached!  ArUco={sorted(self.visited_aruco)}")
            self.stop_robot()
            self.current_state = self.STATE_DONE
            return

        if cmd == 'LEFT':
            self.target_yaw = self.normalize_angle(
                self.current_yaw + math.pi / 2.0)
        elif cmd == 'RIGHT':
            self.target_yaw = self.normalize_angle(
                self.current_yaw - math.pi / 2.0)
        elif cmd == 'INPLACE ROTATION':
            self._rotation_last_yaw    = self.current_yaw
            self._rotation_accumulated = 0.0
        elif cmd == 'STOP':
            self._stop_sign_start_time = self.get_clock().now()
        elif cmd == 'FORWARD':
            self._forward_goal_sent = False

        self.get_logger().info(f"[MANEUVER] Starting '{cmd}'.")

    # ──────────────────────────────────────────────────────────────────────
    # EXECUTE_MANEUVER – per-tick
    # ──────────────────────────────────────────────────────────────────────

    def _run_maneuver(self):
        cmd = self.active_command

        # ── LEFT / RIGHT ──────────────────────────────────────────────────
        if cmd in ('LEFT', 'RIGHT'):
            err = self.normalize_angle(self.target_yaw - self.current_yaw)
            if abs(err) < 0.05:
                self.stop_robot()
                self.set_cooldown(cmd)
                self.get_logger().info(f"[FSM] '{cmd}' done → POST_MANEUVER")
                self._enter_post_maneuver()
            else:
                t = Twist()
                t.angular.z = max(min(1.5 * err, 0.65), -0.65)
                self.cmd_pub.publish(t)

        # ── INPLACE ROTATION (360°) ───────────────────────────────────────
        elif cmd == 'INPLACE ROTATION':
            delta = abs(self.normalize_angle(
                self.current_yaw - self._rotation_last_yaw))
            self._rotation_accumulated += delta
            self._rotation_last_yaw     = self.current_yaw

            if self._rotation_accumulated >= (2.0 * math.pi - 0.15):
                self.stop_robot()
                self.set_cooldown(cmd)
                self.get_logger().info("[FSM] 360° inplace rotation done")
                self.current_state = self.STATE_CORRIDOR

            else:
                t = Twist()
                t.angular.z = 0.8
                self.cmd_pub.publish(t)

        # ── STOP ──────────────────────────────────────────────────────────
        elif cmd == 'STOP':
            elapsed = (self.get_clock().now() -
                       self._stop_sign_start_time).nanoseconds / 1e9
            if elapsed >= STOP_SIGN_DURATION:
                self.set_cooldown(cmd)
                self.active_command  = None
                self.active_distance = 99.0
                self.get_logger().info("[FSM] STOP done → CORRIDOR")
                self.current_state = self.STATE_CORRIDOR
            else:
                self.stop_robot()

        # ── FORWARD ───────────────────────────────────────────────────────
        elif cmd == 'FORWARD':
            if not self._forward_goal_sent:
                gx, gy = self._straight_ahead_goal(FORWARD_OVERRIDE_DIST)
                self.send_nav2_goal(gx, gy, target_yaw=self.current_yaw)
                self._forward_goal_sent = True
                self.get_logger().info(
                    f"[MANEUVER] FORWARD goal sent ({FORWARD_OVERRIDE_DIST}m).")
            elif not self.goal_in_progress:
                self.set_cooldown(cmd)
                self.get_logger().info("[FSM] FORWARD done → POST_MANEUVER")
                self._enter_post_maneuver()

    # ──────────────────────────────────────────────────────────────────────
    # POST_MANEUVER_DRIVE
    # ──────────────────────────────────────────────────────────────────────

    def _enter_post_maneuver(self):
        self.current_state   = self.STATE_POST_MANEUVER
        self.post_start_x    = self.current_x
        self.post_start_y    = self.current_y
        self.active_command  = None
        self.active_distance = 99.0
        self.get_logger().info(
            f"[POST] Clearing {POST_MANEUVER_DISTANCE}m straight.")

    def _run_post_maneuver(self):
        traveled = self.euclidean(
            self.post_start_x, self.post_start_y,
            self.current_x,    self.current_y)

        if traveled >= POST_MANEUVER_DISTANCE:
            self.cancel_current_goal()
            self.get_logger().info("[FSM] POST done → CORRIDOR")
            self.current_state = self.STATE_CORRIDOR
        elif not self.goal_in_progress:
            gx, gy = self._straight_ahead_goal(NAV_STEP_POST)
            self.send_nav2_goal(gx, gy, target_yaw=self.current_yaw)

    # ──────────────────────────────────────────────────────────────────────
    # RECOVERY
    # ──────────────────────────────────────────────────────────────────────
    def check_dead_end(self) -> bool:
        """
        Explicit 'Upside-Down U' check.
        Returns True ONLY if the front, left, and right are all blocked simultaneously.
        """
        if self.latest_scan is None:
            return False

        ranges = np.array(self.latest_scan.ranges)
        # Replace inf/nan with a large number so they don't break our math
        ranges = np.nan_to_num(ranges, posinf=10.0, neginf=0.0)
        
        angle_min = self.latest_scan.angle_min
        angle_inc = self.latest_scan.angle_increment

        front_blocked = False
        left_blocked  = False
        right_blocked = False

        # --- Check the "Ceiling" of the U (Front) ---
        front_fov = math.radians(30)
        front_ranges = [r for i, r in enumerate(ranges) if abs(angle_min + i * angle_inc) <= front_fov]
        
        if front_ranges and max(front_ranges) < DEAD_END_THRESHOLD:  # < 0.7m
            front_blocked = True

        # --- Check the "Left Leg" of the U ---
        left_min  = math.radians(45)
        left_max  = math.radians(90)
        left_ranges = [r for i, r in enumerate(ranges) if left_min <= (angle_min + i * angle_inc) <= left_max]
        
        # We use minimum distance here. If *any* part of the left wall is close, the leg exists.
        if left_ranges and min(left_ranges) < 1.0: 
            left_blocked = True

        # --- Check the "Right Leg" of the U ---
        right_min = math.radians(-90)
        right_max = math.radians(-45)
        right_ranges = [r for i, r in enumerate(ranges) if right_min <= (angle_min + i * angle_inc) <= right_max]
        
        if right_ranges and min(right_ranges) < 1.0:
            right_blocked = True

        # --- The Final Verdict ---
        # It is only a dead end if all three sides enclose the robot
        if front_blocked and left_blocked and right_blocked:
            return True
            
        return False
    def _enter_recovery(self):
        self.current_state       = self.STATE_RECOVERY
        self.recovery_phase      = 0
        self.recovery_start_time = self.get_clock().now()
        self.last_progress_time  = self.get_clock().now()

    def _run_recovery(self):
        elapsed = (self.get_clock().now() -
                   self.recovery_start_time).nanoseconds / 1e9

        if self.recovery_phase == 0:                  # reverse 3 s
            if elapsed < 3.0:
                t = Twist()
                t.linear.x = -0.20
                self.cmd_pub.publish(t)
            else:
                self.stop_robot()
                self.recovery_phase      = 1
                self.recovery_target_yaw = self.normalize_angle(
                    self.current_yaw + math.pi)
                self.recovery_start_time = self.get_clock().now()
                self.get_logger().info("[RECOVERY] Reversing done; rotating 180°.")

        elif self.recovery_phase == 1:                # rotate 180°
            err = self.normalize_angle(
                self.recovery_target_yaw - self.current_yaw)
            if abs(err) < 0.10:
                self.stop_robot()
                self.get_logger().info("[RECOVERY] Done → CORRIDOR")
                self.current_state      = self.STATE_CORRIDOR
                self.last_progress_x    = self.current_x
                self.last_progress_y    = self.current_y
                self.last_progress_yaw  = self.current_yaw
                self.last_progress_time = self.get_clock().now()
            else:
                t = Twist()
                t.angular.z = max(min(1.0 * err, 0.5), -0.5)
                self.cmd_pub.publish(t)


# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = NavigationBrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()