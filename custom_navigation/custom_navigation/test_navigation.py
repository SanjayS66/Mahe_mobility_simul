import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
import math
import numpy as np

class PureExplorationNode(Node):
    def __init__(self):
        super().__init__('pure_exploration_node')

        # --- VARIABLES ---
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # State tracking
        self.goal_in_progress = False
        self.target_distance = 1.2  # Shorter steps for better control
        self.latest_scan = None

        # --- SUBSCRIBERS ---
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/r1_mini/lidar', self.lidar_callback, 10)

        # --- ACTION CLIENT ---
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # Main Decision Timer (checks if we need a new goal at 5Hz)
        self.create_timer(0.2, self.decision_loop)

        self.get_logger().info("Sequential Exploration Node Started. Waiting for Nav2...")

    def euler_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = self.euler_from_quaternion(msg.pose.pose.orientation)

    def lidar_callback(self, msg):
        self.latest_scan = msg

    def decision_loop(self):
        if self.goal_in_progress or self.latest_scan is None:
            return

        ranges = np.array(self.latest_scan.ranges)
        ranges = np.nan_to_num(ranges, posinf=10.0, neginf=0.0)
        num_readings = len(ranges)

        # 1. Map all "open" angles to a list of actual radians first
        open_angles = []
        
        # Define the front cone (-30 to +30 degrees)
        # We iterate through the whole array but only keep angles in the front cone
        for i in range(num_readings):
            angle = self.latest_scan.angle_min + (i * self.latest_scan.angle_increment)
            
            # Only look at the front 60 degree cone
            if abs(angle) < math.radians(30):
                if ranges[i] > 2.5: # Your "open" threshold
                    open_angles.append(angle)

        if open_angles:
            # Average the actual angles (this avoids the index wrap-around issue)
            gap_angle = sum(open_angles) / len(open_angles)
        else:
            # Fallback: Find the index of the absolute max distance in the whole scan
            best_idx = np.argmax(ranges)
            gap_angle = self.latest_scan.angle_min + (best_idx * self.latest_scan.angle_increment)

        # 2. Local Goal Calculation
        local_x = self.target_distance * math.cos(gap_angle)
        local_y = self.target_distance * math.sin(gap_angle)

        # 3. Transform to Global Odom Frame
        gx = self.current_x + (local_x * math.cos(self.current_yaw) - local_y * math.sin(self.current_yaw))
        gy = self.current_y + (local_x * math.sin(self.current_yaw) + local_y * math.cos(self.current_yaw))

        self.send_nav2_goal(gx, gy)
        
    def send_nav2_goal(self, x, y):
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Nav2 Action Server not available!")
            return

        self.goal_in_progress = True
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'odom'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.w = 1.0 

        self.get_logger().info(f"Sending next step: ({x:.2f}, {y:.2f})")
        
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected by Nav2.")
            self.goal_in_progress = False
            return

        self.get_logger().info("Goal accepted. Moving...")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        # This triggers when the robot actually reaches the goal or fails
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Goal reached successfully!")
        else:
            self.get_logger().warn(f"Goal failed with status: {status}")
        
        # Unlock the state machine to allow the next goal
        self.goal_in_progress = False

def main(args=None):
    rclpy.init(args=args)
    node = PureExplorationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()