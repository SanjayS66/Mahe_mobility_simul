#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import FollowPath
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class GoalPoseBridge(Node):
    def __init__(self):
        super().__init__('goal_pose_bridge')
        self._client = ActionClient(self, FollowPath, '/follow_path')
        self._sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.get_logger().info('Goal pose bridge ready — use RViz 2D Goal Pose normally')

    def goal_callback(self, pose: PoseStamped):
        self._client.wait_for_server()

        path = Path()
        path.header = pose.header
        path.header.frame_id = 'odom'
        path.poses.append(pose)

        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = 'FollowPath'
        goal.goal_checker_id = 'general_goal_checker'

        self._client.send_goal_async(goal)
        self.get_logger().info(
            f'Forwarding goal: x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}')

def main():
    rclpy.init()
    rclpy.spin(GoalPoseBridge())
    rclpy.shutdown()

if __name__ == '__main__':
    main()