#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class ObstacleController(Node):
    def __init__(self):
        super().__init__('obstacle_controller')
        
        # Publish to the custom topic you set in the VelocityControl plugin
        self.publisher_ = self.create_publisher(Twist, '/obstacle/cmd_vel', 10)
        
        # --- TUNABLE PARAMETERS ---
        self.speed = 0.2         # Velocity in m/s
        self.travel_time = 12.5   # How many seconds to travel before reversing
        # --------------------------
        
        self.direction = 1.0     # Starts positive (1.0), flips to negative (-1.0)
        self.start_time = self.get_clock().now()
        
        # Publish commands at 10 Hz
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info(f"Obstacle controller started. Speed: {self.speed} m/s, Duration: {self.travel_time}s")

    def timer_callback(self):
        now = self.get_clock().now()
        elapsed_seconds = (now - self.start_time).nanoseconds / 1e9
        
        # Check if it is time to reverse direction
        if elapsed_seconds >= self.travel_time:
            self.direction *= -1.0  # Flip the sign
            self.start_time = now   # Reset the stopwatch
            self.get_logger().info(f"Reversing! Current Y speed: {self.speed * self.direction:.1f} m/s")
            
        # Create and populate the Twist message
        msg = Twist()
        msg.linear.y = self.speed * self.direction
        
        # If your box needs to move forward/backward instead of sideways, 
        # change the line above to: msg.linear.x = self.speed * self.direction
        
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()