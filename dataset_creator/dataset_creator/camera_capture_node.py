import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import time
from datetime import datetime

class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__('camera_capture_node')
        
        # 1. Subscribe to the camera topic
        self.subscription = self.create_subscription(
            Image,
            '/r1_mini/camera/image_raw',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        
        # 2. Setup state variables
        self.is_recording = False
        self.last_capture_time = 0.0
        self.capture_interval = 2.0 # Capture every 2 seconds
        
        # 3. Setup the directory to save images
        # Change this path to wherever you want the images saved
        default_save_path = os.path.join(os.path.expanduser('~'), 'robomanipal_datasets_mahe_mobility_challenge', 'captured_images')
        
        self.declare_parameter('save_dir', default_save_path)
        self.save_dir = self.get_parameter('save_dir').value
        
        # 4. Create the directory safely
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f"Images will be safely saved to: {self.save_dir}")

        # 4. Setup OpenCV Window and Mouse Callback
        self.window_name = "R1 Mini Camera"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Define button coordinates on the image (Top-Left corner)
        self.btn_x1, self.btn_y1 = 10, 10
        self.btn_x2, self.btn_y2 = 160, 60

        self.get_logger().info("Node started. Click the button in the window to start/stop recording.")

    def mouse_callback(self, event, x, y, flags, param):
        """ Handles mouse clicks on the OpenCV window. """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the click is within the button's bounding box
            if self.btn_x1 <= x <= self.btn_x2 and self.btn_y1 <= y <= self.btn_y2:
                self.is_recording = not self.is_recording
                state = "ON" if self.is_recording else "OFF"
                self.get_logger().info(f"Capture Mode: {state}")

    def image_callback(self, msg):
        """ Processes incoming ROS Image messages. """
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Handle capturing logic
        current_time = time.time()
        if self.is_recording and (current_time - self.last_capture_time) >= self.capture_interval:
            self.save_image(cv_image)
            self.last_capture_time = current_time

        # Draw the interactive button on the video feed
        self.draw_ui(cv_image)

        # Display the image
        cv2.imshow(self.window_name, cv_image)
        cv2.waitKey(1)

    def draw_ui(self, image):
        """ Draws the 'button' on the image frame based on the current state. """
        if self.is_recording:
            color = (0, 0, 255)  # Red for recording
            text = "REC: ON"
        else:
            color = (0, 200, 0)  # Green for idle
            text = "REC: OFF"

        # Draw the button background
        cv2.rectangle(image, (self.btn_x1, self.btn_y1), (self.btn_x2, self.btn_y2), color, -1)
        
        # Add a black border for visibility
        cv2.rectangle(image, (self.btn_x1, self.btn_y1), (self.btn_x2, self.btn_y2), (0, 0, 0), 2)
        
        # Add the text
        cv2.putText(image, text, (self.btn_x1 + 15, self.btn_y1 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def save_image(self, img):
        """ Generates a unique filename and saves the image. """
        # Create a unique filename using the current date, time, and milliseconds
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        
        cv2.imwrite(filepath, img)
        self.get_logger().info(f"Saved: {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageCaptureNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        # Clean up windows and ROS nodes safely
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()