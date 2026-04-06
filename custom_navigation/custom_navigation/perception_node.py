#!/usr/bin/env python3
"""
perception_node.py
==================
Handles all visual perception for the UGV challenge.

Improvements:
  [Plane Angle] Probes the depth map on the left and right sides of the YOLO 
                bounding box to calculate the 3D angle of the sign's plane 
                relative to the camera.
"""

import json
import math


import os
from ament_index_python.packages import get_package_share_directory

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray, String
from ultralytics import YOLO

CLASS_MAP = {
    'left':      'LEFT',
    'right':     'RIGHT',
    'straight':  'FORWARD',
    'forward':   'FORWARD',
    'stop':      'STOP',
    'rotation':  'INPLACE ROTATION',
    'inplace':   'INPLACE ROTATION',
    'goal':      'GOAL',
}

class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')
        self.bridge = CvBridge()
        package_share_directory = get_package_share_directory('custom_navigation')
        
        default_model_path = os.path.join(
            package_share_directory, 
            'yolo_models', 
            'bigger_dataset_s.pt'
        )

        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('aruco_dict_id',       cv2.aruco.DICT_4X4_50)
        self.declare_parameter('camera_hfov_deg',     87.0)  
        self.declare_parameter('min_sign_distance',    0.05)
        self.declare_parameter('max_sign_distance',    6.0)
        self.declare_parameter('occlusion_depth_tol',  1.0)
        self.declare_parameter('occlusion_area_ratio', 1.5)
        self.declare_parameter('depth_sample_radius',  3) # Reduced slightly for better edge sampling
        self.declare_parameter('min_confidence',       0.55)
        self.declare_parameter('debug_display',        False)

        model_path          = self.get_parameter('model_path').value
        aruco_dict_id       = self.get_parameter('aruco_dict_id').value
        self.hfov_rad       = math.radians(self.get_parameter('camera_hfov_deg').value)
        self.min_sign_dist  = self.get_parameter('min_sign_distance').value
        self.max_sign_dist  = self.get_parameter('max_sign_distance').value
        self.occ_depth_tol  = self.get_parameter('occlusion_depth_tol').value
        self.occ_area_ratio = self.get_parameter('occlusion_area_ratio').value
        self.depth_radius   = self.get_parameter('depth_sample_radius').value
        self.min_conf       = self.get_parameter('min_confidence').value
        self.debug_display  = self.get_parameter('debug_display').value

        # Camera Focal Length Approximation (fx)
        # Needed to convert pixel distances to physical meters
        self.img_w = 640.0 # Default fallback, updated dynamically in callback
        self.fx = (self.img_w / 2.0) / math.tan(self.hfov_rad / 2.0)

        # ── YOLO & ArUco Setup ────────────────────────────────────────────
        self.model = YOLO(model_path)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # ── Subscribers & Publishers ──────────────────────────────────────
        self.rgb_sub   = message_filters.Subscriber(self, Image, '/r1_mini/camera/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/r1_mini/camera/depth/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.15)
        self.ts.registerCallback(self.sync_callback)

        self.sign_pub  = self.create_publisher(String, '/active_sign', 10)
        self.aruco_pub = self.create_publisher(Int32MultiArray, '/aruco_ids', 10)

        self.get_logger().info("Perception Node started.")

    def sync_callback(self, rgb_msg: Image, depth_msg: Image):
        try:
            cv_rgb   = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        except Exception as e:
            return

        debug_frame = cv_rgb.copy() if self.debug_display else None
        
        # Update focal length dynamically based on actual image size
        self.img_w = cv_rgb.shape[1]
        self.fx = (self.img_w / 2.0) / math.tan(self.hfov_rad / 2.0)

        self._detect_signs(cv_rgb, cv_depth, debug_frame)
        self._detect_aruco(cv_rgb, cv_depth, debug_frame)

        if self.debug_display and debug_frame is not None:
            cv2.imshow("Perception Debug", debug_frame)
            cv2.waitKey(1)

    def _sample_depth(self, cv_depth: np.ndarray, cx: int, cy: int) -> float:
        h, w = cv_depth.shape
        r = self.depth_radius
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        patch = cv_depth[y0:y1, x0:x1].flatten()
        valid = patch[np.isfinite(patch) & (patch > self.min_sign_dist)]
        return float(np.median(valid)) if len(valid) > 0 else float('nan')

    def _pixel_to_angle(self, cx: int, img_width: int) -> float:
        half_w = img_width / 2.0
        offset = cx - half_w
        tan_half = math.tan(self.hfov_rad / 2.0)
        return math.atan((offset / half_w) * tan_half)

    def _detect_signs(self, cv_rgb: np.ndarray, cv_depth: np.ndarray, debug_frame):
        img_h, img_w = cv_rgb.shape[:2]
        results = self.model(cv_rgb, verbose=False)
        candidates = []

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0].item())
                if conf < self.min_conf: continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                raw_name = self.model.names[int(box.cls[0].item())]
                command = CLASS_MAP.get(raw_name.lower())

                if command is None: continue

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)
                dist_center = self._sample_depth(cv_depth, cx, cy)

                if math.isnan(dist_center) or not (self.min_sign_dist <= dist_center <= self.max_sign_dist):
                    continue

                # --- NEW: CALCULATE PLANE ANGLE (Depth Gradient) ---
                # Sample 15% inward from the edges to avoid background bleed
                margin = max(1, int((x2 - x1) * 0.15))
                z_left = self._sample_depth(cv_depth, x1 + margin, cy)
                z_right = self._sample_depth(cv_depth, x2 - margin, cy)

                plane_angle = 0.0
                if not (math.isnan(z_left) or math.isnan(z_right)):
                    # Convert pixel coords to physical camera-frame X coordinates
                    X_left = ((x1 + margin) - img_w / 2.0) * z_left / self.fx
                    X_right = ((x2 - margin) - img_w / 2.0) * z_right / self.fx
                    
                    delta_Z = z_right - z_left
                    delta_X = X_right - X_left
                    
                    # atan2 gives the angle of the wall relative to the camera
                    if delta_X != 0:
                        plane_angle = math.atan2(delta_Z, delta_X)

                angle_to_sign = self._pixel_to_angle(cx, img_w)

                if debug_frame is not None:
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{command} d={dist_center:.1f}m plane={math.degrees(plane_angle):.0f}deg"
                    cv2.putText(debug_frame, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

                candidates.append({
                    'command': command,
                    'distance': dist_center,
                    'angle': angle_to_sign,
                    'plane_angle': plane_angle, # Added to payload
                    'area': area,
                    'conf': conf,
                })

        if not candidates: return

        candidates.sort(key=lambda x: x['distance'])
        active = candidates[0]
        
        # Occlusion logic
        if len(candidates) > 1:
            b = candidates[1]
            if (b['distance'] - active['distance']) < self.occ_depth_tol:
                if b['area'] > active['area'] * self.occ_area_ratio:
                    active = b

        payload = {
            'command': active['command'],
            'distance': round(active['distance'], 3),
            'angle': round(active['angle'], 4),
            'plane_angle': round(active['plane_angle'], 4) # Exported
        }
        
        out = String()
        out.data = json.dumps(payload)
        self.sign_pub.publish(out)

    def _detect_aruco(self, cv_rgb: np.ndarray, cv_depth: np.ndarray, debug_frame):
        gray = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) == 0: return

        detected_ids = []
        for i, marker_id in enumerate(ids.flatten()):
            pts = corners[i][0]
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            dist_str = 'N/A'
            if 0 <= cx < cv_depth.shape[1] and 0 <= cy < cv_depth.shape[0]:
                dist = self._sample_depth(cv_depth, cx, cy)
                if not math.isnan(dist): dist_str = f"{dist:.2f}m"
            detected_ids.append(int(marker_id))

        if debug_frame is not None:
            cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids)

        if detected_ids:
            msg = Int32MultiArray()
            msg.data = detected_ids
            self.aruco_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()