#!/usr/bin/env python3

import cv2
import json
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String

from cv_bridge import CvBridge


class ArucoDetector(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        self.bridge = CvBridge()

        self.marker_size = 0.16  # 16 cm marker

        self.camera_matrix = None
        self.dist_coeffs = None

        self.color_image = None
        self.depth_image = None

        self.depth_scale = 0.001  # uint16 depth image in mm

        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )

        self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            '/aruco_detections',
            10
        )

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            cv2.aruco.DetectorParameters()
        )

        self.marker_points = np.array([
            [-self.marker_size/2,  self.marker_size/2, 0],
            [ self.marker_size/2,  self.marker_size/2, 0],
            [ self.marker_size/2, -self.marker_size/2, 0],
            [-self.marker_size/2, -self.marker_size/2, 0]
        ], dtype=np.float32)

        self.create_timer(0.1, self.process_frame)

    def camera_info_callback(self, msg):

        self.camera_matrix = np.array(
            msg.k,
            dtype=np.float32
        ).reshape(3, 3)

        self.dist_coeffs = np.array(
            msg.d,
            dtype=np.float32
        )

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='passthrough'
        )

    def process_frame(self):

        if self.color_image is None:
            return

        if self.depth_image is None:
            return

        if self.camera_matrix is None:
            return

        corners, ids, _ = self.detector.detectMarkers(
            self.color_image
        )

        if ids is None:
            return

        detections = []

        for i in range(len(ids)):

            image_points = corners[i][0].astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                self.marker_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not success:
                continue

            center_x = int(np.mean(image_points[:, 0]))
            center_y = int(np.mean(image_points[:, 1]))

            if (
                center_x < 0 or
                center_x >= self.depth_image.shape[1] or
                center_y < 0 or
                center_y >= self.depth_image.shape[0]
            ):
                continue

            depth = float(
                self.depth_image[center_y, center_x]
            ) * self.depth_scale

            x = float(tvec[0][0])
            y = float(tvec[1][0])
            z = float(tvec[2][0])

            detection = {
                "id": int(ids[i][0]),
                "depth": round(depth, 3),
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3)
            }

            detections.append(detection)

            self.get_logger().info(
                f"ID={ids[i][0]} "
                f"Depth={depth:.2f} "
                f"X={x:.2f} "
                f"Y={y:.2f} "
                f"Z={z:.2f}"
            )

        msg = String()
        msg.data = json.dumps(detections)

        self.publisher.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = ArucoDetector()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
