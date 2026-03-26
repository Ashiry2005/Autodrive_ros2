import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
import math
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry


class AutoStanley(Node):
    def __init__(self):
        super().__init__("autostanley")

        file_path = '/home/autodrive_devkit/src/control/control/extracted_lap_13.csv'
        df = pd.read_csv(file_path)
        self.waypoints = df[['x', 'y']].to_numpy()

        self.last_idx = 0

        self.steer_pub = self.create_publisher(Float32, "/autodrive/roboracer_1/steering_command", 10)
        self.throttle_pub = self.create_publisher(Float32, "/autodrive/roboracer_1/throttle_command", 10)

        self.odom_ = self.create_subscription(
            Odometry,
            "/autodrive/roboracer_1/odom",
            self.callback,
            10
        )

    def callback(self, msg):
        # =============================
        # Current State
        # =============================
        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        curr_v = abs(msg.twist.twist.linear.x)

        # Yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        curr_yaw = math.atan2(siny_cosp, cosy_cosp)

        # =============================
        # Find Closest Point (optimized)
        # =============================
        search_range = 20
        start = max(0, self.last_idx - search_range)
        end = min(len(self.waypoints), self.last_idx + search_range)

        local_points = self.waypoints[start:end]
        distances = np.linalg.norm(local_points - np.array([curr_x, curr_y]), axis=1)
        closest_local = np.argmin(distances)
        closest_idx = start + closest_local

        self.last_idx = closest_idx

        # =============================
        # Look-ahead (adaptive)
        # =============================
        lookahead = int(5 + curr_v * 3)
        target_idx = (closest_idx + lookahead) % len(self.waypoints)
        target = self.waypoints[target_idx]

        # =============================
        # Cross Track Error (CORRECT)
        # =============================
        p1 = self.waypoints[closest_idx]
        p2 = self.waypoints[(closest_idx + 1) % len(self.waypoints)]

        path_vec = p2 - p1
        car_vec = np.array([curr_x, curr_y]) - p1

        # Projection
        path_norm = path_vec / (np.linalg.norm(path_vec) + 1e-6)
        proj = np.dot(car_vec, path_norm)

        closest_point = p1 + proj * path_norm

        error_vec = np.array([curr_x, curr_y]) - closest_point
        cross_track_error = np.linalg.norm(error_vec)

        # Sign using cross product
        cross = np.cross(path_vec, car_vec)
        if cross < 0:
            cross_track_error *= -1

        # =============================
        # Heading Error
        # =============================
        target_yaw = math.atan2(target[1] - curr_y, target[0] - curr_x)
        yaw_error = target_yaw - curr_yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        # =============================
        # Stanley Control
        # =============================
        k = 0.1 + 0.05 * curr_v

        steering = yaw_error + math.atan2(k * cross_track_error, curr_v + 0.5)

        # Steering limit
        max_steer = 0.4
        steering = max(-max_steer, min(max_steer, steering))

        # =============================
        # Throttle Control
        # =============================
        throttle = 0.2 - 0.3 * abs(steering)
        throttle = max(0.05, throttle)

        # =============================
        # Publish
        # =============================
        steer_msg = Float32()
        steer_msg.data = float(steering)
        self.steer_pub.publish(steer_msg)

        throttle_msg = Float32()
        throttle_msg.data = float(throttle)
        self.throttle_pub.publish(throttle_msg)

        dist_to_start = np.linalg.norm(np.array([curr_x, curr_y]) - self.waypoints[0])
        if dist_to_start < 1.0: 
            self.last_idx = 0

        # Debug
        self.get_logger().info(
            f"Idx: {target_idx} | Steer: {steering:.2f} | Error: {cross_track_error:.2f} | V: {curr_v:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = AutoStanley()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
