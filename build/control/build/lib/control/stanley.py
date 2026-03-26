import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
import math
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry


class Stanley(Node):
    def __init__(self):
        super().__init__("stanley")

        # Load waypoints from CSV
        file_path = '/home/autodrive_devkit/src/control/control/extracted_lap_13.csv'
        df = pd.read_csv(file_path)
        self.waypoints = df[['x', 'y']].to_numpy()
        self.last_idx = 0

        # Publishers
        self.steer_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/steering_command", 10)
        self.throttle_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/throttle_command", 10)

        # Subscriber
        self.odom_sub = self.create_subscription(
            Odometry, "/autodrive/roboracer_1/odom", self.callback, 10
        )

        # Stanley controller parameters
        self.k = 0.15              # Stanley gain (independent of velocity)
        self.max_steer = 0.4          # rad
        self.prev_steer = 0.0
        self.alpha = 0.7              # smoothing factor
        self.constant_velocity = 1.0  # m/s (safe for tight track)

    def callback(self, msg):
        # -----------------------------
        # Current State
        # -----------------------------
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        v = self.constant_velocity

        # Orientation (yaw)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y ** 2 + q.z ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # -----------------------------
        # Find Closest Waypoint
        # -----------------------------
        search_range = 20
        start = max(0, self.last_idx - search_range)
        end = min(len(self.waypoints), self.last_idx + search_range)

        local_wp = self.waypoints[start:end]
        distances = np.linalg.norm(local_wp - np.array([x, y]), axis=1)
        closest_local = np.argmin(distances)
        closest_idx = start + closest_local
        self.last_idx = closest_idx

        # -----------------------------
        # Lookahead target
        # -----------------------------
        lookahead = 5  # fixed small lookahead
        target_idx = (closest_idx + lookahead) % len(self.waypoints)
        target = self.waypoints[target_idx]

        # -----------------------------
        # Cross Track Error
        # -----------------------------
        p1 = self.waypoints[closest_idx]
        p2 = self.waypoints[(closest_idx + 1) % len(self.waypoints)]
        path_vec = p2 - p1
        car_vec = np.array([x, y]) - p1

        path_norm = path_vec / (np.linalg.norm(path_vec) + 1e-6)
        proj = np.dot(car_vec, path_norm)
        closest_point = p1 + proj * path_norm

        error_vec = np.array([x, y]) - closest_point
        cte = np.linalg.norm(error_vec)
        if np.cross(path_vec, car_vec) < 0:
            cte *= -1  # assign sign

        # -----------------------------
        # Heading Error
        # -----------------------------
        target_yaw = math.atan2(target[1] - y, target[0] - x)
        yaw_error = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))

        # -----------------------------
        # Stanley Control Law
        # -----------------------------
        steering_raw = yaw_error + math.atan2(self.k * cte, v)

        # Smooth steering
        steering = self.alpha * self.prev_steer + (1 - self.alpha) * steering_raw
        self.prev_steer = steering

        # Clamp steering
        steering = max(-self.max_steer, min(self.max_steer, steering))

        # -----------------------------
        # Throttle (slow in corners)
        # -----------------------------
        throttle = 0.06

        # -----------------------------
        # Publish
        # -----------------------------
        self.steer_pub.publish(Float32(data=float(steering)))
        self.throttle_pub.publish(Float32(data=float(throttle)))

        # Debug
        self.get_logger().info(f"Steer: {steering:.2f} | CTE: {cte:.2f} | Throttle: {throttle:.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = Stanley()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()