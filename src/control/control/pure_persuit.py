import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
import math
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit")

        # =============================
        # Load Path
        # =============================
        file_path = '/home/autodrive_devkit/src/control/control/extracted_lap_13.csv'
        df = pd.read_csv(file_path)
        self.waypoints = df[['x', 'y']].to_numpy()
        self.num_points = len(self.waypoints)

        self.last_idx = 0

        # =============================
        # Vehicle / Controller Parameters
        # =============================
        self.wheelbase = 0.33          # meters, adjust to your platform
        self.min_lookahead = 0.05      # meters
        self.max_lookahead = 4       # meters
        self.lookahead_gain = 0.5      # how much lookahead grows with speed
        self.max_steer = 0.4           # radians, matches Stanley node's limit

        # Speed targets
        self.max_throttle = 0.25
        self.min_throttle = 0.08
        self.curvature_slowdown_gain = 2.5

        # =============================
        # Publishers / Subscribers
        # =============================
        self.steer_pub = self.create_publisher(Float32, "/autodrive/roboracer_1/steering_command", 10)
        self.throttle_pub = self.create_publisher(Float32, "/autodrive/roboracer_1/throttle_command", 10)

        self.odom_ = self.create_subscription(
            Odometry,
            "/autodrive/roboracer_1/odom",
            self.callback,
            10
        )

    # =============================
    # Helper: closest waypoint (local search)
    # =============================
    def find_closest_idx(self, curr_x, curr_y):
        search_range = 20
        start = max(0, self.last_idx - search_range)
        end = min(self.num_points, self.last_idx + search_range)

        local_points = self.waypoints[start:end]
        distances = np.linalg.norm(local_points - np.array([curr_x, curr_y]), axis=1)
        closest_local = np.argmin(distances)
        closest_idx = start + closest_local
        return closest_idx

    # =============================
    # Helper: find lookahead target point
    # by walking forward along the path until
    # cumulative distance >= lookahead_dist
    # =============================
    def find_lookahead_point(self, curr_x, curr_y, closest_idx, lookahead_dist):
        idx = closest_idx
        accumulated = 0.0
        prev_point = self.waypoints[idx]

        for _ in range(self.num_points):
            next_idx = (idx + 1) % self.num_points
            next_point = self.waypoints[next_idx]

            seg_len = np.linalg.norm(next_point - prev_point)
            accumulated += seg_len

            if accumulated >= lookahead_dist:
                return next_point, next_idx

            prev_point = next_point
            idx = next_idx

        # Fallback: if we somehow looped the whole path, just return current closest
        return self.waypoints[closest_idx], closest_idx

    def callback(self, msg):
        # =============================
        # Current State
        # =============================
        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        curr_v = abs(msg.twist.twist.linear.x)

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        curr_yaw = math.atan2(siny_cosp, cosy_cosp)

        # =============================
        # Closest point on path
        # =============================
        closest_idx = self.find_closest_idx(curr_x, curr_y)
        self.last_idx = closest_idx

        # =============================
        # Adaptive lookahead distance
        # Faster speed -> look further ahead
        # =============================
        lookahead_dist = self.min_lookahead + self.lookahead_gain * curr_v
        lookahead_dist = max(self.min_lookahead, min(self.max_lookahead, lookahead_dist))

        target_point, target_idx = self.find_lookahead_point(
            curr_x, curr_y, closest_idx, lookahead_dist
        )

        # =============================
        # Transform target into vehicle frame
        # =============================
        dx = target_point[0] - curr_x
        dy = target_point[1] - curr_y

        # Rotate global delta into the car's local frame
        cos_yaw = math.cos(-curr_yaw)
        sin_yaw = math.sin(-curr_yaw)
        local_x = dx * cos_yaw - dy * sin_yaw   # forward axis
        local_y = dx * sin_yaw + dy * cos_yaw   # leftward axis

        # =============================
        # Pure Pursuit Steering Law
        # delta = atan2(2 * L * y_local, lookahead_dist^2)
        # =============================
        actual_lookahead = math.hypot(local_x, local_y) + 1e-6
        curvature = 2.0 * local_y / (actual_lookahead ** 2)

        steering = math.atan2(self.wheelbase * curvature, 1.0)
        steering = max(-self.max_steer, min(self.max_steer, steering))

        # =============================
        # Throttle: slow down for sharp curvature
        # =============================
        throttle = self.max_throttle - self.curvature_slowdown_gain * abs(curvature)
        throttle = max(self.min_throttle, min(self.max_throttle, throttle))

        # =============================
        # Publish
        # =============================
        steer_msg = Float32()
        steer_msg.data = float(steering)
        self.steer_pub.publish(steer_msg)

        throttle_msg = Float32()
        throttle_msg.data = float(throttle)
        self.throttle_pub.publish(throttle_msg)

        # =============================
        # Lap reset
        # =============================
        dist_to_start = np.linalg.norm(np.array([curr_x, curr_y]) - self.waypoints[0])
        if dist_to_start < 1.0:
            self.last_idx = 0

        # =============================
        # Debug
        # =============================
        self.get_logger().info(
            f"Idx: {target_idx} | Steer: {steering:.2f} | "
            f"Lookahead: {actual_lookahead:.2f} | V: {curr_v:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
