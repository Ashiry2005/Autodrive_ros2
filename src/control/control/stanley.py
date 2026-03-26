import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
import math
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry

class SimpleStanley(Node):
    def __init__(self):
        super().__init__("simple_stanley")
        df = pd.read_csv('/home/autodrive_devkit/src/control/control/extracted_lap_13.csv')
        self.waypoints = df[['x', 'y']].to_numpy()

        self.steer_pub = self.create_publisher(Float32, "/autodrive/roboracer_1/steering_command", 10)
        self.throttle_pub = self.create_publisher(Float32, "/autodrive/roboracer_1/throttle_command", 10)
        self.create_subscription(Odometry, "/autodrive/roboracer_1/odom", self.callback, 10)

    def callback(self, msg):
        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))

        distances = np.linalg.norm(self.waypoints - np.array([curr_x, curr_y]), axis=1)
        closest_idx = np.argmin(distances)

        target_idx = (closest_idx + 5) % len(self.waypoints)
        tx, ty = self.waypoints[target_idx]

        angle_to_target = math.atan2(ty - curr_y, tx - curr_x)
        
        steering_angle = angle_to_target - yaw

        steering_angle = math.atan2(math.sin(steering_angle), math.cos(steering_angle))

        self.steer_pub.publish(Float32(data=float(steering_angle)))
        self.throttle_pub.publish(Float32(data=0.1)) # سرعة هادية 10%

        self.get_logger().info(f"Target Point: {target_idx} | Steer: {steering_angle:.2f}")

def main():
    rclpy.init()
    rclpy.spin(SimpleStanley())
    rclpy.shutdown()

if __name__ == "__main__":
    main()
