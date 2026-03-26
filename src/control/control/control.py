import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point #geometry_msgs/msg/Point
from std_msgs.msg import Float32

class PID: # move 9m in y axis 
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.setpoint = setpoint

    
    def compute_step(self, measured_value, current_time):
        error = self.setpoint - measured_value
        if self.prev_time is None:
            self.prev_time = current_time
            return 0.0
        
        dt = (current_time - self.prev_time).nanoseconds/1e9
        if dt == 0:
            return 0.0
        
        self.integral += error * dt
        derevative = (error - self.prev_error)/dt

        out = (self.kp * error) + (self.ki * self.integral) + (self.kd * derevative)
        self.prev_error = error
        self.prev_time = current_time

        return max(0.0, min(1.0,out))
        
            
class Manager(Node):
    def __init__(self):
        super().__init__("manager_node")
        self.spawn_pt = None
        self.dist_control = PID(0.58, 0.008, 0.408, 9.0)
        self.pose_sub = self.create_subscription(Point,"/autodrive/roboracer_1/ips",self.posi_callback, 10)
        self.vel_cmd = self.create_publisher(Float32, "/autodrive/roboracer_1/throttle_command", 10)



    def posi_callback(self,msg:Point):
        if self.spawn_pt is None:
            self.spawn_pt = msg.y

        current_time = self.get_clock().now()
        delta_pos = abs(msg.y - self.spawn_pt)
        accel_cmd = self.dist_control.compute_step(delta_pos, current_time)
        if delta_pos >= 9.0:
            accel_cmd = 0.0
            self.dist_control.integral = 0.0

        cmd = Float32()
        cmd.data = accel_cmd

        self.vel_cmd.publish(cmd)
        self.get_logger().info(f"distance: {delta_pos:.2f}, throttle: {accel_cmd:.2f}")




def main (args = None):
    rclpy.init(args=args)
    node = Manager()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ =="__main__":
     main()