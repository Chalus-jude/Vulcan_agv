import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CmdVelCLI(Node):

    def __init__(self):
        super().__init__('cmd_vel_cli')

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

        self.get_logger().info("Type command like: cmd=0.2,0.0")

        self.run_cli()

    def run_cli(self):
        while rclpy.ok():
            try:
                user_input = input(">> ")

                if user_input.startswith("cmd="):
                    data = user_input.replace("cmd=", "")
                    linear, angular = map(float, data.split(","))

                    msg = Twist()
                    msg.linear.x = linear
                    msg.angular.z = angular

                    self.publisher_.publish(msg)

                    self.get_logger().info(
                        f"Sent → linear={linear}, angular={angular}"
                    )

            except Exception as e:
                self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelCLI()
    node.destroy_node()
    rclpy.shutdown()