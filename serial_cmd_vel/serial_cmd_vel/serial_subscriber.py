import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import tf_transformations

import serial
import threading
import time
import math
import re
import numpy as np

# ── Hardware constants ──────────────────────────────────────────────
PORT = '/dev/ttyUSB0'
BAUD = 9600

WHEEL_DIAMETER        = 0.21
WHEEL_BASE            = 0.21
WHEEL_CIRCUMFERENCE   = math.pi * WHEEL_DIAMETER
COUNTS_PER_REVOLUTION = 1196
TICKS_PER_METER       = COUNTS_PER_REVOLUTION / WHEEL_CIRCUMFERENCE

# ── Straight-line controller gains ─────────────────────────────────
TARGET_SPEED    = 0.15          # m/s forward speed
KP_HEADING      = 2.0           # proportional gain on heading error
MAX_OMEGA       = 0.5           # rad/s angular speed clamp

# ── EKF noise parameters ────────────────────────────────────────────
# Process noise (how much we trust the wheel odometry model)
SIGMA_V   = 0.05   # std-dev of linear velocity noise  (m/s)
SIGMA_W   = 0.02   # std-dev of angular velocity noise (rad/s)

# Observation noise
SIGMA_OBS_XY    = 0.05   # std-dev of wheel-odom position (m)
SIGMA_OBS_YAW   = 0.01   # std-dev of IMU yaw (rad)


class EKF:
    """
    Extended Kalman Filter for 2-D differential-drive localisation.

    State vector:  x = [x, y, theta]ᵀ
    Control input: u = [v, omega]ᵀ   (from wheel odometry deltas)
    Observations:
        z1 = [x, y]        – wheel odometry position
        z2 = [theta]       – IMU yaw

    Motion model (Euler integration):
        x'     = x + v·cos(θ)·dt
        y'     = y + v·sin(θ)·dt
        theta' = theta + omega·dt

    Jacobians are derived analytically.
    """

    def __init__(self):
        self.mu    = np.zeros(3)          # [x, y, theta]
        self.Sigma = np.diag([0.0, 0.0, 0.0])  # initial certainty

        # Process-noise covariance  (3×3, expanded from 2×2 control noise)
        self.Q_control = np.diag([SIGMA_V**2, SIGMA_W**2])

        # Observation-noise covariances
        self.R_odom = np.diag([SIGMA_OBS_XY**2, SIGMA_OBS_XY**2])   # x, y
        self.R_imu  = np.array([[SIGMA_OBS_YAW**2]])                 # theta

    # ── Prediction step ─────────────────────────────────────────────
    def predict(self, v: float, omega: float, dt: float):
        """
        Propagate the state forward using the motion model.

        Args:
            v:     linear  velocity  (m/s), computed from wheel counts
            omega: angular velocity (rad/s), computed from wheel counts
            dt:    elapsed time (s) since last encoder reading
        """
        theta = self.mu[2]

        # 1. Predicted mean
        self.mu[0] += v * math.cos(theta) * dt
        self.mu[1] += v * math.sin(theta) * dt
        self.mu[2] += omega * dt
        self.mu[2]  = self._wrap_angle(self.mu[2])

        # 2. Jacobian of motion model w.r.t. state  G (3×3)
        G = np.eye(3)
        G[0, 2] = -v * math.sin(theta) * dt
        G[1, 2] =  v * math.cos(theta) * dt

        # 3. Jacobian of motion model w.r.t. control input  V (3×2)
        V = np.array([
            [math.cos(theta) * dt,  0.0],
            [math.sin(theta) * dt,  0.0],
            [0.0,                   dt ],
        ])

        # 4. Process noise in state space
        Q = V @ self.Q_control @ V.T

        # 5. Propagate covariance
        self.Sigma = G @ self.Sigma @ G.T + Q

    # ── Update step: wheel-odometry position ────────────────────────
    def update_odom(self, ox: float, oy: float):
        """
        Incorporate the raw x,y position from wheel odometry.
        """
        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        z    = np.array([ox, oy])
        z_hat = H @ self.mu                          # predicted observation
        y    = z - z_hat                             # innovation

        S = H @ self.Sigma @ H.T + self.R_odom      # innovation covariance
        K = self.Sigma @ H.T @ np.linalg.inv(S)     # Kalman gain

        self.mu    = self.mu + K @ y
        self.Sigma = (np.eye(3) - K @ H) @ self.Sigma
        self.mu[2] = self._wrap_angle(self.mu[2])

    # ── Update step: IMU yaw ─────────────────────────────────────────
    def update_imu(self, yaw: float):
        """
        Incorporate the yaw angle from the IMU.
        """
        H = np.array([[0.0, 0.0, 1.0]])
        z    = np.array([yaw])
        z_hat = H @ self.mu
        y    = np.array([self._wrap_angle(z[0] - z_hat[0])])  # wrap innovation

        S = H @ self.Sigma @ H.T + self.R_imu
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        self.mu    = self.mu + K @ y
        self.Sigma = (np.eye(3) - K @ H) @ self.Sigma
        self.mu[2] = self._wrap_angle(self.mu[2])

    # ── Helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to [-π, π]."""
        return (a + math.pi) % (2 * math.pi) - math.pi

    @property
    def state(self):
        return self.mu.copy()


class StraightLineController:
    """
    Simple proportional heading controller.

    Given the robot's current EKF state, it outputs a Twist that:
      - Drives forward at TARGET_SPEED.
      - Corrects heading drift to maintain the initial bearing.

    The 'desired heading' is locked to whatever θ is when start() is called,
    which means the robot will drive in a straight line in that direction.
    """

    def __init__(self):
        self.desired_heading: float | None = None
        self.active = False

    def start(self, current_theta: float):
        self.desired_heading = current_theta
        self.active = True

    def stop(self):
        self.active = False
        self.desired_heading = None

    def compute(self, current_theta: float) -> Twist:
        """Return a Twist correcting for heading drift."""
        msg = Twist()
        if not self.active or self.desired_heading is None:
            return msg

        heading_error = EKF._wrap_angle(self.desired_heading - current_theta)
        omega = KP_HEADING * heading_error
        omega = max(-MAX_OMEGA, min(MAX_OMEGA, omega))

        msg.linear.x  = TARGET_SPEED
        msg.angular.z = omega
        return msg


class SerialCmdVel(Node):

    def __init__(self):
        super().__init__('serial_cmd_vel_node')

        # ── Serial ──────────────────────────────────────────────────
        self.ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)

        # ── ROS interfaces ──────────────────────────────────────────
        self.create_subscription(Twist, 'cmd_vel', self.cmd_callback, 10)
        self.odom_pub   = self.create_publisher(Odometry, '/odom', 10)
        self.ekf_pub    = self.create_publisher(Odometry, '/odom/ekf', 10)
        self.imu_pub    = self.create_publisher(Imu,      '/imu/data', 10)

        # ── TF ──────────────────────────────────────────────────────
        self.tf_broadcaster = TransformBroadcaster(self)

        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        static_t = TransformStamped()
        static_t.header.stamp    = self.get_clock().now().to_msg()
        static_t.header.frame_id = 'base_link'
        static_t.child_frame_id  = 'imu_link'
        static_t.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(static_t)

        # ── EKF + controller ────────────────────────────────────────
        self.ekf        = EKF()
        self.controller = StraightLineController()

        # Raw odometry integration (used as EKF observation source)
        self.x_raw, self.y_raw, self.theta_raw = 0.0, 0.0, 0.0
        self.enc_left  = None
        self.enc_right = None
        self.last_enc_time = self.get_clock().now()

        # Latest IMU yaw for EKF update
        self._imu_yaw: float | None = None
        self._imu_lock = threading.Lock()

        # ── Control timer (10 Hz) ───────────────────────────────────
        self.create_timer(0.1, self.control_loop)

        # ── Serial reader thread ─────────────────────────────────────
        threading.Thread(target=self.read_serial, daemon=True).start()

        # Start driving straight immediately on node start.
        # θ=0 means the robot will drive in its initial forward direction.
        self.controller.start(current_theta=0.0)
        self.get_logger().info('EKF localisation node started – driving straight.')

    # ── Serial reader ────────────────────────────────────────────────
    def read_serial(self):
        while True:
            try:
                if not self.ser.in_waiting:
                    continue
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue

                # IMU line: yaw=…,pitch=…,roll=…,ax=…,ay=…,az=…
                imu = re.search(
                    r'yaw=([-\d.]+)\s*,\s*pitch=([-\d.]+)\s*,\s*roll=([-\d.]+)'
                    r'\s*,\s*ax=([-\d.]+)\s*,\s*ay=([-\d.]+)\s*,\s*az=([-\d.]+)',
                    line
                )
                if imu:
                    yaw   = float(imu.group(1))
                    pitch = float(imu.group(2))
                    roll  = float(imu.group(3))
                    ax    = float(imu.group(4))
                    ay    = float(imu.group(5))
                    az    = float(imu.group(6))
                    self.publish_imu(yaw, pitch, roll, ax, ay, az)

                    yaw_rad = math.radians(yaw)
                    with self._imu_lock:
                        self._imu_yaw = yaw_rad

                    # EKF IMU update
                    self.ekf.update_imu(yaw_rad)

                # Encoder line: cpr=LEFT, RIGHT
                m = re.search(r'cpr=([-\d.]+),\s*([-\d.]+)', line)
                if m:
                    cpr_l = int(float(m.group(1)))
                    cpr_r = int(float(m.group(2)))
                    self.update_odom(cpr_l, cpr_r)

            except Exception as e:
                self.get_logger().warn(f'Serial read error: {e}')

    # ── IMU publisher ────────────────────────────────────────────────
    def publish_imu(self, yaw_deg, pitch_deg, roll_deg, ax, ay, az):
        imu_msg = Imu()
        imu_msg.header.stamp    = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'

        q = tf_transformations.quaternion_from_euler(
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(yaw_deg)
        )
        imu_msg.orientation.x = q[0]
        imu_msg.orientation.y = q[1]
        imu_msg.orientation.z = q[2]
        imu_msg.orientation.w = q[3]

        imu_msg.linear_acceleration.x = ax
        imu_msg.linear_acceleration.y = ay
        imu_msg.linear_acceleration.z = az

        self.imu_pub.publish(imu_msg)

    # ── Wheel odometry + EKF predict/update ─────────────────────────
    def update_odom(self, cpr_l: int, cpr_r: int):
        now = self.get_clock().now()

        if self.enc_left is None:
            self.enc_left  = cpr_l
            self.enc_right = cpr_r
            self.last_enc_time = now
            return

        dt = (now - self.last_enc_time).nanoseconds * 1e-9
        self.last_enc_time = now

        if dt <= 0.0:
            return

        d_left  = -1.0 * (cpr_l - self.enc_left)  / TICKS_PER_METER
        d_right =  1.0 * (cpr_r - self.enc_right) / TICKS_PER_METER

        self.enc_left  = cpr_l
        self.enc_right = cpr_r

        d_center = (d_left + d_right) / 2.0
        d_theta  = (d_right - d_left) / WHEEL_BASE

        v     = d_center / dt       # linear  velocity (m/s)
        omega = d_theta  / dt       # angular velocity (rad/s)

        # ── Raw dead-reckoning (unchanged logic) ─────────────────────
        self.x_raw     += math.cos(self.theta_raw + d_theta / 2) * d_center
        self.y_raw     += math.sin(self.theta_raw + d_theta / 2) * d_center
        self.theta_raw += d_theta

        self._publish_raw_odom()

        # ── EKF: predict then update with wheel-odom position ───────
        self.ekf.predict(v, omega, dt)
        self.ekf.update_odom(self.x_raw, self.y_raw)

        self._publish_ekf_odom()

    # ── Raw odometry publisher ───────────────────────────────────────
    def _publish_raw_odom(self):
        odom = Odometry()
        odom.header.stamp    = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'
        odom.pose.pose.position.x = self.x_raw
        odom.pose.pose.position.y = self.y_raw

        q = tf_transformations.quaternion_from_euler(0, 0, self.theta_raw)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        self.odom_pub.publish(odom)

    # ── EKF odometry publisher + TF ─────────────────────────────────
    def _publish_ekf_odom(self):
        ex, ey, et = self.ekf.state

        odom = Odometry()
        odom.header.stamp    = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'
        odom.pose.pose.position.x = ex
        odom.pose.pose.position.y = ey

        q = tf_transformations.quaternion_from_euler(0, 0, et)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        # Fill covariance diagonal from EKF Sigma
        S = self.ekf.Sigma
        odom.pose.covariance[0]  = float(S[0, 0])   # x-x
        odom.pose.covariance[7]  = float(S[1, 1])   # y-y
        odom.pose.covariance[35] = float(S[2, 2])   # yaw-yaw

        self.ekf_pub.publish(odom)

        # Broadcast TF from EKF state
        t = TransformStamped()
        t.header.stamp    = odom.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id  = 'base_link'
        t.transform.translation.x = ex
        t.transform.translation.y = ey
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    # ── Straight-line control loop (10 Hz) ──────────────────────────
    def control_loop(self):
        """
        Issue velocity commands to keep the robot on a straight-line heading.

        Uses the EKF's fused theta (IMU + wheel odom) to compute the
        proportional correction.  The serial command is only sent from
        here; the external cmd_vel topic can still override by disabling
        the controller.
        """
        _, _, et = self.ekf.state
        twist = self.controller.compute(et)
        if self.controller.active:
            self._send_serial(twist)

    # ── cmd_vel subscription ─────────────────────────────────────────
    def cmd_callback(self, msg: Twist):
        """
        If an external cmd_vel arrives the controller is paused and the
        external command is forwarded directly.  The controller resumes
        next time it is manually started.
        """
        self.controller.stop()
        self._send_serial(msg)

    # ── Low-level serial write ───────────────────────────────────────
    def _send_serial(self, msg: Twist):
        vl = msg.linear.x - (msg.angular.z * WHEEL_BASE / 2)
        vr = msg.linear.x + (msg.angular.z * WHEEL_BASE / 2)

        rpm_l = round((vl / WHEEL_CIRCUMFERENCE) * 60, 1)
        rpm_r = round((vr / WHEEL_CIRCUMFERENCE) * 60, 1)

        try:
            self.ser.write(f'{rpm_l},{-rpm_r}\n'.encode())
        except serial.SerialException as e:
            self.get_logger().error(f'Serial write failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SerialCmdVel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
