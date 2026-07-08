#!/usr/bin/env python3
"""
ekf_localization.py — EKF-based localization for straight-line driving.

State vector  x = [x_pos, y_pos, theta, v, omega]  (5-DOF)
              x_pos, y_pos  : position in the odom frame  (metres)
              theta         : heading                       (radians, NED/ENU)
              v             : forward linear velocity       (m/s)
              omega         : yaw rate                      (rad/s)

Sensors fused
─────────────
  • /odom  (nav_msgs/Odometry) → v_odom, omega_odom
  • /imu/data (sensor_msgs/Imu) → omega_imu (gyro z), a_x (forward accel)

Integration with LaneFollowerAruco
────────────────────────────────────
  Replace the DeadReckoningController with EKFStraightLineController.
  All public methods mirror the old API:
      .latch(current_yaw)
      .pause()
      .resume(current_yaw)
      .is_active          (property)
      .reference_yaw      (property)
      .correction(current_yaw, dt) → float   ← angular-velocity correction
      .heading_error_deg(current_yaw) → float

  Additionally, EKFStraightLineController exposes:
      .update_odom(msg)   ← call from /odom subscriber
      .update_imu(msg)    ← call from /imu/data subscriber (replaces yaw_from_imu)
      .pose               ← (x, y, theta) tuple from EKF
      .covariance         ← full 5×5 P matrix (numpy array)

  The LaneFollowerAruco node needs three small changes:
    1. Import EKFStraightLineController instead of DeadReckoningController.
    2. Add an /odom subscriber that calls self._dr.update_odom(msg).
    3. Change the _imu_cb to also call self._dr.update_imu(msg).
  Everything else (including _start_rotation, _rotation_loop, _lane_cb_combined)
  stays identical because the public correction() / is_active / reference_yaw API
  is preserved.

Design decisions
─────────────────
  • Process model: constant-velocity unicycle with additive Gaussian noise.
  • Measurement models: linear in v and omega, so the EKF update is exact.
  • Q (process noise) and R (measurement noise) are tunable parameters.
  • Heading reference is stored as a single float (same as DR) so correction()
    is a pure P+I controller on (ref_yaw - ekf_theta), exactly matching the
    old DR interface.
  • Thread safety: all numpy operations happen under a threading.Lock so the
    node can be upgraded to MultiThreadedExecutor later without data races.
"""

import math
import threading
import time
from typing import Optional, Tuple

import numpy as np

# ── Optional ROS 2 types (imported lazily so the module can be unit-tested
#    without a full ROS 2 install) ────────────────────────────────────────────
try:
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(angle: float) -> float:
    """Wrap angle to (−π, π]."""
    while angle >  math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (rotation about Z) from a unit quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# ─────────────────────────────────────────────────────────────────────────────
# Core EKF
# ─────────────────────────────────────────────────────────────────────────────

class EKF5DOF:
    """
    Extended Kalman Filter — unicycle model, 5-state.

    State:  [x, y, theta, v, omega]

    Process model (Euler integration, dt seconds):
        x'     = x     + v * cos(theta) * dt
        y'     = y     + v * sin(theta) * dt
        theta' = theta + omega * dt
        v'     = v                             (constant-velocity assumption)
        omega' = omega                         (constant turn-rate assumption)

    Measurement models
    ──────────────────
    Odometry   z_odom  = [v_odom, omega_odom]       H_odom  = rows 3,4 of I
    IMU gyro   z_gyro  = [omega_imu]                H_gyro  = row 4 of I
    IMU accel  z_accel = [a_x]   (optional)         used to correct v̇

    Parameters
    ──────────
    q_xy        : position process noise variance       (m²/s)
    q_theta     : heading process noise variance        (rad²/s)
    q_v         : linear-velocity process noise         ((m/s)²/s)
    q_omega     : angular-velocity process noise        ((rad/s)²/s)
    r_v_odom    : odom linear-velocity meas. noise      ((m/s)²)
    r_omega_odom: odom angular-velocity meas. noise     ((rad/s)²)
    r_omega_imu : IMU gyro meas. noise                  ((rad/s)²)
    r_ax_imu    : IMU forward-accel meas. noise         ((m/s²)²)
    """

    # ── Indices ───────────────────────────────────────────────────────────────
    IX, IY, IT, IV, IW = 0, 1, 2, 3, 4   # state indices
    N = 5                                   # state dimension

    def __init__(self,
                 q_xy:          float = 1e-4,
                 q_theta:       float = 1e-4,
                 q_v:           float = 5e-3,
                 q_omega:       float = 5e-3,
                 r_v_odom:      float = 0.02,
                 r_omega_odom:  float = 0.01,
                 r_omega_imu:   float = 0.005,
                 r_ax_imu:      float = 0.50,
                 use_imu_accel: bool  = False):

        self._lock = threading.Lock()

        # state mean and covariance
        self._x = np.zeros(self.N)         # [x, y, theta, v, omega]
        self._P = np.eye(self.N) * 0.1    # start with moderate uncertainty

        # noise matrices (built once, updated if params change)
        self._Q = np.diag([q_xy, q_xy, q_theta, q_v, q_omega])
        self._R_odom = np.diag([r_v_odom, r_omega_odom])
        self._R_gyro = np.array([[r_omega_imu]])
        self._R_ax   = np.array([[r_ax_imu]])

        self._use_imu_accel = use_imu_accel

        # measurement matrices (constant for this linear measurement model)
        # H_odom selects rows IV and IW from the state
        self._H_odom = np.zeros((2, self.N))
        self._H_odom[0, self.IV] = 1.0
        self._H_odom[1, self.IW] = 1.0

        # H_gyro selects row IW
        self._H_gyro = np.zeros((1, self.N))
        self._H_gyro[0, self.IW] = 1.0

        self._initialised = False
        self._prev_predict_time: Optional[float] = None

    # ── Public init ───────────────────────────────────────────────────────────

    def initialise(self, x: float, y: float, theta: float,
                   v: float = 0.0, omega: float = 0.0):
        """Seed the filter with a known pose (call once on first IMU/odom)."""
        with self._lock:
            self._x[:] = [x, y, theta, v, omega]
            self._P    = np.eye(self.N) * 0.05
            self._initialised = True
            self._prev_predict_time = time.monotonic()

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, dt: float):
        """
        Propagate state forward by dt seconds using the unicycle model.
        Thread-safe.
        """
        if not self._initialised or dt <= 0.0:
            return

        with self._lock:
            x, y, th, v, w = self._x

            cos_th = math.cos(th)
            sin_th = math.sin(th)

            # ── State propagation ──────────────────────────────────────────
            x_new  = x  + v * cos_th * dt
            y_new  = y  + v * sin_th * dt
            th_new = _normalize(th + w * dt)
            v_new  = v                       # constant-velocity
            w_new  = w                       # constant-yaw-rate

            self._x[:] = [x_new, y_new, th_new, v_new, w_new]

            # ── Jacobian of f w.r.t. x ────────────────────────────────────
            #  ∂f/∂x:
            #  [ 1  0  -v·sin(θ)·dt   cos(θ)·dt   0  ]
            #  [ 0  1   v·cos(θ)·dt   sin(θ)·dt   0  ]
            #  [ 0  0   1              0           dt  ]
            #  [ 0  0   0              1            0  ]
            #  [ 0  0   0              0            1  ]
            F = np.eye(self.N)
            F[self.IX, self.IT] = -v * sin_th * dt
            F[self.IX, self.IV] =  cos_th * dt
            F[self.IY, self.IT] =  v * cos_th * dt
            F[self.IY, self.IV] =  sin_th * dt
            F[self.IT, self.IW] =  dt

            # ── Covariance propagation ─────────────────────────────────────
            self._P = F @ self._P @ F.T + self._Q * dt

    # ── Update helpers ────────────────────────────────────────────────────────

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        """
        Standard EKF update step.  Modifies self._x and self._P in-place.
        Caller must hold self._lock.
        """
        S  = H @ self._P @ H.T + R
        K  = self._P @ H.T @ np.linalg.inv(S)
        y  = z - H @ self._x
        # wrap heading innovations if H touches the theta row
        if H.shape[1] > self.IT and np.any(H[:, self.IT] != 0.0):
            for i in range(len(y)):
                if H[i, self.IT] != 0.0:
                    y[i] = _normalize(y[i])
        self._x = self._x + K @ y
        self._x[self.IT] = _normalize(self._x[self.IT])
        I_KH   = np.eye(self.N) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T  # Joseph form

    # ── Public update methods ─────────────────────────────────────────────────

    def update_odom(self, v_meas: float, omega_meas: float):
        """
        Fuse an odometry measurement (v, omega).
        Thread-safe.
        """
        if not self._initialised:
            return
        z = np.array([v_meas, omega_meas])
        with self._lock:
            self._update(z, self._H_odom, self._R_odom)

    def update_imu_gyro(self, omega_meas: float):
        """
        Fuse IMU gyroscope yaw-rate.
        Thread-safe.
        """
        if not self._initialised:
            return
        z = np.array([omega_meas])
        with self._lock:
            self._update(z, self._H_gyro, self._R_gyro)

    def update_imu_accel(self, ax_meas: float, dt: float):
        """
        Optional: fuse forward (body-frame x) acceleration to correct v.
        H_ax selects v row; innovation = a_x - (v_new - v_old)/dt.
        Only used when use_imu_accel=True and dt > 0.
        Thread-safe.
        """
        if not self._initialised or not self._use_imu_accel or dt <= 0.0:
            return
        H_ax = np.zeros((1, self.N))
        H_ax[0, self.IV] = 1.0 / dt          # approximate: v ≈ v_prev + a*dt
        z = np.array([ax_meas])
        with self._lock:
            self._update(z, H_ax, self._R_ax)

    # ── State accessors ───────────────────────────────────────────────────────

    @property
    def state(self) -> np.ndarray:
        """Return a copy of the current state vector [x, y, theta, v, omega]."""
        with self._lock:
            return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Return a copy of the 5×5 covariance matrix P."""
        with self._lock:
            return self._P.copy()

    @property
    def pose(self) -> Tuple[float, float, float]:
        """Return (x, y, theta) from the current estimate."""
        with self._lock:
            return float(self._x[self.IX]), float(self._x[self.IY]), float(self._x[self.IT])

    @property
    def theta(self) -> float:
        with self._lock:
            return float(self._x[self.IT])

    @property
    def is_initialised(self) -> bool:
        return self._initialised


# ─────────────────────────────────────────────────────────────────────────────
# Straight-line heading controller built on the EKF
# ─────────────────────────────────────────────────────────────────────────────

class EKFStraightLineController:
    """
    Drop-in replacement for DeadReckoningController that uses the EKF pose
    for its heading reference instead of raw IMU yaw.

    Public API mirrors DeadReckoningController exactly so that no other part
    of LaneFollowerAruco needs to change (except adding the odom subscriber
    and passing IMU data here).

    How it works
    ─────────────
    1.  Every incoming /odom message:
          a. predict(dt)          — advance EKF to now
          b. update_odom(v, ω)    — fuse wheel odometry
    2.  Every incoming /imu/data message:
          a. update_imu_gyro(ω_z) — fuse gyro (low-noise yaw-rate)
          b. optionally update_imu_accel(a_x, dt)
    3.  correction(current_yaw, dt) uses ekf.theta as the best heading
        estimate and runs a P+I controller on (ref_yaw - ekf_theta).
        current_yaw (raw IMU) is accepted for API compatibility but ignored
        in favour of the EKF estimate.

    Parameters
    ──────────
    All EKF5DOF noise parameters are exposed here plus the PID gains.
    """

    def __init__(self,
                 # PID gains (match original DR defaults)
                 kp:             float = 0.80,
                 ki:             float = 0.002,
                 max_correction: float = 0.15,
                 integral_limit: float = 0.30,
                 # EKF process noise
                 q_xy:           float = 1e-4,
                 q_theta:        float = 1e-4,
                 q_v:            float = 5e-3,
                 q_omega:        float = 5e-3,
                 # EKF measurement noise
                 r_v_odom:       float = 0.02,
                 r_omega_odom:   float = 0.01,
                 r_omega_imu:    float = 0.005,
                 r_ax_imu:       float = 0.50,
                 use_imu_accel:  bool  = False,
                 active:         bool  = False):

        self._kp             = kp
        self._ki             = ki
        self._max_correction = max_correction
        self._integral_limit = integral_limit

        self._ref_yaw  : Optional[float] = None
        self._integral : float           = 0.0
        self._active   : bool            = active

        self._ekf = EKF5DOF(
            q_xy=q_xy, q_theta=q_theta, q_v=q_v, q_omega=q_omega,
            r_v_odom=r_v_odom, r_omega_odom=r_omega_odom,
            r_omega_imu=r_omega_imu, r_ax_imu=r_ax_imu,
            use_imu_accel=use_imu_accel,
        )

        self._last_odom_time: Optional[float] = None
        self._last_imu_time:  Optional[float] = None
        self._last_ax:        float           = 0.0

    # ── Sensor update entry points ────────────────────────────────────────────

    def update_odom(self, msg) -> None:
        """
        Call from your /odom subscriber.

        Accepts nav_msgs/Odometry or a simple duck-typed object with:
            .twist.twist.linear.x   → forward velocity (m/s)
            .twist.twist.angular.z  → yaw rate         (rad/s)
            .pose.pose.orientation  → quaternion (used for first-init only)
        """
        now = time.monotonic()

        twist = msg.twist.twist
        v_meas     = float(twist.linear.x)
        omega_meas = float(twist.angular.z)

        # ── First call: initialise EKF from odom pose ─────────────────────
        if not self._ekf.is_initialised:
            q = msg.pose.pose.orientation
            theta0 = _yaw_from_quat(q.x, q.y, q.z, q.w)
            p      = msg.pose.pose.position
            self._ekf.initialise(
                x=float(p.x), y=float(p.y), theta=theta0,
                v=v_meas, omega=omega_meas)
            self._last_odom_time = now
            return

        # ── Subsequent calls ──────────────────────────────────────────────
        dt = now - self._last_odom_time if self._last_odom_time is not None else 0.02
        dt = max(0.001, min(0.5, dt))
        self._last_odom_time = now

        self._ekf.predict(dt)
        self._ekf.update_odom(v_meas, omega_meas)

    def update_imu(self, msg) -> None:
        """
        Call from your /imu/data subscriber.

        Accepts sensor_msgs/Imu or a duck-typed object with:
            .angular_velocity.z       → gyro yaw-rate   (rad/s)
            .linear_acceleration.x    → forward accel   (m/s²) [optional]
        """
        now = time.monotonic()
        dt  = now - self._last_imu_time if self._last_imu_time is not None else 0.02
        dt  = max(0.001, min(0.5, dt))
        self._last_imu_time = now

        omega_imu = float(msg.angular_velocity.z)
        self._ekf.update_imu_gyro(omega_imu)

        ax = float(msg.linear_acceleration.x)
        if math.isfinite(ax):
            self._last_ax = ax
            self._ekf.update_imu_accel(ax, dt)

    # ── DR-compatible public API ──────────────────────────────────────────────

    def latch(self, current_yaw: float) -> None:
        """Lock the EKF heading as the straight-ahead reference."""
        # Prefer EKF theta if available; fall back to raw IMU yaw.
        ref = self._ekf.theta if self._ekf.is_initialised else current_yaw
        self._ref_yaw  = ref
        self._integral = 0.0
        self._active   = True

    def pause(self) -> None:
        """Stop producing corrections."""
        self._active   = False
        self._integral = 0.0

    def resume(self, current_yaw: float) -> None:
        """Re-latch a fresh reference and re-enable corrections."""
        self.latch(current_yaw)

    @property
    def is_active(self) -> bool:
        return self._active and self._ref_yaw is not None

    @property
    def reference_yaw(self) -> Optional[float]:
        return self._ref_yaw

    def correction(self, current_yaw: float, dt: float) -> float:
        """
        Return the angular-velocity correction (rad/s) to ADD to the
        lane-PID output.  Positive = CCW.  Returns 0.0 if not active.

        Uses self._ekf.theta instead of current_yaw for better accuracy.
        current_yaw is kept as fallback when EKF is not yet initialised.
        """
        if not self.is_active:
            return 0.0

        estimated_theta = (self._ekf.theta
                           if self._ekf.is_initialised
                           else current_yaw)

        heading_error = _normalize(self._ref_yaw - estimated_theta)

        p_term = self._kp * heading_error

        self._integral = max(-self._integral_limit,
                             min(self._integral_limit,
                                 self._integral + heading_error * dt))
        i_term = self._ki * self._integral

        output = p_term + i_term
        return max(-self._max_correction, min(self._max_correction, output))

    def heading_error_deg(self, current_yaw: float) -> float:
        if self._ref_yaw is None:
            return 0.0
        estimated_theta = (self._ekf.theta
                           if self._ekf.is_initialised
                           else current_yaw)
        return math.degrees(_normalize(self._ref_yaw - estimated_theta))

    # ── Extra diagnostics not in the old DR API ───────────────────────────────

    @property
    def pose(self) -> Tuple[float, float, float]:
        """(x, y, theta) from the EKF. Use for logging / visualisation."""
        return self._ekf.pose

    @property
    def covariance(self) -> np.ndarray:
        """5×5 P matrix. Use for diagnostics."""
        return self._ekf.covariance

    @property
    def ekf(self) -> EKF5DOF:
        """Direct access to the underlying EKF (for advanced use)."""
        return self._ekf


# ─────────────────────────────────────────────────────────────────────────────
# Minimal integration patch for LaneFollowerAruco
# ─────────────────────────────────────────────────────────────────────────────
# The patch below is shown as a code comment so it can be copy-pasted into
# lane_follower_aruco.py with minimal effort.  No other method needs to change.
#
# STEP 1 ─ Replace the import at the top of lane_follower_aruco.py:
#
#   - from ekf_localization import DeadReckoningController   # old (doesn't exist)
#   + from ekf_localization import EKFStraightLineController
#
# STEP 2 ─ In LaneFollowerAruco.__init__(), replace:
#
#   - self._dr = DeadReckoningController(
#   -     kp=self._dr_kp, ki=self._dr_ki,
#   -     max_correction=self._dr_max_correction,
#   -     integral_limit=self._dr_integral_limit,
#   - )
#   + self._dr = EKFStraightLineController(
#   +     kp=self._dr_kp,
#   +     ki=self._dr_ki,
#   +     max_correction=self._dr_max_correction,
#   +     integral_limit=self._dr_integral_limit,
#   +     # EKF noise — tune per robot:
#   +     q_theta=1e-4,   r_omega_imu=0.005,  r_omega_odom=0.01,
#   + )
#
# STEP 3 ─ Add an /odom subscriber after the existing subscriptions:
#
#   + from nav_msgs.msg import Odometry
#   + self.create_subscription(
#   +     Odometry, '/odom', self._odom_cb, 10)
#
# STEP 4 ─ Add the two callbacks (they can live anywhere in the class):
#
#   + def _odom_cb(self, msg):
#   +     self._assert_single_thread()
#   +     self._dr.update_odom(msg)
#   +
#   + # Update _imu_cb to also feed the EKF:
#   + def _imu_cb(self, msg):
#   +     self._assert_single_thread()
#   +     prev_yaw          = self._current_yaw
#   +     self._current_yaw = yaw_from_imu(msg)      # keep for compatibility
#   +     self._dr.update_imu(msg)                   # <-- NEW
#   +     if prev_yaw is None and self._current_yaw is not None:
#   +         self._dr.latch(self._current_yaw)
#   +         self.get_logger().info(
#   +             f'[EKF-DR] Initial heading latched: '
#   +             f'{math.degrees(self._current_yaw):.2f}°')
#   +     ay = msg.linear_acceleration.y
#   +     if math.isfinite(ay):
#   +         self._imu_accel_y_filt = ema_update(
#   +             self._imu_accel_y_filt, ay, self._imu_lpf_alpha)
#
# That's it.  All rotation-state-machine code and lane-follow PID code is
# unchanged because EKFStraightLineController.correction() / is_active /
# reference_yaw match the old DeadReckoningController signatures exactly.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Standalone ROS 2 node (optional — use when you want EKF pose on its own topic)
# ─────────────────────────────────────────────────────────────────────────────

if _ROS_AVAILABLE:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseWithCovarianceStamped

    class EKFLocalizationNode(Node):
        """
        Standalone ROS 2 node that publishes /ekf/pose
        (geometry_msgs/PoseWithCovarianceStamped) from /odom + /imu/data.

        This node is independent of LaneFollowerAruco and can be run alongside
        it for visualisation or as a reference for the integration.

        Launch:
            ros2 run <your_package> ekf_localization_node
        """

        def __init__(self):
            super().__init__('ekf_localization')

            self.declare_parameter('q_xy',          1e-4)
            self.declare_parameter('q_theta',        1e-4)
            self.declare_parameter('q_v',            5e-3)
            self.declare_parameter('q_omega',        5e-3)
            self.declare_parameter('r_v_odom',       0.02)
            self.declare_parameter('r_omega_odom',   0.01)
            self.declare_parameter('r_omega_imu',    0.005)
            self.declare_parameter('use_imu_accel',  False)
            self.declare_parameter('publish_rate',   50.0)  # Hz

            g = self.get_parameter
            self._ctrl = EKFStraightLineController(
                q_xy=g('q_xy').value,
                q_theta=g('q_theta').value,
                q_v=g('q_v').value,
                q_omega=g('q_omega').value,
                r_v_odom=g('r_v_odom').value,
                r_omega_odom=g('r_omega_odom').value,
                r_omega_imu=g('r_omega_imu').value,
                use_imu_accel=g('use_imu_accel').value,
            )

            self._pub_pose = self.create_publisher(
                PoseWithCovarianceStamped, '/ekf/pose', 10)

            self.create_subscription(Odometry, '/odom',     self._odom_cb, 10)
            self.create_subscription(Imu,      '/imu/data', self._imu_cb,  10)

            rate = g('publish_rate').value
            self.create_timer(1.0 / rate, self._publish_cb)

            self.get_logger().info('[EKFLocalizationNode] Ready — publishing /ekf/pose')

        def _odom_cb(self, msg: Odometry):
            self._ctrl.update_odom(msg)

        def _imu_cb(self, msg: Imu):
            self._ctrl.update_imu(msg)

        def _publish_cb(self):
            if not self._ctrl.ekf.is_initialised:
                return

            x, y, theta = self._ctrl.pose
            P = self._ctrl.covariance

            msg = PoseWithCovarianceStamped()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = 'odom'

            msg.pose.pose.position.x = x
            msg.pose.pose.position.y = y
            msg.pose.pose.position.z = 0.0

            # Convert theta back to quaternion
            msg.pose.pose.orientation.z = math.sin(theta / 2.0)
            msg.pose.pose.orientation.w = math.cos(theta / 2.0)
            msg.pose.pose.orientation.x = 0.0
            msg.pose.pose.orientation.y = 0.0

            # ROS covariance is 6×6 [x,y,z,roll,pitch,yaw]; copy 2D subset
            cov = [0.0] * 36
            cov[0]  = float(P[EKF5DOF.IX, EKF5DOF.IX])   # xx
            cov[1]  = float(P[EKF5DOF.IX, EKF5DOF.IY])   # xy
            cov[6]  = float(P[EKF5DOF.IY, EKF5DOF.IX])   # yx
            cov[7]  = float(P[EKF5DOF.IY, EKF5DOF.IY])   # yy
            cov[35] = float(P[EKF5DOF.IT, EKF5DOF.IT])   # yaw-yaw
            msg.pose.covariance = cov

            self._pub_pose.publish(msg)


    def main(args=None):
        rclpy.init(args=args)
        node = EKFLocalizationNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()


    if __name__ == '__main__':
        main()