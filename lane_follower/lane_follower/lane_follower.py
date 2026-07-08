import math
import time
import json
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry


# ── Mode constants ─────────────────────────────────────────────────────────────
MODE_LANE_FOLLOW = "LANE_FOLLOW"
MODE_PRE_ROTATE_FORWARD = "PRE_ROTATE_FORWARD"
MODE_WAIT_IMU    = "WAIT_IMU"
MODE_ROTATE      = "ROTATE"
MODE_FORWARD_RUN = "FORWARD_RUN"
MODE_FINAL_HOLD  = "FINAL_HOLD"
MODE_RETURN_LANE = "RETURN_LANE"


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def ema_update(current: float, new_sample: float, alpha: float) -> float:
    return alpha * new_sample + (1.0 - alpha) * current


def slew(current: float, target: float, max_rate: float, dt: float) -> float:
    max_delta = max_rate * dt
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)


def yaw_from_imu(msg: Imu) -> float:
    q = msg.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    while angle >  math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle


# ─────────────────────────────────────────────────────────────────────────────
# Dead-Reckoning Straight-Line Controller
# ─────────────────────────────────────────────────────────────────────────────
class DeadReckoningController:
    def __init__(self,
                 kp: float             = 0.80,
                 ki: float             = 0.002,
                 max_correction: float = 0.15,
                 integral_limit: float = 0.30,
                 active: bool          = False):
        self.kp              = kp
        self.ki              = ki
        self.max_correction  = max_correction
        self.integral_limit  = integral_limit
        self._ref_yaw        = None
        self._integral       = 0.0
        self._active         = active

    def latch(self, current_yaw: float):
        self._ref_yaw  = current_yaw
        self._integral = 0.0
        self._active   = True

    def pause(self):
        self._active   = False
        self._integral = 0.0

    def resume(self, current_yaw: float):
        self.latch(current_yaw)

    @property
    def is_active(self) -> bool:
        return self._active and self._ref_yaw is not None

    @property
    def reference_yaw(self) -> float:
        return self._ref_yaw

    def correction(self, current_yaw: float, dt: float) -> float:
        if not self.is_active:
            return 0.0
        heading_error = normalize_angle(self._ref_yaw - current_yaw)
        p_term = self.kp * heading_error
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit,
                                 self._integral + heading_error * dt))
        i_term = self.ki * self._integral
        output = p_term + i_term
        return max(-self.max_correction, min(self.max_correction, output))

    def heading_error_deg(self, current_yaw: float) -> float:
        if self._ref_yaw is None:
            return 0.0
        return math.degrees(normalize_angle(self._ref_yaw - current_yaw))


# ─────────────────────────────────────────────────────────────────────────────
# Odometry-Based Lateral Drift Corrector
# ─────────────────────────────────────────────────────────────────────────────
class OdomDriftCorrector:
    def __init__(self,
                 kp_lat: float        = 0.8,
                 kd_lat: float        = 0.3,
                 max_correction: float = 0.12,
                 active: bool          = False):
        self.kp_lat         = kp_lat
        self.kd_lat         = kd_lat
        self.max_correction = max_correction
        self._x0: float | None = None
        self._y0: float | None = None
        self._yaw0: float | None = None
        self._lateral_error      = 0.0
        self._prev_lateral_error = 0.0
        self._active = active

    def latch(self, x: float, y: float, yaw: float):
        self._x0   = x
        self._y0   = y
        self._yaw0 = yaw
        self._lateral_error      = 0.0
        self._prev_lateral_error = 0.0
        self._active = True

    def pause(self):
        self._active = False

    def resume(self, x: float, y: float, yaw: float):
        self.latch(x, y, yaw)

    def reset(self):
        self._x0   = None
        self._y0   = None
        self._yaw0 = None
        self._lateral_error      = 0.0
        self._prev_lateral_error = 0.0

    @property
    def is_active(self) -> bool:
        return self._active and self._x0 is not None

    def update(self, x: float, y: float):
        if self._x0 is None:
            return
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        dx = x - self._x0
        dy = y - self._y0
        self._prev_lateral_error = self._lateral_error
        # Cross-track error along the robot's "left" axis, in the frame
        # latched at self._yaw0 (REP103: x=forward, y=left, +yaw=CCW/left).
        # Positive => robot has drifted LEFT of the reference line.
        # Negative => robot has drifted RIGHT of the reference line.
        self._lateral_error = -dx * math.sin(self._yaw0) + dy * math.cos(self._yaw0)

    def correction(self, dt: float) -> float:
        if not self.is_active:
            return 0.0
        if dt > 1e-6:
            d_error = (self._lateral_error - self._prev_lateral_error) / dt
        else:
            d_error = 0.0
        # ROOT-CAUSE FIX: this must be NEGATIVE feedback. Under REP103,
        # positive angular.z turns the robot LEFT (CCW). If lateral_error
        # is positive (drifted left), we need to steer RIGHT to converge
        # back to the line, i.e. command a NEGATIVE angular.z. The
        # previous implementation returned +kp*lateral_error (unsigned),
        # which meant a leftward drift commanded a further LEFT turn and
        # a rightward drift commanded a further RIGHT turn — a positive
        # feedback loop. Any small real-world bias (wheel/motor asymmetry,
        # encoder bias, camera mount offset) would get amplified in
        # whichever direction it first appeared, producing exactly the
        # sustained one-sided drift observed on hardware. Negating here
        # makes this a proper stabilizing cross-track controller.
        output = -(self.kp_lat * self._lateral_error + self.kd_lat * d_error)
        return max(-self.max_correction, min(self.max_correction, output))

    @property
    def lateral_error(self) -> float:
        return self._lateral_error


# ─────────────────────────────────────────────────────────────────────────────
# Delta-time PID
# ─────────────────────────────────────────────────────────────────────────────
class DtPID:
    def __init__(self, kp, ki, kd, output_limit, integral_limit,
                 deriv_lpf_alpha=0.2):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.output_limit    = output_limit
        self.integral_limit  = integral_limit
        self.deriv_lpf_alpha = deriv_lpf_alpha
        self._integral    = 0.0
        self._prev_meas   = None
        self._deriv_filt  = 0.0

    def reset(self):
        self._integral   = 0.0
        self._prev_meas  = None
        self._deriv_filt = 0.0

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        error  = setpoint - measurement
        p_term = self.kp * error
        tentative_integral = self._integral + error * dt
        i_term_tentative   = self.ki * tentative_integral
        if abs(p_term + i_term_tentative) < self.output_limit:
            self._integral = max(-self.integral_limit,
                                 min(self.integral_limit, tentative_integral))
        i_term = self.ki * self._integral
        if self._prev_meas is None:
            raw_deriv = 0.0
        else:
            raw_deriv = -(measurement - self._prev_meas) / dt
        self._deriv_filt = ema_update(self._deriv_filt, raw_deriv,
                                      self.deriv_lpf_alpha)
        d_term = self.kd * self._deriv_filt
        self._prev_meas = measurement
        output = p_term + i_term + d_term
        return max(-self.output_limit, min(self.output_limit, output))


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic dead-band
# ─────────────────────────────────────────────────────────────────────────────
class DynamicDeadband:
    def __init__(self, base_width, min_width, max_width, window=20):
        self.base_width = base_width
        self.min_width  = min_width
        self.max_width  = max_width
        self._history   = []
        self._window    = window

    def update_and_check(self, error: float) -> bool:
        self._history.append(error)
        if len(self._history) > self._window:
            self._history.pop(0)
        if len(self._history) < 3:
            width = self.base_width
        else:
            mean       = sum(self._history) / len(self._history)
            variance   = sum((e - mean) ** 2 for e in self._history) / len(self._history)
            normalised = min(1.0, math.sqrt(variance) / (self.base_width * 2))
            width = self.max_width - normalised * (self.max_width - self.min_width)
        return abs(error) <= width


# ─────────────────────────────────────────────────────────────────────────────
# ArUco target-machine navigation map
# ─────────────────────────────────────────────────────────────────────────────
#
# forward_before_rotation: static fallback distance (m) to drive before
#   rotating. At runtime this is OVERRIDDEN by the live depth_m from
#   the depth camera when available (see _aruco_cb). Set to a reasonable
#   fallback in case the depth reading is invalid.
#
# distance_offset: optional per-entry offset (m) applied ONLY to the live
#   depth_m reading before it is used as forward_before_rotation.
#   e.g. distance_offset=0.20 drives 20 cm farther than the raw camera
#   reading; distance_offset=-0.20 stops 20 cm short. Defaults to 0.0 if
#   omitted. Does NOT affect the static map fallback value or
#   forward_after_rotation.
#
# NOTE: both the pre-rotation leg (forward_before_rotation, driven for
# depth_m + distance_offset) and the post-rotation leg
# (forward_after_rotation) now steer using the lane-follow PID whenever
# a lane signal is available, and fall back to driving straight only
# when the lane is not detected. See _drive_distance_leg(follow_lane=True)
# and its two callers in _rotation_loop.
ARUCO_MAP = {
    0: [
        {"machine": "B", "forward_before_rotation": 0.0, "direction": "u-turn",
         "final": True, "forward_after_rotation": 0.0, "distance_offset": -0.5},
        
    ],
    1: [
        {"machine": "D", "forward_before_rotation": 0.00,"direction": "right", "final": False,
         "forward_after_rotation": 0.0, "distance_offset": 2.2},
    ],
    2: [
        {"machine": "B", "forward_before_rotation": 0.0,"direction": "left", "final": False,
         "forward_after_rotation": 0.0, "distance_offset": -0.05},
    ],
    3: [
        {"machine": "B","forward_before_rotation": 0.0, "direction": "right", "final": False,
         "forward_after_rotation": 0, "distance_offset": -1.1},
    ],
    4: [
        {"machine": "C","forward_before_rotation": 0.0, "direction": "left", "final": True,
         "forward_after_rotation": 5.4, "distance_offset": 1.0},
    ],
    5: [
        {"machine": "A","forward_before_rotation": 0.0, "direction": "left", "final": False,
         "forward_after_rotation": 0.10, "distance_offset": 1.35},
    ],
    6: [
        {"machine": "A","forward_before_rotation": 0.0, "direction": "straight", "final": True,
         "forward_after_rotation": 5.0, "distance_offset": 0.0},
    ],
   
}

NAV_TARGETS = ["A", "B", "C","D"]

ARUCO_TRIGGER_COOLDOWN_S = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main node
# ─────────────────────────────────────────────────────────────────────────────
class LaneFollowerAruco(Node):

    def __init__(self):
        super().__init__('lane_follower_aruco')

        self._callback_thread: threading.Thread | None = None

        # ── Lane-follower parameters ──────────────────────────────────────
        self.declare_parameter('kp',                  0.00051)
        self.declare_parameter('ki',                  0.000006)
        self.declare_parameter('kd',                  0.0016)
        self.declare_parameter('target_pixels',      -300.0)
        self.declare_parameter('dead_band_base',       20.0)
        self.declare_parameter('dead_band_min',        12.0)
        self.declare_parameter('dead_band_max',        35.0)
        self.declare_parameter('max_angular',           0.8)
        self.declare_parameter('integral_limit',      120.0)
        self.declare_parameter('deriv_lpf_alpha',       0.30)
        self.declare_parameter('input_ema_alpha',       0.15)
        self.declare_parameter('output_ema_alpha',      0.20)
        self.declare_parameter('angular_slew_rate',     0.35)
        self.declare_parameter('linear_speed',          0.5)
        self.declare_parameter('min_linear_speed',      0.15)
        self.declare_parameter('linear_slew_rate',      0.18)
        self.declare_parameter('speed_angular_scale',   0)
        self.declare_parameter('lane_timeout',          0.2)
        self.declare_parameter('imu_gain',              0.005)
        self.declare_parameter('imu_noise_floor',       0.30)
        self.declare_parameter('imu_lpf_alpha',         0.08)
        self.declare_parameter('angular_trim',          0.005)
        self.declare_parameter('lane_loss_stop_s',      0.0)
        self.declare_parameter('lookahead_far_weight',  0.70)
        self.declare_parameter('lookahead_near_weight', 0.30)

        # ── Dead-reckoning parameters ─────────────────────────────────────
        self.declare_parameter('dr_kp',              0.40)
        self.declare_parameter('dr_ki',              0.001)
        self.declare_parameter('dr_max_correction',  0.05)
        self.declare_parameter('dr_integral_limit',  0.15)
        self.declare_parameter('dr_lane_blend',      0.20)
        self.declare_parameter('dr_no_lane_blend',   0.60)

        # ── Odometry-based lateral drift corrector parameters ─────────────
        # NOTE (tuning pass, root-cause fix for rightward drift):
        # The sign bug in OdomDriftCorrector.correction() has been fixed
        # (see class definition above) — it was previously reinforcing
        # whatever direction the robot drifted in instead of correcting
        # it. Because that loop is now doing real corrective work for the
        # first time, gains below start conservative rather than reusing
        # the old (never-actually-tested) defaults. Recommended field
        # tuning procedure once deployed:
        #   1. Run with lat_drift_enabled=False to confirm camera-only
        #      steering baseline (this isolates whether any residual
        #      drift is a camera/target_pixels calibration issue rather
        #      than an odometry-correction issue).
        #   2. Re-enable with the values below. Watch the [DR] log line
        #      (lat_err / lat_corr) over a straight run.
        #   3. If lat_err grows and lat_corr is clearly opposing it but
        #      too weakly -> raise lat_kp_lat in +0.1 steps.
        #   4. If the robot oscillates/hunts side-to-side -> raise
        #      lat_kd_lat in +0.05 steps, or lower lat_kp_lat.
        #   5. Once stable and tracking well, raise lat_drift_weight
        #      toward 0.5 in +0.05 steps to let it contribute more
        #      relative to lane_angular.
        self.declare_parameter('lat_drift_enabled',      True)
        self.declare_parameter('lat_kp_lat',             0.60)
        self.declare_parameter('lat_kd_lat',             0.25)
        self.declare_parameter('lat_max_correction',     0.10)
        self.declare_parameter('lat_drift_weight',       0.35)

        # ── Rotation parameters ───────────────────────────────────────────
        self.declare_parameter('rot_cruise_speed',      1.5)
        self.declare_parameter('rot_max_speed',         10.0)
        self.declare_parameter('rot_min_speed',         1.5)
        self.declare_parameter('rot_slow_zone',         0.17)
        self.declare_parameter('rot_slow_gain',         1.2)
        self.declare_parameter('rot_smoothing',         0.70)
        self.declare_parameter('rot_target_turn',       1.5708)
        self.declare_parameter('rot_angular_accel',     3.0)

        self.declare_parameter('trigger_distance',      3.0)

        # ── Forward-run parameters ────────────────────────────────────────
        self.declare_parameter('fwd_cruise_speed',      0.32)
        self.declare_parameter('fwd_distance_tolerance', 0.0075)
        self.declare_parameter('fwd_max_duration_s',    40.0)
        self.declare_parameter('final_hold_duration_s', 2.0)

        self._read_params()

        # ── Controllers ───────────────────────────────────────────────────
        self._pid = DtPID(
            kp=self._kp, ki=self._ki, kd=self._kd,
            output_limit=self._max_angular,
            integral_limit=self._integral_limit,
            deriv_lpf_alpha=self._deriv_lpf_alpha,
        )
        self._deadband = DynamicDeadband(
            base_width=self._dead_band_base,
            min_width=self._dead_band_min,
            max_width=self._dead_band_max,
        )
        self._dr = DeadReckoningController(
            kp=self._dr_kp,
            ki=self._dr_ki,
            max_correction=self._dr_max_correction,
            integral_limit=self._dr_integral_limit,
        )
        self._lat_drift = OdomDriftCorrector(
            kp_lat=self._lat_kp_lat,
            kd_lat=self._lat_kd_lat,
            max_correction=self._lat_max_correction,
        )

        # ── Lane-follower state ───────────────────────────────────────────
        self._input_filtered    = 0.0
        self._output_angular    = 0.0
        self._current_linear    = 0.0
        self._current_angular   = 0.0
        self._imu_accel_y_filt  = 0.0
        self._lane_detected     = False
        self._last_msg_time     = self.get_clock().now()
        self._first_meas        = True

        self._lane_prev_time    = time.monotonic()
        self._rot_prev_time     = time.monotonic()
        self._odom_prev_time    = time.monotonic()

        # ── Odometry state ────────────────────────────────────────────────
        self._odom_x:   float | None = None
        self._odom_y:   float | None = None
        self._odom_yaw: float | None = None

        self._px_near: float | None = None
        self._px_far:  float | None = None

        self._lane_loss_start: float | None = None
        self._last_lane_rx_time = None   # rclpy Time of last raw pixel msg (any mode)
        self._distance_leg_lane_active = False  # for edge-triggered log messages

        # ── Rotation state ────────────────────────────────────────────────
        self._mode             = MODE_LANE_FOLLOW
        self._turn_direction   = 0
        self._current_yaw      = None
        self._start_yaw        = None
        self._rot_prev_angular = 0.0
        self._forward_start    = None

        # ── ArUco navigation state ────────────────────────────────────────
        self._nav_targets            = list(NAV_TARGETS)
        self._current_target_index   = 0
        self._pending_direction:str | None = None
        self._pending_final:bool          = False
        self._pending_forward:float       = 0.0
        self._pending_forward_before:float = 0.0
        self._aruco_last_trigger_time: dict[int, float] = {}
        self._fwd_start_x: float | None = None
        self._fwd_start_y: float | None = None
        self._fwd_target_dist: float = 0.0
        self._fwd_start_time: float | None = None
        self._final_hold_start: float | None = None

        # ── ROS interfaces ────────────────────────────────────────────────
        self._pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.create_subscription(
            Float32, '/lane/distance_pixels_near', self._lane_near_cb, 10)
        self.create_subscription(
            Float32, '/lane/distance_pixels_far',  self._lane_far_cb,  10)
        self.create_subscription(
            Float32, '/lane/distance_pixels', self._lane_near_cb, 10)
        self.create_subscription(Imu,      '/imu/data',      self._imu_cb,   10)
        self.create_subscription(Odometry, '/odom',          self._odom_cb,  10)
        # CHANGED: subscribe to the qr_scanner node's actual output topic
        # (previously '/aruco_markers', which qr_scanner never publishes).
        self.create_subscription(String,   '/aruco_detections', self._aruco_cb, 10)

        self.create_timer(0.02, self._rotation_loop)
        self.create_timer(0.05, self._watchdog_cb)

        self.get_logger().info('[LaneFollowerAruco] Ready.')

    # ── Parameter loader ──────────────────────────────────────────────────────
    def _read_params(self):
        g = self.get_parameter
        self._kp                     = g('kp').value
        self._ki                     = g('ki').value
        self._kd                     = g('kd').value
        self._target_px              = float(g('target_pixels').value)
        self._dead_band_base         = g('dead_band_base').value
        self._dead_band_min          = g('dead_band_min').value
        self._dead_band_max          = g('dead_band_max').value
        self._max_angular            = g('max_angular').value
        self._integral_limit         = g('integral_limit').value
        self._deriv_lpf_alpha        = g('deriv_lpf_alpha').value
        self._input_ema_alpha        = g('input_ema_alpha').value
        self._output_ema_alpha       = g('output_ema_alpha').value
        self._angular_slew_rate      = g('angular_slew_rate').value
        self._linear_speed           = g('linear_speed').value
        self._min_linear_speed       = g('min_linear_speed').value
        self._linear_slew_rate       = g('linear_slew_rate').value
        self._speed_angular_scale    = g('speed_angular_scale').value
        self._lane_timeout           = g('lane_timeout').value
        self._imu_gain               = g('imu_gain').value
        self._imu_noise_floor        = g('imu_noise_floor').value
        self._imu_lpf_alpha          = g('imu_lpf_alpha').value
        self._angular_trim           = g('angular_trim').value
        self._lane_loss_stop_s       = g('lane_loss_stop_s').value
        self._lookahead_far_weight   = g('lookahead_far_weight').value
        self._lookahead_near_weight  = g('lookahead_near_weight').value

        self._dr_kp              = g('dr_kp').value
        self._dr_ki              = g('dr_ki').value
        self._dr_max_correction  = g('dr_max_correction').value
        self._dr_integral_limit  = g('dr_integral_limit').value
        self._dr_lane_blend      = g('dr_lane_blend').value
        self._dr_no_lane_blend   = g('dr_no_lane_blend').value

        self._lat_drift_enabled  = g('lat_drift_enabled').value
        self._lat_kp_lat         = g('lat_kp_lat').value
        self._lat_kd_lat         = g('lat_kd_lat').value
        self._lat_max_correction = g('lat_max_correction').value
        self._lat_drift_weight   = g('lat_drift_weight').value

        self._rot_cruise_speed   = g('rot_cruise_speed').value
        self._rot_max_speed      = g('rot_max_speed').value
        self._rot_min_speed      = g('rot_min_speed').value
        self._rot_slow_zone      = g('rot_slow_zone').value
        self._rot_slow_gain      = g('rot_slow_gain').value
        self._rot_smoothing      = g('rot_smoothing').value
        self._rot_target_turn    = g('rot_target_turn').value
        self._rot_angular_accel  = g('rot_angular_accel').value
        self._trigger_distance   = g('trigger_distance').value

        self._fwd_cruise_speed       = g('fwd_cruise_speed').value
        self._fwd_distance_tolerance = g('fwd_distance_tolerance').value
        self._fwd_max_duration_s     = g('fwd_max_duration_s').value
        self._final_hold_duration_s  = g('final_hold_duration_s').value

    # ── Per-consumer dt helpers ───────────────────────────────────────────────
    def _get_lane_dt(self) -> float:
        now = time.monotonic()
        dt  = now - self._lane_prev_time
        self._lane_prev_time = now
        return max(0.001, min(0.5, dt))

    def _get_rot_dt(self) -> float:
        now = time.monotonic()
        dt  = now - self._rot_prev_time
        self._rot_prev_time = now
        return max(0.001, min(0.5, dt))

    def _get_odom_dt(self) -> float:
        now = time.monotonic()
        dt  = now - self._odom_prev_time
        self._odom_prev_time = now
        return max(0.001, min(0.5, dt))

    # ── Single-thread guard ───────────────────────────────────────────────────
    def _assert_single_thread(self):
        t = threading.current_thread()
        if self._callback_thread is None:
            self._callback_thread = t
        elif self._callback_thread is not t:
            raise RuntimeError(
                'LaneFollowerAruco is not thread-safe. '
                'Use SingleThreadedExecutor.')

    # ── IMU callback ───────────────────────────────────────────────────────────
    def _imu_cb(self, msg: Imu):
        self._assert_single_thread()
        prev_yaw          = self._current_yaw
        self._current_yaw = yaw_from_imu(msg)
        if prev_yaw is None and self._current_yaw is not None:
            self._dr.latch(self._current_yaw)
            self.get_logger().info(
                f'[DR] Initial heading latched: '
                f'{math.degrees(self._current_yaw):.2f}°')
        ay = msg.linear_acceleration.y
        if math.isfinite(ay):
            self._imu_accel_y_filt = ema_update(
                self._imu_accel_y_filt, ay, self._imu_lpf_alpha)

    # ── Odometry callback ─────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        self._assert_single_thread()
        _ = self._get_odom_dt()

        pos = msg.pose.pose.position
        x, y = pos.x, pos.y
        if not (math.isfinite(x) and math.isfinite(y)):
            return

        self._odom_x = x
        self._odom_y = y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._odom_yaw = math.atan2(siny_cosp, cosy_cosp)

        if not self._lat_drift_enabled:
            return

        self._lat_drift.update(x, y)

        if self._dr.is_active and not self._lat_drift.is_active:
            yaw_ref = self._current_yaw if self._current_yaw is not None else self._odom_yaw
            if yaw_ref is not None:
                self._lat_drift.latch(x, y, yaw_ref)
        elif not self._dr.is_active and self._lat_drift.is_active:
            self._lat_drift.pause()

    def _imu_trim(self) -> float:
        if self._dr.is_active:
            return 0.0
        ay = self._imu_accel_y_filt
        if abs(ay) < self._imu_noise_floor:
            return 0.0
        trim  = -self._imu_gain * ay
        limit = self._max_angular * 0.10
        return max(-limit, min(limit, trim))

    # ── Lane subscriptions ────────────────────────────────────────────────────
    def _lane_near_cb(self, msg: Float32):
        self._assert_single_thread()
        self._last_lane_rx_time = self.get_clock().now()
        self._px_near = msg.data if math.isfinite(msg.data) else None
        self._lane_cb_combined()

    def _lane_far_cb(self, msg: Float32):
        self._assert_single_thread()
        self._last_lane_rx_time = self.get_clock().now()
        self._px_far = msg.data if math.isfinite(msg.data) else None

    def _lane_signal_fresh(self) -> bool:
        """True if a raw lane pixel message has arrived within lane_timeout,
        regardless of current mode. Used by distance legs (pre-rotate /
        forward-run) to detect a stalled lane topic that _watchdog_cb
        would not otherwise catch outside MODE_LANE_FOLLOW."""
        if self._last_lane_rx_time is None:
            return False
        elapsed = (self.get_clock().now() - self._last_lane_rx_time).nanoseconds * 1e-9
        return elapsed <= self._lane_timeout

    def _blended_pixel(self) -> float | None:
        near = self._px_near
        far  = self._px_far
        fw   = self._lookahead_far_weight
        nw   = self._lookahead_near_weight
        if near is None and far is None:
            return None
        if far is None or fw == 0.0:
            return near
        if near is None:
            return far
        total = fw + nw
        if total <= 0.0:
            return near
        return (fw * far + nw * near) / total

    # ─────────────────────────────────────────────────────────────────────
    # ArUco target-machine navigation
    # ─────────────────────────────────────────────────────────────────────
    def _current_target_machine(self) -> str | None:
        if self._current_target_index >= len(self._nav_targets):
            return None
        return self._nav_targets[self._current_target_index]

    def _aruco_cooldown_active(self, marker_id: int) -> bool:
        last = self._aruco_last_trigger_time.get(marker_id)
        if last is None:
            return False
        return (time.monotonic() - last) < ARUCO_TRIGGER_COOLDOWN_S

    def _aruco_cb(self, msg: String):
        """
        Target-machine-driven ArUco navigation.

        Distance extraction (simplified to match qr_scanner.py):
        ───────────────────────────────────────────────────────────
        qr_scanner.py (ArucoDetector) publishes a JSON list of detections
        on /aruco_detections, each shaped like:
            {"id": <int>, "depth": <float>, "x": <float>,
             "y": <float>, "z": <float>}

        "depth" there is the raw depth-camera reading (metres) sampled at
        the pixel centre of the marker. This node now uses that single
        "depth" field directly as the distance measurement — the previous
        horiz_dist_m / slant_dist_m distinction has been removed since
        qr_scanner.py does not publish those fields.

        Priority order for the trigger distance check:
          1. depth       (from qr_scanner.py — preferred, live camera reading)
          2. distance_m  (legacy key, kept for backward compat)
          3. distance    (legacy key)
          4. float('inf') if none of the above are valid

        forward_before_rotation override:
        ──────────────────────────────────
        When a valid depth (> 0) is available it is used as the actual
        forward_before_rotation distance, replacing the static map value.
        This makes the robot drive exactly to the marker position rather
        than a hardcoded estimate. If the depth reading is invalid (<= 0
        or missing) the static map value is used as a fallback.

        distance_offset override:
        ──────────────────────────
        Each ARUCO_MAP entry may define a "distance_offset" (meters). This
        is added to the live depth reading (only) before it is used as
        forward_before_rotation, letting individual markers/entries drive
        a bit farther (+) or stop a bit short (-) of the raw camera
        reading. Defaults to 0.0 if not present on the entry. Does not
        apply to the static map fallback or to forward_after_rotation.
        """
        self._assert_single_thread()

        if self._mode != MODE_LANE_FOLLOW:
            self.get_logger().debug(
                f'ArUco message ignored — currently in mode {self._mode}')
            return

        target_machine = self._current_target_machine()
        if target_machine is None:
            self.get_logger().info(
                'All nav targets complete — ignoring further ArUco markers.')
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Failed to parse /aruco_detections JSON')
            return

        if isinstance(payload, dict):
            markers = payload.get('markers', [])
        elif isinstance(payload, list):
            markers = payload
        else:
            self.get_logger().warn(
                f'Unexpected /aruco_detections payload type: '
                f'{type(payload).__name__}')
            return

        for marker in markers:
            if not isinstance(marker, dict):
                self.get_logger().warn(
                    f'Skipping non-dict marker entry: {marker!r}')
                continue

            raw_id = marker.get('id')
            try:
                marker_id = int(raw_id)
            except (TypeError, ValueError):
                self.get_logger().warn(
                    f'ArUco marker with non-numeric id: {raw_id!r}')
                continue

            # ── CHANGED: read qr_scanner.py's "depth" field directly ───────
            depth_m = None
            raw_depth = marker.get('depth')
            try:
                v = float(raw_depth)
                if v > 0.0:
                    depth_m = v
            except (TypeError, ValueError):
                pass

            if depth_m is not None:
                distance     = depth_m
                distance_src = 'depth'
            else:
                raw_legacy = marker.get('distance_m', marker.get('distance'))
                try:
                    distance = float(raw_legacy)
                    distance_src = 'legacy'
                except (TypeError, ValueError):
                    distance     = float('inf')
                    distance_src = 'none'
            # ── END CHANGED ───────────────────────────────────────────────

            self.get_logger().info(
                f'ArUco seen: id={marker_id} '
                f'distance={distance:.3f} m [{distance_src}] '
                f'(trigger <= {self._trigger_distance} m) '
                f'target_machine={target_machine}')

            if distance > self._trigger_distance:
                continue

            if self._aruco_cooldown_active(marker_id):
                self.get_logger().debug(
                    f'ArUco id={marker_id} in cooldown — ignoring')
                continue

            entries = ARUCO_MAP.get(marker_id)
            if not entries:
                self.get_logger().info(
                    f'ArUco id={marker_id} has no map entries — ignoring')
                continue

            matched = None
            for entry in entries:
                if entry.get('machine') == target_machine:
                    matched = entry
                    break

            if matched is None:
                self.get_logger().info(
                    f'ArUco id={marker_id} present but no entry matches '
                    f'current target machine "{target_machine}" — ignoring '
                    f'(machines on this marker: '
                    f'{[e.get("machine") for e in entries]})')
                continue

            direction    = matched.get('direction', 'straight')
            final        = bool(matched.get('final', False))
            forward_after = float(matched.get(
                'forward_after_rotation', matched.get('forward', 0.0)))

            # ── CHANGED: forward_before_rotation from depth ± distance_offset,
            # map value as fallback (offset NOT applied to the fallback)
            map_forward_before = float(matched.get('forward_before_rotation', 0.0))
            distance_offset    = float(matched.get('distance_offset', 0.0))
            if depth_m is not None:
                # Use the live depth reading, shifted by this entry's
                # distance_offset, so the robot drives exactly to the
                # marker position (adjusted) before rotating.
                forward_before = depth_m + distance_offset
                forward_before = max(0.0, forward_before)
                self.get_logger().info(
                    f'ArUco id={marker_id}: forward_before_rotation '
                    f'set from depth={depth_m:.3f} m '
                    f'+ distance_offset={distance_offset:+.3f} m '
                    f'= {forward_before:.3f} m '
                    f'(map fallback was {map_forward_before:.3f} m)')
            else:
                forward_before = map_forward_before
                self.get_logger().info(
                    f'ArUco id={marker_id}: depth unavailable — '
                    f'using map forward_before_rotation={forward_before:.3f} m '
                    f'(distance_offset not applied to fallback)')
            # ── END CHANGED ───────────────────────────────────────────────

            self.get_logger().info(
                f'ArUco id={marker_id} matched target "{target_machine}": '
                f'direction={direction} final={final} '
                f'forward_before_rotation={forward_before:.3f} m '
                f'forward_after_rotation={forward_after:.3f} m '
                f'→ starting manoeuvre')

            self._aruco_last_trigger_time[marker_id] = time.monotonic()
            self._start_aruco_maneuver(
                direction, final, forward_after, forward_before)
            return

    def _start_aruco_maneuver(self, direction: str, final: bool,
                               forward_after: float, forward_before: float = 0.0):
        if direction == 'left':
            self._turn_direction = 1
            self._rot_target_turn = math.radians(90)
        elif direction == 'right':
            self._turn_direction = -1
            self._rot_target_turn = math.radians(90)
        elif direction == 'u-turn':
            self._turn_direction = 1
            self._rot_target_turn = math.radians(170)
        else:
            self._turn_direction = 0

        self._pending_direction      = direction
        self._pending_final          = final
        self._pending_forward        = max(0.0, forward_after)
        self._pending_forward_before = max(0.0, forward_before)

        self._start_yaw = self._current_yaw

        self._pid.reset()
        self._first_meas       = True
        self._current_linear   = self._min_linear_speed
        self._current_angular  = 0.0
        self._lane_loss_start  = None

        self._dr.pause()
        self._lat_drift.pause()
        self._lat_drift.reset()

        twist = Twist()
        twist.linear.x  = self._min_linear_speed
        twist.angular.z = 0.0
        self._pub.publish(twist)

        if self._pending_forward_before > 0.0:
            self._mode = MODE_PRE_ROTATE_FORWARD
            self._enter_distance_leg(self._pending_forward_before)
            yaw_str = (f'{math.degrees(self._start_yaw):.1f}°'
                       if self._start_yaw is not None else 'unavailable (no IMU data yet)')
            self.get_logger().info(
                f'[PRE_ROTATE_FORWARD] Driving forward '
                f'{self._pending_forward_before:.3f} m before rotating '
                f'(lane-following enabled if lane signal available; '
                f'start_yaw latched at {yaw_str})')
        else:
            self._mode = MODE_WAIT_IMU

    # ── Generic odom-distance leg helpers ─────────────────────────────────────
    def _enter_distance_leg(self, target_dist: float):
        # Start reference and timeout clock are latched lazily, inside
        # _drive_distance_leg, the first time odometry is actually
        # available — not here. This avoids the leg being silently
        # skipped if an ArUco marker triggers before the first /odom
        # message has arrived (e.g. right at node startup).
        self._fwd_start_x     = None
        self._fwd_start_y     = None
        self._fwd_target_dist = target_dist
        self._fwd_start_time  = None
        self._current_linear  = 0.0
        self._current_angular = 0.0
        self._pid.reset()
        self._distance_leg_lane_active = False

    def _drive_distance_leg(self, twist: Twist, dt: float,
                             follow_lane: bool = False) -> bool:
        if self._fwd_target_dist <= 0.0:
            return True

        if self._odom_x is None or self._odom_y is None:
            # No odometry at all yet — hold position and wait rather
            # than treating this as "leg complete". Previously this
            # returned True immediately, which silently skipped the
            # entire forward distance if an ArUco trigger arrived
            # before the first /odom message.
            self.get_logger().warn(
                '[DISTANCE_LEG] Waiting for odometry before starting '
                f'leg (target {self._fwd_target_dist:.3f} m) — holding position',
                throttle_duration_sec=1.0)
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            return False

        if self._fwd_start_x is None or self._fwd_start_y is None:
            # First call where odometry is actually available — latch
            # the reference point now, so the full target distance is
            # measured from here, and start the safety-timeout clock
            # now too (not back when the leg was nominally "entered").
            self._fwd_start_x    = self._odom_x
            self._fwd_start_y    = self._odom_y
            self._fwd_start_time = time.monotonic()
            self.get_logger().info(
                '[DISTANCE_LEG] Odometry available — starting leg, '
                f'target {self._fwd_target_dist:.3f} m')
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            return False

        traveled  = math.hypot(self._odom_x - self._fwd_start_x,
                                self._odom_y - self._fwd_start_y)
        remaining = self._fwd_target_dist - traveled
        elapsed   = time.monotonic() - self._fwd_start_time
        timed_out = elapsed >= self._fwd_max_duration_s

        if remaining <= self._fwd_distance_tolerance or timed_out:
            if timed_out and remaining > self._fwd_distance_tolerance:
                self.get_logger().warn(
                    f'[DISTANCE_LEG] Safety timeout after {elapsed:.1f}s — '
                    f'traveled {traveled:.3f}/{self._fwd_target_dist:.3f} m')
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            return True

        taper_zone = max(0.05, self._fwd_cruise_speed * 0.3)
        if remaining < taper_zone:
            target_speed = max(self._min_linear_speed * 0.5,
                                self._fwd_cruise_speed * (remaining / taper_zone))
        else:
            target_speed = self._fwd_cruise_speed

        self._current_linear = slew(self._current_linear, target_speed,
                                    self._linear_slew_rate, dt)
        twist.linear.x = float(self._current_linear)

        if follow_lane:
            # Re-evaluated every call (this runs at the _rotation_loop
            # rate, ~50 Hz) — so the leg never "locks in" to blind mode.
            # It steers on the lane PID whenever a lane reading is BOTH
            # present (blended is not None) AND fresh (a pixel message
            # has arrived within lane_timeout, independent of mode —
            # see _lane_signal_fresh). If the lane drops out or the
            # topic stalls, angular slews back to 0 and the robot
            # continues straight (odometry-tracked distance still
            # applies). The moment a fresh reading reappears, this same
            # check flips back to True on the very next cycle and PID
            # control resumes automatically.
            blended    = self._blended_pixel()
            lane_ready = blended is not None and self._lane_signal_fresh()

            if lane_ready != self._distance_leg_lane_active:
                self._distance_leg_lane_active = lane_ready
                if lane_ready:
                    self.get_logger().info(
                        '[DISTANCE_LEG] Lane signal available — '
                        'steering with lane PID')
                else:
                    self.get_logger().warn(
                        '[DISTANCE_LEG] Lane signal lost/stale — '
                        'driving straight (odometry-tracked)')

            if lane_ready:
                error   = blended - self._target_px
                in_band = self._deadband.update_and_check(error)
                if in_band:
                    self._pid.compute(self._target_px, blended, dt)
                    ang = self._angular_trim
                else:
                    pid_out = self._pid.compute(self._target_px, blended, dt)
                    ang = pid_out + self._angular_trim
                    ang = max(-self._max_angular, min(self._max_angular, ang))
                self._current_angular = slew(self._current_angular, ang,
                                             self._angular_slew_rate, dt)
            else:
                self._current_angular = slew(self._current_angular, 0.0,
                                             self._angular_slew_rate, dt)
            twist.angular.z = float(self._current_angular)
        else:
            twist.angular.z = 0.0

        return False

    # ── Rotation / forward-run / final-hold timer (50 Hz) ────────────────────
    def _rotation_loop(self):
        self._assert_single_thread()

        if self._mode == MODE_LANE_FOLLOW:
            return

        twist = Twist()

        if self._mode == MODE_PRE_ROTATE_FORWARD:
            # CHANGED: this leg now follows the lane (when a lane signal
            # is available) instead of always driving blind. This covers
            # the depth-measured distance-to-marker + distance_offset
            # (e.g. marker 6 / machine C: depth_m + 0.80 m) using the
            # same lane-PID steering as the post-rotation forward leg.
            # If the lane is lost mid-leg it falls back to straight
            # (dead-reckoned via odometry) driving automatically inside
            # _drive_distance_leg.
            dt = self._get_rot_dt()
            leg_done = self._drive_distance_leg(twist, dt, follow_lane=True)
            self._pub.publish(twist)
            if leg_done:
                self.get_logger().info(
                    '[PRE_ROTATE_FORWARD] Pre-rotation forward leg complete '
                    '— proceeding to rotation')
                self._fwd_start_x = None
                self._fwd_start_y = None
                self._mode = MODE_WAIT_IMU
            return

        if self._mode == MODE_WAIT_IMU:
            if self._current_yaw is None:
                return
            _ = self._get_rot_dt()

            if self._turn_direction == 0:
                self.get_logger().info(
                    'Direction "straight" — skipping rotation, '
                    'going straight to forward leg')
                self._enter_forward_run()
                return

            if self._start_yaw is None:
                self._start_yaw = self._current_yaw

            self._rot_prev_angular = 0.0
            self._mode             = MODE_ROTATE
            direction_label = 'LEFT' if self._turn_direction == 1 else 'RIGHT'
            self.get_logger().info(
                f'Start yaw: {math.degrees(self._start_yaw):.1f}° '
                f'→ Rotating 90° {direction_label}')

        elif self._mode == MODE_ROTATE:
            if self._current_yaw is None:
                return
            dt = self._get_rot_dt()
            delta  = normalize_angle(
                (self._current_yaw - self._start_yaw) * self._turn_direction)
            turned = abs(delta)
            error  = self._rot_target_turn - turned

            self.get_logger().info(
                f'Turned: {math.degrees(turned):.1f}°',
                throttle_duration_sec=0.2)

            if error > 0.02:
                in_slow_zone = (turned >= self._rot_target_turn - self._rot_slow_zone)
                if not in_slow_zone:
                    target_speed = self._rot_cruise_speed
                else:
                    target_speed = self._rot_slow_gain * error
                    target_speed = min(self._rot_max_speed, target_speed)
                    target_speed = max(self._rot_min_speed, target_speed)
                target_z = self._turn_direction * target_speed
            else:
                target_z = 0.0

            raw_z = slew(self._rot_prev_angular, target_z,
                         self._rot_angular_accel, dt)
            twist.angular.z        = raw_z
            self._rot_prev_angular = raw_z

            if error <= 0.02 and abs(raw_z) < 0.03:
                twist.angular.z        = 0.0
                self._rot_prev_angular = 0.0
                self.get_logger().info('✅ Rotation complete — entering forward leg')
                self._pub.publish(twist)
                self._enter_forward_run()
                return

        elif self._mode == MODE_FORWARD_RUN:
            dt = self._get_rot_dt()
            leg_done = self._drive_distance_leg(twist, dt, follow_lane=True)
            if leg_done:
                self._pub.publish(twist)
                self._finish_forward_run()
                return

        elif self._mode == MODE_FINAL_HOLD:
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            self._pub.publish(twist)

            if self._final_hold_start is None:
                self._final_hold_start = time.monotonic()
                return

            if (time.monotonic() - self._final_hold_start) >= self._final_hold_duration_s:
                self._final_hold_start = None
                self.get_logger().info(
                    f'✅ Final-marker hold complete for target '
                    f'"{self._current_target_machine()}" — advancing target')
                self._current_target_index += 1
                next_target = self._current_target_machine()
                if next_target is not None:
                    self.get_logger().info(f'➡️  New nav target: {next_target}')
                else:
                    self.get_logger().info('🏁 All nav targets reached.')
                self._mode = MODE_RETURN_LANE
            return

        elif self._mode == MODE_RETURN_LANE:
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            self._pub.publish(twist)

            if self._current_yaw is not None:
                self._dr.resume(self._current_yaw)
                if self._odom_x is not None and self._odom_y is not None:
                    self._lat_drift.resume(self._odom_x, self._odom_y, self._current_yaw)
                else:
                    self._lat_drift.reset()
                self.get_logger().info(
                    f'[DR] Resumed with new reference: '
                    f'{math.degrees(self._current_yaw):.2f}°')

            self._last_msg_time    = self.get_clock().now()
            self._lane_detected    = False
            self._lane_loss_start  = time.monotonic()
            self._px_near          = None
            self._px_far           = None
            self._current_linear   = 0.0
            self._current_angular  = 0.0
            self._pending_direction      = None
            self._pending_final          = False
            self._pending_forward        = 0.0
            self._pending_forward_before = 0.0
            self._start_yaw         = None
            self._mode              = MODE_LANE_FOLLOW
            self.get_logger().info(
                '🔄 Holding — waiting for lane signal before moving')
            return

        self._pub.publish(twist)

    def _enter_forward_run(self):
        if self._pending_forward <= 0.0:
            self.get_logger().info(
                '[FORWARD_RUN] No forward distance required — '
                'skipping forward leg')
            self._finish_forward_run()
            return
        self._enter_distance_leg(self._pending_forward)
        self._mode = MODE_FORWARD_RUN
        self.get_logger().info(
            f'[FORWARD_RUN] Driving forward {self._pending_forward:.3f} m '
            f'(odom-tracked, lane-following enabled)')

    def _finish_forward_run(self):
        self._fwd_start_x = None
        self._fwd_start_y = None
        if self._pending_final:
            self._final_hold_start = None
            self._mode = MODE_FINAL_HOLD
            self.get_logger().info(
                f'[FINAL_HOLD] Reached final marker for target '
                f'"{self._current_target_machine()}" — holding '
                f'{self._final_hold_duration_s:.1f}s')
        else:
            self.get_logger().info(
                '[FORWARD_RUN] Intermediate marker complete — target '
                f'unchanged ("{self._current_target_machine()}") — '
                f'resuming lane-follow')
            self._mode = MODE_RETURN_LANE

    # ── Lane-loss watchdog ─────────────────────────────────────────────────────
    def _watchdog_cb(self):
        self._assert_single_thread()
        if self._mode != MODE_LANE_FOLLOW:
            return
        if not self._lane_detected:
            return
        elapsed = (self.get_clock().now() - self._last_msg_time).nanoseconds * 1e-9
        if elapsed > self._lane_timeout:
            self.get_logger().warn(
                f'[LaneFollower] Lane timeout ({elapsed:.2f} s) — stopping.')
            self._on_lane_lost()

    def _on_lane_lost(self):
        self._lane_detected    = False
        self._lane_loss_start  = time.monotonic()
        self._pid.reset()
        ref = self._dr.reference_yaw
        if ref is not None:
            self.get_logger().info(
                '[LaneFollower] Lane lost — DR holding heading '
                f'{math.degrees(ref):.2f}°')
        else:
            self.get_logger().info('[LaneFollower] Lane lost — decelerating.')

    def _on_lane_acquired(self):
        self._lane_detected   = True
        self._lane_loss_start = None
        if self._current_yaw is not None:
            self._dr.latch(self._current_yaw)
        self.get_logger().info('[LaneFollower] Lane acquired — DR reference updated.')

    def _stop_robot(self):
        self._current_linear  = 0.0
        self._current_angular = 0.0
        self._pub.publish(Twist())

    def _combined_dr_correction(self, dt: float) -> float:
        if self._current_yaw is None or not self._dr.is_active:
            return 0.0
        yaw_correction = self._dr.correction(self._current_yaw, dt)
        if self._lat_drift_enabled and self._lat_drift.is_active:
            lat_correction = self._lat_drift.correction(dt)
            w = max(0.0, min(1.0, self._lat_drift_weight))
            return (1.0 - w) * yaw_correction + w * lat_correction
        return yaw_correction

    # ── Combined lane control step ────────────────────────────────────────────
    def _lane_cb_combined(self):
        if self._mode != MODE_LANE_FOLLOW:
            return

        dt      = self._get_lane_dt()
        blended = self._blended_pixel()

        if blended is None:
            if self._lane_detected:
                self._on_lane_lost()
            self._publish_decel(dt)
            return

        self._last_msg_time = self.get_clock().now()
        if not self._lane_detected:
            self._on_lane_acquired()

        if self._first_meas:
            self._input_filtered = blended
            self._first_meas     = False
        else:
            self._input_filtered = ema_update(
                self._input_filtered, blended, self._input_ema_alpha)

        error   = self._input_filtered - self._target_px
        in_band = self._deadband.update_and_check(error)

        if in_band:
            self._pid.compute(self._target_px, self._input_filtered, dt)
            lane_angular = self._angular_trim + self._imu_trim()
        else:
            pid_out      = self._pid.compute(self._target_px, self._input_filtered, dt)
            lane_angular = pid_out + self._angular_trim + self._imu_trim()
            lane_angular = max(-self._max_angular, min(self._max_angular, lane_angular))

        if self._current_yaw is not None and self._dr.is_active:
            yaw_correction = self._dr.correction(self._current_yaw, dt)
            lat_correction = (self._lat_drift.correction(dt)
                              if (self._lat_drift_enabled and self._lat_drift.is_active)
                              else 0.0)
            dr_correction = self._combined_dr_correction(dt)
            blend = (self._dr_lane_blend if self._lane_detected
                     else self._dr_no_lane_blend)
            raw_angular = (1.0 - blend) * lane_angular + blend * dr_correction
            self.get_logger().info(
                f'[DR] err={self._dr.heading_error_deg(self._current_yaw):.2f}°  '
                f'yaw_corr={yaw_correction:.4f}  lat_corr={lat_correction:.4f}  '
                f'lat_err={self._lat_drift.lateral_error:.4f} m  '
                f'lane={lane_angular:.4f}  blend={blend:.2f}  out={raw_angular:.4f}',
                throttle_duration_sec=0.5)
        else:
            raw_angular = lane_angular

        self._output_angular = ema_update(
            self._output_angular, raw_angular, self._output_ema_alpha)
        self._current_angular = slew(
            self._current_angular, self._output_angular,
            self._angular_slew_rate, dt)

        turn_fraction = abs(self._current_angular) / max(self._max_angular, 1e-6)
        speed_scale   = max(0.0, min(1.0, self._speed_angular_scale))
        target_linear = self._linear_speed * (1.0 - speed_scale * turn_fraction)
        target_linear = max(self._min_linear_speed, target_linear)

        self._current_linear = slew(
            self._current_linear, target_linear, self._linear_slew_rate, dt)
        self._current_linear = max(self._min_linear_speed, self._current_linear)

        twist = Twist()
        twist.linear.x  = float(self._current_linear)
        twist.angular.z = float(self._current_angular)
        self._pub.publish(twist)

    def _publish_decel(self, dt: float):
        now = time.monotonic()
        sustained_loss = (
            self._lane_loss_start is not None and
            (now - self._lane_loss_start) >= self._lane_loss_stop_s
        )
        if sustained_loss:
            self._current_linear  = 0.0
            self._current_angular = 0.0
            self._pub.publish(Twist())
            return

        self._current_linear = slew(self._current_linear, self._min_linear_speed,
                                    self._linear_slew_rate, dt)
        self._current_linear = max(self._min_linear_speed, self._current_linear)

        if self._current_yaw is not None and self._dr.is_active:
            target_angular = self._dr_no_lane_blend * self._combined_dr_correction(dt)
        else:
            target_angular = 0.0

        self._current_angular = slew(self._current_angular, target_angular,
                                     self._angular_slew_rate, dt)

        twist = Twist()
        twist.linear.x  = float(self._current_linear)
        twist.angular.z = float(self._current_angular)
        self._pub.publish(twist)


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerAruco()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
