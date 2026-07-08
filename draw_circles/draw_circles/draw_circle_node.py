import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32

import cv2
import numpy as np
from collections import deque

# ==============================================
# LANE DETECTION CONFIGURATION
# ==============================================

FRAME_WIDTH  = 848
FRAME_HEIGHT = 480

ROI_TOP_RATIO   = 0.1
ROI_LEFT_RATIO  = 0.1
ROI_RIGHT_RATIO = 0.1

# HSV yellow tape range
TAPE_HSV_LOW  = np.array([10, 30, 140])
TAPE_HSV_HIGH = np.array([40, 140, 255])

# HLS yellow tape range
TAPE_HLS_LOW  = np.array([15, 80, 60])
TAPE_HLS_HIGH = np.array([35, 220, 255])

# LAB B-channel threshold
LAB_B_LOW  = 135
LAB_B_HIGH = 255

# Blur
BLUR_KERNEL     = (3, 3)
MEDIAN_BLUR_K   = 5
BILATERAL_D     = 9
BILATERAL_SIGMA = 75

# Morphology
MORPH_CLOSE_K = (5, 5)
MORPH_OPEN_K  = (3, 3)
DILATE_ITER   = 1

# Canny
CANNY_LOW  = 30
CANNY_HIGH = 90

# Hough
HOUGH_RHO             = 1
HOUGH_THETA           = np.pi / 180
HOUGH_THRESHOLD       = 30
HOUGH_MIN_LINE_LENGTH = 60
HOUGH_MAX_LINE_GAP    = 40

# Lane
LANE_OFFSET     = 150
SLOPE_THRESHOLD = 0.5
SMOOTH_BUFFER   = 8

# Vanishing point smoothing
VP_SMOOTH_BUFFER = 12

# FLICKER REDUCTION PARAMETERS
ENABLE_TEMPORAL_SMOOTHING = True
BOTTOM_PT_SMOOTH_BUFFER = 8
SEGMENT_CONFIDENCE_THRESH = 0.4
MIN_SEGMENTS_REQUIRED = 2
HOUGH_VOTE_THRESHOLD_ADAPT = True
ADAPTIVE_THRESHOLD_WINDOW = 30

# Distance calibration (pixels to meters)
# Adjust these values based on your camera setup
PIXELS_PER_METER = 100  # Example: 100 pixels = 1 meter
LANE_WIDTH_METERS = 3.5  # Standard lane width in meters

# --- NEW: fitLine / median-edge / distance-smoothing parameters ---
REFERENCE_Y_RATIO   = 0.95   # row (fraction of frame height) to evaluate lane edge at
EDGE_MATCH_TOLERANCE = 40    # px tolerance for a segment to "cover" the reference row
DIST_SMOOTH_BUFFER   = 8     # smoothing buffer length for final published pixel_dist

# FILTER SWITCHES
USE_HSV_MASK      = True
USE_HLS_MASK      = True
USE_LAB_MASK      = True
USE_MEDIAN_BLUR   = True
USE_BILATERAL     = False
USE_GAUSSIAN      = True
USE_MORPH_OPEN    = True
USE_MORPH_CLOSE   = True
USE_DILATE        = True
USE_CLAHE         = True
USE_SOBEL_BOOST   = False
USE_SHADOW_REMOVE = True

# ==============================================
# GLOBAL STATE
# ==============================================
left_buffer = deque(maxlen=SMOOTH_BUFFER)
vp_buffer   = deque(maxlen=VP_SMOOTH_BUFFER)
bottom_pt_buffer = deque(maxlen=BOTTOM_PT_SMOOTH_BUFFER)
dist_buffer = deque(maxlen=DIST_SMOOTH_BUFFER)
segment_confidence = 0.0
last_valid_bottom_pt = None
last_valid_segments = []
frame_counter = 0
hough_threshold_history = deque(maxlen=ADAPTIVE_THRESHOLD_WINDOW)

_k_close = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_CLOSE_K)
_k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_OPEN_K)
_k_dil   = np.ones((3, 3), np.uint8)
_clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ==============================================
# LANE DETECTION FUNCTIONS
# ==============================================

def get_adaptive_hough_threshold():
    if not HOUGH_VOTE_THRESHOLD_ADAPT:
        return HOUGH_THRESHOLD
    
    if segment_confidence < 0.3:
        return max(15, HOUGH_THRESHOLD - 15)
    elif segment_confidence > 0.7:
        return min(50, HOUGH_THRESHOLD + 10)
    else:
        return HOUGH_THRESHOLD

def remove_shadows(bgr: np.ndarray) -> np.ndarray:
    bgr_f = bgr.astype(np.float32) + 1.0
    blur  = cv2.GaussianBlur(bgr_f, (101, 101), 0)
    norm  = (bgr_f / blur * 127).clip(0, 255).astype(np.uint8)
    return norm

def build_morphed_mask(frame_bgr: np.ndarray):
    dbg = {}

    src = remove_shadows(frame_bgr) if USE_SHADOW_REMOVE else frame_bgr.copy()
    dbg["shadow_norm"] = src

    combined = np.zeros(src.shape[:2], dtype=np.uint8)

    if USE_HSV_MASK:
        hsv_mask = cv2.inRange(
            cv2.cvtColor(src, cv2.COLOR_BGR2HSV),
            TAPE_HSV_LOW, TAPE_HSV_HIGH)
        combined = cv2.bitwise_or(combined, hsv_mask)
        dbg["hsv_mask"] = hsv_mask

    if USE_HLS_MASK:
        hls_mask = cv2.inRange(
            cv2.cvtColor(src, cv2.COLOR_BGR2HLS),
            TAPE_HLS_LOW, TAPE_HLS_HIGH)
        combined = cv2.bitwise_or(combined, hls_mask)
        dbg["hls_mask"] = hls_mask

    if USE_LAB_MASK:
        b_ch     = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)[:, :, 2]
        lab_mask = cv2.inRange(b_ch, LAB_B_LOW, LAB_B_HIGH)
        if USE_HSV_MASK or USE_HLS_MASK:
            lab_mask = cv2.bitwise_and(combined, lab_mask)
        combined = cv2.bitwise_or(combined, lab_mask)
        dbg["lab_mask"] = lab_mask

    dbg["fused_colour_mask"] = combined

    proc = combined.copy()
    if USE_MORPH_OPEN:
        proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN,  _k_open)
    if USE_MORPH_CLOSE:
        proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, _k_close)
    if USE_DILATE:
        proc = cv2.dilate(proc, _k_dil, iterations=DILATE_ITER)

    morphed_mask = proc.copy()
    dbg["morphed_mask"] = morphed_mask

    blurred = proc.copy()
    if USE_MEDIAN_BLUR:
        blurred = cv2.medianBlur(blurred, MEDIAN_BLUR_K)
    if USE_BILATERAL:
        blurred = cv2.bilateralFilter(blurred, BILATERAL_D,
                                      BILATERAL_SIGMA, BILATERAL_SIGMA)
    if USE_GAUSSIAN:
        blurred = cv2.GaussianBlur(blurred, BLUR_KERNEL, 0)
    dbg["blurred"] = blurred

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if USE_CLAHE:
        gray = _clahe.apply(gray)
    dbg["clahe_gray"] = gray

    canny_edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    dbg["canny_edges"] = canny_edges

    if USE_SOBEL_BOOST:
        sobelx       = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_abs    = np.uint8(np.clip(np.absolute(sobelx), 0, 255))
        _, sobel_bin = cv2.threshold(sobel_abs, 40, 255, cv2.THRESH_BINARY)
        sobel_masked = cv2.bitwise_and(sobel_bin, morphed_mask)
        canny_edges  = cv2.bitwise_or(canny_edges, sobel_masked)
        dbg["sobel_boost"] = sobel_masked

    final_mask = cv2.bitwise_or(morphed_mask, canny_edges)
    dbg["final_mask"] = final_mask

    return final_mask, morphed_mask, dbg

def region_of_interest(img: np.ndarray) -> np.ndarray:
    h, w = img.shape

    top_y          = int(h * 0.3)
    top_left_x     = int(w * 0.01)
    top_right_x    = int(w * 0.50)
    bottom_y       = h
    bottom_left_x  = 10
    bottom_right_x = int(w * 0.50)

    polygon = np.array([[
        (bottom_left_x,  bottom_y),
        (bottom_right_x, bottom_y),
        (top_right_x,    top_y),
        (top_left_x,     top_y),
    ]], dtype=np.int32)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def filter_left_segments(lines):
    left_segments = []
    if lines is None:
        return left_segments
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < SLOPE_THRESHOLD:
            continue
        if slope < 0:
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 30:
                left_segments.append((x1, y1, x2, y2))
    
    return left_segments

# ==============================================
# NEW: fitLine-based, median-edge bottom point
# ==============================================

def fit_lane_line(left_segments):
    """
    Fit a single robust line through all left-lane Hough segments using
    cv2.fitLine with a Huber loss, which downweights outlier points
    instead of letting one bad segment skew the whole fit the way plain
    least squares would.

    Returns (vx, vy, x0, y0): unit direction vector + a point on the line.
    Returns None if there aren't enough points to fit.
    """
    if not left_segments or len(left_segments) < MIN_SEGMENTS_REQUIRED:
        return None

    pts = []
    for x1, y1, x2, y2 in left_segments:
        pts.append([x1, y1])
        pts.append([x2, y2])

    pts = np.array(pts, dtype=np.float32)

    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    return float(vx), float(vy), float(x0), float(y0)

def point_on_line_at_y(line_params, target_y):
    """
    Given fitLine params (vx, vy, x0, y0), solve for x at a specific y.
    Returns None if the line is ~horizontal (vy ~ 0), since x can't be
    pinned down from y in that case.
    """
    vx, vy, x0, y0 = line_params
    if abs(vy) < 1e-6:
        return None
    t = (target_y - y0) / vy
    return x0 + t * vx

def median_edge_x(left_segments, target_y, tolerance=EDGE_MATCH_TOLERANCE):
    """
    Robust right-edge x estimate at a given row.

    For each segment that spans (or nearly spans, within `tolerance`)
    target_y, linearly interpolate its x-coordinate at that row. Take the
    MEDIAN of those x-values across all qualifying segments.

    Median is used instead of picking a single segment's endpoint because
    one short/noisy Hough segment can no longer single-handedly determine
    the lane edge position -- it just becomes one vote among several.
    """
    xs_at_target = []

    for x1, y1, x2, y2 in left_segments:
        # order points so p1 is the upper point (smaller y), p2 is lower
        if y1 <= y2:
            px1, py1, px2, py2 = x1, y1, x2, y2
        else:
            px1, py1, px2, py2 = x2, y2, x1, y1

        if py2 == py1:
            continue  # degenerate/horizontal segment, skip

        if py1 - tolerance <= target_y <= py2 + tolerance:
            t = (target_y - py1) / (py2 - py1)
            x_interp = px1 + t * (px2 - px1)
            xs_at_target.append(x_interp)

    if not xs_at_target:
        return None

    return float(np.median(xs_at_target))

def get_bottom_point_fitline(left_segments, frame_height):
    """
    Replacement for the old max-y endpoint-picking bottom point logic.

    1. Compute target_y = a fixed reference row near the bottom of frame.
    2. Primary estimate: median x across all segments at that row
       (median_edge_x) -- robust to any single noisy segment.
    3. Fallback: if no segment spans that row closely enough, project the
       fitLine-fitted line (fit_lane_line) down to that row instead.
    4. Apply the same temporal smoothing buffer used previously.
    """
    global last_valid_bottom_pt, segment_confidence

    target_y = int(frame_height * REFERENCE_Y_RATIO)

    if not left_segments:
        segment_confidence = max(0.0, segment_confidence - 0.1)
        return last_valid_bottom_pt

    edge_x = median_edge_x(left_segments, target_y)

    if edge_x is None:
        line_params = fit_lane_line(left_segments)
        if line_params is not None:
            edge_x = point_on_line_at_y(line_params, target_y)

    if edge_x is None:
        segment_confidence = max(0.0, segment_confidence - 0.1)
        return last_valid_bottom_pt

    bottom_pt = (int(edge_x), target_y)

    if len(left_segments) >= MIN_SEGMENTS_REQUIRED:
        segment_confidence = min(1.0, segment_confidence + 0.15)
    else:
        segment_confidence = max(0.0, segment_confidence - 0.05)

    if ENABLE_TEMPORAL_SMOOTHING:
        bottom_pt_buffer.append(bottom_pt)
        if len(bottom_pt_buffer) >= 2:
            weights = np.linspace(0.5, 1.0, len(bottom_pt_buffer))
            weighted_x = sum(pt[0] * w for pt, w in zip(bottom_pt_buffer, weights)) / sum(weights)
            weighted_y = sum(pt[1] * w for pt, w in zip(bottom_pt_buffer, weights)) / sum(weights)
            bottom_pt = (int(weighted_x), int(weighted_y))
        last_valid_bottom_pt = bottom_pt

    return bottom_pt

def smooth_pixel_dist(pixel_dist):
    """
    Apply the same weighted-buffer smoothing pattern used for bottom_pt
    and vp to the FINAL published distance value, so /lane/distance_pixels
    isn't raw single-frame noise riding straight through to the controller.
    """
    if pixel_dist is None:
        return None

    dist_buffer.append(pixel_dist)
    weights = np.linspace(0.5, 1.0, len(dist_buffer))
    smoothed = sum(d * w for d, w in zip(dist_buffer, weights)) / sum(weights)
    return int(round(smoothed))

def compute_vanishing_point(frame, left_segments):
    if not left_segments or len(left_segments) < MIN_SEGMENTS_REQUIRED:
        if vp_buffer:
            return vp_buffer[-1]
        return None

    top_pt = None
    min_y  = frame.shape[0] + 1

    for x1, y1, x2, y2 in left_segments:
        if y2 < min_y:
            min_y  = y2
            top_pt = (x2, y2)
        if y1 < min_y:
            min_y  = y1
            top_pt = (x1, y1)

    if top_pt:
        vp_buffer.append(top_pt)

    if vp_buffer:
        weights = np.exp(np.linspace(-1, 0, len(vp_buffer)))
        weights = weights / weights.sum()
        vp_x = int(sum(pt[0] * w for pt, w in zip(vp_buffer, weights)))
        vp_y = int(sum(pt[1] * w for pt, w in zip(vp_buffer, weights)))
        return (vp_x, vp_y)
    
    return None

def draw_center_line(frame):
    h, w     = frame.shape[:2]
    center_x = w // 2

    dash_len = 15
    gap_len  = 8
    y        = 0
    while y < h:
        y_end = min(y + dash_len, h)
        cv2.line(frame,
                 (center_x, y),
                 (center_x, y_end),
                 (255, 255, 255), 1, cv2.LINE_AA)
        y += dash_len + gap_len

    cv2.putText(frame,
                f"C={center_x}px",
                (center_x + 5, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255, 255, 255), 1, cv2.LINE_AA)

    return center_x

def draw_distance(frame, center_x, bottom_pt):
    if bottom_pt is None or segment_confidence < SEGMENT_CONFIDENCE_THRESH:
        return None

    lane_x, lane_y = bottom_pt
    pixel_dist     = lane_x - center_x
    abs_dist       = abs(pixel_dist)
    direction      = "L" if pixel_dist < 0 else "R"

    h         = frame.shape[0]
    measure_y = min(lane_y, h - 45)

    cv2.line(frame,
             (center_x, measure_y),
             (lane_x,   measure_y),
             (0, 200, 255), 2, cv2.LINE_AA)

    cv2.line(frame,
             (center_x, measure_y - 6),
             (center_x, measure_y + 6),
             (0, 200, 255), 2, cv2.LINE_AA)

    cv2.line(frame,
             (lane_x, measure_y - 6),
             (lane_x, measure_y + 6),
             (0, 200, 255), 2, cv2.LINE_AA)

    cv2.arrowedLine(frame,
                    (center_x, measure_y),
                    (lane_x,   measure_y),
                    (0, 200, 255), 2,
                    tipLength=0.07, line_type=cv2.LINE_AA)
    cv2.arrowedLine(frame,
                    (lane_x,   measure_y),
                    (center_x, measure_y),
                    (0, 200, 255), 2,
                    tipLength=0.07, line_type=cv2.LINE_AA)

    # Display in pixels
    label     = f"{abs_dist}px ({direction})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_x   = (center_x + lane_x) // 2 - tw // 2
    label_y   = measure_y - 10

    cv2.rectangle(frame,
                  (label_x - 3,      label_y - th - 2),
                  (label_x + tw + 3, label_y + 2),
                  (20, 20, 20), -1)
    cv2.putText(frame, label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 200, 255), 2, cv2.LINE_AA)
    
    # Also display in meters if calibrated
    if PIXELS_PER_METER > 0:
        dist_meters = abs_dist / PIXELS_PER_METER
        meter_label = f"{dist_meters:.2f}m ({direction})"
        (mw, mh), _ = cv2.getTextSize(meter_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame,
                      (label_x - 3, label_y - th - mh - 15),
                      (label_x + mw + 3, label_y - th - 5),
                      (20, 20, 20), -1)
        cv2.putText(frame, meter_label,
                    (label_x, label_y - th - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 200, 255), 1, cv2.LINE_AA)

    cv2.circle(frame, (lane_x, lane_y), 7, (0, 140, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (lane_x, lane_y), 7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (center_x, lane_y), 5, (255, 255, 255), -1, cv2.LINE_AA)

    return pixel_dist

def draw_lanes(frame, left_segments, vp):
    output    = frame.copy()
    bottom_pt = get_bottom_point_fitline(left_segments, frame.shape[0])

    if segment_confidence >= SEGMENT_CONFIDENCE_THRESH:
        for x1, y1, x2, y2 in left_segments:
            color = (0, int(255 * (0.5 + segment_confidence/2)), 0)
            cv2.line(output, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
            cv2.circle(output, (x1, y1), 4, (0, 200, 0), -1)
            cv2.circle(output, (x2, y2), 4, (0, 200, 0), -1)

    if vp is not None and segment_confidence >= SEGMENT_CONFIDENCE_THRESH:
        vx, vy = vp

        if left_segments and len(left_segments) >= MIN_SEGMENTS_REQUIRED:
            top_x, top_y = min(
                [(x2, y2) for x1, y1, x2, y2 in left_segments],
                key=lambda p: p[1]
            )
            num_dashes = 12
            for i in range(num_dashes):
                t0  = i       / num_dashes
                t1  = (i+0.5) / num_dashes
                px0 = int(top_x + t0 * (vx - top_x))
                py0 = int(top_y + t0 * (vy - top_y))
                px1 = int(top_x + t1 * (vx - top_x))
                py1 = int(top_y + t1 * (vy - top_y))
                cv2.line(output, (px0, py0), (px1, py1),
                         (0, 255, 255), 2, cv2.LINE_AA)

        cv2.circle(output, (vx, vy), 18, (0, 200, 255),  2, cv2.LINE_AA)
        cv2.circle(output, (vx, vy), 12, (0, 230, 255),  2, cv2.LINE_AA)
        cv2.circle(output, (vx, vy),  6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.line(output, (vx - 20, vy), (vx + 20, vy), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(output, (vx, vy - 20), (vx, vy + 20), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output, f"VP ({vx},{vy})",
                    (vx + 10, vy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)

    return output, bottom_pt

def draw_hud(frame, pixel_dist, left_segments, vp):
    w    = frame.shape[1]
    h    = frame.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)

    seg_text  = f"Seg: {len(left_segments)}"
    conf_text = f"Conf: {segment_confidence:.2f}"
    vp_text   = f"VP: ({vp[0]},{vp[1]})" if vp is not None else "VP: --"

    if pixel_dist is not None and segment_confidence >= SEGMENT_CONFIDENCE_THRESH:
        direction  = "LEFT" if pixel_dist < 0 else "RIGHT"
        dist_text  = f"Offset: {abs(pixel_dist)}px {direction}"
        if PIXELS_PER_METER > 0:
            dist_meters = abs(pixel_dist) / PIXELS_PER_METER
            dist_text += f" ({dist_meters:.2f}m)"
    else:
        dist_text  = "Offset: --"

    cv2.putText(frame, seg_text,  (10,  24), font, 0.50, (0, 220, 0),   1, cv2.LINE_AA)
    cv2.putText(frame, conf_text, (100, 24), font, 0.50, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, dist_text, (200, 24), font, 0.50, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, vp_text,   (500, 24), font, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

    bar_width = int(w * segment_confidence)
    cv2.rectangle(frame, (0, h - 5), (bar_width, h), (0, 255, 0), -1)
    cv2.rectangle(frame, (0, h - 5), (w, h), (100, 100, 100), 1)

    active = []
    if USE_HSV_MASK:      active.append("HSV")
    if USE_HLS_MASK:      active.append("HLS")
    if USE_LAB_MASK:      active.append("LAB")
    if USE_MEDIAN_BLUR:   active.append("MED")
    if USE_BILATERAL:     active.append("BIL")
    if USE_MORPH_OPEN:    active.append("OPEN")
    if USE_MORPH_CLOSE:   active.append("CLOSE")
    if USE_DILATE:        active.append("DIL")
    if USE_CLAHE:         active.append("CLAHE")
    if USE_SOBEL_BOOST:   active.append("SOBEL")
    if USE_SHADOW_REMOVE: active.append("SHDW")

    cv2.rectangle(frame, (0, h-22), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "Filters: " + "+".join(active),
                (6, h-6), font, 0.35, (180, 180, 60), 1, cv2.LINE_AA)

# ==============================================
# ROS NODE
# ==============================================

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # Subscriber for compressed image
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.image_callback,
            10
        )

        # Publisher for annotated image
        self.publisher = self.create_publisher(
            Image,
            '/lane/output',
            10
        )

        # Publisher for distance in pixels
        self.distance_publisher = self.create_publisher(
            Float32,
            '/lane/distance_pixels',
            10
        )

        # Publisher for distance in meters (if calibrated)
        self.distance_meters_publisher = self.create_publisher(
            Float32,
            '/lane/distance_meters',
            10
        )

        # Publisher for lane detection confidence
        self.confidence_publisher = self.create_publisher(
            Float32,
            '/lane/confidence',
            10
        )

        self.bridge = CvBridge()
        self.frame_counter = 0

        self.get_logger().info("Lane Detection Node Started 🚀")
        self.get_logger().info(f"Publishing distances to: /lane/distance_pixels and /lane/distance_meters")
        self.get_logger().info(f"Publishing confidence to: /lane/confidence")

    def image_callback(self, msg):
        global frame_counter, segment_confidence, last_valid_segments, bottom_pt_buffer, vp_buffer

        # Convert ROS → OpenCV
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        frame_counter += 1
        self.frame_counter = frame_counter

        # Resize if needed to match expected dimensions
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Lane detection pipeline
        final_mask, morphed_mask, dbg = build_morphed_mask(frame)

        roi = region_of_interest(morphed_mask)

        current_hough_thresh = get_adaptive_hough_threshold()

        lines = cv2.HoughLinesP(
            roi,
            rho=HOUGH_RHO,
            theta=HOUGH_THETA,
            threshold=current_hough_thresh,
            minLineLength=HOUGH_MIN_LINE_LENGTH,
            maxLineGap=HOUGH_MAX_LINE_GAP,
        )

        left_segments = filter_left_segments(lines)

        if left_segments:
            last_valid_segments = left_segments

        vp = compute_vanishing_point(frame, left_segments)

        output, bottom_pt = draw_lanes(frame, left_segments, vp)

        center_x = draw_center_line(output)

        pixel_dist = draw_distance(output, center_x, bottom_pt)

        # Smooth the final distance value before publishing / using downstream.
        # bottom_pt/vp are already smoothed individually, but pixel_dist itself
        # had no dedicated smoothing pass -- this closes that gap.
        pixel_dist = smooth_pixel_dist(pixel_dist)

        draw_hud(output, pixel_dist, left_segments, vp)

        # ==============================================
        # PUBLISH DISTANCE AND CONFIDENCE DATA
        # ==============================================
        
        # Convert numpy types to Python native types
        confidence_value = float(segment_confidence) if not isinstance(segment_confidence, float) else segment_confidence
        
        # Publish distance in pixels
        if pixel_dist is not None and segment_confidence >= SEGMENT_CONFIDENCE_THRESH:
            distance_msg = Float32()
            distance_msg.data = float(pixel_dist)  # Ensure it's Python float
            self.distance_publisher.publish(distance_msg)
            
            # Publish distance in meters if calibrated
            if PIXELS_PER_METER > 0:
                distance_meters = float(pixel_dist) / PIXELS_PER_METER
                distance_meters_msg = Float32()
                distance_meters_msg.data = distance_meters
                self.distance_meters_publisher.publish(distance_meters_msg)
                
                # Log occasionally
                if frame_counter % 60 == 0:
                    direction = "LEFT" if pixel_dist < 0 else "RIGHT"
                    self.get_logger().info(
                        f"Frame {frame_counter}: Distance = {abs(pixel_dist)}px "
                        f"({abs(distance_meters):.2f}m) {direction}, "
                        f"Confidence = {confidence_value:.2f}"
                    )
        else:
            # Publish NaN when no lane detected
            nan_msg = Float32()
            nan_msg.data = float('nan')
            self.distance_publisher.publish(nan_msg)
            
            if PIXELS_PER_METER > 0:
                self.distance_meters_publisher.publish(nan_msg)
        
        # Publish confidence (ensure it's Python float)
        confidence_msg = Float32()
        confidence_msg.data = confidence_value
        self.confidence_publisher.publish(confidence_msg)

        # Publish annotated image
        ros_image = self.bridge.cv2_to_imgmsg(output, encoding='bgr8')
        self.publisher.publish(ros_image)

        # Debug view
        cv2.imshow("Lane Detection (ROS2)", output)
        cv2.waitKey(1)

    def __del__(self):
        cv2.destroyAllWindows()

# ==============================================
# MAIN
# ==============================================

def main(args=None):
    rclpy.init(args=args)

    node = LaneDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
