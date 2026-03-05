#!/usr/bin/env python3

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from booster_interface.msg import BoosterApiReqMsg
from booster_interface.srv import RpcService


def create_msg(api_id: int, param_dict=None) -> BoosterApiReqMsg:
    msg = BoosterApiReqMsg()
    msg.api_id = api_id
    msg.body = json.dumps(param_dict) if param_dict is not None else ""
    return msg


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def lerp(a: float, b: float, t: float) -> float:
    t = clamp(float(t), 0.0, 1.0)
    return (1.0 - t) * float(a) + t * float(b)


def apply_deadband(value: float, deadband: float) -> float:
    deadband = max(0.0, float(deadband))
    if abs(value) <= deadband:
        return 0.0
    return math.copysign(abs(value) - deadband, value)


@dataclass
class PIDState:
    prev_error: float = 0.0
    integral: float = 0.0
    prev_time: Optional[float] = None


def pid_step(
    *,
    error: float,
    state: PIDState,
    kp: float,
    ki: float,
    kd: float,
    i_limit: float,
    step_limit: float,
    now: float,
    integral_decay: float,
) -> float:
    dt = 0.0
    if state.prev_time is not None:
        dt = max(1e-3, float(now - state.prev_time))

    if dt > 0.0:
        d_error = (error - state.prev_error) / dt
    else:
        d_error = 0.0

    if dt > 0.0 and ki != 0.0:
        state.integral += error * dt
        if i_limit > 0.0:
            state.integral = clamp(state.integral, -i_limit, i_limit)
    else:
        # Keep the integrator from hanging onto stale bias.
        if integral_decay < 1.0:
            state.integral *= clamp(integral_decay, 0.0, 1.0)

    step = (kp * error) + (ki * state.integral) + (kd * d_error)
    step = clamp(step, -step_limit, step_limit)

    state.prev_error = error
    state.prev_time = now
    return step


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


class HeadTrackPeople(Node):
    def __init__(self):
        super().__init__("head_track_people")

        self.DEBUG_DISPLAY = False  # Set True only when you have a display
        self.declare_parameter("rgb_topic", "/StereoNetNode/rectified_image")
        self.declare_parameter("depth_topic", "/StereoNetNode/stereonet_depth")
        self.declare_parameter("camera_info_topic", "/image_left_raw/camera_info")
        self.declare_parameter(
            "camera_info_topics",
            [
                "/image_left_raw/camera_info",
                "/StereoNetNode/stereonet_depth/camera_info",
            ],
        )
        self.declare_parameter("rpc_service_name", "booster_rpc_service")

        self.declare_parameter("enable_viz", True)
        self.declare_parameter("viz_window", "head_track_people")
        self.declare_parameter("control_hz", 10.0)
        self.declare_parameter("stale_detection_sec", 0.75)
        self.declare_parameter("hold_bbox_sec", 1.5)
        self.declare_parameter("show_bbox_when_missing", True)

        self.declare_parameter("track_pitch", True)
        self.declare_parameter("pitch_hold", 0.0)
        self.declare_parameter("target_smoothing_alpha", 0.6)

        # Control mode: pid (smooth continuous) or quantized (6 fixed directions)
        self.declare_parameter("control_mode", "pid")  # pid|quantized
        self.declare_parameter("quantized_bins", 6)
        self.declare_parameter("quantized_hysteresis_px", 20)
        self.declare_parameter("quantized_use_slew", True)

        # Pixel tolerance window around the image center.
        # If the target stays within +/- tolerance pixels, don't move the head.
        self.declare_parameter("yaw_tolerance_px", 55)
        self.declare_parameter("pitch_tolerance_px", 200)
        self.declare_parameter("tolerance_integral_decay", 0.85)

        # Command pacing to avoid overshoot when actuators lag.
        self.declare_parameter("min_command_interval_sec", 0.12)
        self.declare_parameter("enable_command_hold", False)
        self.declare_parameter("head_yaw_rate_rad_s", 1.2)
        self.declare_parameter("head_pitch_rate_rad_s", 1.2)
        self.declare_parameter("command_settle_sec", 0.05)
        self.declare_parameter("command_min_hold_sec", 0.0)

        self.declare_parameter("lock_camera_info", True)
        self.declare_parameter("require_camera_info_match_rgb", True)

        self.declare_parameter("yaw_kp", 0.9)
        self.declare_parameter("yaw_ki", 0.0)
        self.declare_parameter("yaw_kd", 0.12)
        self.declare_parameter("pitch_kp", 0.9)
        self.declare_parameter("pitch_ki", 0.0)
        self.declare_parameter("pitch_kd", 0.12)
        self.declare_parameter("max_step_rad", 0.08)

        # Stabilization near center (reduces left-right hunting)
        self.declare_parameter("yaw_deadband_rad", 0.03)
        self.declare_parameter("pitch_deadband_rad", 0.03)
        self.declare_parameter("pid_integral_limit", 0.6)
        self.declare_parameter("pid_integral_decay", 0.90)
        self.declare_parameter("yaw_limit_left", 0.785)
        self.declare_parameter("yaw_limit_right", -0.785)
        # Pitch conventions: +down, -up
        # Requested range: [-0.3, 0.0]
        self.declare_parameter("pitch_limit_up", -0.28)
        self.declare_parameter("pitch_limit_down", 0.0)

        # Distance-based pitch behavior
        self.declare_parameter("pitch_track_by_distance", True)
        self.declare_parameter("pitch_fixed_on_detect", True)
        self.declare_parameter("pitch_detected_rad", -0.30)
        self.declare_parameter("pitch_instant_set", True)
        self.declare_parameter("pitch_home_rad", 0.0)
        self.declare_parameter("pitch_near_m", 1.2)
        self.declare_parameter("pitch_far_m", 1.5)
        self.declare_parameter("pitch_near_rad", -0.30)
        self.declare_parameter("pitch_mid_rad", -0.15)
        self.declare_parameter("pitch_far_rad", 0.0)
        self.declare_parameter("pitch_return_on_missing", True)
        # If we were pitched up and then the target gets far / disappears,
        # hold the last up-pitch angle for this long before returning home.
        self.declare_parameter("pitch_hold_after_lost_sec", 3.0)

        self.declare_parameter("min_depth_m", 0.2)
        self.declare_parameter("max_depth_m", 6.0)
        self.declare_parameter("depth_median_window", 7)

        self.declare_parameter("hog_stride", 8)
        self.declare_parameter("hog_padding", 8)
        self.declare_parameter("hog_scale", 1.05)
        self.declare_parameter("use_roi", True)
        self.declare_parameter("roi_margin", 0.5)

        # Orin-optimized detector options
        self.declare_parameter("detector_backend", "yolo")  # yolo|hog
        self.declare_parameter("detector_hz", 10.0)

        # Ultralytics YOLO (supports TensorRT .engine on Jetson)
        self.declare_parameter("yolo_model_path", "yolov8n.engine")
        self.declare_parameter("yolo_fallback_pt", "yolov8n.pt")
        self.declare_parameter("yolo_imgsz", 640)
        self.declare_parameter("yolo_conf", 0.35)
        self.declare_parameter("yolo_iou", 0.6)
        self.declare_parameter("yolo_device", "cuda:0")
        self.declare_parameter("yolo_half", True)

        # Tracker to bridge detection dropouts
        self.declare_parameter("enable_tracker", True)
        self.declare_parameter("tracker_type", "CSRT")  # CSRT|KCF|MOSSE
        self.declare_parameter("tracker_max_age_sec", 1.0)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.camera_info_topics = list(self.get_parameter("camera_info_topics").value)
        self.rpc_service_name = self.get_parameter("rpc_service_name").value

        self.enable_viz = bool(self.get_parameter("enable_viz").value)
        self.viz_window = str(self.get_parameter("viz_window").value)
        self.control_hz = float(self.get_parameter("control_hz").value)
        self.stale_detection_sec = float(
            self.get_parameter("stale_detection_sec").value
        )
        self.hold_bbox_sec = float(self.get_parameter("hold_bbox_sec").value)
        self.show_bbox_when_missing = bool(
            self.get_parameter("show_bbox_when_missing").value
        )

        self.track_pitch = bool(self.get_parameter("track_pitch").value)
        self.pitch_hold = float(self.get_parameter("pitch_hold").value)
        self.target_smoothing_alpha = float(
            self.get_parameter("target_smoothing_alpha").value
        )
        self.target_smoothing_alpha = clamp(self.target_smoothing_alpha, 0.0, 1.0)

        self.control_mode = str(self.get_parameter("control_mode").value).strip().lower()
        if self.control_mode not in ("pid", "quantized"):
            self.get_logger().warn(
                f"Unknown control_mode='{self.control_mode}', using 'pid'"
            )
            self.control_mode = "pid"
        self.quantized_bins = int(self.get_parameter("quantized_bins").value)
        self.quantized_bins = max(2, self.quantized_bins)
        self.quantized_hysteresis_px = int(
            self.get_parameter("quantized_hysteresis_px").value
        )
        self.quantized_hysteresis_px = max(0, self.quantized_hysteresis_px)
        self.quantized_use_slew = bool(self.get_parameter("quantized_use_slew").value)

        self.yaw_tolerance_px = int(self.get_parameter("yaw_tolerance_px").value)
        self.pitch_tolerance_px = int(self.get_parameter("pitch_tolerance_px").value)
        self.yaw_tolerance_px = max(0, self.yaw_tolerance_px)
        self.pitch_tolerance_px = max(0, self.pitch_tolerance_px)
        self.tolerance_integral_decay = float(
            self.get_parameter("tolerance_integral_decay").value
        )
        self.tolerance_integral_decay = clamp(self.tolerance_integral_decay, 0.0, 1.0)

        self.min_command_interval_sec = float(
            self.get_parameter("min_command_interval_sec").value
        )
        self.min_command_interval_sec = max(0.0, self.min_command_interval_sec)
        self.enable_command_hold = bool(self.get_parameter("enable_command_hold").value)
        self.head_yaw_rate_rad_s = float(self.get_parameter("head_yaw_rate_rad_s").value)
        self.head_pitch_rate_rad_s = float(
            self.get_parameter("head_pitch_rate_rad_s").value
        )
        self.head_yaw_rate_rad_s = max(1e-3, self.head_yaw_rate_rad_s)
        self.head_pitch_rate_rad_s = max(1e-3, self.head_pitch_rate_rad_s)
        self.command_settle_sec = float(self.get_parameter("command_settle_sec").value)
        self.command_settle_sec = max(0.0, self.command_settle_sec)
        self.command_min_hold_sec = float(
            self.get_parameter("command_min_hold_sec").value
        )
        self.command_min_hold_sec = max(0.0, self.command_min_hold_sec)

        self.lock_camera_info = bool(self.get_parameter("lock_camera_info").value)
        self.require_camera_info_match_rgb = bool(
            self.get_parameter("require_camera_info_match_rgb").value
        )

        self.yaw_kp = float(self.get_parameter("yaw_kp").value)
        self.yaw_ki = float(self.get_parameter("yaw_ki").value)
        self.yaw_kd = float(self.get_parameter("yaw_kd").value)
        self.pitch_kp = float(self.get_parameter("pitch_kp").value)
        self.pitch_ki = float(self.get_parameter("pitch_ki").value)
        self.pitch_kd = float(self.get_parameter("pitch_kd").value)
        self.max_step_rad = float(self.get_parameter("max_step_rad").value)

        self.yaw_deadband_rad = float(self.get_parameter("yaw_deadband_rad").value)
        self.pitch_deadband_rad = float(
            self.get_parameter("pitch_deadband_rad").value
        )
        self.pid_integral_limit = float(
            self.get_parameter("pid_integral_limit").value
        )
        self.pid_integral_limit = max(0.0, self.pid_integral_limit)
        self.pid_integral_decay = float(
            self.get_parameter("pid_integral_decay").value
        )
        self.pid_integral_decay = clamp(self.pid_integral_decay, 0.0, 1.0)

        self.yaw_limit_left = float(self.get_parameter("yaw_limit_left").value)
        self.yaw_limit_right = float(self.get_parameter("yaw_limit_right").value)
        self.pitch_limit_up = float(self.get_parameter("pitch_limit_up").value)
        self.pitch_limit_down = float(self.get_parameter("pitch_limit_down").value)

        self.pitch_track_by_distance = bool(
            self.get_parameter("pitch_track_by_distance").value
        )
        self.pitch_fixed_on_detect = bool(
            self.get_parameter("pitch_fixed_on_detect").value
        )
        self.pitch_detected_rad = float(
            self.get_parameter("pitch_detected_rad").value
        )
        self.pitch_instant_set = bool(
            self.get_parameter("pitch_instant_set").value
        )
        self.pitch_home_rad = float(self.get_parameter("pitch_home_rad").value)
        self.pitch_near_m = float(self.get_parameter("pitch_near_m").value)
        self.pitch_far_m = float(self.get_parameter("pitch_far_m").value)
        self.pitch_near_rad = float(self.get_parameter("pitch_near_rad").value)
        self.pitch_mid_rad = float(self.get_parameter("pitch_mid_rad").value)
        self.pitch_far_rad = float(self.get_parameter("pitch_far_rad").value)
        self.pitch_return_on_missing = bool(
            self.get_parameter("pitch_return_on_missing").value
        )
        self.pitch_hold_after_lost_sec = float(
            self.get_parameter("pitch_hold_after_lost_sec").value
        )
        self.pitch_hold_after_lost_sec = max(0.0, self.pitch_hold_after_lost_sec)

        # Ensure distance thresholds are sane.
        self.pitch_near_m = max(0.01, self.pitch_near_m)
        self.pitch_far_m = max(self.pitch_near_m + 1e-3, self.pitch_far_m)

        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.depth_median_window = int(
            self.get_parameter("depth_median_window").value
        )
        self.depth_median_window = max(3, self.depth_median_window | 1)  # odd >= 3

        hog_stride = int(self.get_parameter("hog_stride").value)
        hog_padding = int(self.get_parameter("hog_padding").value)
        hog_scale = float(self.get_parameter("hog_scale").value)
        self.use_roi = bool(self.get_parameter("use_roi").value)
        self.roi_margin = float(self.get_parameter("roi_margin").value)
        self.roi_margin = clamp(self.roi_margin, 0.0, 2.0)

        self.detector_backend = str(self.get_parameter("detector_backend").value)
        self.detector_backend = self.detector_backend.strip().lower()
        self.detector_hz = float(self.get_parameter("detector_hz").value)
        self.detector_hz = max(0.1, self.detector_hz)

        self.yolo_model_path = str(self.get_parameter("yolo_model_path").value)
        self.yolo_fallback_pt = str(self.get_parameter("yolo_fallback_pt").value)
        self.yolo_imgsz = int(self.get_parameter("yolo_imgsz").value)
        self.yolo_conf = float(self.get_parameter("yolo_conf").value)
        self.yolo_iou = float(self.get_parameter("yolo_iou").value)
        self.yolo_device = str(self.get_parameter("yolo_device").value)
        self.yolo_half = bool(self.get_parameter("yolo_half").value)

        self.enable_tracker = bool(self.get_parameter("enable_tracker").value)
        self.tracker_type = str(self.get_parameter("tracker_type").value).strip().upper()
        self.tracker_max_age_sec = float(self.get_parameter("tracker_max_age_sec").value)
        self.tracker_max_age_sec = max(0.0, self.tracker_max_age_sec)

        self.bridge = CvBridge()
        self.intrinsics: Optional[Intrinsics] = None
        self._intrinsics_topic: Optional[str] = None
        self._last_rgb_shape: Optional[Tuple[int, int]] = None  # (h, w)

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.hog_stride = (hog_stride, hog_stride)
        self.hog_padding = (hog_padding, hog_padding)
        self.hog_scale = hog_scale

        self._yolo = None
        self._yolo_source = None
        if self.detector_backend == "yolo":
            self._init_yolo()
            if self._yolo is None:
                self.get_logger().warn(
                    "YOLO backend requested but not available; falling back to HOG"
                )
                self.detector_backend = "hog"

        self._last_detector_time = 0.0
        self._last_bbox_source: str = "none"  # yolo|hog|tracker|none

        self._tracker = None
        self._tracker_bbox: Optional[Tuple[int, int, int, int]] = None
        self._tracker_last_ok_time: Optional[float] = None

        self.latest_target_px: Optional[Tuple[int, int]] = None
        self._smoothed_target_px: Optional[Tuple[float, float]] = None
        self.latest_target_depth_m: Optional[float] = None
        self.latest_detection_time: Optional[float] = None
        self.latest_bbox: Optional[Tuple[int, int, int, int]] = None

        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.last_rpc_time = 0.0
        self.last_sent_yaw: Optional[float] = None
        self.last_sent_pitch: Optional[float] = None

        self._yaw_pid = PIDState()
        self._pitch_pid = PIDState()

        # Pitch-up hold state (monotonic time).
        self._pitch_up_last_time_mono: Optional[float] = None
        self._pitch_up_hold_pitch: Optional[float] = None

        self._quantized_region: Optional[int] = None

        self._last_command_time_mono: Optional[float] = None
        self._hold_until_mono: float = 0.0

        # Wave-hand configuration
        self.declare_parameter("wave_enable", True)
        self.declare_parameter("wave_min_distance_m", 2.0)
        self.declare_parameter("wave_max_distance_m", 4.0)
        self.declare_parameter("wave_min_detection_s", 3.0)
        self.declare_parameter("wave_rpc_api_id", 2005)
        self.declare_parameter("wave_topic", "/wave_cmd")

        self.wave_enable = bool(self.get_parameter("wave_enable").value)
        self.wave_min_distance_m = float(self.get_parameter("wave_min_distance_m").value)
        self.wave_max_distance_m = float(self.get_parameter("wave_max_distance_m").value)
        self.wave_min_detection_s = float(self.get_parameter("wave_min_detection_s").value)
        self.wave_rpc_api_id = int(self.get_parameter("wave_rpc_api_id").value)
        self.wave_topic = str(self.get_parameter("wave_topic").value)

        # Ensure wave distance thresholds are sane.
        self.wave_min_distance_m = max(0.0, float(self.wave_min_distance_m))
        self.wave_max_distance_m = max(
            float(self.wave_min_distance_m) + 1e-3, float(self.wave_max_distance_m)
        )

        # Internal wave state
        self._wave_first_seen_ts: Optional[float] = None
        self._waving_side: Optional[str] = None
        # If no RPC api id is configured, publish simple string commands on `wave_topic`.
        self.wave_pub = self.create_publisher(String, self.wave_topic, 10)

        # Subscribe to one-or-more camera info topics.
        # By default we lock to the first valid intrinsics that match the RGB stream.
        topics = [t for t in self.camera_info_topics if isinstance(t, str) and t]
        if self.camera_info_topic and self.camera_info_topic not in topics:
            topics.insert(0, self.camera_info_topic)
        if not topics:
            topics = ["/image_left_raw/camera_info"]

        self._camera_info_subs = [
            self.create_subscription(
                CameraInfo, t, lambda msg, topic=t: self.camera_info_cb(msg, topic), 10
            )
            for t in topics
        ]

        self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.15
        )
        self.sync.registerCallback(self.synced_cb)

        self.rpc_client = self.create_client(RpcService, self.rpc_service_name)
        self.create_timer(1.0 / max(1.0, self.control_hz), self.control_timer_cb)

        self.get_logger().info(
            "HeadTrackPeople started. Topics: rgb=%s depth=%s info=%s service=%s"
            % (
                self.rgb_topic,
                self.depth_topic,
                ",".join(topics),
                self.rpc_service_name,
            )
        )
        self.get_logger().info(
            f"Detector backend: {self.detector_backend} (tracker={self.enable_tracker}, type={self.tracker_type})"
        )

    def _rpc_wavehand(self, hand_index: int, hand_action: int) -> bool:
        """Send WaveHand RPC (api_id=2005). Returns True if sent."""
        if not self._ensure_rpc_ready():
            return False

        request = RpcService.Request()
        request.msg = create_msg(int(self.wave_rpc_api_id), {
            "hand_index": int(hand_index),
            "hand_action": int(hand_action),
        })
        self.rpc_client.call_async(request)
        return True

    def _start_waving(self, side: str):
        if not self.wave_enable:
            return

        # Side -> hand index per Booster SDK headers:
        #   kLeftHand=0, kRightHand=1
        hand_index = 1 if side == "right" else 0

        # Start wave: open hand (kHandOpen=0) as in simple_wave_test.py
        sent = False
        if self.wave_rpc_api_id > 0:
            sent = self._rpc_wavehand(hand_index=hand_index, hand_action=0)

        if (not sent) and self.wave_pub is not None:
            msg = String()
            msg.data = f"wave_{side}"
            self.wave_pub.publish(msg)

        self._waving_side = side
        self.get_logger().info(f"Waving {side} hand")

    def _stop_waving(self):
        if self._waving_side is None:
            return

        # Stop wave: close the currently waving hand (kHandClose=1)
        side = self._waving_side
        hand_index = 1 if side == "right" else 0

        sent = False
        if self.wave_rpc_api_id > 0:
            sent = self._rpc_wavehand(hand_index=hand_index, hand_action=1)

        if (not sent) and self.wave_pub is not None:
            msg = String()
            msg.data = "stop_wave"
            self.wave_pub.publish(msg)

        self._waving_side = None
        self._wave_first_seen_ts = None
        self.get_logger().info("Stop waving")

    def _update_waving(self, *, target_valid: bool, du_px: Optional[float], distance_m: Optional[float]):
        """Implements the spec:

        - When detected continuously > wave_min_detection_s and distance is in (wave_min_distance_m, wave_max_distance_m): start waving
        - Wave right hand if target is on right side; left hand if on left side
        - If person disappears or distance outside thresholds: stop waving
        """
        if not self.wave_enable:
            return

        if (
            (not target_valid)
            or (distance_m is None)
            or (distance_m <= self.wave_min_distance_m)
            or (distance_m >= self.wave_max_distance_m)
            or (du_px is None)
        ):
            if self._waving_side is not None:
                self._stop_waving()
            self._wave_first_seen_ts = None
            return

        now = time.time()
        if self._wave_first_seen_ts is None:
            self._wave_first_seen_ts = now

        if (now - self._wave_first_seen_ts) < self.wave_min_detection_s:
            # Not seen long enough yet.
            if self._waving_side is not None:
                self._stop_waving()
            return

        # Image convention: u > cx => target on the right side of image
        if du_px > 0.0:
            desired_side = "right"
        elif du_px < 0.0:
            desired_side = "left"
        else:
            desired_side = None

        if desired_side is None:
            if self._waving_side is not None:
                self._stop_waving()
            return

        if self._waving_side != desired_side:
            # Switching sides: stop previous, then start new.
            if self._waving_side is not None:
                self._stop_waving()
            self._start_waving(desired_side)

    def _init_yolo(self):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            self.get_logger().warn(f"Ultralytics not available: {e}")
            self._yolo = None
            self._yolo_source = None
            return

        candidate = self.yolo_model_path
        if candidate and os.path.exists(candidate):
            model_path = candidate
        else:
            fallback = self.yolo_fallback_pt
            if fallback and os.path.exists(fallback):
                model_path = fallback
            else:
                # Allow ultralytics to download default .pt if given a known name.
                model_path = fallback or "yolov8n.pt"

        try:
            self._yolo = YOLO(model_path)
            self._yolo_source = model_path
            self.get_logger().info(f"Loaded YOLO model: {model_path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to load YOLO model '{model_path}': {e}")
            self._yolo = None
            self._yolo_source = None

    def _create_tracker(self):
        t = self.tracker_type
        try:
            if t == "CSRT" and hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
            if t == "KCF" and hasattr(cv2, "TrackerKCF_create"):
                return cv2.TrackerKCF_create()
            if t == "MOSSE" and hasattr(cv2, "TrackerMOSSE_create"):
                return cv2.TrackerMOSSE_create()

            # Fallbacks
            if hasattr(cv2, "TrackerKCF_create"):
                self.get_logger().warn("Requested tracker not available; using KCF")
                return cv2.TrackerKCF_create()
            if hasattr(cv2, "TrackerCSRT_create"):
                self.get_logger().warn("Requested tracker not available; using CSRT")
                return cv2.TrackerCSRT_create()
            if hasattr(cv2, "TrackerMOSSE_create"):
                self.get_logger().warn("Requested tracker not available; using MOSSE")
                return cv2.TrackerMOSSE_create()
        except Exception:
            return None
        return None

    def _init_tracker_from_bbox(self, bgr: np.ndarray, bbox: Tuple[int, int, int, int]):
        if not self.enable_tracker:
            return
        tracker = self._create_tracker()
        if tracker is None:
            return
        x, y, w, h = bbox
        ok = tracker.init(bgr, (float(x), float(y), float(w), float(h)))
        if ok:
            self._tracker = tracker
            self._tracker_bbox = bbox
            self._tracker_last_ok_time = time.time()

    def _tracker_update(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if not self.enable_tracker or self._tracker is None:
            return None
        try:
            ok, box = self._tracker.update(bgr)
            if not ok:
                return None
            x, y, w, h = box
            bbox = (int(x), int(y), int(w), int(h))
            self._tracker_bbox = bbox
            self._tracker_last_ok_time = time.time()
            return bbox
        except Exception:
            return None

    def camera_info_cb(self, msg: CameraInfo, topic: str):
        try:
            if self.lock_camera_info and self._intrinsics_topic is not None:
                return

            if (
                self.require_camera_info_match_rgb
                and self._last_rgb_shape is not None
                and msg.width
                and msg.height
            ):
                h, w = self._last_rgb_shape
                if int(msg.width) != int(w) or int(msg.height) != int(h):
                    return

            fx = float(msg.k[0])
            fy = float(msg.k[4])
            cx = float(msg.k[2])
            cy = float(msg.k[5])
            if fx > 0.0 and fy > 0.0:
                self.intrinsics = Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
                self._intrinsics_topic = topic
        except Exception as e:
            self.get_logger().error(f"camera_info_cb error: {e}")

    def _rosimg_to_bgr(self, msg: Image) -> Optional[np.ndarray]:
        try:
            if msg.encoding in ("bgr8", "rgb8", "mono8"):
                enc = "bgr8" if msg.encoding != "mono8" else "mono8"
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=enc)
                if enc == "mono8":
                    return cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
                if msg.encoding == "rgb8":
                    return cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                return cv_img

            # Many Booster stereo topics use NV12 without a helpful encoding string.
            yuv = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                (msg.height * 3 // 2, msg.width)
            )
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        except Exception as e:
            self.get_logger().warn(f"Failed to convert RGB image: {e}")
            return None

    def _rosimg_to_depth(self, msg: Image) -> Optional[np.ndarray]:
        try:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"Failed to convert depth image: {e}")
            return None

    def _depth_at(self, depth: np.ndarray, u: int, v: int) -> Optional[float]:
        half = self.depth_median_window // 2
        v0 = max(0, v - half)
        v1 = min(depth.shape[0], v + half + 1)
        u0 = max(0, u - half)
        u1 = min(depth.shape[1], u + half + 1)

        patch = depth[v0:v1, u0:u1]
        if patch.size == 0:
            return None

        patch = patch.astype(np.float32)

        if depth.dtype in (np.uint16, np.uint32, np.int32, np.int16):
            patch_m = patch * 0.001
        else:
            patch_m = patch

        patch_m = patch_m[np.isfinite(patch_m)]
        patch_m = patch_m[(patch_m > self.min_depth_m) & (patch_m < self.max_depth_m)]
        if patch_m.size == 0:
            return None
        return float(np.median(patch_m))

    def _depth_for_bbox(
        self, depth: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[float]:
        x, y, w, h = bbox
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(depth.shape[1], int(x + w))
        y1 = min(depth.shape[0], int(y + h))
        if x1 <= x0 or y1 <= y0:
            return None

        patch = depth[y0:y1:4, x0:x1:4]
        if patch.size == 0:
            return None

        patch = patch.astype(np.float32)
        if depth.dtype in (np.uint16, np.uint32, np.int32, np.int16):
            patch_m = patch * 0.001
        else:
            patch_m = patch

        patch_m = patch_m[np.isfinite(patch_m)]
        patch_m = patch_m[(patch_m > self.min_depth_m) & (patch_m < self.max_depth_m)]
        if patch_m.size == 0:
            return None
        return float(np.median(patch_m))

    def _detect_person_hog(
        self, bgr: np.ndarray, prev_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int]]]:
        if bgr is None:
            return None, None

        # Optionally detect in a ROI around previous bbox to reduce misses.
        roi_offset_x = 0
        roi_offset_y = 0
        detect_img = bgr
        if self.use_roi and prev_bbox is not None:
            x, y, w, h = prev_bbox
            pad_x = int(w * self.roi_margin)
            pad_y = int(h * self.roi_margin)
            x0 = max(0, int(x - pad_x))
            y0 = max(0, int(y - pad_y))
            x1 = min(bgr.shape[1], int(x + w + pad_x))
            y1 = min(bgr.shape[0], int(y + h + pad_y))
            if x1 - x0 > 32 and y1 - y0 > 32:
                detect_img = bgr[y0:y1, x0:x1]
                roi_offset_x = x0
                roi_offset_y = y0

        # Light resize for performance; keep aspect ratio.
        max_w = 640
        scale = 1.0
        if detect_img.shape[1] > max_w:
            scale = max_w / float(detect_img.shape[1])
            bgr_small = cv2.resize(
                detect_img,
                (int(detect_img.shape[1] * scale), int(detect_img.shape[0] * scale)),
            )
        else:
            bgr_small = detect_img

        rects, _weights = self.hog.detectMultiScale(
            bgr_small,
            winStride=self.hog_stride,
            padding=self.hog_padding,
            scale=self.hog_scale,
        )

        if rects is None or len(rects) == 0:
            return None, None

        # Pick the largest detection.
        rects = list(rects)
        rects.sort(key=lambda r: int(r[2]) * int(r[3]), reverse=True)
        x, y, w, h = rects[0]

        # Map back to original image.
        if scale != 1.0:
            inv = 1.0 / scale
            x = int(x * inv)
            y = int(y * inv)
            w = int(w * inv)
            h = int(h * inv)

        # Map from ROI coordinates back to full image coordinates.
        x += roi_offset_x
        y += roi_offset_y

        u = int(x + w / 2)
        v = int(y + h / 2)
        return (x, y, w, h), (u, v)

    def _detect_person_yolo(
        self, bgr: np.ndarray, prev_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int]]]:
        if bgr is None or self._yolo is None:
            return None, None

        roi_offset_x = 0
        roi_offset_y = 0
        detect_img = bgr
        if self.use_roi and prev_bbox is not None:
            x, y, w, h = prev_bbox
            pad_x = int(w * self.roi_margin)
            pad_y = int(h * self.roi_margin)
            x0 = max(0, int(x - pad_x))
            y0 = max(0, int(y - pad_y))
            x1 = min(bgr.shape[1], int(x + w + pad_x))
            y1 = min(bgr.shape[0], int(y + h + pad_y))
            if x1 - x0 > 64 and y1 - y0 > 64:
                detect_img = bgr[y0:y1, x0:x1]
                roi_offset_x = x0
                roi_offset_y = y0

        try:
            results = self._yolo.predict(
                source=detect_img,
                imgsz=self.yolo_imgsz,
                conf=self.yolo_conf,
                iou=self.yolo_iou,
                device=self.yolo_device,
                half=self.yolo_half,
                verbose=False,
                classes=[0],  # person
                max_det=10,
            )
        except TypeError:
            # Older ultralytics versions may not support some kwargs
            results = self._yolo.predict(
                source=detect_img,
                imgsz=self.yolo_imgsz,
                conf=self.yolo_conf,
                iou=self.yolo_iou,
                device=self.yolo_device,
                verbose=False,
            )
        except Exception as e:
            self.get_logger().warn(f"YOLO predict failed: {e}")
            return None, None

        if not results:
            return None, None
        r0 = results[0]
        if r0 is None or r0.boxes is None or len(r0.boxes) == 0:
            return None, None

        boxes = r0.boxes
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
            clss = boxes.cls.cpu().numpy() if boxes.cls is not None else None
        except Exception:
            # If boxes are already numpy
            xyxy = np.array(boxes.xyxy)
            confs = np.array(boxes.conf) if hasattr(boxes, "conf") else None
            clss = np.array(boxes.cls) if hasattr(boxes, "cls") else None

        best_i = None
        best_score = -1.0
        for i in range(xyxy.shape[0]):
            if clss is not None and int(clss[i]) != 0:
                continue
            x1, y1, x2, y2 = xyxy[i]
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            c = float(confs[i]) if confs is not None else 1.0
            score = c * math.sqrt(area + 1.0)
            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            return None, None

        x1, y1, x2, y2 = xyxy[best_i]
        x = int(x1) + roi_offset_x
        y = int(y1) + roi_offset_y
        w = int(x2 - x1)
        h = int(y2 - y1)
        if w <= 2 or h <= 2:
            return None, None

        u = int(x + w / 2)
        v = int(y + h / 2)
        return (x, y, w, h), (u, v)

    def synced_cb(self, rgb_msg: Image, depth_msg: Image):
        bgr = self._rosimg_to_bgr(rgb_msg)
        depth = self._rosimg_to_depth(depth_msg)
        if bgr is None or depth is None:
            return

        self._last_rgb_shape = (int(bgr.shape[0]), int(bgr.shape[1]))

        prev_bbox = None
        if self.latest_bbox is not None and self.latest_detection_time is not None:
            if (time.time() - self.latest_detection_time) <= self.hold_bbox_sec:
                prev_bbox = self.latest_bbox

        now = time.time()

        # 1) Try tracker every frame (cheap) to avoid missing between detector frames.
        bbox = None
        center = None
        tracker_bbox = self._tracker_update(bgr)
        if tracker_bbox is not None:
            bbox = tracker_bbox
            x, y, w, h = bbox
            center = (int(x + w / 2), int(y + h / 2))
            self._last_bbox_source = "tracker"

        # 2) Run detector at a limited rate, or when tracker is stale/failing.
        detector_due = (now - self._last_detector_time) >= (1.0 / self.detector_hz)
        tracker_stale = (
            self._tracker_last_ok_time is None
            or (now - self._tracker_last_ok_time) > self.tracker_max_age_sec
        )
        need_detector = (center is None) or detector_due or tracker_stale

        if need_detector:
            if self.detector_backend == "yolo":
                bbox_d, center_d = self._detect_person_yolo(bgr, prev_bbox)
                src = "yolo"
            else:
                bbox_d, center_d = self._detect_person_hog(bgr, prev_bbox)
                src = "hog"
            self._last_detector_time = now

            if center_d is not None and bbox_d is not None:
                bbox, center = bbox_d, center_d
                self._last_bbox_source = src
                self._init_tracker_from_bbox(bgr, bbox_d)
        if center is None:
            if self.enable_viz:
                vis = bgr.copy()
                now = time.time()
                if (
                    self.show_bbox_when_missing
                    and self.latest_bbox is not None
                    and self.latest_detection_time is not None
                    and (now - self.latest_detection_time) <= self.hold_bbox_sec
                ):
                    x, y, w, h = self.latest_bbox
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(
                        vis,
                        "SEARCHING (holding last box)",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        vis,
                        "SEARCHING",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                if self.DEBUG_DISPLAY:
                    cv2.imshow(self.viz_window, vis)
                    cv2.waitKey(1)
            return

        u, v = center
        if bbox is not None:
            d_m = self._depth_for_bbox(depth, bbox)
        else:
            d_m = self._depth_at(depth, u, v)

        self.latest_bbox = bbox
        self.latest_target_px = center

        if self._smoothed_target_px is None:
            self._smoothed_target_px = (float(u), float(v))
        else:
            su, sv = self._smoothed_target_px
            a = self.target_smoothing_alpha
            self._smoothed_target_px = (a * float(u) + (1.0 - a) * su, a * float(v) + (1.0 - a) * sv)

        self.latest_target_depth_m = d_m
        self.latest_detection_time = time.time()

        if self.enable_viz:
            vis = bgr.copy()
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(vis, (u, v), 5, (0, 0, 255), -1)
            if d_m is not None:
                cv2.putText(
                    vis,
                    f"distance={d_m:.2f}m",
                    (max(0, u - 60), max(0, v - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            status = "LOCKED" if self.latest_detection_time is not None else "SEARCHING"
            info = f"{status}  src={self._last_bbox_source}  yaw_only={not self.track_pitch}  cam_info={self._intrinsics_topic or 'unset'}"
            cv2.putText(
                vis,
                info,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            if self.DEBUG_DISPLAY:
                cv2.imshow(self.viz_window, vis)
                cv2.waitKey(1)

    def _ensure_rpc_ready(self) -> bool:
        if self.rpc_client.service_is_ready():
            return True

        # Non-blocking wait (short) so we don't stall image callbacks.
        if self.rpc_client.wait_for_service(timeout_sec=0.0):
            return True

        # Log occasionally.
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_rpc_time > 2.0:
            self.get_logger().warn(
                f"RPC service '{self.rpc_service_name}' not ready yet"
            )
            self.last_rpc_time = now
        return False

    def _send_head(self, yaw: float, pitch: float):
        now_mono = time.monotonic()
        if self._last_command_time_mono is not None and self.min_command_interval_sec > 0.0:
            if (now_mono - self._last_command_time_mono) < self.min_command_interval_sec:
                return

        if not self._ensure_rpc_ready():
            return

        yaw = clamp(yaw, self.yaw_limit_right, self.yaw_limit_left)
        pitch = clamp(pitch, self.pitch_limit_up, self.pitch_limit_down)

        if self.last_sent_yaw is not None and self.last_sent_pitch is not None:
            if abs(yaw - self.last_sent_yaw) < 1e-3 and abs(pitch - self.last_sent_pitch) < 1e-3:
                return

        request = RpcService.Request()
        request.msg = create_msg(2004, {"pitch": float(pitch), "yaw": float(yaw)})
        self.rpc_client.call_async(request)

        self._last_command_time_mono = now_mono

        if self.enable_command_hold:
            dy = abs(yaw - self.current_yaw)
            dp = abs(pitch - self.current_pitch)
            est = max(dy / self.head_yaw_rate_rad_s, dp / self.head_pitch_rate_rad_s)
            hold = max(self.command_min_hold_sec, est + self.command_settle_sec)
            self._hold_until_mono = max(self._hold_until_mono, now_mono + hold)

        self.last_sent_yaw = yaw
        self.last_sent_pitch = pitch
        self.current_yaw = yaw
        self.current_pitch = pitch

    def control_timer_cb(self):
        now_mono = time.monotonic()
        now_wall = time.time()
        target_valid = (
            self.latest_target_px is not None
            and self.latest_detection_time is not None
            and (now_wall - self.latest_detection_time) <= self.stale_detection_sec
        )

        du_px_for_wave: Optional[float] = None
        distance_for_wave: Optional[float] = None

        # Even if we're holding head commands, still update waving state.
        if now_mono < self._hold_until_mono:
            if target_valid and self.intrinsics is not None:
                if self._smoothed_target_px is not None:
                    u = int(round(self._smoothed_target_px[0]))
                else:
                    u = int(self.latest_target_px[0])
                du_px_for_wave = float(u) - float(self.intrinsics.cx)
                distance_for_wave = self.latest_target_depth_m
            self._update_waving(
                target_valid=target_valid,
                du_px=du_px_for_wave,
                distance_m=distance_for_wave,
            )
            return

        new_yaw = self.current_yaw
        new_pitch = self.current_pitch if self.track_pitch else self.pitch_hold

        if target_valid:
            if self.intrinsics is None:
                self._update_waving(target_valid=False, du_px=None, distance_m=None)
                return

            if self._smoothed_target_px is not None:
                u = int(round(self._smoothed_target_px[0]))
                v = int(round(self._smoothed_target_px[1]))
            else:
                u, v = self.latest_target_px
            intr = self.intrinsics

            # Pixel offsets from center.
            du_px = float(u) - float(intr.cx)
            dv_px = float(v) - float(intr.cy)

            du_px_for_wave = du_px
            distance_for_wave = self.latest_target_depth_m

            # Apply a pixel tolerance window to avoid micro-corrections.
            du_px_eff = apply_deadband(du_px, float(self.yaw_tolerance_px))
            dv_px_eff = apply_deadband(dv_px, float(self.pitch_tolerance_px))

            # Normalize pixel offsets.
            x_n = du_px_eff / intr.fx
            y_n = dv_px_eff / intr.fy

            # Head conventions (from simple_head_test.py):
            # yaw: +left, -right ; pitch: +down, -up
            yaw_error = -math.atan(x_n)
            pitch_error_img = math.atan(y_n)

            # Deadband in radians (small) for additional stability.
            yaw_error = apply_deadband(yaw_error, self.yaw_deadband_rad)
            pitch_error_img = apply_deadband(pitch_error_img, self.pitch_deadband_rad)

            if self.control_mode == "quantized":
                # Split image into N equal vertical bins. Pick a fixed yaw target per bin.
                # Person on left => positive yaw (turn left). Person on right => negative yaw.
                width = None
                if self._last_rgb_shape is not None:
                    width = int(self._last_rgb_shape[1])
                if width is None or width <= 0:
                    width = int(round(intr.cx * 2.0))
                width = max(1, width)

                bins = self.quantized_bins
                bin_w = float(width) / float(bins)

                raw_region = int(math.floor(float(u) / bin_w))
                raw_region = int(clamp(raw_region, 0, bins - 1))

                region = raw_region
                if self._quantized_region is None:
                    self._quantized_region = raw_region
                elif raw_region != self._quantized_region:
                    cur = int(self._quantized_region)
                    hyst = float(self.quantized_hysteresis_px)

                    # Only update region if we move sufficiently past the boundary.
                    if raw_region > cur:
                        boundary = float(cur + 1) * bin_w
                        if float(u) > (boundary + hyst):
                            self._quantized_region = raw_region
                    else:
                        boundary = float(cur) * bin_w
                        if float(u) < (boundary - hyst):
                            self._quantized_region = raw_region
                    region = int(self._quantized_region)
                else:
                    region = int(self._quantized_region)

                yaw_targets = np.linspace(
                    self.yaw_limit_left, self.yaw_limit_right, bins
                )
                desired_yaw = float(yaw_targets[region])
                if self.quantized_use_slew:
                    dy = desired_yaw - self.current_yaw
                    new_yaw = self.current_yaw + clamp(
                        dy, -self.max_step_rad, self.max_step_rad
                    )
                else:
                    new_yaw = desired_yaw

                # Keep yaw PID from accumulating in quantized mode.
                self._yaw_pid.prev_time = now_mono
                self._yaw_pid.prev_error = 0.0
                self._yaw_pid.integral *= self.pid_integral_decay
            else:
                # If we're within pixel tolerance (effective offsets are zero), don't command motion.
                # Also bleed integrators slightly so we don't "creep".
                if du_px_eff == 0.0:
                    self._yaw_pid.prev_time = now_mono
                    self._yaw_pid.prev_error = 0.0
                    self._yaw_pid.integral *= self.tolerance_integral_decay
                    new_yaw = self.current_yaw
                else:
                    yaw_step = pid_step(
                        error=yaw_error,
                        state=self._yaw_pid,
                        kp=self.yaw_kp,
                        ki=self.yaw_ki,
                        kd=self.yaw_kd,
                        i_limit=self.pid_integral_limit,
                        step_limit=self.max_step_rad,
                        now=now_mono,
                        integral_decay=self.pid_integral_decay,
                    )
                    new_yaw = self.current_yaw + yaw_step

            # Pitch tracking
            if self.track_pitch:
                if self.pitch_fixed_on_detect:
                    # Only pitch up when the target is near enough.
                    # If we were already pitched up and the target becomes far / depth
                    # disappears, hold the last up pitch for a short time.
                    d_m = self.latest_target_depth_m
                    pitch_up_condition = (d_m is not None) and (d_m < self.pitch_near_m)
                    hold_active = (
                        self._pitch_up_last_time_mono is not None
                        and self._pitch_up_hold_pitch is not None
                        and (now_mono - float(self._pitch_up_last_time_mono))
                        < float(self.pitch_hold_after_lost_sec)
                    )

                    if pitch_up_condition:
                        desired_pitch = float(self.pitch_detected_rad)
                        desired_pitch = clamp(
                            float(desired_pitch),
                            self.pitch_limit_up,
                            self.pitch_limit_down,
                        )

                        if self.pitch_instant_set:
                            new_pitch = desired_pitch
                            # Reset PID state so it doesn't fight the fixed command later.
                            self._pitch_pid.prev_time = now_mono
                            self._pitch_pid.prev_error = 0.0
                            self._pitch_pid.integral = 0.0
                        else:
                            pitch_err = desired_pitch - self.current_pitch
                            pitch_err = apply_deadband(
                                pitch_err, self.pitch_deadband_rad
                            )
                            pitch_step = pid_step(
                                error=pitch_err,
                                state=self._pitch_pid,
                                kp=self.pitch_kp,
                                ki=self.pitch_ki,
                                kd=self.pitch_kd,
                                i_limit=self.pid_integral_limit,
                                step_limit=self.max_step_rad,
                                now=now_mono,
                                integral_decay=self.pid_integral_decay,
                            )
                            new_pitch = self.current_pitch + pitch_step

                        # Update hold state while we're in the up region.
                        self._pitch_up_last_time_mono = now_mono
                        self._pitch_up_hold_pitch = float(new_pitch)
                    elif hold_active:
                        # Hold last up pitch.
                        new_pitch = clamp(
                            float(self._pitch_up_hold_pitch),
                            self.pitch_limit_up,
                            self.pitch_limit_down,
                        )
                        self._pitch_pid.prev_time = now_mono
                        self._pitch_pid.prev_error = 0.0
                        self._pitch_pid.integral *= self.pid_integral_decay
                    else:
                        # After the hold expires (or if we never pitched up), return home.
                        desired_pitch = float(self.pitch_home_rad)
                        desired_pitch = clamp(
                            float(desired_pitch),
                            self.pitch_limit_up,
                            self.pitch_limit_down,
                        )

                        if self.pitch_instant_set:
                            new_pitch = desired_pitch
                            self._pitch_pid.prev_time = now_mono
                            self._pitch_pid.prev_error = 0.0
                            self._pitch_pid.integral = 0.0
                        else:
                            pitch_err = desired_pitch - self.current_pitch
                            pitch_err = apply_deadband(
                                pitch_err, self.pitch_deadband_rad
                            )
                            pitch_step = pid_step(
                                error=pitch_err,
                                state=self._pitch_pid,
                                kp=self.pitch_kp,
                                ki=self.pitch_ki,
                                kd=self.pitch_kd,
                                i_limit=self.pid_integral_limit,
                                step_limit=self.max_step_rad,
                                now=now_mono,
                                integral_decay=self.pid_integral_decay,
                            )
                            new_pitch = self.current_pitch + pitch_step

                        self._pitch_up_last_time_mono = None
                        self._pitch_up_hold_pitch = None
                elif self.pitch_track_by_distance:
                    d_m = self.latest_target_depth_m
                    pitch_up_condition = (d_m is not None) and (d_m < self.pitch_near_m)
                    hold_active = (
                        self._pitch_up_last_time_mono is not None
                        and self._pitch_up_hold_pitch is not None
                        and (now_mono - float(self._pitch_up_last_time_mono))
                        < float(self.pitch_hold_after_lost_sec)
                    )

                    if pitch_up_condition:
                        desired_pitch = float(self.pitch_near_rad)
                    elif hold_active:
                        new_pitch = clamp(
                            float(self._pitch_up_hold_pitch),
                            self.pitch_limit_up,
                            self.pitch_limit_down,
                        )
                        self._pitch_pid.prev_time = now_mono
                        self._pitch_pid.prev_error = 0.0
                        self._pitch_pid.integral *= self.pid_integral_decay
                        # Skip PID update while holding.
                        desired_pitch = None
                    else:
                        desired_pitch = float(self.pitch_home_rad)
                        self._pitch_up_last_time_mono = None
                        self._pitch_up_hold_pitch = None

                    if desired_pitch is not None:
                        pitch_err = desired_pitch - self.current_pitch
                        pitch_err = apply_deadband(pitch_err, self.pitch_deadband_rad)
                        pitch_step = pid_step(
                            error=pitch_err,
                            state=self._pitch_pid,
                            kp=self.pitch_kp,
                            ki=self.pitch_ki,
                            kd=self.pitch_kd,
                            i_limit=self.pid_integral_limit,
                            step_limit=self.max_step_rad,
                            now=now_mono,
                            integral_decay=self.pid_integral_decay,
                        )
                        new_pitch = self.current_pitch + pitch_step
                        if pitch_up_condition:
                            self._pitch_up_last_time_mono = now_mono
                            self._pitch_up_hold_pitch = float(new_pitch)
                else:
                    # Image-based pitch (legacy): keep centered vertically.
                    if dv_px_eff == 0.0:
                        self._pitch_pid.prev_time = now_mono
                        self._pitch_pid.prev_error = 0.0
                        self._pitch_pid.integral *= self.tolerance_integral_decay
                        new_pitch = self.current_pitch
                    else:
                        pitch_step = pid_step(
                            error=pitch_error_img,
                            state=self._pitch_pid,
                            kp=self.pitch_kp,
                            ki=self.pitch_ki,
                            kd=self.pitch_kd,
                            i_limit=self.pid_integral_limit,
                            step_limit=self.max_step_rad,
                            now=now_mono,
                            integral_decay=self.pid_integral_decay,
                        )
                        new_pitch = self.current_pitch + pitch_step
            else:
                # Keep pitch PID from accumulating when pitch tracking is disabled.
                self._pitch_pid.prev_time = now_mono
                self._pitch_pid.prev_error = 0.0
                self._pitch_pid.integral *= self.pid_integral_decay
                new_pitch = self.pitch_hold
                self._pitch_up_last_time_mono = None
                self._pitch_up_hold_pitch = None

        # If target is missing/stale: return pitch to home (0.0 rad) if requested.
        if (not target_valid) and self.track_pitch and self.pitch_return_on_missing:
            hold_active = (
                self._pitch_up_last_time_mono is not None
                and self._pitch_up_hold_pitch is not None
                and (now_mono - float(self._pitch_up_last_time_mono))
                < float(self.pitch_hold_after_lost_sec)
            )
            if hold_active:
                new_pitch = clamp(
                    float(self._pitch_up_hold_pitch),
                    self.pitch_limit_up,
                    self.pitch_limit_down,
                )
                self._pitch_pid.prev_time = now_mono
                self._pitch_pid.prev_error = 0.0
                self._pitch_pid.integral *= self.pid_integral_decay
            else:
                desired_pitch = float(self.pitch_home_rad)
                desired_pitch = clamp(
                    float(desired_pitch), self.pitch_limit_up, self.pitch_limit_down
                )
                if self.pitch_instant_set:
                    new_pitch = desired_pitch
                    self._pitch_pid.prev_time = now_mono
                    self._pitch_pid.prev_error = 0.0
                    self._pitch_pid.integral = 0.0
                else:
                    pitch_err = desired_pitch - self.current_pitch
                    pitch_err = apply_deadband(pitch_err, self.pitch_deadband_rad)
                    pitch_step = pid_step(
                        error=pitch_err,
                        state=self._pitch_pid,
                        kp=self.pitch_kp,
                        ki=self.pitch_ki,
                        kd=self.pitch_kd,
                        i_limit=self.pid_integral_limit,
                        step_limit=self.max_step_rad,
                        now=now_mono,
                        integral_decay=self.pid_integral_decay,
                    )
                    new_pitch = self.current_pitch + pitch_step

                self._pitch_up_last_time_mono = None
                self._pitch_up_hold_pitch = None

        self._send_head(new_yaw, new_pitch)

        # Wave-hand state machine
        self._update_waving(
            target_valid=target_valid,
            du_px=du_px_for_wave,
            distance_m=distance_for_wave,
        )


def main(args=None):
    rclpy.init(args=args)
    node = HeadTrackPeople()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._stop_waving()
        except Exception:
            pass
        try:
            if node.enable_viz:
                cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
