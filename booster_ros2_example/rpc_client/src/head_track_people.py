#!/usr/bin/env python3

import json
import math
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

from booster_interface.msg import BoosterApiReqMsg
from booster_interface.srv import RpcService


def create_msg(api_id: int, param_dict=None) -> BoosterApiReqMsg:
    msg = BoosterApiReqMsg()
    msg.api_id = api_id
    msg.body = json.dumps(param_dict) if param_dict is not None else ""
    return msg


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


class HeadTrackPeople(Node):
    def __init__(self):
        super().__init__("head_track_people")

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

        self.declare_parameter("track_pitch", False)
        self.declare_parameter("pitch_hold", 0.0)
        self.declare_parameter("target_smoothing_alpha", 0.6)

        self.declare_parameter("lock_camera_info", True)
        self.declare_parameter("require_camera_info_match_rgb", True)

        self.declare_parameter("yaw_kp", 0.9)
        self.declare_parameter("pitch_kp", 0.9)
        self.declare_parameter("max_step_rad", 0.08)
        self.declare_parameter("yaw_limit_left", 0.785)
        self.declare_parameter("yaw_limit_right", -0.785)
        self.declare_parameter("pitch_limit_up", -0.3)
        self.declare_parameter("pitch_limit_down", 1.0)

        self.declare_parameter("min_depth_m", 0.2)
        self.declare_parameter("max_depth_m", 6.0)
        self.declare_parameter("depth_median_window", 7)

        self.declare_parameter("hog_stride", 8)
        self.declare_parameter("hog_padding", 8)
        self.declare_parameter("hog_scale", 1.05)
        self.declare_parameter("use_roi", True)
        self.declare_parameter("roi_margin", 0.5)

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

        self.lock_camera_info = bool(self.get_parameter("lock_camera_info").value)
        self.require_camera_info_match_rgb = bool(
            self.get_parameter("require_camera_info_match_rgb").value
        )

        self.yaw_kp = float(self.get_parameter("yaw_kp").value)
        self.pitch_kp = float(self.get_parameter("pitch_kp").value)
        self.max_step_rad = float(self.get_parameter("max_step_rad").value)

        self.yaw_limit_left = float(self.get_parameter("yaw_limit_left").value)
        self.yaw_limit_right = float(self.get_parameter("yaw_limit_right").value)
        self.pitch_limit_up = float(self.get_parameter("pitch_limit_up").value)
        self.pitch_limit_down = float(self.get_parameter("pitch_limit_down").value)

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

        self.bridge = CvBridge()
        self.intrinsics: Optional[Intrinsics] = None
        self._intrinsics_topic: Optional[str] = None
        self._last_rgb_shape: Optional[Tuple[int, int]] = None  # (h, w)

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.hog_stride = (hog_stride, hog_stride)
        self.hog_padding = (hog_padding, hog_padding)
        self.hog_scale = hog_scale

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

    def _detect_person(
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

        bbox, center = self._detect_person(bgr, prev_bbox)
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
            info = f"{status}  yaw_only={not self.track_pitch}  cam_info={self._intrinsics_topic or 'unset'}"
            cv2.putText(
                vis,
                info,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
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

        self.last_sent_yaw = yaw
        self.last_sent_pitch = pitch
        self.current_yaw = yaw
        self.current_pitch = pitch

    def control_timer_cb(self):
        if self.intrinsics is None:
            return

        if self.latest_target_px is None or self.latest_detection_time is None:
            return

        age = time.time() - self.latest_detection_time
        if age > self.stale_detection_sec:
            return

        if self._smoothed_target_px is not None:
            u = int(round(self._smoothed_target_px[0]))
            v = int(round(self._smoothed_target_px[1]))
        else:
            u, v = self.latest_target_px
        intr = self.intrinsics

        # Normalize pixel offsets.
        x_n = (u - intr.cx) / intr.fx
        y_n = (v - intr.cy) / intr.fy

        # Head conventions (from simple_head_test.py):
        # yaw: +left, -right ; pitch: +down, -up
        yaw_error = -math.atan(x_n)
        pitch_error = math.atan(y_n)

        yaw_step = clamp(self.yaw_kp * yaw_error, -self.max_step_rad, self.max_step_rad)
        new_yaw = self.current_yaw + yaw_step
        if self.track_pitch:
            pitch_step = clamp(
                self.pitch_kp * pitch_error, -self.max_step_rad, self.max_step_rad
            )
            new_pitch = self.current_pitch + pitch_step
        else:
            new_pitch = self.pitch_hold

        self._send_head(new_yaw, new_pitch)


def main(args=None):
    rclpy.init(args=args)
    node = HeadTrackPeople()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node.enable_viz:
                cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
