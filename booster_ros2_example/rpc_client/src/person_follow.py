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
from sensor_msgs.msg import Image

from booster_interface.msg import BoosterApiReqMsg
from booster_interface.srv import RpcService


def create_msg(api_id: int, param_dict=None) -> BoosterApiReqMsg:
    msg = BoosterApiReqMsg()
    msg.api_id = int(api_id)
    msg.body = json.dumps(param_dict) if param_dict is not None else ""
    return msg


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class TargetState:
    bbox: Optional[Tuple[int, int, int, int]] = None
    center_px: Optional[Tuple[int, int]] = None
    distance_m: Optional[float] = None
    last_seen_wall: Optional[float] = None
    last_rgb_shape: Optional[Tuple[int, int]] = None  # (h, w)


class PersonFollowYolo(Node):
    def __init__(self):
        super().__init__("person_follow_yolo")

        self.DEBUG_DISPLAY = False  # Set True only when you have a display


        self.declare_parameter("rgb_topic", "/StereoNetNode/rectified_image")
        self.declare_parameter("depth_topic", "/StereoNetNode/stereonet_depth")
        self.declare_parameter("rpc_service_name", "booster_rpc_service")

        self.declare_parameter("control_hz", 10.0)
        self.declare_parameter("detector_hz", 10.0)
        self.declare_parameter("stale_detection_sec", 0.75)

        self.declare_parameter("enable_viz", True)
        self.declare_parameter("viz_window", "person_follow")

        # Follow distance band (meters)
        self.declare_parameter("follow_min_distance_m", 0.45)
        self.declare_parameter("follow_max_distance_m", 1.0)
        self.declare_parameter("follow_max_max_distance_m", 6.0)

        # Depth filtering (meters)
        self.declare_parameter("min_depth_m", 0.2)
        self.declare_parameter("max_depth_m", 6.0)
        self.declare_parameter("depth_median_window", 7)

        # Motion limits
        self.declare_parameter("max_vx", 0.5)
        self.declare_parameter("max_vyaw", 0.5)
        self.declare_parameter("vx_kp", 0.6)
        self.declare_parameter("vyaw_kp", 0.0025)  # rad/s per pixel
        self.declare_parameter("yaw_deadband_px", 30)
        self.declare_parameter("min_command_interval_sec", 0.12)
        self.declare_parameter("movement_wait_interval", 0.2)

        # YOLO (Ultralytics)
        self.declare_parameter("yolo_model_path", "yolov8n.engine")
        self.declare_parameter("yolo_fallback_pt", "yolov8n.pt")
        self.declare_parameter("yolo_imgsz", 640)
        self.declare_parameter("yolo_conf", 0.35)
        self.declare_parameter("yolo_iou", 0.6)
        self.declare_parameter("yolo_device", "cuda:0")
        self.declare_parameter("yolo_half", True)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.rpc_service_name = str(self.get_parameter("rpc_service_name").value)

        self.control_hz = float(self.get_parameter("control_hz").value)
        self.detector_hz = float(self.get_parameter("detector_hz").value)
        self.stale_detection_sec = float(self.get_parameter("stale_detection_sec").value)

        self.enable_viz = bool(self.get_parameter("enable_viz").value)
        self.viz_window = str(self.get_parameter("viz_window").value)

        self.follow_min_distance_m = float(self.get_parameter("follow_min_distance_m").value)
        self.follow_max_distance_m = float(self.get_parameter("follow_max_distance_m").value)
        self.follow_max_max_distance_m = float(
            self.get_parameter("follow_max_max_distance_m").value
        )
        self.follow_min_distance_m = max(0.0, self.follow_min_distance_m)
        self.follow_max_distance_m = max(self.follow_min_distance_m + 1e-3, self.follow_max_distance_m)
        self.follow_max_max_distance_m = max(
            self.follow_max_distance_m + 1e-3, self.follow_max_max_distance_m
        )

        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.min_depth_m = max(0.0, self.min_depth_m)
        self.max_depth_m = max(self.min_depth_m + 1e-3, self.max_depth_m)
        self.depth_median_window = int(self.get_parameter("depth_median_window").value)
        self.depth_median_window = max(3, self.depth_median_window | 1)  # odd >= 3

        self.max_vx = float(self.get_parameter("max_vx").value)
        self.max_vyaw = float(self.get_parameter("max_vyaw").value)
        self.vx_kp = float(self.get_parameter("vx_kp").value)
        self.vyaw_kp = float(self.get_parameter("vyaw_kp").value)
        self.yaw_deadband_px = int(self.get_parameter("yaw_deadband_px").value)
        self.min_command_interval_sec = float(self.get_parameter("min_command_interval_sec").value)
        self.movement_wait_interval = float(self.get_parameter("movement_wait_interval").value)
        self.movement_wait_interval = max(0.0, self.movement_wait_interval)

        self.yolo_model_path = str(self.get_parameter("yolo_model_path").value)
        self.yolo_fallback_pt = str(self.get_parameter("yolo_fallback_pt").value)
        self.yolo_imgsz = int(self.get_parameter("yolo_imgsz").value)
        self.yolo_conf = float(self.get_parameter("yolo_conf").value)
        self.yolo_iou = float(self.get_parameter("yolo_iou").value)
        self.yolo_device = str(self.get_parameter("yolo_device").value)
        self.yolo_half = bool(self.get_parameter("yolo_half").value)

        self.bridge = CvBridge()
        self.target = TargetState()

        self._yolo = None
        self._yolo_source = None
        self._init_yolo()

        self._last_detector_time = 0.0
        self._last_command_time_mono: Optional[float] = None

        self._next_command_not_before_mono: float = 0.0

        self._last_cmd_vx: float = 0.0
        self._last_cmd_vyaw: float = 0.0

        self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.15
        )
        self.sync.registerCallback(self.synced_cb)

        self.rpc_client = self.create_client(RpcService, self.rpc_service_name)
        self.create_timer(1.0 / max(1.0, self.control_hz), self.control_timer_cb)

        self.get_logger().info(
            f"PersonFollowYolo started. rgb={self.rgb_topic} depth={self.depth_topic} service={self.rpc_service_name}"
        )
        self.get_logger().info(
            f"Follow band: [{self.follow_min_distance_m:.2f}, {self.follow_max_distance_m:.2f}] m"
        )
        if self._yolo_source is not None:
            self.get_logger().info(f"YOLO model: {self._yolo_source}")

    def _init_yolo(self):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            self.get_logger().error(f"Ultralytics not available: {e}")
            self._yolo = None
            self._yolo_source = None
            return

        candidate = self.yolo_model_path
        if candidate and os.path.exists(candidate):
            model_path = candidate
        else:
            fallback = self.yolo_fallback_pt
            model_path = fallback or "yolov8n.pt"

        try:
            self._yolo = YOLO(model_path)
            self._yolo_source = model_path
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model '{model_path}': {e}")
            self._yolo = None
            self._yolo_source = None

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

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0.0:
            return 0.0
        area_a = float(max(0, aw) * max(0, ah))
        area_b = float(max(0, bw) * max(0, bh))
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 0.0 else 0.0

    def _detect_person_yolo(
        self, bgr: np.ndarray, prev_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int]]]:
        if bgr is None or self._yolo is None:
            return None, None

        try:
            results = self._yolo.predict(
                source=bgr,
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
            results = self._yolo.predict(
                source=bgr,
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
            xyxy = np.array(boxes.xyxy)
            confs = np.array(boxes.conf) if hasattr(boxes, "conf") else None
            clss = np.array(boxes.cls) if hasattr(boxes, "cls") else None

        candidates: list[Tuple[Tuple[int, int, int, int], float]] = []
        for i in range(int(xyxy.shape[0])):
            if clss is not None and int(clss[i]) != 0:
                continue
            x1, y1, x2, y2 = xyxy[i]
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)
            if w <= 2 or h <= 2:
                continue
            c = float(confs[i]) if confs is not None else 1.0
            score = c * math.sqrt(float(w * h) + 1.0)
            candidates.append(((x, y, w, h), score))

        if not candidates:
            return None, None

        # "Tracking": if we had a previous bbox, keep the detection with the best IoU (then score).
        if prev_bbox is not None:
            best = None
            best_key = (-1.0, -1.0)
            for bbox, score in candidates:
                iou = self._iou(prev_bbox, bbox)
                key = (iou, score)
                if key > best_key:
                    best_key = key
                    best = bbox
            if best is not None and best_key[0] > 0.05:
                x, y, w, h = best
                return best, (int(x + w / 2), int(y + h / 2))

        # Otherwise pick the best overall score.
        candidates.sort(key=lambda t: t[1], reverse=True)
        x, y, w, h = candidates[0][0]
        return (x, y, w, h), (int(x + w / 2), int(y + h / 2))

    def synced_cb(self, rgb_msg: Image, depth_msg: Image):
        bgr = self._rosimg_to_bgr(rgb_msg)
        depth = self._rosimg_to_depth(depth_msg)
        if bgr is None or depth is None:
            return

        self.target.last_rgb_shape = (int(bgr.shape[0]), int(bgr.shape[1]))
        now = time.time()

        detector_due = (now - self._last_detector_time) >= (
            1.0 / max(0.1, self.detector_hz)
        )
        if detector_due:
            self._last_detector_time = now
            bbox, center = self._detect_person_yolo(bgr, self.target.bbox)
            if bbox is not None and center is not None:
                d_m = self._depth_for_bbox(depth, bbox)
                self.target.bbox = bbox
                self.target.center_px = center
                self.target.distance_m = d_m
                self.target.last_seen_wall = now

        if self.enable_viz:
            self._show_viz(bgr)

    def _show_viz(self, bgr: np.ndarray):
        try:
            vis = bgr.copy()

            now_wall = time.time()
            target_valid = (
                self.target.center_px is not None
                and self.target.last_seen_wall is not None
                and (now_wall - float(self.target.last_seen_wall))
                <= self.stale_detection_sec
            )

            status = "LOCKED" if target_valid else "SEARCHING"
            if self.target.bbox is not None:
                x, y, w, h = self.target.bbox
                color = (0, 255, 0) if target_valid else (0, 255, 255)
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

            if self.target.center_px is not None:
                u, v = self.target.center_px
                cv2.circle(vis, (int(u), int(v)), 5, (0, 0, 255), -1)

            if self.target.distance_m is not None:
                cv2.putText(
                    vis,
                    f"distance={float(self.target.distance_m):.2f}m",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    vis,
                    "distance=NA",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.putText(
                vis,
                f"{status}  vx={self._last_cmd_vx:+.2f}  vyaw={self._last_cmd_vyaw:+.2f}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            if self.DEBUG_DISPLAY:
                cv2.imshow(self.viz_window, vis)
                cv2.waitKey(1)
        except Exception as e:
            # Avoid crashing on headless systems.
            self.get_logger().warn(f"Visualization disabled (imshow failed): {e}")
            self.enable_viz = False

    def _ensure_rpc_ready(self) -> bool:
        if self.rpc_client.service_is_ready():
            return True
        return bool(self.rpc_client.wait_for_service(timeout_sec=0.0))

    def _send_move(self, vx: float, vyaw: float):
        now_mono = time.monotonic()

        if now_mono < float(self._next_command_not_before_mono):
            return

        if self._last_command_time_mono is not None and self.min_command_interval_sec > 0.0:
            if (now_mono - self._last_command_time_mono) < self.min_command_interval_sec:
                return
        if not self._ensure_rpc_ready():
            return

        vx = clamp(float(vx), -abs(self.max_vx), abs(self.max_vx))
        vyaw = clamp(float(vyaw), -abs(self.max_vyaw), abs(self.max_vyaw))

        request = RpcService.Request()
        request.msg = create_msg(2001, {"vx": vx, "vy": 0.0, "vyaw": vyaw})
        self.rpc_client.call_async(request)
        self._last_command_time_mono = now_mono
        self._last_cmd_vx = float(vx)
        self._last_cmd_vyaw = float(vyaw)

        if self.movement_wait_interval > 0.0:
            self._next_command_not_before_mono = now_mono + float(self.movement_wait_interval)

    def control_timer_cb(self):
        now_wall = time.time()
        target_valid = (
            self.target.center_px is not None
            and self.target.last_seen_wall is not None
            and (now_wall - float(self.target.last_seen_wall)) <= self.stale_detection_sec
        )

        if not target_valid:
            self._send_move(0.0, 0.0)
            return

        if self.target.distance_m is None:
            # No usable depth => stop (simplest, safe behavior).
            self._send_move(0.0, 0.0)
            return

        d = float(self.target.distance_m)

        # Hard cutoff: do not follow beyond this distance.
        if d > self.follow_max_max_distance_m:
            self._send_move(0.0, 0.0)
            return

        follow_center = 0.5 * (self.follow_min_distance_m + self.follow_max_distance_m)

        # Distance control: keep inside [min, max].
        if d < self.follow_min_distance_m:
            dist_err = d - self.follow_min_distance_m  # negative => back up
        elif d > self.follow_max_distance_m:
            dist_err = d - self.follow_max_distance_m  # positive => move forward
        else:
            dist_err = 0.0

        vx = clamp(self.vx_kp * dist_err, -abs(self.max_vx), abs(self.max_vx))

        # Yaw control: keep person centered in image.
        vyaw = 0.0
        if self.target.last_rgb_shape is not None and self.target.center_px is not None:
            _h, w = self.target.last_rgb_shape
            u, _v = self.target.center_px
            du = float(u) - (0.5 * float(w))
            if abs(du) > float(self.yaw_deadband_px):
                # du>0 means target on right => rotate right (clockwise) => negative vyaw.
                vyaw = clamp(-self.vyaw_kp * du, -abs(self.max_vyaw), abs(self.max_vyaw))

        # If in-range, you can optionally reduce yaw aggression.
        if dist_err == 0.0:
            vyaw = clamp(vyaw, -0.5 * abs(self.max_vyaw), 0.5 * abs(self.max_vyaw))

        self._send_move(vx, vyaw)

        # Quiet unused variable warning if you later want center distance.
        _ = follow_center

    def stop(self):
        try:
            self._send_move(0.0, 0.0)
        except Exception:
            pass
        try:
            if self.enable_viz:
                cv2.destroyAllWindows()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = PersonFollowYolo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
