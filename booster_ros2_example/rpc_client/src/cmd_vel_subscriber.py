#!/usr/bin/env python3

import json
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from booster_interface.msg import BoosterApiReqMsg
from booster_interface.srv import RpcService


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def create_msg(api_id: int, param_dict: Optional[dict] = None) -> BoosterApiReqMsg:
    msg = BoosterApiReqMsg()
    msg.api_id = int(api_id)
    msg.body = json.dumps(param_dict) if param_dict is not None else ""
    return msg


class CmdVelRpcBridge(Node):
    """Subscribe to /cmd_vel and forward it to Booster RPC kMove (api_id=2001)."""

    def __init__(self):
        super().__init__("cmd_vel_rpc_bridge")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("rpc_service_name", "booster_rpc_service")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("cmd_timeout_sec", 0.5)
        self.declare_parameter("min_command_interval_sec", 0.02)
        self.declare_parameter("max_vx", 1.0)
        self.declare_parameter("max_vy", 0.5)
        self.declare_parameter("max_vyaw", 1.5)
        self.declare_parameter("log_rpc", True)
        self.declare_parameter("log_cmd_epsilon", 1e-3)

        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._rpc_service_name = str(self.get_parameter("rpc_service_name").value)
        self._control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self._cmd_timeout_sec = float(self.get_parameter("cmd_timeout_sec").value)
        self._min_command_interval_sec = float(self.get_parameter("min_command_interval_sec").value)
        self._max_vx = float(self.get_parameter("max_vx").value)
        self._max_vy = float(self.get_parameter("max_vy").value)
        self._max_vyaw = float(self.get_parameter("max_vyaw").value)
        self._log_rpc = bool(self.get_parameter("log_rpc").value)
        self._log_cmd_epsilon = float(self.get_parameter("log_cmd_epsilon").value)

        self._rpc_client = self.create_client(RpcService, self._rpc_service_name)

        self._last_twist: Twist = Twist()
        self._last_twist_time_mono: Optional[float] = None
        self._last_sent_time_mono: Optional[float] = None
        self._last_logged_cmd: Optional[tuple] = None

        self.create_subscription(Twist, self._cmd_vel_topic, self._cmd_vel_cb, 10)

        period = 1.0 / self._control_rate_hz if self._control_rate_hz > 0.0 else 0.05
        self._control_timer = self.create_timer(period, self._control_timer_cb)

        self.get_logger().info(
            f"Listening on {self._cmd_vel_topic} -> RPC service {self._rpc_service_name} (api_id=2001)"
        )

    def _ensure_rpc_ready(self) -> bool:
        if self._rpc_client.service_is_ready():
            return True
        return bool(self._rpc_client.wait_for_service(timeout_sec=0.0))

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self._last_twist = msg
        self._last_twist_time_mono = time.monotonic()

    def _send_move(self, vx: float, vy: float, vyaw: float) -> None:
        now_mono = time.monotonic()

        if self._last_sent_time_mono is not None and self._min_command_interval_sec > 0.0:
            if (now_mono - self._last_sent_time_mono) < self._min_command_interval_sec:
                return

        if not self._ensure_rpc_ready():
            return

        vx = clamp(float(vx), -abs(self._max_vx), abs(self._max_vx))
        vy = clamp(float(vy), -abs(self._max_vy), abs(self._max_vy))
        vyaw = clamp(float(vyaw), -abs(self._max_vyaw), abs(self._max_vyaw))

        if self._log_rpc:
            cmd = (vx, vy, vyaw)
            eps = abs(self._log_cmd_epsilon)
            should_log = self._last_logged_cmd is None
            if not should_log:
                lvx, lvy, lvyaw = self._last_logged_cmd
                should_log = (abs(vx - lvx) > eps) or (abs(vy - lvy) > eps) or (abs(vyaw - lvyaw) > eps)
            if should_log:
                self.get_logger().info(
                    f"RPC send kMove(api_id=2001): vx={vx:+.3f} vy={vy:+.3f} vyaw={vyaw:+.3f}"
                )
                self._last_logged_cmd = cmd

        request = RpcService.Request()
        request.msg = create_msg(2001, {"vx": vx, "vy": vy, "vyaw": vyaw})

        future = self._rpc_client.call_async(request)

        if self._log_rpc:
            def _on_done(fut):
                try:
                    resp = fut.result()
                    if resp is None:
                        self.get_logger().warn("RPC reply: None")
                        return

                    body = ""
                    try:
                        body = str(resp.msg.body)
                    except Exception:
                        body = "<unavailable>"

                    if len(body) > 200:
                        body = body[:200] + "..."

                    self.get_logger().info(f"RPC reply: status={getattr(resp.msg, 'status', 'NA')} body={body}")
                except Exception as e:
                    self.get_logger().error(f"RPC call failed: {e}")

            future.add_done_callback(_on_done)

        self._last_sent_time_mono = now_mono

    def _control_timer_cb(self) -> None:
        now_mono = time.monotonic()
        stale = (
            self._last_twist_time_mono is None
            or (now_mono - float(self._last_twist_time_mono)) > self._cmd_timeout_sec
        )

        if stale:
            self._send_move(0.0, 0.0, 0.0)
            return

        vx = float(self._last_twist.linear.x)
        vy = float(self._last_twist.linear.y)
        vyaw = float(self._last_twist.angular.z)
        self._send_move(vx, vy, vyaw)

    def stop(self) -> None:
        try:
            self._send_move(0.0, 0.0, 0.0)
        except Exception:
            pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelRpcBridge()
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
