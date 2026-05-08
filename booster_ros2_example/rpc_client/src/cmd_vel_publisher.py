#!/usr/bin/env python3

import math
import time
from dataclasses import dataclass
from typing import List

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


@dataclass(frozen=True)
class Step:
    name: str
    vx: float
    vy: float
    vyaw: float


class CmdVelSequencePublisher(Node):
    def __init__(self):
        super().__init__("cmd_vel_sequence_publisher")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("command_duration_sec", 1.0)
        self.declare_parameter("pause_after_command_sec", 1.0)

        # Distances/angles requested are converted into velocities for 1 second.
        # - Move forward/backward 0.2m in 1s => vx = +/-0.2 m/s
        # - Move left/right 0.1m in 1s => vy = +/-0.1 m/s
        # - Turn 15deg in 1s => vyaw = +/- (15deg/s) = +/-0.261799 rad/s
        deg15_per_sec = math.radians(15.0)

        self._steps: List[Step] = [
            Step("forward_0.2m", vx=0.2, vy=0.0, vyaw=0.0),
            Step("backward_0.2m", vx=-0.2, vy=0.0, vyaw=0.0),
            Step("left_0.1m", vx=0.0, vy=0.1, vyaw=0.0),
            Step("right_0.1m", vx=0.0, vy=-0.1, vyaw=0.0),
            Step("turn_left_15deg", vx=0.0, vy=0.0, vyaw=deg15_per_sec),
            Step("turn_right_15deg", vx=0.0, vy=0.0, vyaw=-deg15_per_sec),
        ]

        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._command_duration_sec = float(self.get_parameter("command_duration_sec").value)
        self._pause_after_command_sec = float(self.get_parameter("pause_after_command_sec").value)

        self._pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)

        self.get_logger().info(f"Publishing cmd_vel sequence to {self._cmd_vel_topic}")

    def _publish_twist(self, vx: float, vy: float, vyaw: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(vyaw)
        self._pub.publish(msg)

    def _run_step(self, step: Step) -> None:
        rate_hz = self._publish_rate_hz if self._publish_rate_hz > 0.0 else 20.0
        period = 1.0 / rate_hz

        self.get_logger().info(
            f"Step {step.name}: vx={step.vx:+.3f} vy={step.vy:+.3f} vyaw={step.vyaw:+.3f} for {self._command_duration_sec:.2f}s"
        )

        end_time = time.monotonic() + max(0.0, self._command_duration_sec)
        while rclpy.ok() and time.monotonic() < end_time:
            self._publish_twist(step.vx, step.vy, step.vyaw)
            time.sleep(period)

        # Stop, then wait.
        self._publish_twist(0.0, 0.0, 0.0)
        self.get_logger().info(f"Step {step.name}: stop, wait {self._pause_after_command_sec:.2f}s")
        time.sleep(max(0.0, self._pause_after_command_sec))

    def run_sequence(self) -> None:
        # Small delay to let subscribers connect.
        time.sleep(0.2)
        for step in self._steps:
            if not rclpy.ok():
                break
            self._run_step(step)

        # Final stop for safety.
        self._publish_twist(0.0, 0.0, 0.0)
        self.get_logger().info("Sequence complete")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelSequencePublisher()
    try:
        node.run_sequence()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._publish_twist(0.0, 0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
