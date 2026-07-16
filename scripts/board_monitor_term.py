#!/usr/bin/env python3
"""Terminal race dashboard — run beside auto_driving / debug_monitor.

Web monitor (http://<board>:5000) shows camera + White/IN/OUT BEV.
This script prints planner / control / ArUco lines in the terminal.

    # Terminal A
    ros2 launch inference debug_monitor.launch.py route_mode:=out

    # Terminal B
    python3 scripts/board_monitor_term.py
    python3 scripts/board_monitor_term.py --hz
    python3 scripts/board_monitor_term.py --control-only

Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import rclpy
from control_msgs.msg import Control
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String


def _jpeg_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


class BoardTermMonitor(Node):
    def __init__(
        self,
        *,
        planner_topic: str,
        control_topic: str,
        aruco_topic: str,
        show_planner: bool,
        show_control: bool,
        show_aruco: bool,
        show_hz: bool,
        bev_topics: list[str],
    ) -> None:
        super().__init__('board_term_monitor')
        self._last_planner = ''
        self._tick_counts: dict[str, deque[float]] = {
            'planner': deque(maxlen=64),
            'control': deque(maxlen=64),
            'aruco': deque(maxlen=64),
            'white': deque(maxlen=64),
            'in': deque(maxlen=64),
            'out': deque(maxlen=64),
        }

        if show_planner and planner_topic:
            self.create_subscription(String, planner_topic, self._on_planner, 10)
        if show_control:
            self.create_subscription(Control, control_topic, self._on_control, 10)
        if show_aruco:
            self.create_subscription(String, aruco_topic, self._on_aruco, 10)

        if show_hz:
            from sensor_msgs.msg import CompressedImage

            qos = _jpeg_qos()
            for key, topic in zip(
                ('white', 'in', 'out'), bev_topics, strict=True
            ):
                self.create_subscription(
                    CompressedImage,
                    topic,
                    lambda _msg, k=key: self._on_bev(k),
                    qos,
                )
            self.create_timer(2.0, self._print_hz)

    def _stamp(self, key: str) -> None:
        self._tick_counts[key].append(time.monotonic())

    def _on_planner(self, msg: String) -> None:
        self._stamp('planner')
        line = msg.data.strip()
        if line == self._last_planner:
            return
        self._last_planner = line
        print(f'[planner] {line}', flush=True)

    def _on_control(self, msg: Control) -> None:
        self._stamp('control')
        print(
            f'[control] steer={msg.steering:+.3f} throttle={msg.throttle:+.3f}',
            flush=True,
        )

    def _on_aruco(self, msg: String) -> None:
        self._stamp('aruco')
        print(f'[aruco] {msg.data.strip()}', flush=True)

    def _on_bev(self, key: str) -> None:
        self._stamp(key)

    def _rate(self, key: str) -> float:
        stamps = self._tick_counts[key]
        if len(stamps) < 2:
            return 0.0
        dt = stamps[-1] - stamps[0]
        if dt <= 1e-6:
            return 0.0
        return (len(stamps) - 1) / dt

    def _print_hz(self) -> None:
        parts = [
            f'planner={self._rate("planner"):.1f}',
            f'control={self._rate("control"):.1f}',
            f'aruco={self._rate("aruco"):.1f}',
            f'bev_w={self._rate("white"):.1f}',
            f'bev_in={self._rate("in"):.1f}',
            f'bev_out={self._rate("out"):.1f}',
        ]
        print(f'[hz] {" ".join(parts)}', flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planner-topic', default='/debug/planner')
    parser.add_argument('--control-topic', default='/control')
    parser.add_argument('--aruco-topic', default='/debug/aruco')
    parser.add_argument(
        '--control-only',
        action='store_true',
        help='Only print /control',
    )
    parser.add_argument('--no-control', action='store_true')
    parser.add_argument('--no-aruco', action='store_true')
    parser.add_argument(
        '--hz',
        action='store_true',
        help='Every 2s print topic rates including BEV JPEG topics',
    )
    parser.add_argument(
        '--bev-white', default='/debug/bev/white/compressed'
    )
    parser.add_argument('--bev-in', default='/debug/bev/in/compressed')
    parser.add_argument('--bev-out', default='/debug/bev/out/compressed')
    args = parser.parse_args()

    if args.control_only:
        show_planner, show_control, show_aruco = False, True, False
    else:
        show_planner = True
        show_control = not args.no_control
        show_aruco = not args.no_aruco

    rclpy.init()
    node = BoardTermMonitor(
        planner_topic=args.planner_topic,
        control_topic=args.control_topic,
        aruco_topic=args.aruco_topic,
        show_planner=show_planner,
        show_control=show_control,
        show_aruco=show_aruco,
        show_hz=args.hz,
        bev_topics=[args.bev_white, args.bev_in, args.bev_out],
    )
    print(
        'board_monitor_term: listening '
        f'(planner={show_planner} control={show_control} aruco={show_aruco} '
        f'hz={args.hz}). Web UI: http://<board-ip>:5000',
        flush=True,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
