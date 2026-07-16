#!/usr/bin/env python3
"""Demote per-frame camera INFO spam to DEBUG."""
from __future__ import annotations
import sys
from pathlib import Path

def main() -> int:
    if len(sys.argv) != 2:
        return 2
    path = Path(sys.argv[1]).expanduser().resolve() / "src/camera/camera/camera_node.py"
    if not path.is_file():
        print(f"[SEA-Me board] skip camera quiet: missing {path}")
        return 0
    text = path.read_text(encoding="utf-8")
    old = "self.get_logger().info(f'Published frame: {len(msg.data)} bytes')"
    new = "self.get_logger().debug(f'Published frame: {len(msg.data)} bytes')"
    if new in text:
        print("[SEA-Me board] camera quiet logs already applied")
        return 0
    if old not in text:
        print("[SEA-Me board] WARNING: camera Published-frame log pattern mismatch", file=sys.stderr)
        return 1
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("[SEA-Me board] applied camera quiet logs")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
