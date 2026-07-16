#!/usr/bin/env python3
"""Idempotent Kit HTML tweak: Grayscale/Blur/Edge → Lane/Road BEV panels."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <D-Racer-Kit-root>', file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).expanduser().resolve()
    html = root / 'src' / 'monitor' / 'monitor' / 'templates' / 'index.html'
    if not html.is_file():
        print(f'[SEA-Me board] skip monitor labels: missing {html}')
        return 0
    raw = html.read_bytes()
    text = raw.decode('utf-8')
    nl = '\r\n' if '\r\n' in text else '\n'

    def block(*lines: str) -> str:
        return nl.join(lines) + nl

    old = block(
        '            <article class="debug-panel">',
        '              <p class="debug-panel__title">Grayscale</p>',
        '              <img id="debug-frame-grayscale" class="debug-panel__frame" '
        'src="{{ placeholder_url }}" alt="Grayscale preview">',
        '              <p class="debug-panel__topic">{{ opencv_grayscale_topic }}</p>',
        '            </article>',
        '            <article class="debug-panel">',
        '              <p class="debug-panel__title">Blur</p>',
        '              <img id="debug-frame-blur" class="debug-panel__frame" '
        'src="{{ placeholder_url }}" alt="Blur preview">',
        '              <p class="debug-panel__topic">{{ opencv_blur_topic }}</p>',
        '            </article>',
        '            <article class="debug-panel">',
        '              <p class="debug-panel__title">Edge</p>',
        '              <img id="debug-frame-edge" class="debug-panel__frame" '
        'src="{{ placeholder_url }}" alt="Edge preview">',
        '              <p class="debug-panel__topic">{{ opencv_edge_topic }}</p>',
        '            </article>',
    )
    new = block(
        '            <article class="debug-panel">',
        '              <p class="debug-panel__title">Lane (HSV paint)</p>',
        '              <img id="debug-frame-grayscale" class="debug-panel__frame" '
        'src="{{ placeholder_url }}" alt="Lane mask BEV">',
        '              <p class="debug-panel__topic">{{ opencv_grayscale_topic }}</p>',
        '            </article>',
        '            <article class="debug-panel">',
        '              <p class="debug-panel__title">Road (drivable)</p>',
        '              <img id="debug-frame-blur" class="debug-panel__frame" '
        'src="{{ placeholder_url }}" alt="Road mask BEV">',
        '              <p class="debug-panel__topic">{{ opencv_blur_topic }}</p>',
        '            </article>',
    )
    if 'Lane (HSV paint)' in text:
        print('[SEA-Me board] monitor BEV panel labels already applied')
        return 0
    if old not in text:
        print(
            '[SEA-Me board] WARNING: monitor index.html pattern mismatch; '
            'labels not updated',
            file=sys.stderr,
        )
        return 1
    html.write_bytes(text.replace(old, new, 1).encode('utf-8'))
    print('[SEA-Me board] applied monitor BEV panel labels')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
