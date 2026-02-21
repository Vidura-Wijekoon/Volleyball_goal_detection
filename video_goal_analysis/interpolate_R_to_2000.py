#!/usr/bin/env python3
"""
Interpolate View R from 1200 to 2000 frames.
For each target frame, linearly blend the two nearest source frames.
Output: frames/R_2000/
"""

import cv2
import numpy as np
from pathlib import Path

SOURCE_DIR = Path('frames/R')
TARGET_DIR = Path('frames/R_2000')
TARGET_FRAMES = 2000

source_files = sorted(SOURCE_DIR.glob('*.jpg'))
source_count = len(source_files)

print(f"Source: {source_count} frames in {SOURCE_DIR}")
print(f"Target: {TARGET_FRAMES} frames -> {TARGET_DIR}")

if source_count == 0:
    print("ERROR: No source frames found in frames/R/")
    exit(1)

TARGET_DIR.mkdir(exist_ok=True)

# Cache last two loaded frames to avoid redundant disk reads
cached = {}

def load_frame(idx):
    if idx not in cached:
        # Keep cache small
        if len(cached) > 4:
            cached.clear()
        cached[idx] = cv2.imread(str(source_files[idx]))
    return cached[idx]

print(f"\nGenerating {TARGET_FRAMES} frames...")

for t in range(TARGET_FRAMES):
    # Map target index to fractional source position
    src_pos = t * (source_count - 1) / (TARGET_FRAMES - 1)
    lo = int(src_pos)
    hi = min(lo + 1, source_count - 1)
    alpha = src_pos - lo  # blend weight for hi frame

    frame_lo = load_frame(lo)
    if frame_lo is None:
        print(f"  WARNING: Could not read source frame {lo+1}, skipping")
        continue

    if alpha < 0.001 or lo == hi:
        # No blending needed
        out = frame_lo
    else:
        frame_hi = load_frame(hi)
        if frame_hi is None:
            out = frame_lo
        else:
            out = cv2.addWeighted(frame_lo, 1.0 - alpha, frame_hi, alpha, 0)

    out_path = TARGET_DIR / f'frame_{t+1:05d}.jpg'
    cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 92])

    if (t + 1) % 200 == 0:
        print(f"  {t+1}/{TARGET_FRAMES} frames written...")

print(f"\nDone. {TARGET_FRAMES} frames saved to {TARGET_DIR}")
