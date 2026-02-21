#!/usr/bin/env python3
"""
Resample views = ['A', 'H', 'M', 'R'] to exactly 2000 frames by evenly sampling from source.
- A: 9661 -> 2000  (downsample, copy every ~4.8th frame)
- H: 5000 -> 2000  (downsample, copy every 2.5th frame)
- M: 2295 -> 2000  (downsample, copy every ~1.1th frame)
- R: XXXX -> 2000  (add R view)
Output: frames/A_2000/, frames/H_2000/, frames/M_2000/, frames/R_2000/
"""

import shutil
from pathlib import Path

TARGET = 2000

views = {
    'A': Path('frames/A_original_1200'),
    'H': Path('frames/H'),
    'M': Path('frames/M'),
    'R': Path('frames/R'),
}

for view, src_dir in views.items():
    out_dir = Path(f'frames/{view}_2000')
    out_dir.mkdir(exist_ok=True)

    src_files = sorted(src_dir.glob('*.jpg'))
    src_count = len(src_files)

    if src_count == 0:
        print(f"View {view}: ERROR - no frames found in {src_dir}")
        continue

    print(f"View {view}: {src_count} -> {TARGET} frames ({src_dir} -> {out_dir})")

    for t in range(TARGET):
        # Evenly map target index to source index
        src_idx = int(t * (src_count - 1) / (TARGET - 1))
        src_idx = min(src_idx, src_count - 1)

        dst_path = out_dir / f'frame_{t+1:05d}.jpg'
        shutil.copy2(src_files[src_idx], dst_path)

        if (t + 1) % 400 == 0:
            print(f"  {t+1}/{TARGET}...")

    print(f"  Done -> {out_dir}")

print("\nAll views resampled to 2000 frames.")
print("Outputs: frames/A_2000/  frames/H_2000/  frames/M_2000/")
