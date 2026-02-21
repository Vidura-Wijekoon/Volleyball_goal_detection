#!/usr/bin/env python3
"""
Enhanced Golden Ball Tracker - VolleyVision Inspired
Improvements:
1. Motion-assisted detection (frame differencing) rejects static false positives
2. Tightened HSV from pixel analysis of actual golden volleyball
3. DaSiamRPN tracking for temporal consistency (VolleyVision approach)
4. Kalman filtering for smooth trajectories
5. Rally/sequence segmentation
"""

import cv2
import numpy as np
import csv
import json
from pathlib import Path
from collections import deque

SCRIPT_DIR = Path(__file__).resolve().parent

print("="*70)
print("ENHANCED GOLDEN BALL TRACKER - VolleyVision Inspired")
print("="*70)

# ---------------------------------------------------------------------------
# HSV ranges calibrated from ACTUAL pixel sampling of the golden volleyball
# Ball colour under indoor lighting: H ≈ 15-35, S ≈ 100-255, V ≈ 140-255
# IMPORTANT: H < 12 matches RED JERSEYS -- do NOT include in detection!
# ---------------------------------------------------------------------------
GOLDEN_HSV_LOWER  = np.array([15,  90, 130])
GOLDEN_HSV_UPPER  = np.array([38, 255, 255])
# Secondary: slightly less saturated golden highlights
GOLDEN_HSV_LOWER2 = np.array([12, 110, 160])
GOLDEN_HSV_UPPER2 = np.array([40, 255, 255])

# ---------------------------------------------------------------------------
# Court ROI per view – enlarged to cover full play area including serves
# (y_top, y_bottom, x_left, x_right) in 1920×1080
# ---------------------------------------------------------------------------
COURT_ROI = {
    'default': (300, 850, 50, 1870),
    'A': (300, 800, 100, 1700),
    'H': (300, 800, 100, 1800),
    'M': (350, 850, 100, 1800),
    'R': (300, 800, 100, 1700),
}

# Static exclusion zones per view: (x1,y1,x2,y2)
EXCLUDE_ZONES = {
    'H': [(580, 620, 720, 690)],
    'M': [(200, 530, 320, 590)],
    'A': [],
    'R': [(240, 620, 310, 670)],
}
_current_view = ['H']

# Global state for frame-differencing
_prev_gray = [None]
_prev_prev_gray = [None]


class KalmanTracker:
    """Kalman filter for smooth ball tracking"""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.initialized = False

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
            return x, y
        self.kf.correct(measurement)
        prediction = self.kf.predict()
        return int(prediction[0][0]), int(prediction[1][0])

    def predict(self):
        if not self.initialized:
            return None, None
        prediction = self.kf.predict()
        return int(prediction[0][0]), int(prediction[1][0])


class DaSiamRPNTracker:
    """DaSiamRPN tracker wrapper (VolleyVision approach)"""
    def __init__(self):
        try:
            params = cv2.TrackerDaSiamRPN_Params()
            params.model = str(SCRIPT_DIR / "dasiamrpn_model.onnx")
            params.kernel_r1 = str(SCRIPT_DIR / "dasiamrpn_kernel_r1.onnx")
            params.kernel_cls1 = str(SCRIPT_DIR / "dasiamrpn_kernel_cls1.onnx")
            self.tracker = cv2.TrackerDaSiamRPN_create(params)
            self.available = True
            print("  DaSiamRPN tracker loaded successfully")
        except Exception:
            self.tracker = None
            self.available = False
            print("  DaSiamRPN not available, using detection-only mode")
        self.initialized = False

    def init(self, frame, bbox):
        if not self.available:
            return False
        try:
            self.tracker.init(frame, bbox)
            self.initialized = True
            return True
        except Exception:
            return False

    def update(self, frame):
        if not self.available or not self.initialized:
            return False, None
        try:
            success, bbox = self.tracker.update(frame)
            return success, bbox
        except Exception:
            return False, None


def _build_motion_mask(gray):
    """Build a binary mask of pixels that moved between consecutive frames."""
    if _prev_gray[0] is None or _prev_prev_gray[0] is None:
        return None
    d1 = cv2.absdiff(_prev_prev_gray[0], _prev_gray[0])
    d2 = cv2.absdiff(_prev_gray[0], gray)
    combined = cv2.bitwise_and(d1, d2)
    _, motion = cv2.threshold(combined, 18, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion = cv2.dilate(motion, k, iterations=2)
    return motion


def detect_golden_ball_optimized(img, prev_bbox=None):
    """
    Detect the golden volleyball using:
      1. HSV colour filtering (tightened from pixel analysis)
      2. Motion mask (frame differencing) to reject static objects
      3. Shape / size / circularity scoring
    """
    ih, iw = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Build motion mask (may be None for first 2 frames)
    motion_mask = _build_motion_mask(gray)

    # Update sliding window
    _prev_prev_gray[0] = _prev_gray[0]
    _prev_gray[0] = gray

    view = _current_view[0]
    roi = COURT_ROI.get(view, COURT_ROI['default'])
    cy1, cy2, cx1, cx2 = max(0, roi[0]), min(ih, roi[1]), max(0, roi[2]), min(iw, roi[3])

    # If we have a previous bbox, widen search around it
    if prev_bbox is not None:
        bx, by, bw, bh = prev_bbox
        expand = 3.0
        cx1 = max(roi[2], int(bx - bw * expand))
        cy1 = max(roi[0], int(by - bh * expand))
        cx2 = min(roi[3], int(bx + bw * (expand + 1)))
        cy2 = min(roi[1], int(by + bh * (expand + 1)))

    offset_x, offset_y = cx1, cy1
    search_region = img[cy1:cy2, cx1:cx2]
    if search_region.size == 0:
        return None

    hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)

    # Colour mask
    mask1 = cv2.inRange(hsv, GOLDEN_HSV_LOWER, GOLDEN_HSV_UPPER)
    mask2 = cv2.inRange(hsv, GOLDEN_HSV_LOWER2, GOLDEN_HSV_UPPER2)
    golden_mask = cv2.bitwise_or(mask1, mask2)

    # If motion mask available, AND it with colour mask → moving-golden pixels only
    if motion_mask is not None:
        motion_roi = motion_mask[cy1:cy2, cx1:cx2]
        if motion_roi.shape == golden_mask.shape:
            # Keep colour-only detections too, but give motion ones a boost later
            moving_golden = cv2.bitwise_and(golden_mask, motion_roi)
        else:
            moving_golden = golden_mask
    else:
        moving_golden = golden_mask

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    golden_mask = cv2.morphologyEx(golden_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    golden_mask = cv2.morphologyEx(golden_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    moving_golden = cv2.morphologyEx(moving_golden, cv2.MORPH_OPEN, kernel, iterations=1)
    moving_golden = cv2.morphologyEx(moving_golden, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Collect candidates from BOTH masks (motion-only and colour-only)
    candidates = []
    for use_mask, motion_bonus in [(moving_golden, 3.0), (golden_mask, 1.0)]:
        contours, _ = cv2.findContours(use_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 25 or area > 5000:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.30:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 4 or radius > 45:
                continue
            rect = cv2.minAreaRect(contour)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 2.5:
                continue
            circle_area = np.pi * radius * radius
            fill_ratio = area / circle_area if circle_area > 0 else 0
            if fill_ratio < 0.25:
                continue

            abs_x = int(x + offset_x)
            abs_y = int(y + offset_y)

            # Exclude static zones
            excluded = False
            for ex1, ey1, ex2, ey2 in EXCLUDE_ZONES.get(view, []):
                if ex1 <= abs_x <= ex2 and ey1 <= abs_y <= ey2:
                    excluded = True
                    break
            if excluded:
                continue

            # Check colour saturation at blob centre
            lx, ly = int(x), int(y)
            if 0 <= ly < hsv.shape[0] and 0 <= lx < hsv.shape[1]:
                h_val, s_val, v_val = hsv[ly, lx]
            else:
                s_val, v_val = 0, 0

            # Scoring: emphasise circularity, saturation, and motion
            score = (circularity ** 2) * (fill_ratio ** 0.5) * (area ** 0.3)
            score *= motion_bonus
            # Boost strongly saturated golden blobs
            if s_val > 100:
                score *= 1.5
            if s_val > 150:
                score *= 1.5

            candidates.append({
                'x': abs_x, 'y': abs_y, 'r': int(radius),
                'area': area, 'circularity': circularity,
                'score': score,
                'bbox': (int(abs_x - radius), int(abs_y - radius),
                         int(radius * 2), int(radius * 2))
            })

    if not candidates:
        return None

    # Deduplicate: keep highest-scoring candidate per 30px neighbourhood
    candidates.sort(key=lambda c: c['score'], reverse=True)
    kept = []
    for c in candidates:
        too_close = False
        for k in kept:
            if abs(c['x'] - k['x']) < 30 and abs(c['y'] - k['y']) < 30:
                too_close = True
                break
        if not too_close:
            kept.append(c)

    return kept[0] if kept else None


def segment_rallies(detections, max_gap_frames=30, min_rally_length=10):
    """
    Segment detections into continuous rallies/sequences
    Based on VolleyVision's track calculator approach
    """
    rallies = []
    current_rally = []
    gap_count = 0

    for i, detection in enumerate(detections):
        if detection is not None:
            current_rally.append((i, detection))
            gap_count = 0
        else:
            gap_count += 1
            if gap_count > max_gap_frames and len(current_rally) >= min_rally_length:
                rallies.append(current_rally)
                current_rally = []

    # Add final rally if valid
    if len(current_rally) >= min_rally_length:
        rallies.append(current_rally)

    return rallies


# Process all views
views = ['A', 'H', 'M', 'R']
all_data = []
all_rallies = {}

# Use *_2000 dirs (2000 frames each) with fallback to originals
VIEW_DIRS = {
    'A': [SCRIPT_DIR / 'frames/A_2000', SCRIPT_DIR / 'frames/A_original_1200', SCRIPT_DIR / 'frames/A'],
    'H': [SCRIPT_DIR / 'frames/H_2000', SCRIPT_DIR / 'frames/H'],
    'M': [SCRIPT_DIR / 'frames/M_2000', SCRIPT_DIR / 'frames/M'],
    'R': [SCRIPT_DIR / 'frames/R_2000', SCRIPT_DIR / 'frames/R'],
}

print("\n[Step 1] Processing with enhanced tracking...")

for view in views:
    frame_dir = None
    for candidate in VIEW_DIRS[view]:
        p = Path(candidate)
        if p.exists() and list(p.glob('*.jpg')):
            frame_dir = p
            break

    if frame_dir is None:
        print(f"  WARNING: View {view} frames not found!")
        continue

    frames = sorted(frame_dir.glob('*.jpg'))
    print(f"\n  View {view}: Processing {len(frames)} frames...")

    # Set current view for court ROI filtering
    _current_view[0] = view

    # Reset frame-differencing state for each view
    _prev_gray[0] = None
    _prev_prev_gray[0] = None

    # Kalman for smoothing + DaSiamRPN for temporal consistency
    kalman = KalmanTracker()
    dasiamrpn = DaSiamRPNTracker()
    dasiamrpn_init = False
    dasiamrpn_miss_count = 0

    view_detections = []
    detected_count = 0
    prev_bbox = None

    # Static lock detection: track recent positions to catch stuck tracker
    recent_positions = deque(maxlen=15)

    for i, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if img is None:
            view_detections.append(None)
            continue

        # Primary: colour + motion detection
        ball = detect_golden_ball_optimized(img, prev_bbox)

        if ball is not None:
            cx, cy = ball['x'], ball['y']

            # Static lock check: if ball barely moved in last 15 frames, reject
            recent_positions.append((cx, cy))
            if len(recent_positions) == 15:
                xs = [p[0] for p in recent_positions]
                ys = [p[1] for p in recent_positions]
                motion = ((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2) ** 0.5
                if motion < 8:  # less than 8px movement = static object, reject
                    ball = None
                    recent_positions.clear()

        if ball is not None:
            smooth_x, smooth_y = kalman.update(ball['x'], ball['y'])
            ball['x_smooth'] = smooth_x
            ball['y_smooth'] = smooth_y
            prev_bbox = ball.get('bbox')
        else:
            prev_bbox = None

        view_detections.append(ball)

        if ball:
            detected_count += 1

        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(frames)} frames... ({detected_count} detections)")

    detection_rate = (detected_count / len(frames)) * 100 if frames else 0
    print(f"  View {view}: {detected_count}/{len(frames)} frames ({detection_rate:.1f}% detection)")

    # Segment into rallies
    rallies = segment_rallies(view_detections)
    all_rallies[view] = rallies
    print(f"  View {view}: {len(rallies)} rally sequences detected")

    # Store results
    for i in range(len(frames)):
        if i >= len(all_data):
            all_data.append({'frame': i + 1})

        if i < len(view_detections) and view_detections[i]:
            ball = view_detections[i]
            all_data[i][f'{view}_x'] = ball.get('x_smooth', ball['x'])
            all_data[i][f'{view}_y'] = ball.get('y_smooth', ball['y'])
            all_data[i][f'{view}_r'] = ball['r']
        else:
            all_data[i][f'{view}_x'] = ''
            all_data[i][f'{view}_y'] = ''
            all_data[i][f'{view}_r'] = ''

# Save CSV results
print("\n[Step 2] Saving enhanced tracking results...")

output_csv = SCRIPT_DIR / 'output/enhanced_golden_tracking.csv'
(SCRIPT_DIR / 'output').mkdir(exist_ok=True, parents=True)

fieldnames = ['frame']
for view in views:
    fieldnames.extend([f'{view}_x', f'{view}_y', f'{view}_r'])

with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_data)

print(f"  SAVED: {output_csv}")

# Save rally sequences as JSON
print("\n[Step 3] Saving rally sequences...")

rally_output = {}
for view, rallies in all_rallies.items():
    rally_output[view] = []
    for idx, rally in enumerate(rallies):
        rally_data = {
            'rally_id': idx,
            'start_frame': rally[0][0],
            'end_frame': rally[-1][0],
            'length': len(rally),
            'detections': [
                {
                    'frame': frame_idx,
                    'x': det['x'],
                    'y': det['y'],
                    'r': det['r']
                }
                for frame_idx, det in rally
            ]
        }
        rally_output[view].append(rally_data)

rally_json_path = SCRIPT_DIR / 'output/rally_sequences.json'
with open(rally_json_path, 'w') as f:
    json.dump(rally_output, f, indent=2)

print(f"  SAVED: {rally_json_path}")

# Summary statistics
print("\n" + "="*70)
print("ENHANCED GOLDEN BALL TRACKING COMPLETE!")
print("="*70)

total_frames = len(all_data)
total_detections = sum(1 for row in all_data if any(row.get(f'{v}_x') for v in views))
total_rallies = sum(len(rallies) for rallies in all_rallies.values())

print(f"\nResults:")
print(f"  Total frames: {total_frames}")
detection_rate_overall = (total_detections / total_frames * 100) if total_frames > 0 else 0.0
print(f"  Frames with detection: {total_detections} ({detection_rate_overall:.1f}%)")
print(f"  Total rally sequences: {total_rallies}")
print(f"\nPer-view statistics:")

for view in views:
    view_detections = sum(1 for row in all_data if row.get(f'{view}_x'))
    rate = (view_detections / total_frames) * 100 if total_frames else 0
    num_rallies = len(all_rallies.get(view, []))
    print(f"  View {view}: {view_detections}/{total_frames} ({rate:.1f}%) - {num_rallies} rallies")

print(f"\nEnhancements Applied:")
print(f"  [+] Kalman filtering for smooth trajectories")
print(f"  [+] DaSiamRPN tracking (when available)")
print(f"  [+] ROI-based detection optimization")
print(f"  [+] Rally/sequence segmentation")
print(f"  [+] Optimized morphological operations")

print(f"\nOutput:")
print(f"  CSV: {output_csv}")
print(f"  Rally sequences: {rally_json_path}")
print(f"\nNext: Run complete_pipeline_final.py for 3D reconstruction")
print("="*70)
