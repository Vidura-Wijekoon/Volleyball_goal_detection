# 🏐 Enhanced Golden Volleyball Tracking System

A high-fidelity 4-view volleyball tracking and 3D reconstruction system designed for detecting goals/passages in golden volleyball matches.

https://github.com/user-attachments/assets/3835437b-ed21-4f66-92c1-41216dc495c8

## 🚀 Overview
This system processes 4 synchronized camera views (A, H, M, R) to:
1.  **Detect** the golden volleyball using a hybrid HSV + Motion-differencing algorithm.
2.  **Filter** out false positives (red jerseys, floor markings).
3.  **Triangulate** the ball's position into 3D space.
4.  **Analyze** velocities and detect specific goal/line crossings.
5.  **Visualize** results in a high-performance 4-view synchronized GUI.

This project incorporates advanced tracking techniques inspired by **VolleyVision** (open-source volleyball tracking) to achieve high accuracy and real-time performance.

## 🛠️ Technology Stack
- **OpenCV 4.x**: Core library for HSV detection, morphological operations, and image processing.
- **NumPy**: Numerical computing for 3D reconstruction (DLT) and matrix operations.
- **Kalman Filter (4-state)**: State estimation tracking `[x, y, vx, vy]` for trajectory smoothing.
- **DaSiamRPN Tracker**: Deep learning tracker (ONNX) for correlation-based ball tracking.
- **DLT (Direct Linear Transform)**: Multi-view triangulation for 3D point reconstruction.
- **Tkinter GUI**: 4-panel interactive interface with trajectory visualization and playback controls.

## 🎨 Detection Parameters (HSV)
The system uses the HSV color space for robust detection:
- **Hue (H)**: 15° - 45° (Yellow/gold range)
- **Saturation (S)**: 80 - 255 (Color intensity)
- **Value (V)**: 80 - 255 (Brightness)
- **Constraints**: Contour area (50-3000 px²), Circularity (>0.5), Aspect Ratio (<2.0).

## 📊 Performance Metrics
- **Detection Accuracy**: 99.3% across 1200 frames.
- **3D Success Rate**: 76.8% (922 points generated).
- **Processing Speed**: 60 FPS (with DaSiamRPN + ROI optimization).
- **False Positives**: < 1% error rate.

## 📁 System Architecture
- `gui/golden_ball_gui.py`: The interactive playback and analysis interface.
- `video_goal_analysis/enhanced_golden_tracker.py`: Core detection and 2D tracking logic.
- `video_goal_analysis/complete_pipeline_final.py`: 3D reconstruction and velocity pipeline.

## 📽️ Video Processing (Splitting Method)
To ensure perfect synchronization across all 4 views, the system expects frames extracted directly from the master MP4 recordings. 

If you have the source videos (`A-E.mp4`, `H-E.mp4`, `M-E.mp4`, `R-E.mp4`), use the following method to prepare the frames:

```python
import cv2
from pathlib import Path

# Extract frames from 4-view synced videos
def extract_synced_frames(video_paths, output_base, target_frames=2000):
    for view, vpath in video_paths.items():
        cap = cv2.VideoCapture(str(vpath))
        out_dir = Path(output_base) / f"{view}_2000"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        while count < target_frames:
            ret, frame = cap.read()
            if not ret: break
            count += 1
            cv2.imwrite(str(out_dir / f"frame_{count:05d}.jpg"), frame)
        cap.release()

# Example usage:
# videos = {'A': 'A-E.mp4', 'H': 'H-E.mp4', 'M': 'M-E.mp4', 'R': 'R-E.mp4'}
# extract_synced_frames(videos, 'video_goal_analysis/frames/')
```

## 🛠️ Performance Features
- **Motion-Assisted Detection**: Eliminates static background noise by requiring object movement.
- **Red Hue Exclusion**: Specifically tuned to ignore red team jerseys that frequent the court.
- **ROI Optimization**: Search region expansion by 1.5x for 3x faster detection.
- **Rally Segmentation**: Automatic detection of continuous play sequences (max gap: 30 frames).
- **Background Pre-loading**: Multi-threaded pre-loader to maintain 30+ FPS during playback.

## 🚦 Getting Started
1. Install dependencies: `pip install -r video_goal_analysis/requirements.txt`
2. Prepare frames using the splitting method above.
3. Run tracking: `python video_goal_analysis/enhanced_golden_tracker.py`
4. Run 3D pipeline: `python video_goal_analysis/complete_pipeline_final.py`
5. Launch GUI: `python gui/golden_ball_gui.py`

## 📚 References
- **VolleyVision**: Open-source volleyball tracking (shukkkur/VolleyVision).
- **DaSiamRPN**: Distractor-aware Siamese Region Proposal Network (ECCV 2018).
