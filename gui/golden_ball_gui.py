#!/usr/bin/env python3
"""
High-Performance Golden Ball 4-View Tracking GUI
Optimized with background pre-loading for smooth real-time playback.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import csv
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk
import threading
import queue
import time
from collections import deque

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
VGA_DIR = PROJECT_DIR / "video_goal_analysis"

VIEW_FRAME_DIRS = {}
for _v in ("A", "H", "M", "R"):
    for _candidate in (
        VGA_DIR / f"frames/{_v}_2000",
        VGA_DIR / f"frames/{_v}_original_1200",
        VGA_DIR / f"frames/{_v}",
    ):
        if _candidate.exists() and list(_candidate.glob("*.jpg"))[:1]:
            VIEW_FRAME_DIRS[_v] = _candidate
            break

TRACKING_CSV = VGA_DIR / "output" / "enhanced_golden_tracking.csv"
if not TRACKING_CSV.exists():
    TRACKING_CSV = VGA_DIR / "output" / "golden_ball_tracking.csv"

# ---------------------------------------------------------------------------
# Preloader Thread
# ---------------------------------------------------------------------------

class FramePreloader(threading.Thread):
    def __init__(self, frame_paths, tracking, quadrants, canvas_size, buffer_size=60):
        super().__init__(daemon=True)
        self.frame_paths = frame_paths
        self.tracking = tracking
        self.quadrants = quadrants
        self.canvas_size = canvas_size # (w, h)
        self.buffer_size = buffer_size
        
        self.queue = queue.Queue(maxsize=buffer_size)
        self.target_idx = 0
        self.running = True
        self.lock = threading.Lock()
        
    def set_index(self, idx):
        with self.lock:
            self.target_idx = idx
            # Clear queue when jumping to a new position
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
                    
    def stop(self):
        self.running = False

    def run(self):
        current_load_idx = 0
        while self.running:
            with self.lock:
                # If we are far away from target, jump
                if abs(current_load_idx - self.target_idx) > self.buffer_size:
                    current_load_idx = self.target_idx
                
            if self.queue.full():
                time.sleep(0.01)
                continue
                
            # Load and process frame `current_load_idx`
            composite = self.process_frame(current_load_idx)
            if composite is not None:
                self.queue.put((current_load_idx, composite))
                current_load_idx += 1
            else:
                time.sleep(0.1)

    def process_frame(self, idx):
        # We need to check the total count from one of the views
        view_a = next(iter(self.frame_paths.keys()))
        if idx >= len(self.frame_paths[view_a]):
            return None
            
        cw, ch = self.canvas_size
        half_w, half_h = cw // 2, ch // 2
        composite = np.zeros((ch, cw, 3), dtype=np.uint8)
        
        track_row = self.tracking[idx] if idx < len(self.tracking) else {}
        
        for view, (qc, qr) in self.quadrants.items():
            x_off = qc * half_w
            y_off = qr * half_h
            
            p_list = self.frame_paths.get(view, [])
            if idx < len(p_list):
                img = cv2.imread(str(p_list[idx]))
                if img is not None:
                    tile = cv2.resize(img, (half_w, half_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Draw tracking
                    bx_raw = track_row.get(f"{view}_x", "")
                    by_raw = track_row.get(f"{view}_y", "")
                    br_raw = track_row.get(f"{view}_r", "")
                    
                    if bx_raw and by_raw:
                        try:
                            orig_h, orig_w = img.shape[:2]
                            sx, sy = half_w / orig_w, half_h / orig_h
                            bx, by = int(float(bx_raw) * sx), int(float(by_raw) * sy)
                            br = max(6, int(float(br_raw) * min(sx, sy))) if br_raw else 10
                            
                            # Golden ball highlight
                            cv2.circle(tile, (bx, by), br + 4, (0, 200, 255), 2)
                            cv2.circle(tile, (bx, by), br, (0, 215, 255), 2)
                            
                            # Trail (expensive to do in preloader if we don't have global tracking access easily,
                            # but we do have self.tracking here)
                            for back in range(1, 10):
                                ti = idx - back
                                if ti >= 0 and ti < len(self.tracking):
                                    t_row = self.tracking[ti]
                                    tx_raw, ty_raw = t_row.get(f"{view}_x", ""), t_row.get(f"{view}_y", "")
                                    if tx_raw and ty_raw:
                                        tx, ty = int(float(tx_raw) * sx), int(float(ty_raw) * sy)
                                        cv2.circle(tile, (tx, ty), 2, (0, 165, 255), -1)
                        except: pass
                    
                    # Labels
                    label = f"{view} | {p_list[idx].name}"
                    cv2.putText(tile, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                    cv2.putText(tile, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    composite[y_off:y_off+half_h, x_off:x_off+half_w] = tile
            
        return composite

# ---------------------------------------------------------------------------
# Optimized GUI
# ---------------------------------------------------------------------------

class OptimizedGoldenBallGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Golden Ball Tracker - 4-View SYNCED")
        self.root.geometry("1300x850")
        self.root.configure(bg="#0f172a")

        self.canvas_w, self.canvas_h = 1280, 720
        self.quadrants = {"A": (0, 0), "H": (1, 0), "M": (0, 1), "R": (1, 1)}
        
        self.frame_paths = {}
        self.tracking = []
        self.total_frames = 0
        self.current_idx = 0
        self.is_playing = False
        self.target_fps = 30
        
        self._load_data()
        
        # UI
        self._setup_ui()
        
        # Preloader
        self.preloader = FramePreloader(
            self.frame_paths, self.tracking, self.quadrants, 
            (self.canvas_w, self.canvas_h)
        )
        self.preloader.start()
        
        self.root.after(100, self._update_loop)

    def _load_data(self):
        min_count = float('inf')
        for v in ("A", "H", "M", "R"):
            if v in VIEW_FRAME_DIRS:
                paths = sorted(VIEW_FRAME_DIRS[v].glob("*.jpg"))
                self.frame_paths[v] = paths
                min_count = min(min_count, len(paths))
        self.total_frames = int(min_count) if min_count != float('inf') else 0
        
        if TRACKING_CSV.exists():
            with open(TRACKING_CSV, "r") as f:
                self.tracking = list(csv.DictReader(f))
        
    def _setup_ui(self):
        # Canvas
        self.canvas = tk.Canvas(self.root, width=self.canvas_w, height=self.canvas_h, bg="#000", highlightthickness=0)
        self.canvas.pack(pady=10)
        
        # Controls
        ctrl_frame = tk.Frame(self.root, bg="#0f172a")
        ctrl_frame.pack(fill=tk.X, padx=20)
        
        self.btn_play = tk.Button(ctrl_frame, text="▶ Play", command=self._toggle_play, width=10, bg="#1e293b", fg="white")
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.lbl_fps_ctrl = tk.Label(ctrl_frame, text="FPS:", bg="#0f172a", fg="white")
        self.lbl_fps_ctrl.pack(side=tk.LEFT, padx=5)
        self.speed_spin = tk.Spinbox(ctrl_frame, from_=1, to=120, width=5, command=self._on_speed_change)
        self.speed_spin.delete(0, "end")
        self.speed_spin.insert(0, "30")
        self.speed_spin.pack(side=tk.LEFT, padx=5)

        self.slider_var = tk.DoubleVar()
        self.slider = ttk.Scale(ctrl_frame, from_=0, to=self.total_frames-1, orient=tk.HORIZONTAL, variable=self.slider_var, command=self._on_slider)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)
        
        self.lbl_info = tk.Label(ctrl_frame, text="Frame: 0 / 0", bg="#0f172a", fg="#94a3b8", font=("Arial", 10))
        self.lbl_info.pack(side=tk.RIGHT, padx=10)
        
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("<Right>", lambda e: self._step(1))
        self.root.bind("<Left>", lambda e: self._step(-1))

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.config(text="⏸ Pause" if self.is_playing else "▶ Play")
        if self.is_playing:
            self.last_update_time = time.time()

    def _on_slider(self, val):
        idx = int(float(val))
        if idx != self.current_idx:
            self.current_idx = idx
            self.preloader.set_index(idx)
            self._render_current()

    def _step(self, delta):
        new_idx = max(0, min(self.total_frames - 1, self.current_idx + delta))
        self.slider_var.set(new_idx)
        self._on_slider(new_idx)

    def _render_current(self):
        # Try to get from queue
        found = False
        # We peek at the queue to see if it has our current index
        # This is a bit tricky with Queue. Simple approach: pull until we find idx or queue empty
        while not self.preloader.queue.empty():
            q_idx, composite = self.preloader.queue.get()
            if q_idx == self.current_idx:
                self._display(composite)
                found = True
                break
            elif q_idx > self.current_idx:
                # We missed it? (Shouldn't happen if playing forward)
                # But if we did, just keep the latest?
                # For now, if we are behind, we discard.
                pass
        
        if not found:
            # If not in queue, force a load (maybe slow)
            comp = self.preloader.process_frame(self.current_idx)
            if comp is not None:
                self._display(comp)
        
        self.lbl_info.config(text=f"Frame: {self.current_idx + 1} / {self.total_frames}")

    def _display(self, composite):
        img = Image.fromarray(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def _update_loop(self):
        if self.is_playing:
            now = time.time()
            dt = now - self.last_update_time
            target_dt = 1.0 / self.target_fps
            if dt >= target_dt:
                if self.current_idx < self.total_frames - 1:
                    # Catch-up logic if laggy
                    steps = int(dt / target_dt)
                    self.current_idx = min(self.total_frames - 1, self.current_idx + steps)
                    
                    self.slider_var.set(self.current_idx)
                    self._render_current()
                    self.last_update_time = now
                else:
                    self.is_playing = False
                    self.btn_play.config(text="▶ Play")
        else:
            self._render_current()

        self.root.after(5, self._update_loop)

    def _on_speed_change(self):
        try:
            self.target_fps = int(self.speed_spin.get())
        except:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizedGoldenBallGUI(root)
    root.mainloop()
