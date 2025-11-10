#!/usr/bin/env python3
"""
Video + Telemetry Sync Viewer (DriveTest edition, hardcoded sync + SEEK SLIDER)
Now plots TWO telemetry signals from the CSV:
  1) motor_speed.quadrature_current   [top plot]
  2) motor_power.measured_dc_voltage_v [bottom plot]

Defaults:
  --video "drivetest.mp4"
  --csv   "DriveTestMain - output.csv"
  --current-col "motor_speed.quadrature_current"
  --voltage-col "motor_power.measured_dc_voltage_v"

If no time column is provided/found, time is synthesized using a fixed step.
A moving vertical line and dot show the current telemetry time as the video plays.

Controls:
  A/D    : offset  -0.10s / +0.10s
  Z/X    : offset  -1.00s / +1.00s
  W/S    : scale   +1% / -1%
  C/V    : scale   +10% / -10%
  [ / ]  : step    -10% / +10%   (only with synthetic time)
  { / }  : step    -1% / +1%     (Shift + [ or ])
  SPACE  : pause / resume
  R      : reset sync to hardcoded (if enabled) or neutral (0,1)
  Q/ESC  : quit
  Slider "Seek": drag to jump to a frame (viewer pauses on seek)

Tip: Use --lock-sync to ignore any hotkeys that change offset/scale/step.
"""

import argparse
import time
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # render plots off-screen for OpenCV blitting
from matplotlib import pyplot as plt

# --- Hardcoded sync (edit these as you like) ---
HARD_SYNC_ENABLED = True         # set False to ignore the hardcoded defaults
HARD_CSV_STEP_SEC = 0.0352       # telemetry logging period [s]
HARD_INIT_OFFSET  = 285.0         # seconds added after scaling
HARD_INIT_SCALE   = 1.0          # multiply video time before adding offset

def parse_args():
    p = argparse.ArgumentParser(description="Sync telemetry CSV to a video and visualize with moving time markers.")
    # File defaults for your setup
    p.add_argument("--video", default="drivetest.mp4", help='Path to video file (default: "drivetest.mp4")')
    p.add_argument("--csv",   default="DriveTestMain - output.csv", help='Path to telemetry CSV (default: "DriveTestMain - output.csv")')

    # Time column is optional; if missing we synthesize evenly spaced time
    p.add_argument("--time-col", default="", help="Time column name in CSV (leave empty to synthesize)")
    p.add_argument("--csv-time-scale", type=float, default=1.0, help="Multiply CSV time by this (e.g., 0.001 if CSV is in ms)")

    # Telemetry columns
    p.add_argument("--current-col", default="motor_speed.quadrature_current", help="CSV column for current (top plot)")
    p.add_argument("--voltage-col", default="motor_power.measured_dc_voltage_v", help="CSV column for DC bus voltage (bottom plot)")

    # Defaults come from hard-coded constants (or neutral if disabled)
    p.add_argument("--csv-step-sec", type=float,
                   default=(HARD_CSV_STEP_SEC if HARD_SYNC_ENABLED else 0.01),
                   help="Sample period to use if time column is missing (seconds)")
    p.add_argument("--init-offset", type=float,
                   default=(HARD_INIT_OFFSET if HARD_SYNC_ENABLED else 0.0),
                   help="Initial time offset in seconds (adds to scaled video time)")
    p.add_argument("--init-scale", type=float,
                   default=(HARD_INIT_SCALE if HARD_SYNC_ENABLED else 1.0),
                   help="Initial time scale multiplier")

    p.add_argument("--plot-width", type=int, default=640, help="Width of plot panel (pixels)")
    p.add_argument("--dpi", type=int, default=100, help="Matplotlib figure DPI (higher = sharper, slower)")
    p.add_argument("--y-pad", type=float, default=0.1, help="Fractional vertical padding for y-limits")
    p.add_argument("--fps-cap", type=float, default=0.0, help="Optional FPS cap for display loop (0=uncapped)")

    # Freeze interactive tweaks if desired
    p.add_argument("--lock-sync", action="store_true",
                   help="Disable hotkeys that modify offset/scale/step during playback")
    return p.parse_args()

def load_csv(csv_path, time_col, cur_col, volt_col, time_scale, fallback_step):
    df = pd.read_csv(csv_path)

    missing = [c for c in (cur_col, volt_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Available columns include: {list(df.columns)[:12]} ...")

    cur = df[cur_col].astype(float).to_numpy()
    volt = df[volt_col].astype(float).to_numpy()

    if time_col and time_col in df.columns:
        t = df[time_col].astype(float).to_numpy() * time_scale
        # Sort & dedup to keep interpolation stable
        order = np.argsort(t)
        t = t[order]
        cur = cur[order]
        volt = volt[order]
        uniq = np.concatenate(([True], np.diff(t) > 0))
        t = t[uniq]
        cur = cur[uniq]
        volt = volt[uniq]
        time_mode = "csv"
    else:
        # Synthesize time: 0, step, 2*step, ...
        n = len(cur)
        t = np.arange(n, dtype=float) * float(fallback_step)
        # Ensure volt matches length
        if len(volt) != n:
            # Trim/pad to match
            m = min(n, len(volt))
            cur = cur[:m]
            volt = volt[:m]
            t = t[:m]
        time_mode = "synthetic"

    return t, cur, volt, time_mode

def make_figure_img(times, cur_values, volt_values, current_time,
                    width_px, height_px, dpi, ypad_frac,
                    artists=None, fig=None, axes=None, title_suffix=""):
    """
    Draw two stacked plots sharing time:
      Top  : current (A)
      Bottom: DC voltage (V)
    artists: dict with keys 'cur_line', 'cur_vline', 'cur_dot', 'volt_line', 'volt_vline', 'volt_dot'
    Returns: (img_bgr, artists, fig, axes)
    """
    if fig is None or axes is None or artists is None:
        fig_w = width_px / dpi
        fig_h = height_px / dpi
        fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=dpi, sharex=True,
                                gridspec_kw={"height_ratios": [1, 1], "hspace": 0.15})
        ax_cur, ax_volt = axs

        # Current plot
        ax_cur.set_title(f"Quadrature Current vs Time{title_suffix}")
        ax_cur.set_ylabel("Current [A]")
        cur_line, = ax_cur.plot(times, cur_values, lw=1)
        cur_vline = ax_cur.axvline(current_time, lw=2)
        y_now_cur = np.interp(current_time, times, cur_values, left=np.nan, right=np.nan)
        cur_dot, = ax_cur.plot([current_time], [y_now_cur], marker="o", ms=6)

        # Voltage plot
        ax_volt.set_xlabel("Time [s]")
        ax_volt.set_ylabel("DC Voltage [V]")
        volt_line, = ax_volt.plot(times, volt_values, lw=1)
        volt_vline = ax_volt.axvline(current_time, lw=2)
        y_now_volt = np.interp(current_time, times, volt_values, left=np.nan, right=np.nan)
        volt_dot, = ax_volt.plot([current_time], [y_now_volt], marker="o", ms=6)

        # Y limits with padding
        def set_ylims(ax, y):
            y_min, y_max = np.nanmin(y), np.nanmax(y)
            yr = (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0
            pad = yr * ypad_frac
            ax.set_ylim(y_min - pad, y_max + pad)

        set_ylims(ax_cur, cur_values)
        set_ylims(ax_volt, volt_values)

        # X limits
        x0, x1 = times[0], times[-1] if times[-1] > times[0] else (times[0] + 1.0)
        ax_cur.set_xlim(x0, x1)

        fig.tight_layout()

        artists = {
            "cur_line": cur_line, "cur_vline": cur_vline, "cur_dot": cur_dot,
            "volt_line": volt_line, "volt_vline": volt_vline, "volt_dot": volt_dot,
        }
        axes = (ax_cur, ax_volt)
    else:
        ax_cur, ax_volt = axes
        # Move markers
        artists["cur_vline"].set_xdata([current_time, current_time])
        y_now_cur = np.interp(current_time, times, cur_values, left=np.nan, right=np.nan)
        artists["cur_dot"].set_data([current_time], [y_now_cur])

        artists["volt_vline"].set_xdata([current_time, current_time])
        y_now_volt = np.interp(current_time, times, volt_values, left=np.nan, right=np.nan)
        artists["volt_dot"].set_data([current_time], [y_now_volt])

    # Render figure to RGBA, convert to BGR
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    try:
        img_rgba = np.asarray(fig.canvas.buffer_rgba())
    except Exception:
        renderer = fig.canvas.get_renderer()
        buf = renderer.buffer_rgba()
        img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)

    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    if img_bgr.shape[1] != width_px or img_bgr.shape[0] != height_px:
        img_bgr = cv2.resize(img_bgr, (width_px, height_px), interpolation=cv2.INTER_AREA)

    return img_bgr, artists, fig, axes

def main():
    args = parse_args()

    # Basic existence check
    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")
    if not os.path.exists(args.video):
        raise SystemExit(f"Video not found: {args.video}")

    # Load telemetry
    times, cur_values, volt_values, time_mode = load_csv(
        args.csv, args.time_col, args.current_col, args.voltage_col,
        args.csv_time_scale, args.csv_step_sec
    )

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1000  # fallback

    # Window + seek slider
    win_name = "Video + Telemetry Sync"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    seek_requested = False
    seek_target_frame = 0
    ignore_trackbar = False
    paused = False

    def on_seek(val):
        nonlocal seek_requested, seek_target_frame, paused
        if ignore_trackbar:
            return
        seek_target_frame = int(val)
        seek_requested = True
        paused = True  # pause on user scrub

    cv2.createTrackbar("Seek", win_name, 0, max(0, total_frames - 1), on_seek)

    # Canvas sizes
    plot_w = args.plot_width
    out_h = vid_h
    out_w = vid_w + plot_w

    # Alignment state
    time_offset = args.init_offset
    time_scale  = args.init_scale

    # Title suffix to show time mode
    title_suffix = f"  ({'CSV time' if time_mode=='csv' else f'step={args.csv_step_sec:.4f}s'})"

    # Pre-render plots
    artists = None
    fig = None
    axes = None
    plot_img, artists, fig, axes = make_figure_img(
        times, cur_values, volt_values, current_time=0.0,
        width_px=plot_w, height_px=out_h, dpi=args.dpi, ypad_frac=args.y_pad,
        artists=artists, fig=fig, axes=axes, title_suffix=title_suffix
    )

    def draw_hud(img, fps, t_vid, t_tele, csv_step_sec, locked):
        # Show interpolated values at the marker (nice for alignment)
        cur_now = float(np.interp(t_tele, times, cur_values, left=np.nan, right=np.nan))
        volt_now = float(np.interp(t_tele, times, volt_values, left=np.nan, right=np.nan))

        txt = [
            f"Video FPS: {fps:.2f}   Frames: {total_frames}",
            f"Video t: {t_vid:.3f} s   Telemetry t: {t_tele:.3f} s"
            f"   Iq: {cur_now:+.2f} A   Vdc: {volt_now:.1f} V",
            f"Scale: {time_scale:.4f}   Offset: {time_offset:+.3f} s   Step: {csv_step_sec:.4f}s ({'csv' if time_mode=='csv' else 'synthetic'})   Lock:{'ON' if locked else 'OFF'}",
            "Keys: [A/D] off ±0.10s  [Z/X] off ±1s   [W/S] scale ±1%  [C/V] scale ±10%",
            "      [[/]] step −10%/+10%   { / } step −1%/+1%   [SPACE] pause   [R] reset   [Q] quit",
            "Slider: drag 'Seek' to jump to a frame (video pauses on seek)"
        ]
        y = 22
        for line in txt:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
            y += 22

    last_t = time.perf_counter()
    frame_delay = 1.0/float(args.fps_cap) if args.fps_cap and args.fps_cap > 0 else 0.0
    csv_step_sec = args.csv_step_sec  # live-adjustable if synthetic

    while True:
        # Handle seek requests
        if seek_requested:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(seek_target_frame, total_frames - 1)))
            seek_requested = False

        # Read frame (pause keeps current frame visible)
        ok, frame = cap.read() if not paused else (True, cap.retrieve()[1] if cap.grab() else None)
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break

        cur_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if cur_frame_idx < 0:
            cur_frame_idx = 0

        # Update slider position without re-triggering seek
        ignore_trackbar = True
        try:
            cv2.setTrackbarPos("Seek", win_name, max(0, min(cur_frame_idx, total_frames - 1)))
        finally:
            ignore_trackbar = False

        # Current video time
        t_video_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if t_video_ms and t_video_ms > 0:
            t_video = t_video_ms / 1000.0
        else:
            fps_safe = native_fps if native_fps and native_fps > 0 else 30.0
            t_video = cur_frame_idx / fps_safe

        # If using synthetic time and step changed, recompute 'times'
        if time_mode == "synthetic":
            n = len(cur_values)
            times = np.arange(n, dtype=float) * float(csv_step_sec)

        # Map to telemetry time: t_tele = scale * t_video + offset
        t_tele = time_scale * t_video + time_offset

        # Update plots image
        plot_img, artists, fig, axes = make_figure_img(
            times, cur_values, volt_values, current_time=t_tele,
            width_px=plot_w, height_px=out_h, dpi=args.dpi, ypad_frac=args.y_pad,
            artists=artists, fig=fig, axes=axes, title_suffix=title_suffix
        )

        # Compose side-by-side
        if frame.shape[0] != out_h:
            frame = cv2.resize(frame, (vid_w, out_h))
        combo = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        combo[:, :vid_w, :] = frame
        combo[:, vid_w:, :] = plot_img

        draw_hud(combo, native_fps if native_fps else 0.0, t_video, t_tele, csv_step_sec, args.lock_sync)
        cv2.imshow(win_name, combo)

        key = cv2.waitKey(1 if not paused else 10) & 0xFF
        if key in (ord('q'), 27):     # q or ESC
            break
        elif key == ord(' '):         # pause/resume
            paused = not paused
        elif key == ord('r') and not args.lock_sync:
            time_offset = HARD_INIT_OFFSET if HARD_SYNC_ENABLED else 0.0
            time_scale  = HARD_INIT_SCALE  if HARD_SYNC_ENABLED else 1.0
            if time_mode == "synthetic":
                csv_step_sec = HARD_CSV_STEP_SEC if HARD_SYNC_ENABLED else args.csv_step_sec
        elif not args.lock_sync:
            if key == ord('a'):
                time_offset -= 0.10
            elif key == ord('d'):
                time_offset += 0.10
            elif key == ord('z'):
                time_offset -= 1.0
            elif key == ord('x'):
                time_offset += 1.0
            elif key == ord('w'):
                time_scale *= 1.01
            elif key == ord('s'):
                time_scale /= 1.01
            elif key == ord('c'):
                time_scale *= 1.10
            elif key == ord('v'):
                time_scale /= 1.10
            elif time_mode == "synthetic":
                if key == ord('['):       # -10%
                    csv_step_sec *= 0.90
                elif key == ord(']'):     # +10%
                    csv_step_sec /= 0.90
                elif key == ord('{'):     # -1% (Shift + [)
                    csv_step_sec *= 0.99
                elif key == ord('}'):     # +1% (Shift + ])
                    csv_step_sec /= 0.99

        if frame_delay > 0:
            now = time.perf_counter()
            dt = now - last_t
            if dt < frame_delay:
                time.sleep(frame_delay - dt)
            last_t = time.perf_counter()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
