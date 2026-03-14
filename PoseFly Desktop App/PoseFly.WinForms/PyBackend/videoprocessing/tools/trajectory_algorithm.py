import os
import re
import sys
import shutil
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


# =========================================================
# HELPERS
# =========================================================
def open_with_default_app(path: str) -> None:
    """Open a file with the system default application."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"Could not open output automatically: {e}")


def extract_first_number(text: str):
    """Extract first numeric value from text, return float or None."""
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        return float(match.group(0))
    return None


def parse_log_file(file_path: str):
    """
    Parse PoseFly-style result logs.

    Expected line examples:
      ANGLE (45 92.6%)
      DISTANCE (5m 0.2%)
      DISTANCE (no)

    Returns:
      points -> list of (distance, angle_deg)
    """
    points = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            angle_match = re.search(r"ANGLE\s*\(([^)]*)\)", line, re.IGNORECASE)
            dist_match = re.search(r"DISTANCE\s*\(([^)]*)\)", line, re.IGNORECASE)

            if not angle_match or not dist_match:
                continue

            angle_text = angle_match.group(1).strip()
            dist_text = dist_match.group(1).strip()

            # Skip rows with "no"
            if angle_text.lower().startswith("no") or dist_text.lower().startswith("no"):
                continue

            angle_value = extract_first_number(angle_text)
            dist_value = extract_first_number(dist_text)

            if angle_value is None or dist_value is None:
                continue

            # angle_value = first number in ANGLE(...)
            # dist_value  = first number in DISTANCE(...)
            points.append((dist_value, angle_value))

    return points


# =========================================================
# MAIN
# =========================================================
def main():
    # Hide root Tk window
    root = tk.Tk()
    root.withdraw()

    # -----------------------------------------------------
    # Select input txt file
    # -----------------------------------------------------
    input_path = filedialog.askopenfilename(
        title="Select PoseFly TXT Log File",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )

    if not input_path:
        print("No input file selected. Exiting.")
        return

    # -----------------------------------------------------
    # Parse points
    # -----------------------------------------------------
    points = parse_log_file(input_path)

    if not points:
        messagebox.showerror(
            "No Valid Data",
            "No valid (distance, angle) points were found in the selected file."
        )
        return

    print(f"Loaded {len(points)} valid points from:\n{input_path}")

    # -----------------------------------------------------
    # Convert data
    # -----------------------------------------------------
    r = np.array([d for d, a in points], dtype=float)
    theta_deg = np.array([a for d, a in points], dtype=float)
    theta_deg = np.mod(theta_deg, 360.0)
    theta_rad = np.deg2rad(theta_deg)

    max_r = max(1.0, float(np.ceil(np.max(r))))

    # -----------------------------------------------------
    # Save output path
    # -----------------------------------------------------
    has_ffmpeg = shutil.which("ffmpeg") is not None

    default_name = os.path.splitext(os.path.basename(input_path))[0]
    default_ext = ".mp4" if has_ffmpeg else ".gif"

    output_path = filedialog.asksaveasfilename(
        title="Save Animation As",
        defaultextension=default_ext,
        initialfile=f"{default_name}_trajectory{default_ext}",
        filetypes=(
            [("MP4 video", "*.mp4"), ("GIF animation", "*.gif"), ("All files", "*.*")]
            if has_ffmpeg
            else [("GIF animation", "*.gif"), ("All files", "*.*")]
        )
    )

    if not output_path:
        print("No output file selected. Exiting.")
        return

    # If ffmpeg is not available and user typed .mp4 manually, switch to .gif
    if not has_ffmpeg and output_path.lower().endswith(".mp4"):
        output_path = os.path.splitext(output_path)[0] + ".gif"

    # -----------------------------------------------------
    # Create figure
    # -----------------------------------------------------
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    ax.set_rlim(0, max_r)
    ax.set_rticks(list(range(1, int(max_r) + 1)))

    angle_ticks = np.arange(0, 360, 45)
    ax.set_xticks(np.deg2rad(angle_ticks))
    ax.set_xticklabels([f"{d}°" for d in angle_ticks])

    ax.grid(True)

    ax.scatter([0], [0], s=80, color="black")
    ax.annotate("Camera", (0, 0))

    title = ax.set_title("Drone Trajectory", va="bottom")

    # -----------------------------------------------------
    # Plot objects
    # -----------------------------------------------------
    old_point, = ax.plot([], [], marker="o", linestyle="None",
                         markersize=7, color="red", alpha=0.18)

    prev_point, = ax.plot([], [], marker="o", linestyle="None",
                          markersize=9, color="red", alpha=0.40)

    curr_point, = ax.plot([], [], marker="o", linestyle="None",
                          markersize=12, color="darkred", alpha=1.0)

    trail_line, = ax.plot([], [], linewidth=2.2, color="red", alpha=0.75)

    # -----------------------------------------------------
    # Init
    # -----------------------------------------------------
    def init():
        old_point.set_data([], [])
        prev_point.set_data([], [])
        curr_point.set_data([], [])
        trail_line.set_data([], [])
        title.set_text("Drone Trajectory")
        return old_point, prev_point, curr_point, trail_line, title

    # -----------------------------------------------------
    # Update
    # -----------------------------------------------------
    def update(frame):
        ths = []
        rs = []

        if frame >= 2:
            old_point.set_data([theta_rad[frame - 2]], [r[frame - 2]])
            ths.append(theta_rad[frame - 2])
            rs.append(r[frame - 2])
        else:
            old_point.set_data([], [])

        if frame >= 1:
            prev_point.set_data([theta_rad[frame - 1]], [r[frame - 1]])
            ths.append(theta_rad[frame - 1])
            rs.append(r[frame - 1])
        else:
            prev_point.set_data([], [])

        curr_point.set_data([theta_rad[frame]], [r[frame]])
        ths.append(theta_rad[frame])
        rs.append(r[frame])

        trail_line.set_data(ths, rs)
        title.set_text(
            f"Drone Trajectory | Frame {frame + 1}/{len(points)} | "
            f"Angle={theta_deg[frame]:.1f}° | Distance={r[frame]:.2f}m"
        )

        return old_point, prev_point, curr_point, trail_line, title

    # -----------------------------------------------------
    # Animation
    # -----------------------------------------------------
    ani = FuncAnimation(
        fig,
        update,
        frames=len(points),
        init_func=init,
        interval=350,
        blit=True,
        repeat=False
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    try:
        if output_path.lower().endswith(".mp4"):
            if not has_ffmpeg:
                raise RuntimeError(
                    "ffmpeg is not installed. Install it with: sudo apt install ffmpeg"
                )
            writer = FFMpegWriter(fps=4)
            ani.save(output_path, writer=writer)
        else:
            writer = PillowWriter(fps=4)
            ani.save(output_path, writer=writer)

    except Exception as e:
        plt.close(fig)
        messagebox.showerror("Save Failed", f"Could not save animation:\n\n{e}")
        return

    # -----------------------------------------------------
    # Done
    # -----------------------------------------------------
    plt.close(fig)

    messagebox.showinfo(
        "Done",
        f"Animation saved successfully:\n\n{output_path}"
    )

    print(f"Saved animation to:\n{output_path}")

    # Open saved video/gif automatically
    open_with_default_app(output_path)


if __name__ == "__main__":
    main()