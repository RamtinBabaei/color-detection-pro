"""
utils.py
========

Shared utilities used across the application: logging setup, FPS
measurement, HUD drawing helpers, filesystem helpers, and CSV export of
detection statistics.

Keeping these small, single-purpose helpers here avoids duplicating logic
across `main.py`, `detector.py`, and `tracker.py`.
"""

from __future__ import annotations

import csv
import logging
import math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

import config


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def ensure_directories(paths: Iterable[Path]) -> None:
    """Create every directory in `paths` if it does not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def timestamped_filename(prefix: str, extension: str) -> str:
    """Build a collision-safe filename such as `screenshot_20260710_142233.png`."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}{extension}"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(log_file: Path = config.LOG_FILE) -> logging.Logger:
    """Configure a logger that writes to both console and a session log file."""
    ensure_directories([log_file.parent])

    logger = logging.getLogger("color_detection_pro")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        # Avoid duplicate handlers if setup_logging() is called more than once.
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# --------------------------------------------------------------------------- #
# FPS measurement
# --------------------------------------------------------------------------- #
class FPSCounter:
    """Rolling-average FPS counter based on a sliding window of frame times."""

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: Deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        """Register a new frame and return the current smoothed FPS."""
        now = cv2.getTickCount() / cv2.getTickFrequency()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def euclidean_distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    """Standard 2D Euclidean distance between two points."""
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


# --------------------------------------------------------------------------- #
# Drawing helpers (dark-themed HUD)
# --------------------------------------------------------------------------- #
def draw_translucent_panel(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int] = config.COLOR_PANEL_BG,
    alpha: float = config.PANEL_ALPHA,
) -> None:
    """Draw a semi-transparent dark rectangle used as a backdrop for HUD text."""
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    cv2.rectangle(frame, top_left, bottom_right, config.COLOR_PANEL_BORDER, 1)


def put_text(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int] = config.COLOR_TEXT_PRIMARY,
    scale: float = config.FONT_SCALE_MEDIUM,
    thickness: int = config.FONT_THICKNESS,
) -> None:
    """Draw anti-aliased text with the project's default font settings."""
    cv2.putText(frame, text, origin, config.FONT, scale, color, thickness, cv2.LINE_AA)


def draw_lines_block(
    frame: np.ndarray,
    lines: Sequence[str],
    origin: Tuple[int, int],
    line_height: int = 22,
    color: Tuple[int, int, int] = config.COLOR_TEXT_PRIMARY,
    scale: float = config.FONT_SCALE_SMALL,
) -> int:
    """Draw multiple lines of text stacked vertically. Returns the y after the block."""
    x, y = origin
    for line in lines:
        put_text(frame, line, (x, y), color=color, scale=scale)
        y += line_height
    return y


# --------------------------------------------------------------------------- #
# CSV export of detection statistics
# --------------------------------------------------------------------------- #
class CSVStatsLogger:
    """Appends per-frame detection statistics to a CSV file for later analysis."""

    FIELDS = [
        "timestamp",
        "object_id",
        "color",
        "center_x",
        "center_y",
        "width",
        "height",
        "area",
        "contour_area",
        "speed_px_s",
    ]

    def __init__(self, csv_path: Path = config.CSV_EXPORT_FILE) -> None:
        self.csv_path = csv_path
        ensure_directories([csv_path.parent])
        if not self.csv_path.exists():
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.FIELDS)
                writer.writeheader()

    def log_rows(self, rows: List[dict]) -> None:
        """Append a batch of rows (one per tracked object) to the CSV file."""
        if not rows:
            return
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDS)
            for row in rows:
                writer.writerow(row)
