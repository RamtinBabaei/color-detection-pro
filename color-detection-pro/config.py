"""
config.py
=========

Centralized configuration for the Real-Time Multi-Color Object Detection
and Tracking System.

Every tunable parameter in the application lives here so the rest of the
codebase never hard-codes a "magic number". This keeps the system easy to
calibrate for different cameras, lighting conditions, and hardware.
"""

from __future__ import annotations

from pathlib import Path

import cv2

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent

OUTPUT_DIR: Path = BASE_DIR / "outputs"
OUTPUT_IMAGE_DIR: Path = OUTPUT_DIR / "images"
OUTPUT_VIDEO_DIR: Path = OUTPUT_DIR / "videos"
SCREENSHOT_DIR: Path = BASE_DIR / "screenshots"
ASSETS_DIR: Path = BASE_DIR / "assets"
LOG_DIR: Path = BASE_DIR / "logs"
LOG_FILE: Path = LOG_DIR / "session.log"
CSV_EXPORT_FILE: Path = OUTPUT_DIR / "detection_statistics.csv"

REQUIRED_DIRS = (
    OUTPUT_IMAGE_DIR,
    OUTPUT_VIDEO_DIR,
    SCREENSHOT_DIR,
    ASSETS_DIR,
    LOG_DIR,
)

# --------------------------------------------------------------------------- #
# Camera settings
# --------------------------------------------------------------------------- #
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
CAMERA_WARMUP_FRAMES: int = 5
MAX_CAMERA_INDEX_TO_PROBE: int = 5  # used for auto camera detection

# --------------------------------------------------------------------------- #
# Detection settings
# --------------------------------------------------------------------------- #
MIN_CONTOUR_AREA: int = 800
GAUSSIAN_BLUR_KERNEL: tuple[int, int] = (11, 11)
MORPH_KERNEL_SIZE: tuple[int, int] = (5, 5)
ERODE_ITERATIONS: int = 2
DILATE_ITERATIONS: int = 2

# --------------------------------------------------------------------------- #
# Tracking settings
# --------------------------------------------------------------------------- #
TRACKER_MAX_DISAPPEARED: int = 20      # frames before a track is dropped
TRACKER_MAX_DISTANCE: int = 90         # px, max centroid jump to keep same ID
TRAIL_MAX_LENGTH: int = 40             # points kept for the motion trail
SPEED_SMOOTHING_WINDOW: int = 5        # frames averaged for speed estimation

# --------------------------------------------------------------------------- #
# Recording settings
# --------------------------------------------------------------------------- #
RECORDING_FPS: float = 20.0
VIDEO_CODEC: str = "mp4v"
VIDEO_EXTENSION: str = ".mp4"

# --------------------------------------------------------------------------- #
# GUI / overlay settings
# --------------------------------------------------------------------------- #
WINDOW_NAME: str = "Color Detection Pro"
MASK_WINDOW_NAME: str = "Binary Mask"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SMALL: float = 0.5
FONT_SCALE_MEDIUM: float = 0.6
FONT_SCALE_LARGE: float = 0.9
FONT_THICKNESS: int = 1
FONT_THICKNESS_BOLD: int = 2

# Dark theme palette (BGR)
COLOR_PANEL_BG = (18, 18, 18)
COLOR_PANEL_BORDER = (60, 60, 60)
COLOR_TEXT_PRIMARY = (235, 235, 235)
COLOR_TEXT_ACCENT = (0, 215, 255)      # amber accent
COLOR_TEXT_SUCCESS = (0, 200, 0)
COLOR_TEXT_DANGER = (0, 0, 255)
COLOR_BOX = (0, 255, 0)
COLOR_CENTER_DOT = (0, 0, 255)
COLOR_TRAIL = (255, 200, 0)

PANEL_ALPHA: float = 0.55  # transparency for the HUD overlay panel

# --------------------------------------------------------------------------- #
# Behavior flags
# --------------------------------------------------------------------------- #
AUTO_SCREENSHOT_ON_NEW_OBJECT: bool = True
AUTO_SCREENSHOT_COOLDOWN_SEC: float = 3.0
FULLSCREEN_DEFAULT: bool = False

# --------------------------------------------------------------------------- #
# Keyboard shortcuts (documented, single source of truth)
# --------------------------------------------------------------------------- #
KEY_BINDINGS = {
    "1": "yellow",
    "2": "orange",
    "3": "red",
    "4": "green",
    "5": "blue",
    "a": "all",
    "s": "screenshot",
    "r": "record",
    "c": "clear_tracking",
    "m": "toggle_mask",
    "h": "toggle_calibration",
    "f": "toggle_fullscreen",
    "q": "quit",
}
