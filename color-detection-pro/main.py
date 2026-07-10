"""
main.py
=======

Entry point for the Real-Time Multi-Color Object Detection and Tracking
System.

Run with:

    python main.py

Keyboard controls are documented in the README and in `config.KEY_BINDINGS`.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

import config
from colors import ALL_COLOR_NAMES, get_profile
from detector import ColorDetector, DetectedObject
from tracker import CentroidTracker, TrackedObject
from utils import (
    CSVStatsLogger,
    FPSCounter,
    draw_lines_block,
    draw_translucent_panel,
    ensure_directories,
    put_text,
    setup_logging,
    timestamped_filename,
)

logger = setup_logging()


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read from."""


def find_working_camera_index(preferred_index: int) -> int:
    """
    Try the preferred camera index first, then probe a small range of other
    indices. Returns the first index that successfully opens and reads a
    frame. Raises CameraError if none work.
    """
    candidates = [preferred_index] + [
        i for i in range(config.MAX_CAMERA_INDEX_TO_PROBE) if i != preferred_index
    ]
    for index in candidates:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, _ = cap.read()
            cap.release()
            if ok:
                return index
    raise CameraError("No working webcam was found on this system.")


class ColorDetectionApp:
    """Top-level application: owns the camera, detector, tracker, and GUI state."""

    def __init__(self) -> None:
        ensure_directories(config.REQUIRED_DIRS)

        self.detector = ColorDetector()
        self.tracker = CentroidTracker()
        self.fps_counter = FPSCounter()
        self.csv_logger = CSVStatsLogger()

        self.active_colors: Set[str] = {"red"}
        self.show_mask_window: bool = False
        self.show_calibration_window: bool = False
        self.fullscreen: bool = config.FULLSCREEN_DEFAULT

        self.is_recording: bool = False
        self.video_writer: Optional[cv2.VideoWriter] = None

        self.capture: Optional[cv2.VideoCapture] = None
        self._known_object_ids: Set[int] = set()
        self._last_auto_screenshot_time: float = 0.0

        # Calibration trackbar defaults come from the currently active color.
        self._calibration_color = "red"

    # ------------------------------------------------------------------ #
    # Camera lifecycle
    # ------------------------------------------------------------------ #
    def open_camera(self) -> None:
        """Open the webcam, handling the case where it is unavailable."""
        try:
            index = find_working_camera_index(config.CAMERA_INDEX)
        except CameraError as exc:
            logger.error("Camera detection failed: %s", exc)
            raise

        capture = cv2.VideoCapture(index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        if not capture.isOpened():
            raise CameraError(f"Could not open camera index {index}.")

        # Warm up: some webcams return black/garbage frames for the first
        # few reads while auto-exposure settles.
        for _ in range(config.CAMERA_WARMUP_FRAMES):
            capture.read()

        self.capture = capture
        logger.info("Camera opened successfully (index=%s).", index)

    def release_camera(self) -> None:
        if self.capture is not None:
            self.capture.release()
            logger.info("Camera released.")

    # ------------------------------------------------------------------ #
    # Recording / screenshots
    # ------------------------------------------------------------------ #
    def _start_recording(self, frame_shape: tuple) -> None:
        height, width = frame_shape[:2]
        filename = timestamped_filename("recording", config.VIDEO_EXTENSION)
        path = config.OUTPUT_VIDEO_DIR / filename
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)

        writer = cv2.VideoWriter(str(path), fourcc, config.RECORDING_FPS, (width, height))
        if not writer.isOpened():
            logger.error("Failed to open VideoWriter for %s", path)
            return

        self.video_writer = writer
        self.is_recording = True
        logger.info("Recording started: %s", path)

    def _stop_recording(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        logger.info("Recording stopped.")

    def toggle_recording(self, frame_shape: tuple) -> None:
        try:
            if self.is_recording:
                self._stop_recording()
            else:
                self._start_recording(frame_shape)
        except cv2.error as exc:
            logger.error("Recording error: %s", exc)
            self.is_recording = False
            self.video_writer = None

    def save_screenshot(self, frame: np.ndarray, auto: bool = False) -> None:
        try:
            prefix = "auto_screenshot" if auto else "screenshot"
            filename = timestamped_filename(prefix, ".png")
            path = config.OUTPUT_IMAGE_DIR / filename
            success = cv2.imwrite(str(path), frame)
            if success:
                logger.info("Screenshot saved: %s", path)
            else:
                logger.error("cv2.imwrite failed for screenshot: %s", path)
        except Exception as exc:  # noqa: BLE001 - log and continue, never crash the loop
            logger.error("Screenshot error: %s", exc)

    # ------------------------------------------------------------------ #
    # Calibration window (bonus feature)
    # ------------------------------------------------------------------ #
    def _init_calibration_window(self) -> None:
        cv2.namedWindow("HSV Calibration")
        profile = get_profile(self._calibration_color)
        h_lo, s_lo, v_lo = profile.lower1
        h_hi, s_hi, v_hi = profile.upper1
        cv2.createTrackbar("H min", "HSV Calibration", h_lo, 179, lambda _: None)
        cv2.createTrackbar("H max", "HSV Calibration", h_hi, 179, lambda _: None)
        cv2.createTrackbar("S min", "HSV Calibration", s_lo, 255, lambda _: None)
        cv2.createTrackbar("S max", "HSV Calibration", s_hi, 255, lambda _: None)
        cv2.createTrackbar("V min", "HSV Calibration", v_lo, 255, lambda _: None)
        cv2.createTrackbar("V max", "HSV Calibration", v_hi, 255, lambda _: None)

    def _read_calibration_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        h_lo = cv2.getTrackbarPos("H min", "HSV Calibration")
        h_hi = cv2.getTrackbarPos("H max", "HSV Calibration")
        s_lo = cv2.getTrackbarPos("S min", "HSV Calibration")
        s_hi = cv2.getTrackbarPos("S max", "HSV Calibration")
        v_lo = cv2.getTrackbarPos("V min", "HSV Calibration")
        v_hi = cv2.getTrackbarPos("V max", "HSV Calibration")
        lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        return cv2.inRange(hsv_frame, lower, upper)

    # ------------------------------------------------------------------ #
    # Drawing / HUD
    # ------------------------------------------------------------------ #
    def _draw_object(self, frame: np.ndarray, obj: TrackedObject) -> None:
        x, y, w, h = obj.bbox
        profile = get_profile(obj.color_name)
        box_color = profile.display_bgr

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.circle(frame, obj.centroid, 5, config.COLOR_CENTER_DOT, -1)

        # Motion trail: fading line through recent centroids.
        points = list(obj.trail)
        for i in range(1, len(points)):
            thickness = max(1, int(np.sqrt(config.TRAIL_MAX_LENGTH / float(i + 1)) * 1.5))
            cv2.line(frame, points[i - 1], points[i], config.COLOR_TRAIL, thickness)

        label = f"ID {obj.object_id} | {obj.color_name.upper()}"
        cv2.putText(
            frame, label, (x, max(y - 10, 15)),
            config.FONT, config.FONT_SCALE_SMALL, box_color, config.FONT_THICKNESS_BOLD,
            cv2.LINE_AA,
        )

        info_lines = [
            f"Center: ({obj.centroid[0]}, {obj.centroid[1]})",
            f"W x H: {w} x {h}",
            f"Area: {int(obj.area)}  Contour: {int(obj.contour_area)}",
            f"Speed: {obj.speed_px_s:.1f} px/s",
            f"Detected: {obj.time_visible:.2f} sec",
        ]
        draw_lines_block(
            frame, info_lines, (x, y + h + 15), line_height=16,
            color=config.COLOR_TEXT_PRIMARY, scale=config.FONT_SCALE_SMALL,
        )

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        tracked_objects: Dict[int, TrackedObject],
    ) -> None:
        h, w = frame.shape[:2]

        # Top-left status panel.
        draw_translucent_panel(frame, (10, 10), (330, 150))
        now = datetime.now()
        active = ", ".join(sorted(self.active_colors)).upper()
        lines = [
            f"Color Mode: {active}",
            f"FPS: {fps:.0f}",
            f"Objects: {len(tracked_objects)}",
            f"Date: {now.strftime('%Y-%m-%d')}",
            f"Time: {now.strftime('%H:%M:%S')}",
            f"Resolution: {w}x{h}",
        ]
        draw_lines_block(frame, lines, (20, 32), line_height=20)

        # Top-right recording / tracking status panel.
        draw_translucent_panel(frame, (w - 230, 10), (w - 10, 90))
        rec_color = config.COLOR_TEXT_DANGER if self.is_recording else config.COLOR_TEXT_PRIMARY
        rec_text = "RECORDING" if self.is_recording else "Not recording"
        put_text(frame, rec_text, (w - 220, 35), color=rec_color, scale=config.FONT_SCALE_MEDIUM)
        put_text(
            frame, "Tracking: ACTIVE", (w - 220, 60),
            color=config.COLOR_TEXT_SUCCESS, scale=config.FONT_SCALE_SMALL,
        )
        put_text(
            frame, "Press Q to quit", (w - 220, 80),
            color=config.COLOR_TEXT_PRIMARY, scale=config.FONT_SCALE_SMALL,
        )

        if self.is_recording:
            cv2.circle(frame, (w - 205, 27), 6, config.COLOR_TEXT_DANGER, -1)

    # ------------------------------------------------------------------ #
    # Keyboard handling
    # ------------------------------------------------------------------ #
    def _handle_key(self, key_char: str, frame: np.ndarray) -> bool:
        """Handle a single keypress. Returns False if the app should quit."""
        binding = config.KEY_BINDINGS.get(key_char)
        if binding is None:
            return True

        if binding == "quit":
            return False
        if binding == "all":
            self.active_colors = set(ALL_COLOR_NAMES)
            logger.info("Switched to detecting ALL colors.")
        elif binding in ALL_COLOR_NAMES:
            self.active_colors = {binding}
            self._calibration_color = binding
            logger.info("Switched to detecting color: %s", binding)
        elif binding == "screenshot":
            self.save_screenshot(frame)
        elif binding == "record":
            self.toggle_recording(frame.shape)
        elif binding == "clear_tracking":
            self.tracker.reset()
            self._known_object_ids.clear()
            logger.info("Tracking state cleared by user.")
        elif binding == "toggle_mask":
            self.show_mask_window = not self.show_mask_window
            if not self.show_mask_window:
                cv2.destroyWindow(config.MASK_WINDOW_NAME)
        elif binding == "toggle_calibration":
            self.show_calibration_window = not self.show_calibration_window
            if self.show_calibration_window:
                self._init_calibration_window()
            else:
                cv2.destroyWindow("HSV Calibration")
        elif binding == "toggle_fullscreen":
            self.fullscreen = not self.fullscreen
            prop = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)

        return True

    # ------------------------------------------------------------------ #
    # CSV export helper
    # ------------------------------------------------------------------ #
    def _export_stats(self, tracked_objects: Dict[int, TrackedObject]) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        rows = []
        for obj in tracked_objects.values():
            rows.append(
                {
                    "timestamp": timestamp,
                    "object_id": obj.object_id,
                    "color": obj.color_name,
                    "center_x": obj.centroid[0],
                    "center_y": obj.centroid[1],
                    "width": obj.bbox[2],
                    "height": obj.bbox[3],
                    "area": obj.area,
                    "contour_area": obj.contour_area,
                    "speed_px_s": round(obj.speed_px_s, 2),
                }
            )
        self.csv_logger.log_rows(rows)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        logger.info("Application starting.")
        try:
            self.open_camera()
        except CameraError as exc:
            print(f"[ERROR] {exc}")
            print("Please check that a webcam is connected and not in use by another app.")
            return

        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C).")
        finally:
            self._shutdown()

    def _main_loop(self) -> None:
        assert self.capture is not None
        frame_export_counter = 0

        while True:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                logger.error("Invalid frame received from camera; stopping.")
                break

            frame = cv2.flip(frame, 1)  # mirror view feels more natural to users
            fps = self.fps_counter.tick()

            detections, mask = self.detector.detect(frame, self.active_colors)
            tracked_objects = self.tracker.update(detections)

            for obj in tracked_objects.values():
                self._draw_object(frame, obj)

            self._draw_hud(frame, fps, tracked_objects)

            # Auto screenshot bonus: fires shortly after a brand-new object appears.
            current_ids = set(tracked_objects.keys())
            new_ids = current_ids - self._known_object_ids
            if (
                config.AUTO_SCREENSHOT_ON_NEW_OBJECT
                and new_ids
                and (time.time() - self._last_auto_screenshot_time)
                > config.AUTO_SCREENSHOT_COOLDOWN_SEC
            ):
                self.save_screenshot(frame, auto=True)
                self._last_auto_screenshot_time = time.time()
            self._known_object_ids = current_ids

            # Export stats roughly once per second instead of every frame.
            frame_export_counter += 1
            if frame_export_counter >= max(int(fps), 1) and tracked_objects:
                self._export_stats(tracked_objects)
                frame_export_counter = 0

            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(frame)

            cv2.imshow(config.WINDOW_NAME, frame)

            if self.show_mask_window:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.imshow(config.MASK_WINDOW_NAME, mask_bgr)

            if self.show_calibration_window:
                hsv_frame = cv2.cvtColor(cv2.GaussianBlur(frame, (11, 11), 0), cv2.COLOR_BGR2HSV)
                calib_mask = self._read_calibration_mask(hsv_frame)
                cv2.imshow("HSV Calibration", calib_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:  # no key pressed
                continue
            key_char = chr(key).lower()
            if not self._handle_key(key_char, frame):
                logger.info("Quit requested by user.")
                break

    def _shutdown(self) -> None:
        if self.is_recording:
            self._stop_recording()
        self.release_camera()
        cv2.destroyAllWindows()
        logger.info("Application shut down cleanly.")


def main() -> int:
    app = ColorDetectionApp()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
