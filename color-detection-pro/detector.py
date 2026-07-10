"""
detector.py
===========

Core computer-vision detection logic: HSV mask generation, noise removal,
and contour extraction. This module is intentionally free of any GUI or
tracking concerns -- it takes a BGR frame and returns a clean list of
`DetectedObject` instances plus the binary mask used, so it can be reused
or unit-tested independently of the rest of the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

import config
from colors import ColorProfile, get_profile


@dataclass
class DetectedObject:
    """A single detected blob for one frame, before any tracking is applied."""

    color_name: str
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    center: Tuple[int, int]
    area: float                        # bounding-box area (w * h)
    contour_area: float                # cv2.contourArea, i.e. true blob area


class ColorDetector:
    """Detects colored objects in a frame using HSV thresholding + contours."""

    def __init__(
        self,
        min_contour_area: int = config.MIN_CONTOUR_AREA,
        blur_kernel: Tuple[int, int] = config.GAUSSIAN_BLUR_KERNEL,
        morph_kernel_size: Tuple[int, int] = config.MORPH_KERNEL_SIZE,
        erode_iterations: int = config.ERODE_ITERATIONS,
        dilate_iterations: int = config.DILATE_ITERATIONS,
    ) -> None:
        self.min_contour_area = min_contour_area
        self.blur_kernel = blur_kernel
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
        self.erode_iterations = erode_iterations
        self.dilate_iterations = dilate_iterations

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def _to_blurred_hsv(self, frame: np.ndarray) -> np.ndarray:
        """Smooth the frame to reduce sensor noise, then convert to HSV."""
        blurred = cv2.GaussianBlur(frame, self.blur_kernel, 0)
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological opening/closing + erosion/dilation to denoise a mask."""
        # Opening removes small bright speckles (false positives).
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        # Closing fills small dark holes inside the detected blob.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        mask = cv2.erode(mask, self.morph_kernel, iterations=self.erode_iterations)
        mask = cv2.dilate(mask, self.morph_kernel, iterations=self.dilate_iterations)
        return mask

    def build_mask_for_color(self, hsv_frame: np.ndarray, profile: ColorProfile) -> np.ndarray:
        """Build a cleaned binary mask for a single color profile."""
        bounds = profile.as_arrays()
        mask = cv2.inRange(hsv_frame, bounds[0][0], bounds[0][1])
        for lower, upper in bounds[1:]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower, upper))
        return self._clean_mask(mask)

    # ------------------------------------------------------------------ #
    # Contour -> DetectedObject
    # ------------------------------------------------------------------ #
    def _contours_to_objects(
        self, mask: np.ndarray, color_name: str
    ) -> List[DetectedObject]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects: List[DetectedObject] = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)

            objects.append(
                DetectedObject(
                    color_name=color_name,
                    contour=contour,
                    bbox=(x, y, w, h),
                    center=center,
                    area=float(w * h),
                    contour_area=float(contour_area),
                )
            )
        return objects

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(
        self, frame: np.ndarray, color_names: Sequence[str]
    ) -> Tuple[List[DetectedObject], np.ndarray]:
        """
        Detect all objects matching any color in `color_names`.

        Returns a tuple of (detected_objects, combined_binary_mask) so the
        caller can both draw results and preview the raw mask (e.g. in the
        bonus "Binary Mask" debug window).
        """
        hsv_frame = self._to_blurred_hsv(frame)
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        all_objects: List[DetectedObject] = []

        for name in color_names:
            profile = get_profile(name)
            mask = self.build_mask_for_color(hsv_frame, profile)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            all_objects.extend(self._contours_to_objects(mask, name))

        return all_objects, combined_mask
