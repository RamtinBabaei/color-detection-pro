"""
tracker.py
==========

Lightweight centroid-based multi-object tracker.

The tracker is deliberately dependency-free (no external tracking library)
so the whole detection -> tracking pipeline stays transparent and easy to
explain in a portfolio / thesis context. It assigns a stable integer ID to
each detected blob across frames by matching centroids greedily on
Euclidean distance, and derives:

- a motion trail (recent centroid history)
- an instantaneous / smoothed speed estimate (pixels per second)
- a "time visible" duration for each object
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

import config
from detector import DetectedObject
from utils import euclidean_distance


@dataclass
class TrackedObject:
    """State maintained for a single tracked object across frames."""

    object_id: int
    color_name: str
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    area: float
    contour_area: float
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    disappeared: int = 0
    trail: Deque[Tuple[int, int]] = field(
        default_factory=lambda: deque(maxlen=config.TRAIL_MAX_LENGTH)
    )
    speed_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=config.SPEED_SMOOTHING_WINDOW)
    )
    speed_px_s: float = 0.0

    @property
    def time_visible(self) -> float:
        """Seconds since this object was first detected."""
        return time.time() - self.first_seen

    def update(self, detection: DetectedObject) -> None:
        """Refresh this track with a newly matched detection."""
        now = time.time()
        dt = max(now - self.last_seen, 1e-6)

        distance = euclidean_distance(self.centroid, detection.center)
        instantaneous_speed = distance / dt
        self.speed_history.append(instantaneous_speed)
        self.speed_px_s = float(np.mean(self.speed_history))

        self.centroid = detection.center
        self.bbox = detection.bbox
        self.area = detection.area
        self.contour_area = detection.contour_area
        self.color_name = detection.color_name
        self.last_seen = now
        self.disappeared = 0
        self.trail.append(detection.center)


class CentroidTracker:
    """Assigns and maintains stable IDs for detected objects across frames."""

    def __init__(
        self,
        max_disappeared: int = config.TRACKER_MAX_DISAPPEARED,
        max_distance: int = config.TRACKER_MAX_DISTANCE,
    ) -> None:
        self.next_object_id = 0
        self.objects: "OrderedDict[int, TrackedObject]" = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _register(self, detection: DetectedObject) -> None:
        obj = TrackedObject(
            object_id=self.next_object_id,
            color_name=detection.color_name,
            centroid=detection.center,
            bbox=detection.bbox,
            area=detection.area,
            contour_area=detection.contour_area,
        )
        obj.trail.append(detection.center)
        self.objects[self.next_object_id] = obj
        self.next_object_id += 1

    def _deregister(self, object_id: int) -> None:
        del self.objects[object_id]

    def reset(self) -> None:
        """Clear all tracking state (used by the 'clear tracking' hotkey)."""
        self.objects.clear()
        self.next_object_id = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(self, detections: List[DetectedObject]) -> Dict[int, TrackedObject]:
        """
        Match the current frame's detections against existing tracks and
        return the up-to-date dictionary of {object_id: TrackedObject}.
        """
        if not detections:
            # No detections this frame: age out existing tracks.
            for object_id in list(self.objects.keys()):
                self.objects[object_id].disappeared += 1
                if self.objects[object_id].disappeared > self.max_disappeared:
                    self._deregister(object_id)
            return self.objects

        if not self.objects:
            for detection in detections:
                self._register(detection)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid].centroid for oid in object_ids]

        # Distance matrix: existing tracks (rows) x new detections (cols).
        distance_matrix = np.zeros((len(object_centroids), len(detections)), dtype=np.float32)
        for i, existing_centroid in enumerate(object_centroids):
            for j, detection in enumerate(detections):
                distance_matrix[i, j] = euclidean_distance(existing_centroid, detection.center)

        # Greedy matching: smallest distances first.
        rows_sorted = distance_matrix.min(axis=1).argsort()
        cols_for_row = distance_matrix.argmin(axis=1)

        used_rows: set = set()
        used_cols: set = set()

        for row in rows_sorted:
            col = cols_for_row[row]
            if row in used_rows or col in used_cols:
                continue
            if distance_matrix[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id].update(detections[col])
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_centroids))) - used_rows
        unused_cols = set(range(len(detections))) - used_cols

        # Tracks that found no match this frame: mark as disappeared.
        for row in unused_rows:
            object_id = object_ids[row]
            self.objects[object_id].disappeared += 1
            if self.objects[object_id].disappeared > self.max_disappeared:
                self._deregister(object_id)

        # Detections that matched no existing track: register as new objects.
        for col in unused_cols:
            self._register(detections[col])

        return self.objects
