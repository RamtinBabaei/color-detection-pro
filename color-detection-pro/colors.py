"""
colors.py
=========

Defines the HSV color profiles used by the detector.

Each `ColorProfile` describes the lower/upper HSV bounds used to build a
binary mask for a given color, plus a BGR value used purely for drawing
(bounding boxes, labels, trails) so each color is visually distinguishable
in the HUD.

Red requires two HSV ranges because red hue wraps around 0/180 in OpenCV's
HSV representation, so `upper2`/`lower2` are optional fields used only by
red (or any other wrap-around color you may add later).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

HSVBound = Tuple[int, int, int]


@dataclass(frozen=True)
class ColorProfile:
    """Describes a single detectable color."""

    name: str
    lower1: HSVBound
    upper1: HSVBound
    display_bgr: Tuple[int, int, int]
    lower2: Optional[HSVBound] = None
    upper2: Optional[HSVBound] = None

    @property
    def has_wraparound(self) -> bool:
        """True if this color needs a second HSV range (e.g. red)."""
        return self.lower2 is not None and self.upper2 is not None

    def as_arrays(self):
        """Return numpy array pairs ready for cv2.inRange."""
        bounds = [(np.array(self.lower1, dtype=np.uint8),
                   np.array(self.upper1, dtype=np.uint8))]
        if self.has_wraparound:
            bounds.append((np.array(self.lower2, dtype=np.uint8),
                            np.array(self.upper2, dtype=np.uint8)))
        return bounds


# --------------------------------------------------------------------------- #
# Registered color profiles
# --------------------------------------------------------------------------- #
COLOR_PROFILES: Dict[str, ColorProfile] = {
    "yellow": ColorProfile(
        name="yellow",
        lower1=(20, 100, 100),
        upper1=(35, 255, 255),
        display_bgr=(0, 255, 255),
    ),
    "orange": ColorProfile(
        name="orange",
        lower1=(10, 100, 100),
        upper1=(19, 255, 255),
        display_bgr=(0, 140, 255),
    ),
    "red": ColorProfile(
        name="red",
        lower1=(0, 120, 70),
        upper1=(10, 255, 255),
        lower2=(170, 120, 70),
        upper2=(180, 255, 255),
        display_bgr=(0, 0, 255),
    ),
    "green": ColorProfile(
        name="green",
        lower1=(36, 80, 60),
        upper1=(89, 255, 255),
        display_bgr=(0, 255, 0),
    ),
    "blue": ColorProfile(
        name="blue",
        lower1=(90, 100, 60),
        upper1=(130, 255, 255),
        display_bgr=(255, 0, 0),
    ),
}

ALL_COLOR_NAMES = tuple(COLOR_PROFILES.keys())


def get_profile(name: str) -> ColorProfile:
    """Look up a color profile by name, raising a clear error if missing."""
    try:
        return COLOR_PROFILES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown color '{name}'. Available colors: {ALL_COLOR_NAMES}"
        ) from exc
