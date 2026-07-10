# Color Detection Pro

**Professional Real-Time Multi-Color Object Detection and Tracking System**
built with Python and OpenCV.

Color Detection Pro detects colored objects from a live webcam feed,
assigns each one a stable tracking ID, follows it across frames with a
motion trail, estimates its speed, and overlays a dark-themed heads-up
display (HUD) with live statistics — all in real time.

> Built as a portfolio-grade computer vision project: modular architecture,
> type hints, docstrings, structured logging, and configurable HSV
> thresholds throughout.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [How It Works](#how-it-works)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)

---

## Features

### Core

- **Automatic webcam detection** with graceful error handling if no camera is found
- **Real-time HSV-based color detection** for Yellow, Orange, Red, Green, and Blue
- **Noise-robust masks** using Gaussian blur, morphological opening/closing, erosion, and dilation
- **Contour-based bounding boxes** via `cv2.findContours` (no external ML models)
- **Centroid-based multi-object tracking** with stable IDs across frames
- **Motion trails** showing each object's recent path
- **Speed estimation** in pixels/second (smoothed over a rolling window)
- **Per-object "time visible" timer**
- **Live FPS counter**, object counter, date/time display
- **Screenshot capture** (`S`) and **video recording** (`R`)
- **Dark-themed, semi-transparent HUD overlay**
- **Centralized configuration** (`config.py`) — no magic numbers scattered in the code
- **Structured logging** to both console and a rotating session log file

### Bonus

- 🎛️ **Live HSV calibration window** with trackbars for custom color tuning
- 🖼️ **Binary mask preview window** (toggle on demand)
- 📸 **Automatic screenshot** the moment a brand-new object is detected
- 📊 **CSV export** of per-object detection statistics for later analysis
- 🖥️ **Fullscreen toggle**
- 🔍 **Displays webcam resolution** in the HUD
- 🧭 **Automatic fallback camera detection** if the configured index fails

---

## Project Structure

```
color-detection-pro/
│
├── main.py              # Application entry point & GUI/event loop
├── detector.py           # HSV masking + contour-based detection
├── tracker.py             # Centroid tracker: IDs, trails, speed, timers
├── colors.py              # HSV color profile definitions
├── config.py              # All tunable configuration values
├── utils.py               # Logging, FPS counter, drawing helpers, CSV export
│
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── outputs/
│   ├── images/            # Manual & auto screenshots
│   └── videos/            # Recorded sessions (.mp4)
│
├── screenshots/           # README screenshots
├── assets/                # Static assets
└── logs/                  # session.log
```

---

## Installation

**Requirements:** Python 3.12+, a working webcam

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/color-detection-pro.git
cd color-detection-pro

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

- The application will attempt to open the configured camera (`config.CAMERA_INDEX`,
  default `0`). If that fails, it automatically probes other indices before
  reporting a clear error.
- By default it starts in **Red** detection mode — press a color key or `A`
  to switch modes.
- Screenshots are saved to `outputs/images/`, recordings to `outputs/videos/`,
  and a running session log is written to `logs/session.log`.

---

## Keyboard Shortcuts

| Key | Action                          |
|-----|----------------------------------|
| `1` | Detect **Yellow**                |
| `2` | Detect **Orange**                |
| `3` | Detect **Red**                   |
| `4` | Detect **Green**                 |
| `5` | Detect **Blue**                  |
| `A` | Detect **all colors** at once     |
| `S` | Save a screenshot                 |
| `R` | Start / stop video recording      |
| `C` | Clear tracking state (reset IDs)  |
| `M` | Toggle the binary mask window *(bonus)* |
| `H` | Toggle the HSV calibration window *(bonus)* |
| `F` | Toggle fullscreen *(bonus)*        |
| `Q` | Quit the application               |

---

## How It Works

1. **Preprocessing** — each frame is Gaussian-blurred and converted to HSV,
   which is far more lighting-robust than raw BGR thresholding.
2. **Masking** — `cv2.inRange` builds a binary mask per active color (red
   uses two ranges to handle the 0°/180° hue wraparound). Morphological
   opening removes speckle noise, closing fills holes, and erosion/dilation
   further clean the blob boundaries.
3. **Contour extraction** — `cv2.findContours` finds blob outlines; contours
   below `config.MIN_CONTOUR_AREA` are discarded as noise.
4. **Tracking** — a lightweight centroid tracker greedily matches each new
   detection to the closest existing track (within `TRACKER_MAX_DISTANCE`
   pixels). Unmatched tracks age out after `TRACKER_MAX_DISAPPEARED` frames;
   unmatched detections become new tracks with a fresh ID.
5. **Analytics** — for every tracked object the app derives a motion trail,
   a smoothed pixels/second speed estimate, and how long it has been
   continuously visible.
6. **HUD rendering** — a dark, semi-transparent overlay renders FPS, object
   count, per-object stats, and recording/tracking status directly onto the
   frame.

---

## Screenshots

`<img width="1928" height="1097" alt="image" src="https://github.com/user-attachments/assets/be53c935-c9bf-43f9-adf7-86baa6fa802e" />

---------------------------------------

## how It works

<img width="2720" height="2400" alt="color_detection_pro_feature_overview" src="https://github.com/user-attachments/assets/e02a0d31-ae3c-4397-941f-395357d5162b" />


---

## Future Improvements

- Swap the centroid tracker for a Kalman-filter or SORT/DeepSORT tracker for
  more robust handling of occlusion and crossing paths
- Add a lightweight web dashboard (Flask/FastAPI) for remote monitoring
- Support multiple simultaneous camera streams
- Add unit tests (pytest) for `detector.py` and `tracker.py`
- Package as a pip-installable CLI tool
- Add CPU/GPU usage overlay via `psutil`

---
