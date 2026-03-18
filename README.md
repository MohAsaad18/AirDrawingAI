# AirDrawingAI ✍️

Draw in the air using hand tracking with OpenCV and MediaPipe.

![Python Version](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features
- Draw in the air with finger gestures
- 8 colors + eraser
- Adjustable brush size (+/- keys)
- Smooth tip tracking
- Gesture debouncing (no accidental triggers)
- Take screenshots (canvas + combined)
- Modern dark theme UI

## 🎮 Gesture Controls
| Gesture | Action |
|---------|--------|
| 👆 Index finger | Draw |
| ✌️ Index + Middle | Select color from palette |
| ✊ Fist | Clear canvas |
| 🖐️ All fingers | Take screenshot |

## 🎯 Keyboard Controls
- `+` / `-` : Increase/decrease brush size
- `s` : Manual screenshot
- `q` : Quit

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/MohAsaad18/AirDrawingAI.git

# Install requirements
pip install opencv-python mediapipe numpy

# Run
python main.py
