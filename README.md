# AirDrawingAI ✍️

Draw in the air using hand tracking with OpenCV and MediaPipe.

![Python Version](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

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
- `h` : Toggle help overlay
- `[` / `]` : Adjust detection confidence

## 📦 System Requirements
- **Python 3.11+**
- **Webcam/USB camera**
- **Minimum 4GB RAM**
- **OpenCV-compatible OS** (Windows, macOS, Linux)

## 🚀 Quick Start

### Basic Installation
```bash
# Clone repository
git clone https://github.com/MohAsaad18/AirDrawingAI.git
cd AirDrawingAI

# Install requirements
pip install -r requirements.txt

# Run
python main.py
```

### Advanced Usage
```bash
# Run with higher FPS (60 instead of default 30)
python main.py --fps-limit 60

# Run with stricter hand detection
python main.py --detection-conf 0.9

# Run with custom resolution
python main.py --width 1920 --height 1080

# Run with alternative camera (if you have multiple cameras)
python main.py --camera 1

# Enable debug mode with confidence display
python main.py --debug --show-confidence
```

## 🔧 Troubleshooting

### Camera Not Found
- Check available camera ID: Try `--camera 0`, `--camera 1`, `--camera 2`
- Verify your webcam is connected and working
- Restart your application

### Hand Not Detected
- Improve lighting in your environment (well-lit room preferred)
- Adjust detection confidence lower: `python main.py --detection-conf 0.5`
- Ensure your hand is fully visible to the camera
- Try different camera angles

### Lag or Low FPS
- Reduce resolution: `python main.py --width 960 --height 540`
- Lower FPS cap: `python main.py --fps-limit 15`
- Close other applications consuming CPU/GPU
- Update your graphics drivers

### Drawing Accuracy Issues
- Increase smoothing by adjusting `tip_smooth` in config
- Improve lighting conditions
- Reduce camera distance
- Use `--detection-conf` lower if hands are not being tracked consistently

## 📋 File Structure
```
AirDrawingAI/
├── main.py              # Main application
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── LICENSE             # MIT License
├── CONTRIBUTING.md     # Contribution guidelines
├── .gitignore         # Git ignore patterns
└── Screenshots/       # Generated screenshots (not in repo)
```

## 🤝 Contributing
We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Technologies Used
- **OpenCV** - Computer vision library
- **MediaPipe** - Hand tracking solution by Google
- **NumPy** - Numerical computing

## 💡 Tips for Best Results
1. Use in a well-lit environment
2. Keep your hand at a reasonable distance from the camera (30-60cm)
3. Ensure your entire hand is visible in the frame
4. Adjust brush size for better control
5. Experiment with different confidence levels for your environment