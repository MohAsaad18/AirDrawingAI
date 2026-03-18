"""
AIR DRAWING AI  -  Modern Edition
===================================
GESTURE GUIDE
  [1 finger]   Index only        ->  Draw
  [2 fingers]  Index + Middle    ->  Select color
  [0 fingers]  Fist              ->  Clear canvas
  [5 fingers]  All fingers up    ->  Screenshot

KEYS:
  q         = quit
  s         = screenshot
  h         = toggle help
  + / =     = brush bigger
  -         = brush smaller
  [ / ]     = adjust detection confidence
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import logging
import argparse
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Tuple
from pathlib import Path


# ════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════
@dataclass
class Config:
    # Camera
    cam_w:        int   = 1280
    cam_h:        int   = 720
    camera_id:    int   = 0
    fps_limit:    int   = 30

    # Hand tracking
    max_hands:       int   = 1
    detection_conf:  float = 0.75
    tracking_conf:   float = 0.75

    # Brush
    default_brush: int = 15
    min_brush:     int = 3
    max_brush:     int = 60
    brush_step:    int = 2
    eraser_size:   int = 60
    tip_smooth:    int = 6

    # Gesture
    gesture_hold:  int   = 3
    shot_cooldown: float = 2.0
    shot_dir:      str   = "Screenshots"

    # Palette  (label, BGR)
    palette: list = field(default_factory=lambda: [
        ("PINK",   (220,   0, 255)),
        ("SKY",    ( 50, 200, 255)),
        ("LIME",   (100, 255,  40)),
        ("ORANGE", (  0, 140, 255)),
        ("WHITE",  (255, 255, 255)),
        ("ROSE",   ( 80,  80, 255)),
        ("GOLD",   (  0, 215, 255)),
        ("ERASE",  (  0,   0,   0)),
    ])

    # UI
    header_h:       int   = 92
    tile_gap:       int   = 7
    accent:         tuple = (255, 255, 255)
    show_confidence: bool = False
    debug_mode:     bool  = False


CFG = Config()


# ════════════════════════════════════════════════════════
#  MEDIAPIPE
# ════════════════════════════════════════════════════════
_mph  = mp.solutions.hands
_mpd  = mp.solutions.drawing_utils
_mpds = mp.solutions.drawing_styles
TIP   = [4, 8, 12, 16, 20]


class HandDetector:
    def __init__(self):
        try:
            self._h = _mph.Hands(
                static_image_mode=False,
                max_num_hands=CFG.max_hands,
                min_detection_confidence=CFG.detection_conf,
                min_tracking_confidence=CFG.tracking_conf,
            )
            logger.info("Hand detector ready")
        except Exception as e:
            logger.error(f"Hand detector init failed: {e}")
            raise

    def process(self, bgr: np.ndarray) -> Tuple[list, float]:
        """Return (landmarks, confidence). Draws skeleton on bgr."""
        try:
            res = self._h.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks and res.multi_handedness:
                _mpd.draw_landmarks(
                    bgr, res.multi_hand_landmarks[0],
                    _mph.HAND_CONNECTIONS,
                    _mpds.get_default_hand_landmarks_style(),
                    _mpds.get_default_hand_connections_style(),
                )
                h, w = bgr.shape[:2]
                conf = res.multi_handedness[0].classification[0].score
                lm   = [
                    [i, int(lm.x * w), int(lm.y * h)]
                    for i, lm in enumerate(res.multi_hand_landmarks[0].landmark)
                ]
                return lm, conf
            return [], 0.0
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return [], 0.0

    @staticmethod
    def fingers_up(lm: list) -> list:
        if len(lm) < 21:
            return []
        f = [1 if lm[4][1] > lm[3][1] else 0]
        for i in range(1, 5):
            f.append(1 if lm[TIP[i]][2] < lm[TIP[i] - 2][2] else 0)
        return f


# ════════════════════════════════════════════════════════
#  GESTURE ENGINE  (debounced)
# ════════════════════════════════════════════════════════
_GESTURES = [
    ("screenshot", lambda f: f[1] == 1 and f[2] == 1 and f[3] == 1 and f[4] == 1),
    ("select",     lambda f: f[1] == 1 and f[2] == 1 and f[3] == 0 and f[4] == 0),
    ("draw",       lambda f: f[1] == 1 and f[2] == 0 and f[3] == 0 and f[4] == 0),
    ("clear",      lambda f: sum(f) == 0),
]


class GestureEngine:
    def __init__(self):
        self._counts = {name: 0 for name, _ in _GESTURES}
        self.active: Optional[str] = None

    def update(self, fingers: list) -> Optional[str]:
        if not fingers:
            self._counts = {k: 0 for k in self._counts}
            self.active = None
            return None

        detected = next((n for n, p in _GESTURES if p(fingers)), None)

        for k in self._counts:
            self._counts[k] = (
                min(self._counts[k] + 1, CFG.gesture_hold + 1)
                if k == detected
                else max(self._counts[k] - 1, 0)
            )

        self.active = (
            detected
            if detected and self._counts[detected] >= CFG.gesture_hold
            else None
        )
        return self.active


# ════════════════════════════════════════════════════════
#  TIP SMOOTHER
# ════════════════════════════════════════════════════════
class TipSmoother:
    def __init__(self):
        self._x: deque = deque(maxlen=CFG.tip_smooth)
        self._y: deque = deque(maxlen=CFG.tip_smooth)

    def update(self, x: int, y: int) -> Tuple[int, int]:
        self._x.append(x); self._y.append(y)
        return int(np.mean(self._x)), int(np.mean(self._y))

    def reset(self):
        self._x.clear(); self._y.clear()


# ════════════════════════════════════════════════════════
#  UI RENDERER  (ASCII only — OpenCV cannot render emoji)
# ════════════════════════════════════════════════════════
F  = cv2.FONT_HERSHEY_SIMPLEX
F2 = cv2.FONT_HERSHEY_DUPLEX

_GESTURE_DISPLAY = {
    "draw":       ("DRAW",     ( 80, 220,  80)),
    "select":     ("SELECT",   ( 80, 220, 220)),
    "clear":      ("CLEAR",    ( 60,  60, 255)),
    "screenshot": ("SNAPSHOT", ( 60, 200, 255)),
}

_LEGEND = [
    ("[1] Index        DRAW",       ( 80, 220,  80)),
    ("[2] Index+Mid    SELECT",     ( 80, 220, 220)),
    ("[0] Fist         CLEAR",      ( 60,  60, 255)),
    ("[4] 4 Fingers    SNAP",       ( 60, 200, 255)),
]

_HELP_LINES = [
    ("--- AIR DRAWING AI  HELP ---", (255, 200, 50)),
    ("",                              (200, 200, 200)),
    ("GESTURES:",                     (180, 180, 220)),
    ("  1 Finger    Draw",            (200, 200, 200)),
    ("  2 Fingers   Select Color",    (200, 200, 200)),
    ("  Fist        Clear Canvas",    (200, 200, 200)),
    ("  4 Fingers   Screenshot",      (200, 200, 200)),
    ("",                              (200, 200, 200)),
    ("KEYBOARD:",                     (180, 180, 220)),
    ("  Q           Quit",            (200, 200, 200)),
    ("  S           Screenshot",      (200, 200, 200)),
    ("  + / -       Brush size",      (200, 200, 200)),
    ("  [ / ]       Detection conf",  (200, 200, 200)),
    ("  H           Toggle Help",     (200, 200, 200)),
]


def _blend(img, x1, y1, x2, y2, bgr, alpha=0.6):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    fill = np.full_like(roi, bgr)
    cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def _put(img, text, x, y, scale=0.55, color=(255, 255, 255), thick=1):
    """Text with black drop-shadow for readability on any background."""
    cv2.putText(img, text, (x + 1, y + 1), F, scale, (0, 0, 0),   thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x,     y),     F, scale, color, thick,            cv2.LINE_AA)


class UIRenderer:
    def __init__(self):
        n            = len(CFG.palette)
        self._n      = n
        self._tile_w = (CFG.cam_w - CFG.tile_gap * (n + 1)) // n

    # ── Palette bar ──────────────────────────────────────────────────
    def draw_palette(self, img: np.ndarray, active: tuple):
        H, g = CFG.header_h, CFG.tile_gap
        _blend(img, 0, 0, CFG.cam_w, H, (12, 12, 18), alpha=0.88)
        cv2.line(img, (0, H), (CFG.cam_w, H), (55, 55, 75), 1)

        for i, (label, col) in enumerate(CFG.palette):
            x1 = g + i * (self._tile_w + g)
            x2 = x1 + self._tile_w
            y1, y2 = g, H - 18

            selected = (col == active)
            _blend(img, x1, y1, x2, y2, col, alpha=0.92)

            if selected:
                cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), CFG.accent, 2)
                for cx_, cy_ in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    cv2.circle(img, (cx_, cy_), 4, CFG.accent, -1)

            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            if label == "ERASE":
                cv2.line(img, (mx - 9, my - 9), (mx + 9, my + 9), (200, 200, 200), 2)
                cv2.line(img, (mx + 9, my - 9), (mx - 9, my + 9), (200, 200, 200), 2)
            else:
                cv2.circle(img, (mx, my), 5, (255, 255, 255), -1)

            (lw, _), _ = cv2.getTextSize(label, F, 0.35, 1)
            lx = x1 + (self._tile_w - lw) // 2
            cv2.putText(img, label, (lx, H - 4), F, 0.35,
                        (255, 255, 255) if selected else (160, 160, 200),
                        1, cv2.LINE_AA)

    # ── HUD overlays ────────────────────────────────────────────────
    def draw_hud(self, img: np.ndarray, gesture: Optional[str],
                 brush: int, active: tuple, fps: float, confidence: float = 0.0):
        W, H, hh = CFG.cam_w, CFG.cam_h, CFG.header_h

        # Bottom strip
        _blend(img, 0, H - 30, W, H, (10, 10, 16), alpha=0.88)
        _put(img,
             "q=quit  s=snap  h=help  +=bigger  -=smaller  [/]=confidence  |  1=draw  2=select  fist=clear  4fingers=snap",
             14, H - 10, 0.35, (130, 130, 170))

        # FPS (+ optional confidence)
        fps_label = f"FPS {int(fps)}"
        if CFG.show_confidence:
            fps_label += f"  |  Conf {confidence:.2f}"
        _put(img, fps_label, 14, hh + 24, 0.55, (80, 220, 180))

        # ── Brush size panel ─────────────────────────────────────────
        px, py     = 14, hh + 40
        panel_w    = 160
        panel_h    = 70
        _blend(img, px, py, px + panel_w, py + panel_h, (12, 12, 20), alpha=0.75)
        cv2.rectangle(img, (px, py), (px + panel_w, py + panel_h), (45, 45, 65), 1)

        _put(img, "SIZE  +/-", px + 8, py + 18, 0.42, (160, 160, 200))

        # Track
        sx1, sx2 = px + 10, px + panel_w - 10
        sy        = py + 38
        cv2.line(img, (sx1, sy), (sx2, sy), (55, 55, 75), 3)

        # Fill + thumb
        ratio  = (brush - CFG.min_brush) / max(CFG.max_brush - CFG.min_brush, 1)
        fill_x = int(sx1 + ratio * (sx2 - sx1))
        d_col  = active if active != (0, 0, 0) else (150, 150, 150)
        cv2.line(img, (sx1, sy), (fill_x, sy), d_col, 3)
        cv2.circle(img, (fill_x, sy), 7, d_col, -1)
        cv2.circle(img, (fill_x, sy), 7, (200, 200, 200), 1)

        # Brush dot preview
        r = max(brush // 2, 2)
        cv2.circle(img, (px + 20, py + 57), r, d_col, -1)
        cv2.circle(img, (px + 20, py + 57), r, (50, 50, 50), 1)
        _put(img, f"{brush}px", px + 20 + r + 6, py + 62, 0.44, (190, 190, 190))

        # ── Gesture badge (centre) ───────────────────────────────────
        if gesture and gesture in _GESTURE_DISPLAY:
            label, g_col = _GESTURE_DISPLAY[gesture]
            (tw, th), _  = cv2.getTextSize(label, F2, 0.95, 2)
            gx, gy, pad  = (W - tw) // 2, hh + 52, 16
            _blend(img, gx - pad, gy - th - pad, gx + tw + pad, gy + pad // 2,
                   g_col, alpha=0.28)
            cv2.rectangle(img,
                          (gx - pad, gy - th - pad),
                          (gx + tw + pad, gy + pad // 2), g_col, 1)
            cv2.putText(img, label, (gx + 1, gy + 1), F2, 0.95, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(img, label, (gx,     gy),     F2, 0.95, g_col,   2, cv2.LINE_AA)

        # ── Right legend panel ───────────────────────────────────────
        lx = W - 248
        ph = len(_LEGEND) * 26 + 16
        _blend(img, lx - 12, hh + 8, W - 10, hh + 8 + ph, (12, 12, 20), alpha=0.75)
        cv2.rectangle(img, (lx - 12, hh + 8), (W - 10, hh + 8 + ph), (45, 45, 65), 1)
        for j, (txt, col) in enumerate(_LEGEND):
            _put(img, txt, lx, hh + 30 + j * 26, 0.44, col)

    # ── Help overlay ─────────────────────────────────────────────────
    def draw_help(self, img: np.ndarray):
        line_h   = 22
        panel_w  = 340
        panel_h  = len(_HELP_LINES) * line_h + 24
        px       = (CFG.cam_w - panel_w) // 2
        py       = (CFG.cam_h - panel_h) // 2
        _blend(img, px - 14, py - 14, px + panel_w + 14, py + panel_h + 14,
               (18, 18, 28), alpha=0.88)
        cv2.rectangle(img,
                      (px - 14, py - 14),
                      (px + panel_w + 14, py + panel_h + 14),
                      (90, 90, 140), 2)
        for i, (line, col) in enumerate(_HELP_LINES):
            if line:
                _put(img, line, px, py + 18 + i * line_h, 0.44, col)

    # ── Toast banner ─────────────────────────────────────────────────
    def draw_toast(self, img: np.ndarray, text: str, color: tuple):
        scale       = 1.2
        (tw, th), _ = cv2.getTextSize(text, F2, scale, 3)
        cx, cy, pad = (CFG.cam_w - tw) // 2, CFG.cam_h // 2, 20
        _blend(img, cx - pad, cy - th - pad, cx + tw + pad, cy + pad,
               (8, 8, 12), alpha=0.82)
        cv2.rectangle(img, (cx - pad, cy - th - pad), (cx + tw + pad, cy + pad), color, 2)
        cv2.putText(img, text, (cx + 1, cy + 1), F2, scale, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(img, text, (cx,     cy),     F2, scale, color,   3, cv2.LINE_AA)

    def hit_palette(self, x: int, y: int) -> Optional[tuple]:
        if not (0 < y < CFG.header_h):
            return None
        g = CFG.tile_gap
        for i, (_, col) in enumerate(CFG.palette):
            x1 = g + i * (self._tile_w + g)
            if x1 <= x <= x1 + self._tile_w:
                return col
        return None


# ════════════════════════════════════════════════════════
#  TOAST
# ════════════════════════════════════════════════════════
class Toast:
    def __init__(self):
        self._msg   = ""
        self._color = (255, 255, 255)
        self._exp   = 0.0

    def show(self, msg: str, color=(255, 255, 255), dur=1.2):
        self._msg, self._color, self._exp = msg, color, time.time() + dur
        logger.info(f"Toast: {msg}")

    def render(self, img: np.ndarray, ui: UIRenderer):
        if self._msg and time.time() < self._exp:
            ui.draw_toast(img, self._msg, self._color)


# ════════════════════════════════════════════════════════
#  MAIN APP  (context manager)
# ════════════════════════════════════════════════════════
class AirDrawingApp:
    def __init__(self):
        self.cap      = None
        self.detector = None
        self.gestures = None
        self.smoother = None
        self.ui       = None
        self.toast    = None
        self.canvas   = None
        self.color    = None
        self.brush    = None

        self._xp = self._yp = 0
        self._drawing      = False
        self._last_shot    = 0.0
        self._ptime        = time.time()
        self._frame_count  = 0
        self._show_help    = False
        self._confidence   = 0.0

        self._init()

    def _init(self):
        try:
            self.cap = cv2.VideoCapture(CFG.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {CFG.camera_id}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.cam_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.cam_h)
            logger.info(f"Camera {CFG.camera_id} opened  ({CFG.cam_w}x{CFG.cam_h})")

            self.detector = HandDetector()
            self.gestures = GestureEngine()
            self.smoother = TipSmoother()
            self.ui       = UIRenderer()
            self.toast    = Toast()

            self.canvas = np.zeros((CFG.cam_h, CFG.cam_w, 3), np.uint8)
            self.color  = CFG.palette[0][1]
            self.brush  = CFG.default_brush

            Path(CFG.shot_dir).mkdir(exist_ok=True)
            logger.info("All components initialised")
        except Exception as e:
            logger.error(f"Init failed: {e}")
            self._cleanup()
            raise

    def _cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
        if exc_type:
            logger.error(f"Exception on exit: {exc_val}")
        return False

    # ── helpers ─────────────────────────────────────────────────────
    def _merge(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_or(cv2.bitwise_and(frame, inv), self.canvas)

    def _end_stroke(self):
        self._drawing = False
        self._xp = self._yp = 0
        self.smoother.reset()

    def _screenshot(self, frame: np.ndarray):
        now = time.time()
        if now - self._last_shot < CFG.shot_cooldown:
            return
        ts = int(now)
        cv2.imwrite(f"{CFG.shot_dir}/canvas_{ts}.png",   self.canvas)
        cv2.imwrite(f"{CFG.shot_dir}/combined_{ts}.png", self._merge(frame))
        self._last_shot = now
        self.toast.show("SNAPSHOT SAVED", (60, 210, 255), 1.4)
        logger.info(f"Saved to {CFG.shot_dir}/combined_{ts}.png")

    def _set_confidence(self, delta: float):
        new = round(max(0.3, min(1.0, CFG.detection_conf + delta)), 2)
        if new != CFG.detection_conf:
            CFG.detection_conf = new
            self.detector = HandDetector()   # re-init with new threshold
            self.toast.show(f"Conf: {new:.2f}", (100, 200, 255), 1.0)
            logger.info(f"Detection confidence -> {new}")

    # ── main loop ────────────────────────────────────────────────────
    def run(self):
        logger.info("Running — press H for help, Q to quit")
        target_dt = 1.0 / CFG.fps_limit

        try:
            while True:
                t0 = time.time()

                ok, frame = self.cap.read()
                if not ok:
                    logger.warning("Frame read failed")
                    break
                frame = cv2.flip(frame, 1)

                lm, conf  = self.detector.process(frame)
                self._confidence = conf
                fingers   = HandDetector.fingers_up(lm)
                g         = self.gestures.update(fingers)

                if lm:
                    x1, y1 = self.smoother.update(lm[8][1], lm[8][2])

                    if g == "clear":
                        self._end_stroke()
                        self.canvas = np.zeros_like(self.canvas)
                        self.toast.show("CANVAS CLEARED", (60, 60, 255))

                    elif g == "screenshot":
                        self._end_stroke()
                        self._screenshot(frame)

                    elif g == "select":
                        self._end_stroke()
                        cv2.circle(frame, (x1, y1), 18, (255, 255, 255), 2)
                        cv2.circle(frame, (x1, y1),  4, (255, 255, 255), -1)
                        hit = self.ui.hit_palette(x1, y1)
                        if hit is not None:
                            self.color = hit
                            logger.debug("Color changed")

                    elif g == "draw":
                        if not self._drawing:
                            self._drawing = True
                            self._xp, self._yp = x1, y1

                        thick = CFG.eraser_size if self.color == (0, 0, 0) else self.brush
                        if self._xp and self._yp:
                            cv2.line(frame,       (self._xp, self._yp), (x1, y1), self.color, thick)
                            cv2.line(self.canvas, (self._xp, self._yp), (x1, y1), self.color, thick)
                        tip_col = self.color if self.color != (0, 0, 0) else (140, 140, 140)
                        cv2.circle(frame, (x1, y1), thick // 2, tip_col, -1)
                        self._xp, self._yp = x1, y1

                    else:
                        self._end_stroke()
                else:
                    self._end_stroke()

                # Composite + UI
                frame = self._merge(frame)

                now   = time.time()
                fps   = 1.0 / max(now - self._ptime, 1e-5)
                self._ptime = now

                self.ui.draw_palette(frame, self.color)
                self.ui.draw_hud(frame, g, self.brush, self.color, fps, self._confidence)
                self.toast.render(frame, self.ui)
                if self._show_help:
                    self.ui.draw_help(frame)

                cv2.imshow("Air Drawing AI", frame)

                # Keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit")
                    break
                elif key == ord('s'):
                    self._screenshot(frame)
                elif key == ord('h'):
                    self._show_help = not self._show_help
                elif key in (ord('+'), ord('=')):
                    self.brush = min(self.brush + CFG.brush_step, CFG.max_brush)
                    logger.debug(f"Brush -> {self.brush}")
                elif key == ord('-'):
                    self.brush = max(self.brush - CFG.brush_step, CFG.min_brush)
                    logger.debug(f"Brush -> {self.brush}")
                elif key == ord('['):
                    self._set_confidence(-0.05)
                elif key == ord(']'):
                    self._set_confidence(+0.05)

                # Frame-rate cap
                elapsed = time.time() - t0
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

                self._frame_count += 1

        except KeyboardInterrupt:
            logger.info("Interrupted")
        except Exception as e:
            logger.error(f"Runtime error: {e}", exc_info=True)
        finally:
            logger.info(f"Closed after {self._frame_count} frames")


# ════════════════════════════════════════════════════════
#  ENTRY
# ════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Air Drawing AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python air_drawing.py
  python air_drawing.py --camera 1 --width 1280 --height 720
  python air_drawing.py --fps-limit 60 --debug
  python air_drawing.py --detection-conf 0.8 --show-confidence
        """,
    )
    parser.add_argument("--camera",           type=int,   default=0,    help="Camera ID (default 0)")
    parser.add_argument("--width",            type=int,   default=1280, help="Frame width")
    parser.add_argument("--height",           type=int,   default=720,  help="Frame height")
    parser.add_argument("--fps-limit",        type=int,   default=30,   help="FPS cap")
    parser.add_argument("--detection-conf",   type=float, default=0.75, help="Detection confidence 0-1")
    parser.add_argument("--tracking-conf",    type=float, default=0.75, help="Tracking confidence 0-1")
    parser.add_argument("--debug",            action="store_true",      help="Verbose logging")
    parser.add_argument("--show-confidence",  action="store_true",      help="Show confidence in HUD")

    args = parser.parse_args()

    CFG.camera_id       = args.camera
    CFG.cam_w           = args.width
    CFG.cam_h           = args.height
    CFG.fps_limit       = args.fps_limit
    CFG.detection_conf  = args.detection_conf
    CFG.tracking_conf   = args.tracking_conf
    CFG.debug_mode      = args.debug
    CFG.show_confidence = args.show_confidence

    if CFG.debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode on")

    print(__doc__)

    try:
        with AirDrawingApp() as app:
            app.run()
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
