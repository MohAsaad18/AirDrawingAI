"""
AIR DRAWING AI  -  Modern Edition
===================================
GESTURE GUIDE
  [1 finger]   Index only        ->  Draw
  [2 fingers]  Index + Middle    ->  Select color
  [0 fingers]  Fist              ->  Clear canvas
  [5 fingers]  All fingers up    ->  Screenshot

KEYS:  q = quit  |  s = screenshot
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


# ════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════
@dataclass
class Config:
    cam_w: int = 1280
    cam_h: int = 720

    max_hands: int = 1
    detection_conf: float = 0.75
    tracking_conf: float  = 0.75

    default_brush: int = 15
    min_brush:     int = 3
    max_brush:     int = 60
    brush_step:    int = 2
    eraser_size:   int = 60
    tip_smooth:    int = 6

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

    header_h: int   = 92
    tile_gap:  int   = 7
    accent:    tuple = (255, 255, 255)


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
        self._h = _mph.Hands(
            static_image_mode=False,
            max_num_hands=CFG.max_hands,
            min_detection_confidence=CFG.detection_conf,
            min_tracking_confidence=CFG.tracking_conf,
        )

    def process(self, bgr: np.ndarray) -> list:
        res = self._h.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            _mpd.draw_landmarks(
                bgr, res.multi_hand_landmarks[0],
                _mph.HAND_CONNECTIONS,
                _mpds.get_default_hand_landmarks_style(),
                _mpds.get_default_hand_connections_style(),
            )
            h, w = bgr.shape[:2]
            return [
                [i, int(lm.x * w), int(lm.y * h)]
                for i, lm in enumerate(res.multi_hand_landmarks[0].landmark)
            ]
        return []

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
    ("draw",       lambda f: f[1] == 1 and f[2] == 0 and f[3] == 0 and f[4] == 0),
    ("select",     lambda f: f[1] == 1 and f[2] == 1 and f[3] == 0 and f[4] == 0),
    ("clear",      lambda f: sum(f) == 0),
    ("screenshot", lambda f: sum(f) == 5),
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

    def update(self, x: int, y: int) -> tuple:
        self._x.append(x); self._y.append(y)
        return int(np.mean(self._x)), int(np.mean(self._y))

    def reset(self):
        self._x.clear(); self._y.clear()


# ════════════════════════════════════════════════════════
#  UI RENDERER  — ASCII only (OpenCV cannot render emoji)
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
    ("[1] Index         DRAW",    ( 80, 220,  80)),
    ("[2] Index+Mid     SELECT",  ( 80, 220, 220)),
    ("[0] Fist          CLEAR",   ( 60,  60, 255)),
    ("[5] All up        SNAP",    ( 60, 200, 255)),
]


def _blend(img, x1, y1, x2, y2, bgr, alpha=0.6):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    fill = np.full_like(roi, bgr)
    cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def _put(img, text, x, y, scale=0.55, color=(255, 255, 255), thick=1):
    """Text with black drop-shadow."""
    cv2.putText(img, text, (x + 1, y + 1), F, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),          F, scale, color,     thick,     cv2.LINE_AA)


class UIRenderer:
    def __init__(self):
        n = len(CFG.palette)
        self._n      = n
        self._tile_w = (CFG.cam_w - CFG.tile_gap * (n + 1)) // n

    def draw_palette(self, img: np.ndarray, active: tuple):
        H = CFG.header_h
        g = CFG.tile_gap

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

    def draw_hud(self, img: np.ndarray, gesture: Optional[str],
                 brush: int, active: tuple, fps: float):
        W, H, hh = CFG.cam_w, CFG.cam_h, CFG.header_h

        # Bottom strip
        _blend(img, 0, H - 30, W, H, (10, 10, 16), alpha=0.88)
        _put(img,
             "q=quit  s=snap  +=bigger  -=smaller  |  1-finger=draw  2-finger=select  fist=clear  5-finger=snap",
             14, H - 10, 0.38, (130, 130, 170))

        # FPS
        _put(img, f"FPS {int(fps)}", 14, hh + 24, 0.55, (80, 220, 180))

        # ── Brush size panel (left side) ─────────────────────────
        px, py = 14, hh + 40
        panel_w, panel_h = 130, 70
        _blend(img, px, py, px + panel_w, py + panel_h, (12, 12, 20), alpha=0.75)
        cv2.rectangle(img, (px, py), (px + panel_w, py + panel_h), (45, 45, 65), 1)

        _put(img, "SIZE  +/-", px + 8, py + 18, 0.42, (160, 160, 200))

        # Slider track
        sx1, sx2 = px + 10, px + panel_w - 10
        sy = py + 38
        cv2.line(img, (sx1, sy), (sx2, sy), (55, 55, 75), 3)

        # Slider fill
        ratio   = (brush - CFG.min_brush) / max(CFG.max_brush - CFG.min_brush, 1)
        fill_x  = int(sx1 + ratio * (sx2 - sx1))
        d_col   = active if active != (0, 0, 0) else (150, 150, 150)
        cv2.line(img, (sx1, sy), (fill_x, sy), d_col, 3)

        # Slider thumb
        cv2.circle(img, (fill_x, sy), 7, d_col, -1)
        cv2.circle(img, (fill_x, sy), 7, (200, 200, 200), 1)

        # Brush dot preview + px label
        r = max(brush // 2, 2)
        cv2.circle(img, (px + 20, py + 57), r, d_col, -1)
        cv2.circle(img, (px + 20, py + 57), r, (50, 50, 50), 1)
        _put(img, f"{brush}px", px + 20 + r + 6, py + 62, 0.44, (190, 190, 190))

        # Gesture badge (centred)
        if gesture and gesture in _GESTURE_DISPLAY:
            label, g_col = _GESTURE_DISPLAY[gesture]
            (tw, th), _ = cv2.getTextSize(label, F2, 0.95, 2)
            gx  = (W - tw) // 2
            gy  = hh + 52
            pad = 16
            _blend(img, gx - pad, gy - th - pad, gx + tw + pad, gy + pad // 2,
                   g_col, alpha=0.28)
            cv2.rectangle(img,
                          (gx - pad, gy - th - pad),
                          (gx + tw + pad, gy + pad // 2),
                          g_col, 1)
            cv2.putText(img, label, (gx + 1, gy + 1), F2, 0.95, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(img, label, (gx,     gy),     F2, 0.95, g_col,   2, cv2.LINE_AA)

        # Right legend panel
        lx   = W - 248
        ph   = len(_LEGEND) * 26 + 16
        _blend(img, lx - 12, hh + 8, W - 10, hh + 8 + ph, (12, 12, 20), alpha=0.75)
        cv2.rectangle(img, (lx - 12, hh + 8), (W - 10, hh + 8 + ph), (45, 45, 65), 1)
        for j, (txt, col) in enumerate(_LEGEND):
            _put(img, txt, lx, hh + 30 + j * 26, 0.44, col)

    def draw_toast(self, img: np.ndarray, text: str, color: tuple):
        scale = 1.2
        (tw, th), _ = cv2.getTextSize(text, F2, scale, 3)
        cx  = (CFG.cam_w - tw) // 2
        cy  = CFG.cam_h // 2
        pad = 20
        _blend(img, cx - pad, cy - th - pad, cx + tw + pad, cy + pad,
               (8, 8, 12), alpha=0.82)
        cv2.rectangle(img,
                      (cx - pad, cy - th - pad),
                      (cx + tw + pad, cy + pad),
                      color, 2)
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

    def render(self, img: np.ndarray, ui: UIRenderer):
        if self._msg and time.time() < self._exp:
            ui.draw_toast(img, self._msg, self._color)


# ════════════════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════════════════
class AirDrawingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.cam_h)

        self.detector = HandDetector()
        self.gestures = GestureEngine()
        self.smoother = TipSmoother()
        self.ui       = UIRenderer()
        self.toast    = Toast()

        self.canvas = np.zeros((CFG.cam_h, CFG.cam_w, 3), np.uint8)
        self.color  = CFG.palette[0][1]
        self.brush  = CFG.default_brush

        self._xp = self._yp = 0
        self._drawing   = False
        self._last_shot = 0.0
        self._ptime     = time.time()

        os.makedirs(CFG.shot_dir, exist_ok=True)
        print(__doc__)

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
        print(f"  Saved: {CFG.shot_dir}/combined_{ts}.png")

    def run(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            lm      = self.detector.process(frame)
            fingers = HandDetector.fingers_up(lm)
            g       = self.gestures.update(fingers)

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

            frame = self._merge(frame)

            now  = time.time()
            fps  = 1.0 / max(now - self._ptime, 1e-5)
            self._ptime = now

            self.ui.draw_palette(frame, self.color)
            self.ui.draw_hud(frame, g, self.brush, self.color, fps)
            self.toast.render(frame, self.ui)

            cv2.imshow("Air Drawing AI", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._screenshot(frame)
            elif key in (ord('+'), ord('=')):
                self.brush = min(self.brush + CFG.brush_step, CFG.max_brush)
            elif key == ord('-'):
                self.brush = max(self.brush - CFG.brush_step, CFG.min_brush)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    AirDrawingApp().run()