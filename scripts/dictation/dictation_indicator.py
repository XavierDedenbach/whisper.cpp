"""System tray recording LED for whisper-dictation (GNOME/KDE top panel).

Bold solid dot (classic record LED): bright silver when idle, vivid red when recording.
No dark bezel — tray scales icons small; simple high-contrast circles read clearly.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from PIL import Image

try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    pystray = None
    Image = None  # type: ignore[misc, assignment]
    ImageDraw = None  # type: ignore[misc, assignment]

LedState = Literal["idle", "on", "dim"]

# Tray icons are shown ~16-22 px; draw at 48 px with a large filled circle.
ICON_SIZE = 48
TITLE_IDLE = "Whisper dictation - idle"
TITLE_RECORDING = "Whisper dictation - recording"


def tray_indicator_available() -> bool:
    return pystray is not None and Image is not None


def _draw_led_icon(state: LedState) -> Image.Image:
    """Solid record-style LED dot — high contrast on dark GNOME panels."""
    size = ICON_SIZE
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = 1
    outer = (pad, pad, size - pad - 1, size - pad - 1)
    inner = (pad + 4, pad + 4, size - pad - 5, size - pad - 5)

    if state == "idle":
        # Off LED: light silver, visible like other white tray glyphs
        draw.ellipse(outer, fill=(210, 210, 215, 255))
        draw.ellipse(inner, fill=(175, 175, 182, 255))
    elif state == "on":
        # Lit LED: saturated record red
        draw.ellipse(outer, fill=(255, 42, 38, 255))
        draw.ellipse(inner, fill=(255, 72, 64, 255))
        draw.ellipse((pad + 7, pad + 6, pad + 14, pad + 12), fill=(255, 235, 225, 255))
    else:
        # Blink off-phase: still clearly red
        draw.ellipse(outer, fill=(220, 38, 34, 255))
        draw.ellipse(inner, fill=(240, 58, 50, 255))

    return img


def _build_icon_cache() -> dict[LedState, Image.Image]:
    return {
        "idle": _draw_led_icon("idle"),
        "on": _draw_led_icon("on"),
        "dim": _draw_led_icon("dim"),
    }


_ICON_CACHE: dict[LedState, Image.Image] | None = None


def _icons() -> dict[LedState, Image.Image]:
    global _ICON_CACHE
    if _ICON_CACHE is None:
        _ICON_CACHE = _build_icon_cache()
    return _ICON_CACHE


class TrayIndicator:
    """Top-panel tray LED updated from dictation recording state."""

    def __init__(self, blink_interval_s: float = 1.0) -> None:
        self._blink_interval_s = max(0.5, blink_interval_s)
        self._recording = False
        self._blink_red = True
        self._lock = threading.Lock()
        self._icon: pystray.Icon | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_led_state: LedState | None = None

    def start(self) -> bool:
        if not tray_indicator_available():
            return False
        try:
            icons = _icons()
            self._icon = pystray.Icon(
                "whisper-dictation",
                icons["idle"],
                TITLE_IDLE,
            )
            self._icon.run_detached()
            self._last_led_state = "idle"
        except OSError as exc:
            print(
                f"whisper-dictation: tray indicator failed ({exc})",
                file=sys.stderr,
            )
            self._icon = None
            return False
        self._running = True
        self._thread = threading.Thread(
            target=self._tick_loop,
            name="whisper-dictation-tray",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._icon is not None:
            self._icon.stop()

    def set_recording(self, active: bool) -> None:
        with self._lock:
            self._recording = active
            if active:
                self._blink_red = True
                state: LedState = "on"
                title = TITLE_RECORDING
            else:
                state = "idle"
                title = TITLE_IDLE
        self._apply_led(state, title=title)

    def _apply_led(self, state: LedState, title: str | None = None) -> None:
        if state == self._last_led_state and title is None:
            return
        icon = self._icon
        if icon is None:
            return
        try:
            with self._lock:
                icon.icon = _icons()[state]
                self._last_led_state = state
                if title is not None:
                    icon.title = title
        except OSError as exc:
            print(
                f"whisper-dictation: tray icon update failed ({exc})",
                file=sys.stderr,
            )

    def _tick_loop(self) -> None:
        while self._running:
            time.sleep(self._blink_interval_s)
            with self._lock:
                if not self._recording:
                    continue
                self._blink_red = not self._blink_red
                state: LedState = "on" if self._blink_red else "dim"
            self._apply_led(state)
