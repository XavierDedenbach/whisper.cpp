"""Check desktop capabilities without changing focus, clipboard, or input."""

from __future__ import annotations

import os


def check_desktop() -> tuple[bool, str]:
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" or os.environ.get(
        "WAYLAND_DISPLAY"
    ):
        return (
            False,
            "Wayland is unsupported; log into an X11 session for global dictation.",
        )
    if not os.environ.get("DISPLAY"):
        return (
            False,
            "DISPLAY is missing; run this check from the graphical X11 session.",
        )
    try:
        from Xlib import XK, display
    except ImportError:
        return False, "python-xlib is missing; run scripts/dictation/install.sh."

    connection = None
    try:
        connection = display.Display()
        for extension in ("XTEST", "RECORD"):
            if not connection.has_extension(extension):
                return (
                    False,
                    f"X11 extension {extension} is missing; global dictation is unavailable.",
                )
        for symbol in ("Control_L", "v"):
            if not connection.keysym_to_keycode(XK.string_to_keysym(symbol)):
                return (
                    False,
                    f"X11 keyboard mapping lacks {symbol}; automatic paste is unavailable.",
                )
        connection.query_keymap()
        from pynput import keyboard, mouse  # noqa: F401

        return (
            True,
            "X11 keyboard/mouse hooks and XTEST paste available (read-only check).",
        )
    except Exception as exc:
        return False, (
            f"Cannot use X11 ({type(exc).__name__}); check DISPLAY, XAUTHORITY, "
            "and the pynput installation in the graphical session."
        )
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    available, message = check_desktop()
    print(message)
    raise SystemExit(0 if available else 1)
