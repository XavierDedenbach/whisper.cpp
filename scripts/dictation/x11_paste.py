"""Focus-checked X11 paste without an interruptible key-down/key-up helper."""

from __future__ import annotations

import time
from collections.abc import Callable

from Xlib import X, XK, display
from Xlib.ext import xtest

Focus = tuple[int, int]


def _focus(connection) -> Focus | None:
    focused = connection.get_input_focus().focus
    focused = getattr(focused, "id", focused)
    if focused in (X.NONE, X.PointerRoot):
        return None
    active = connection.screen().root.get_full_property(
        connection.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType
    )
    return (
        int(active.value[0]) if active is not None and len(active.value) else 0,
        int(focused),
    )


def capture_focus() -> Focus | None:
    connection = None
    try:
        connection = display.Display()
        return _focus(connection)
    except Exception:
        return None
    finally:
        if connection is not None:
            connection.close()


def _input_down(connection) -> bool:
    buttons = (
        X.Button1Mask | X.Button2Mask | X.Button3Mask | X.Button4Mask | X.Button5Mask
    )
    return any(connection.query_keymap()) or bool(
        connection.screen().root.query_pointer().mask & buttons
    )


def paste_if_focused(
    target: Focus | None,
    permitted: Callable[[], bool],
    *,
    release_timeout: float = 1.5,
) -> str:
    """Paste once after hotkey release, or leave delivery to the user.

    No modifiers are cleared or restored: restoring a key that the user has
    since released can itself leave Ctrl stuck. Key-down and key-up requests
    are queued together, without sleeps or a subprocess timeout between them.
    """
    if target is None:
        return "unknown-focus"
    connection = None
    try:
        connection = display.Display()
        deadline = time.monotonic() + release_timeout
        quiet_since = None
        while time.monotonic() < deadline:
            if not permitted() or _focus(connection) != target:
                return "focus-changed"
            if _input_down(connection):
                quiet_since = None
            elif quiet_since is None:
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since >= 0.05:
                break
            time.sleep(0.01)
        else:
            return "input-active"

        ctrl = connection.keysym_to_keycode(XK.string_to_keysym("Control_L"))
        key = connection.keysym_to_keycode(XK.string_to_keysym("v"))
        if not ctrl or not key:
            return "missing-keymap"
        # Keep another application's focus request out of the final check and
        # the four queued key events. Never sleep while the server is grabbed.
        connection.grab_server()
        pressed = False
        try:
            if not permitted() or _focus(connection) != target:
                return "focus-changed"
            if _input_down(connection):
                return "input-active"
            pressed = True
            xtest.fake_input(connection, X.KeyPress, ctrl)
            xtest.fake_input(connection, X.KeyPress, key)
        finally:
            if pressed:
                xtest.fake_input(connection, X.KeyRelease, key)
                xtest.fake_input(connection, X.KeyRelease, ctrl)
            connection.ungrab_server()
            connection.sync()
        return "pasted"
    except Exception as exc:
        return f"x11-error:{type(exc).__name__}"
    finally:
        if connection is not None:
            connection.close()
