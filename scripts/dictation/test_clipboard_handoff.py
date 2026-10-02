#!/usr/bin/env python3
"""Exercise clipboard ownership on a private X server, without desktop input."""

from __future__ import annotations

import importlib.util
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dictation import Dictation
from Xlib import X, XK, display
from Xlib.ext import xfixes
from Xlib.ext import xtest
from x11_paste import capture_focus, paste_if_focused


@unittest.skipUnless(
    shutil.which("Xvfb") and shutil.which("xclip"), "requires Xvfb and xclip"
)
class ClipboardHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.log = tempfile.TemporaryFile()
        read_fd, write_fd = os.pipe()
        try:
            cls.server = subprocess.Popen(
                ["Xvfb", "-displayfd", str(write_fd), "-nolisten", "tcp"],
                pass_fds=(write_fd,),
                stdout=cls.log,
                stderr=cls.log,
            )
        finally:
            os.close(write_fd)
        try:
            if not select.select([read_fd], [], [], 5)[0]:
                cls.server.terminate()
                cls.server.wait(timeout=3)
                raise RuntimeError("private X server did not start")
            number = os.read(read_fd, 32).decode().strip()
        finally:
            os.close(read_fd)
        if not number.isdigit():
            raise RuntimeError("private X server did not return a display number")
        cls.environment = mock.patch.dict(os.environ, {"DISPLAY": f":{number}"})
        cls.environment.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.environment.stop()
        cls.server.terminate()
        cls.server.wait(timeout=3)
        cls.log.close()

    def setUp(self) -> None:
        self.app = object.__new__(Dictation)
        self.app._clipboard_lock = threading.Lock()
        self.app._clipboard_proc = None
        self.app._clipboard_closed = threading.Event()
        self.app._interaction_serial = 0
        self.app._session_targets = {}
        self.app._delivery_attempted = set()
        self.app._delivery_lock = threading.Lock()
        self.app._last_delivery_reason = ""
        self.app._notify = mock.Mock()
        self.connection = display.Display()
        self.clipboard = self.connection.intern_atom("CLIPBOARD")
        self.connection.xfixes_query_version()
        self.connection.xfixes_select_selection_input(
            self.connection.screen().root,
            self.clipboard,
            xfixes.XFixesSetSelectionOwnerNotifyMask
            | xfixes.XFixesSelectionWindowDestroyNotifyMask
            | xfixes.XFixesSelectionClientCloseNotifyMask,
        )
        self.connection.sync()

    def tearDown(self) -> None:
        self.app._stop_clipboard_owner()
        self.connection.close()

    def clipboard_text(self) -> bytes:
        return subprocess.run(
            ["xclip", "-selection", "clipboard", "-out"],
            capture_output=True,
            check=True,
            timeout=2,
        ).stdout

    def owners(self) -> list[int]:
        self.connection.sync()
        owners = []
        while self.connection.pending_events():
            event = self.connection.next_event()
            if isinstance(event, xfixes.SelectionNotify):
                owners.append(getattr(event.owner, "id", event.owner))
        return owners

    def input_window(self):
        window = self.connection.screen().root.create_window(
            0,
            0,
            200,
            100,
            0,
            self.connection.screen().root_depth,
            event_mask=X.KeyPressMask | X.KeyReleaseMask,
        )
        window.map()
        window.set_input_focus(X.RevertToParent, X.CurrentTime)
        self.connection.sync()
        return window

    def test_paste_does_not_leave_keys_down_when_old_helper_is_slow(self) -> None:
        self.input_window()
        real_run = subprocess.run

        def slow_key(command, **kwargs):
            if command[:2] == ["xdotool", "key"]:
                command = command[:2] + ["--delay", "400"] + command[2:]
            return real_run(command, **kwargs)

        try:
            with mock.patch("dictation.subprocess.run", side_effect=slow_key):
                delivered = self.app._insert("One complete paste.")
            keys = self.connection.query_keymap()
            held = [
                name
                for name in ("v", "Control_L", "Control_R")
                if (lambda code: keys[code // 8] & (1 << (code % 8)))(
                    self.connection.keysym_to_keycode(XK.string_to_keysym(name))
                )
            ]
            self.assertEqual(held, [], "paste must not leave autorepeating keys down")
            self.assertTrue(delivered)
        finally:
            real_run(["xdotool", "keyup", "v", "Control_L", "Control_R"], check=True)

    def test_changed_window_keeps_text_without_pasting(self) -> None:
        original = self.input_window()
        self.app._session_targets[1] = ((0, original.id), 0)
        other = self.input_window()
        self.app._store = mock.Mock()
        self.app._store.completed_text.return_value = "Private dictated text"
        self.app._deliver_completed_session(Path("/tmp/saved-session"), 1)
        self.assertEqual(self.clipboard_text(), b"Private dictated text")
        self.connection.sync()
        presses = []
        while self.connection.pending_events():
            event = self.connection.next_event()
            if event.type == X.KeyPress and event.window.id == other.id:
                presses.append(event.detail)
        self.assertEqual(presses, [], "changed focus must never receive dictation")

    def test_duplicate_completion_does_not_dispatch_twice(self) -> None:
        self.app._store = mock.Mock()
        self.app._store.completed_text.return_value = "One message"
        self.app._insert = mock.Mock(return_value=True)
        session = Path("/tmp/saved-session")
        self.app._deliver_completed_session(session, 1)
        self.app._deliver_completed_session(session, 1)
        self.assertEqual(self.app._insert.call_count, 1)

    def test_click_within_same_window_defers_paste(self) -> None:
        window = self.input_window()
        self.app._session_targets[1] = ((0, window.id), 0)
        self.app.on_click(20, 20, None, True)
        self.app._store = mock.Mock()
        self.app._store.completed_text.return_value = "Saved text"
        self.app._deliver_completed_session(Path("/tmp/saved-session"), 1)
        self.assertEqual(self.clipboard_text(), b"Saved text")
        self.assertEqual(self.app._last_delivery_reason, "focus-changed")
        self.connection.sync()
        while self.connection.pending_events():
            self.assertNotEqual(self.connection.next_event().type, X.KeyPress)

    def test_waits_for_hotkey_release_without_restoring_modifiers(self) -> None:
        self.input_window()
        target = capture_focus()
        ctrl = self.connection.keysym_to_keycode(XK.string_to_keysym("Control_L"))
        space = self.connection.keysym_to_keycode(XK.string_to_keysym("space"))
        xtest.fake_input(self.connection, X.KeyPress, ctrl)
        xtest.fake_input(self.connection, X.KeyPress, space)
        self.connection.sync()
        outcome = []
        worker = threading.Thread(
            target=lambda: outcome.append(paste_if_focused(target, lambda: True))
        )
        worker.start()
        time.sleep(0.1)
        self.assertEqual(outcome, [])
        xtest.fake_input(self.connection, X.KeyRelease, space)
        xtest.fake_input(self.connection, X.KeyRelease, ctrl)
        self.connection.sync()
        worker.join(2)
        self.assertEqual(outcome, ["pasted"])
        self.assertFalse(any(self.connection.query_keymap()))

    def test_busy_keyboard_is_not_cleared_or_pasted_through(self) -> None:
        self.input_window()
        ctrl = self.connection.keysym_to_keycode(XK.string_to_keysym("Control_L"))
        xtest.fake_input(self.connection, X.KeyPress, ctrl)
        self.connection.sync()
        try:
            result = paste_if_focused(
                capture_focus(), lambda: True, release_timeout=0.05
            )
            self.assertEqual(result, "input-active")
            keys = self.connection.query_keymap()
            self.assertTrue(keys[ctrl // 8] & (1 << (ctrl % 8)))
        finally:
            xtest.fake_input(self.connection, X.KeyRelease, ctrl)
            self.connection.sync()

    def test_replacement_never_leaves_clipboard_unowned(self) -> None:
        self.assertTrue(self.app._start_clipboard_owner("previous transcript"))
        self.owners()
        for index in range(20):
            text = f"Transcript {index}: café, 日本語, and a long message. " * 100
            self.assertTrue(self.app._start_clipboard_owner(text))
            self.assertEqual(self.clipboard_text(), text.encode())
            self.assertNotIn(
                0,
                self.owners(),
                "an empty clipboard lets a history manager restore stale text",
            )

    def test_failed_replacement_preserves_previous_owner(self) -> None:
        self.assertTrue(self.app._start_clipboard_owner("previous transcript"))
        previous = self.app._clipboard_proc
        with mock.patch("dictation.subprocess.Popen", side_effect=OSError("test")):
            self.assertFalse(self.app._start_clipboard_owner("new transcript"))
        self.assertIsNone(previous.poll())
        self.assertIs(self.app._clipboard_proc, previous)
        self.assertEqual(self.clipboard_text(), b"previous transcript")
        self.assertNotIn(0, self.owners())

    @unittest.skipUnless(
        shutil.which("xdotool") and importlib.util.find_spec("gi"),
        "requires xdotool and GTK Python bindings",
    )
    def test_complete_messages_paste_once_into_text_field(self) -> None:
        # GTK runs in a separate process so its display connection cannot be
        # inherited from the desktop used when pynput/pystray was imported.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ready = root / "ready"
            result = root / "result"
            done = root / "done"
            sink = """
import sys
from pathlib import Path
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib
root = Path(sys.argv[1])
window = Gtk.Window()
view = Gtk.TextView()
window.add(view)
window.show_all()
view.grab_focus()
window.get_window().focus(Gdk.CURRENT_TIME)
buffer = view.get_buffer()
def snapshot():
    text = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
    (root / 'result').write_text(text)
    if (root / 'done').exists():
        Gtk.main_quit()
        return False
    return True
GLib.timeout_add(10, snapshot)
GLib.timeout_add_seconds(10, lambda: Gtk.main_quit() or False)
(root / 'ready').touch()
Gtk.main()
"""
            with tempfile.TemporaryFile() as log:
                receiver = subprocess.Popen(
                    [sys.executable, "-c", sink, str(root)],
                    stdout=log,
                    stderr=log,
                    env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
                )
                try:
                    deadline = time.monotonic() + 3
                    while not ready.exists() and time.monotonic() < deadline:
                        if receiver.poll() is not None:
                            log.seek(0)
                            self.fail(log.read().decode(errors="replace"))
                        time.sleep(0.01)
                    self.assertTrue(ready.exists(), "text field did not start")
                    expected = ""
                    for text in (
                        "First complete message. ",
                        "Long Unicode dictation: café 日本語. " * 2000,
                        "Final complete message.",
                    ):
                        self.assertTrue(self.app._insert(text))
                        expected += text
                        deadline = time.monotonic() + 3
                        while time.monotonic() < deadline:
                            if result.exists() and result.read_text() == expected:
                                break
                            time.sleep(0.01)
                        self.assertEqual(result.read_text(), expected)
                    done.touch()
                    receiver.wait(timeout=3)
                    self.assertEqual(result.read_text(), expected)
                finally:
                    if receiver.poll() is None:
                        receiver.terminate()
                        receiver.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
