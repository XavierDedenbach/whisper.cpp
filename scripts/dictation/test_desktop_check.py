"""Read-only desktop checks on real isolated X servers."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

CHECK = Path(__file__).with_name("desktop_check.py")


class DesktopCheckTests(unittest.TestCase):
    def check(self, *, overrides=None, python_args=(), server_args=None):
        env = dict(os.environ, XDG_SESSION_TYPE="x11", WAYLAND_DISPLAY="")
        env.update(overrides or {})
        command = [sys.executable, *python_args, str(CHECK)]
        if server_args is not None:
            command = ["xvfb-run", "-a", "-s", server_args, *command]
        return subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=10
        )

    def test_x11_capabilities_available(self):
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("X11", result.stdout)

    def test_missing_display_is_actionable(self):
        result = self.check(overrides={"DISPLAY": ""})
        self.assertEqual(result.returncode, 1)
        self.assertIn("DISPLAY", result.stdout)

    def test_wayland_with_xwayland_is_not_claimed_supported(self):
        result = self.check(overrides={"XDG_SESSION_TYPE": "wayland"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("Wayland", result.stdout)

    def test_wayland_socket_without_session_type_is_not_claimed_supported(self):
        result = self.check(
            overrides={"XDG_SESSION_TYPE": "", "WAYLAND_DISPLAY": "wayland-0"}
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Wayland", result.stdout)

    def test_unreachable_display_is_actionable(self):
        result = self.check(overrides={"DISPLAY": "unix:65530"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("XAUTHORITY", result.stdout)

    def test_missing_dependency_is_actionable(self):
        result = self.check(python_args=("-S",), overrides={"PYTHONPATH": ""})
        self.assertEqual(result.returncode, 1)
        self.assertIn("python-xlib", result.stdout)

    def test_server_without_xtest_is_rejected(self):
        result = self.check(server_args="-screen 0 800x600x24 -extension XTEST")
        self.assertEqual(result.returncode, 1)
        self.assertIn("XTEST", result.stdout)


if __name__ == "__main__":
    unittest.main()
