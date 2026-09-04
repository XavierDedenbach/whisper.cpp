#!/usr/bin/env python3
"""Focused integration tests for dictation runtime/build selection."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from dictation import Dictation  # noqa: E402


def write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


class RuntimeSelectionTests(unittest.TestCase):
    def run_fake_server_check(
        self,
        temp: Path,
        *,
        accelerator: str = "sycl",
        cli_reports_accelerator: bool = True,
        sycl_device: str = "0",
        sycl_device_name: str = "Intel Iris Xe Graphics",
        sycl_expected_device: str = "",
        curl_fails: bool = False,
        server_matches: bool = True,
        malformed_response: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        home = temp / "home"
        config_dir = home / ".config/whisper-dictation"
        config_dir.mkdir(parents=True)
        repo = temp / "repo"
        bin_dir = repo / f"build-{accelerator}/bin"
        bin_dir.mkdir(parents=True)
        (repo / "models").mkdir()
        (repo / "samples").mkdir()
        (repo / "scripts/dictation/.venv/bin").mkdir(parents=True)
        (repo / "models/ggml-small.en.bin").write_bytes(b"model")
        (repo / "samples/jfk.wav").write_bytes(b"wav")
        setvars = temp / "setvars.sh"
        setvars.write_text("export FAKE_ONEAPI_READY=1\n", encoding="utf-8")
        (config_dir / "install.env").write_text(
            f'WHISPER_REPO_ROOT="{repo}"\n', encoding="utf-8"
        )
        (config_dir / "config.env").write_text(
            'WHISPER_MODEL="small.en"\n'
            f'WHISPER_BUILD_DIR="build-{accelerator}"\n'
            f'WHISPER_ACCELERATOR="{accelerator}"\n'
            f'WHISPER_ONEAPI_SETVARS="{setvars}"\n'
            f'WHISPER_SYCL_DEVICE="{sycl_device}"\n'
            f'WHISPER_SYCL_EXPECTED_DEVICE="{sycl_expected_device}"\n'
            'WHISPER_BACKEND="server"\n'
            'WHISPER_SERVER_URL="http://127.0.0.1:18178"\n',
            encoding="utf-8",
        )
        if accelerator == "sycl":
            backend_log = (
                'printf "whisper_backend_init_gpu: using SYCL0 backend\\n" >&2\n'
                f'printf "[level_zero:gpu:{sycl_device}] {sycl_device_name}\\n" >&2\n'
            )
        else:
            backend_log = (
                'printf "ggml_cuda_init: found 1 CUDA devices\\n" >&2\n'
                'printf "whisper_backend_init_gpu: using CUDA0 backend\\n" >&2\n'
            )
        if not cli_reports_accelerator:
            backend_log = ""
        write_executable(
            bin_dir / "whisper-cli",
            "#!/usr/bin/env bash\n" + backend_log + 'printf "transcript\\n"\n',
        )
        write_executable(bin_dir / "whisper-server", "#!/usr/bin/env bash\nexit 0\n")
        write_executable(
            repo / "scripts/dictation/.venv/bin/python",
            "#!/usr/bin/env bash\nexit 0\n",
        )
        fake_bin = temp / "bin"
        for command in ("pw-record", "xdotool", "xclip"):
            write_executable(fake_bin / command, "#!/usr/bin/env bash\nexit 0\n")
        write_executable(
            fake_bin / "sycl-ls",
            "#!/usr/bin/env bash\n"
            f'printf "[level_zero:gpu][level_zero:{sycl_device}] {sycl_device_name}\\n"\n',
        )
        write_executable(
            fake_bin / "systemctl",
            "#!/usr/bin/env bash\n"
            'if [[ "$*" == *"show"* ]]; then printf "1234\\n"; fi\n'
            "exit 0\n",
        )
        write_executable(
            fake_bin / "pgrep",
            "#!/usr/bin/env bash\n"
            'if [[ "$*" == *"-P 1234"* ]]; then printf "4321\\n"; fi\n',
        )
        selected_server = str(bin_dir / "whisper-server")
        reported_server = (
            selected_server if server_matches else str(temp / "cpu/whisper-server")
        )
        write_executable(
            fake_bin / "readlink",
            f'#!/usr/bin/env bash\nprintf "%s\\n" "{reported_server}"\n',
        )
        if curl_fails:
            curl_body = "exit 7\n"
        elif malformed_response:
            curl_body = "printf '{\"text\":'\n"
        else:
            curl_body = 'printf \'{"text":"ok"}\'\n'
        write_executable(fake_bin / "curl", f"#!/usr/bin/env bash\n{curl_body}")
        env = os.environ.copy()
        env.update({"HOME": str(home), "PATH": f"{fake_bin}:/usr/bin:/bin"})
        return subprocess.run(
            ["bash", str(SCRIPT_DIR / "check.sh")],
            text=True,
            capture_output=True,
            env=env,
            timeout=10,
        )

    def run_autostart_install(self, temp: Path) -> tuple[Path, list[str]]:
        home = temp / "home"
        config_dir = home / ".config/whisper-dictation"
        config_dir.mkdir(parents=True)
        (config_dir / "config.env").write_text(
            'WHISPER_BACKEND="server"\n'
            'WHISPER_BUILD_DIR="build-sycl"\n'
            'WHISPER_ACCELERATOR="sycl"\n',
            encoding="utf-8",
        )
        event_log = temp / "systemctl.log"
        fake_bin = temp / "bin"
        write_executable(
            fake_bin / "systemctl",
            '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "${EVENT_LOG}"\n',
        )
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(home),
                "EVENT_LOG": str(event_log),
                "PATH": f"{fake_bin}:/usr/bin:/bin",
                "XDG_RUNTIME_DIR": str(temp / "runtime"),
            }
        )
        result = subprocess.run(
            ["bash", str(SCRIPT_DIR / "install.sh"), "--autostart-only"],
            text=True,
            capture_output=True,
            env=env,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        events = event_log.read_text(encoding="utf-8").splitlines()
        return home, events

    def test_dictation_build_directory_defaults_to_build(self) -> None:
        with mock.patch("dictation.build_recorder_cmd", return_value=["parecord"]):
            app = Dictation({"WHISPER_HOME": str(REPO_ROOT)})
        self.assertEqual(app.cli, REPO_ROOT / "build/bin/whisper-cli")

    def test_dictation_build_directory_override(self) -> None:
        with mock.patch("dictation.build_recorder_cmd", return_value=["parecord"]):
            app = Dictation(
                {
                    "WHISPER_HOME": str(REPO_ROOT),
                    "WHISPER_BUILD_DIR": "build-sycl",
                }
            )
        self.assertEqual(app.cli, REPO_ROOT / "build-sycl/bin/whisper-cli")

    def test_runtime_env_selects_sycl_and_sources_oneapi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            home = temp / "home"
            config_dir = home / ".config/whisper-dictation"
            config_dir.mkdir(parents=True)
            repo = temp / "repo"
            repo.mkdir()
            event_log = temp / "events.log"
            setvars = temp / "setvars.sh"
            setvars.write_text(
                'export FAKE_ONEAPI_READY="1"\nprintf "oneapi\\n" >> "${EVENT_LOG}"\n',
                encoding="utf-8",
            )
            (config_dir / "install.env").write_text(
                f'WHISPER_REPO_ROOT="{repo}"\n', encoding="utf-8"
            )
            (config_dir / "config.env").write_text(
                'WHISPER_BUILD_DIR="build-sycl"\n'
                'WHISPER_ACCELERATOR="sycl"\n'
                f'WHISPER_ONEAPI_SETVARS="{setvars}"\n'
                'WHISPER_ONEAPI_DEVICE_SELECTOR="level_zero:gpu"\n'
                'WHISPER_SYCL_DEVICE="0"\n',
                encoding="utf-8",
            )

            command = (
                f'source "{SCRIPT_DIR / "runtime-env.sh"}"; '
                "whisper_dictation_load_runtime; "
                'printf "bin=%s\\noneapi=%s\\nselector=%s\\ndevice=%s\\n" '
                '"${WHISPER_BIN_DIR}" "${FAKE_ONEAPI_READY:-0}" '
                '"${ONEAPI_DEVICE_SELECTOR:-}" "${GGML_SYCL_DEVICE:-}"'
            )
            env = os.environ.copy()
            env.update({"HOME": str(home), "EVENT_LOG": str(event_log)})
            result = subprocess.run(
                ["bash", "-c", command],
                text=True,
                capture_output=True,
                env=env,
                timeout=5,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"bin={repo / 'build-sycl/bin'}", result.stdout)
            self.assertIn("oneapi=1", result.stdout)
            self.assertIn("selector=level_zero:gpu", result.stdout)
            self.assertIn("device=0", result.stdout)
            self.assertEqual(event_log.read_text(encoding="utf-8"), "oneapi\n")

    def test_runtime_env_auto_selects_cpu_from_build_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            home = temp / "home"
            config_dir = home / ".config/whisper-dictation"
            config_dir.mkdir(parents=True)
            repo = temp / "repo"
            repo.mkdir()
            (config_dir / "install.env").write_text(
                f'WHISPER_REPO_ROOT="{repo}"\n', encoding="utf-8"
            )
            (config_dir / "config.env").write_text(
                'WHISPER_BUILD_DIR="build"\n'
                'WHISPER_ACCELERATOR="auto"\n'
                'WHISPER_ONEAPI_SETVARS="/definitely/missing/setvars.sh"\n',
                encoding="utf-8",
            )
            command = (
                f'source "{SCRIPT_DIR / "runtime-env.sh"}"; '
                "whisper_dictation_load_runtime; "
                'printf "bin=%s\\naccelerator=%s\\n" '
                '"${WHISPER_BIN_DIR}" "${WHISPER_ACCELERATOR}"'
            )
            env = os.environ.copy()
            env["HOME"] = str(home)
            result = subprocess.run(
                ["bash", "-c", command],
                text=True,
                capture_output=True,
                env=env,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"bin={repo / 'build/bin'}", result.stdout)
            self.assertIn("accelerator=\n", result.stdout)

    def test_runtime_env_auto_selects_cuda_from_build_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            home = temp / "home"
            config_dir = home / ".config/whisper-dictation"
            config_dir.mkdir(parents=True)
            repo = temp / "repo"
            (repo / "build").mkdir(parents=True)
            (repo / "build/CMakeCache.txt").write_text(
                "GGML_CUDA:BOOL=ON\n", encoding="utf-8"
            )
            (config_dir / "install.env").write_text(
                f'WHISPER_REPO_ROOT="{repo}"\n', encoding="utf-8"
            )
            (config_dir / "config.env").write_text(
                'WHISPER_BUILD_DIR="build"\n'
                'WHISPER_ACCELERATOR="auto"\n'
                'WHISPER_SYCL_EXPECTED_DEVICE="Wrong device"\n',
                encoding="utf-8",
            )
            command = (
                f'source "{SCRIPT_DIR / "runtime-env.sh"}"; '
                "whisper_dictation_load_runtime; "
                'printf "accelerator=%s\\n" "${WHISPER_ACCELERATOR}"'
            )
            env = os.environ.copy()
            env["HOME"] = str(home)
            result = subprocess.run(
                ["bash", "-c", command],
                text=True,
                capture_output=True,
                env=env,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("accelerator=cuda", result.stdout)

    def test_runtime_env_caller_build_override_wins_over_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            home = temp / "home"
            config_dir = home / ".config/whisper-dictation"
            config_dir.mkdir(parents=True)
            (config_dir / "config.env").write_text(
                'WHISPER_BUILD_DIR="build"\nWHISPER_ACCELERATOR="auto"\n',
                encoding="utf-8",
            )
            setvars = temp / "setvars.sh"
            setvars.write_text("export OVERRIDE_ONEAPI=1\n", encoding="utf-8")
            custom_build = temp / "custom-sycl"
            command = (
                f'source "{SCRIPT_DIR / "runtime-env.sh"}"; '
                "whisper_dictation_load_runtime; "
                'printf "build=%s\\naccelerator=%s\\noneapi=%s\\nexpected=%s\\n" '
                '"${WHISPER_RESOLVED_BUILD_DIR}" "${WHISPER_ACCELERATOR}" '
                '"${OVERRIDE_ONEAPI:-0}" "${WHISPER_SYCL_EXPECTED_DEVICE}"'
            )
            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "WHISPER_BUILD_DIR": str(custom_build),
                    "WHISPER_ACCELERATOR": "sycl",
                    "WHISPER_ONEAPI_SETVARS": str(setvars),
                    "WHISPER_SYCL_EXPECTED_DEVICE": "Intel Arc Graphics",
                }
            )
            result = subprocess.run(
                ["bash", "-c", command],
                text=True,
                capture_output=True,
                env=env,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"build={custom_build}", result.stdout)
            self.assertIn("accelerator=sycl", result.stdout)
            self.assertIn("oneapi=1", result.stdout)
            self.assertIn("expected=Intel Arc Graphics", result.stdout)

    def test_build_sycl_helper_configures_supported_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            fake_bin = temp / "bin"
            event_log = temp / "cmake.log"
            setvars = temp / "setvars.sh"
            setvars.write_text("export FAKE_ONEAPI_READY=1\n", encoding="utf-8")
            write_executable(
                fake_bin / "sycl-ls",
                "#!/usr/bin/env bash\n"
                'printf "[level_zero:gpu][level_zero:1] Intel(R) Arc(TM) Graphics\\n"\n',
            )
            write_executable(
                fake_bin / "cmake",
                '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "${EVENT_LOG}"\n',
            )
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:/usr/bin:/bin",
                    "EVENT_LOG": str(event_log),
                    "WHISPER_ONEAPI_SETVARS": str(setvars),
                    "WHISPER_SYCL_DEVICE": "1",
                    "WHISPER_SYCL_EXPECTED_DEVICE": "Intel Arc Graphics",
                }
            )
            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT_DIR / "build-sycl.sh"),
                    "--build-dir",
                    str(temp / "build-sycl"),
                ],
                text=True,
                capture_output=True,
                env=env,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = event_log.read_text(encoding="utf-8")
            self.assertIn("-DGGML_SYCL=ON", calls)
            self.assertIn("-DGGML_SYCL_SUPPORT_LEVEL_ZERO=ON", calls)
            self.assertIn("-DGGML_SYCL_F16=OFF", calls)
            self.assertIn("--target whisper-cli whisper-server", calls)
            self.assertNotIn("ls-sycl-device", calls)

    def test_build_cuda_helper_configures_supported_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            fake_bin = temp / "bin"
            event_log = temp / "cmake.log"
            write_executable(fake_bin / "nvcc", "#!/usr/bin/env bash\nexit 0\n")
            write_executable(
                fake_bin / "cmake",
                '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "${EVENT_LOG}"\n',
            )
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:/usr/bin:/bin",
                    "EVENT_LOG": str(event_log),
                }
            )
            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT_DIR / "build-cuda.sh"),
                    "--build-dir",
                    str(temp / "build-cuda"),
                ],
                text=True,
                capture_output=True,
                env=env,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = event_log.read_text(encoding="utf-8")
            self.assertIn("-DGGML_CUDA=ON", calls)
            self.assertIn("--target whisper-cli whisper-server", calls)

    def test_check_rejects_cpu_binary_when_sycl_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            home = temp / "home"
            config_dir = home / ".config/whisper-dictation"
            config_dir.mkdir(parents=True)
            (config_dir / "install.env").write_text(
                f'WHISPER_REPO_ROOT="{REPO_ROOT}"\n', encoding="utf-8"
            )
            (config_dir / "config.env").write_text(
                'WHISPER_MODEL="small.en"\n'
                'WHISPER_BUILD_DIR="build"\n'
                'WHISPER_ACCELERATOR="sycl"\n'
                'WHISPER_ONEAPI_SETVARS="/opt/intel/oneapi/setvars.sh"\n'
                'WHISPER_ONEAPI_DEVICE_SELECTOR="level_zero:gpu"\n'
                'WHISPER_SYCL_DEVICE="0"\n'
                'WHISPER_BACKEND="cli"\n',
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update({"HOME": str(home), "DISPLAY": os.environ.get("DISPLAY", ":0")})
            result = subprocess.run(
                ["bash", str(SCRIPT_DIR / "check.sh")],
                text=True,
                capture_output=True,
                env=env,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("selected binary did not report SYCL", result.stdout)

    def test_check_accepts_nonzero_intel_sycl_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_fake_server_check(
                Path(tmp),
                accelerator="sycl",
                sycl_device="1",
                sycl_device_name="Intel(R) Arc(TM) Graphics",
                sycl_expected_device="Intel Arc Graphics",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(
                "selected binary reports SYCL0 on [level_zero:gpu:1]",
                result.stdout,
            )

    def test_check_enforces_optional_sycl_device_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_fake_server_check(
                Path(tmp),
                accelerator="sycl",
                sycl_device_name="Intel Arc Graphics",
                sycl_expected_device="Iris Xe",
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("expected SYCL device name not found", result.stdout)

    def test_shared_server_response_validator_rejects_malformed_json(self) -> None:
        validator = SCRIPT_DIR / "validate-server-response.py"
        malformed = subprocess.run(
            [sys.executable, str(validator)],
            input='{"text":',
            text=True,
            capture_output=True,
            timeout=5,
        )
        empty = subprocess.run(
            [sys.executable, str(validator)],
            input='{"text":"  "}',
            text=True,
            capture_output=True,
            timeout=5,
        )
        valid = subprocess.run(
            [sys.executable, str(validator)],
            input='{"text":"ready"}',
            text=True,
            capture_output=True,
            timeout=5,
        )
        self.assertNotEqual(malformed.returncode, 0)
        self.assertNotEqual(empty.returncode, 0)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        run_server = (SCRIPT_DIR / "run-server.sh").read_text(encoding="utf-8")
        self.assertIn("validate-server-response.py", run_server)

    def test_check_rejects_cpu_binary_when_cuda_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_fake_server_check(
                Path(tmp), accelerator="cuda", cli_reports_accelerator=False
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("selected binary did not report CUDA", result.stdout)

    def test_check_accepts_selected_cuda_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_fake_server_check(Path(tmp), accelerator="cuda")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("selected binary reports CUDA0", result.stdout)

    def test_check_rejects_unreachable_configured_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_fake_server_check(Path(tmp), curl_fails=True)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("server inference failed", result.stdout)

    def test_check_rejects_mismatched_running_server_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_fake_server_check(Path(tmp), server_matches=False)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(
                "running server does not match selected binary", result.stdout
            )

    def test_check_rejects_malformed_server_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_fake_server_check(Path(tmp), malformed_response=True)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("invalid or empty JSON response", result.stdout)

    def test_run_server_warms_before_notifying_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            home = temp / "home"
            config_dir = home / ".config/whisper-dictation"
            config_dir.mkdir(parents=True)
            repo = temp / "repo"
            (repo / "build-sycl/bin").mkdir(parents=True)
            (repo / "models").mkdir()
            (repo / "samples").mkdir()
            (repo / "models/ggml-small.en.bin").write_bytes(b"model")
            (repo / "samples/jfk.wav").write_bytes(b"wav")
            event_log = temp / "events.log"
            server_pid_file = temp / "server.pid"
            setvars = temp / "setvars.sh"
            setvars.write_text(
                'export FAKE_ONEAPI_READY="1"\nprintf "oneapi\\n" >> "${EVENT_LOG}"\n',
                encoding="utf-8",
            )
            (config_dir / "install.env").write_text(
                f'WHISPER_REPO_ROOT="{repo}"\n', encoding="utf-8"
            )
            (config_dir / "config.env").write_text(
                'WHISPER_MODEL="small.en"\n'
                'WHISPER_BUILD_DIR="build-sycl"\n'
                'WHISPER_ACCELERATOR="sycl"\n'
                f'WHISPER_ONEAPI_SETVARS="{setvars}"\n'
                'WHISPER_SERVER_URL="http://127.0.0.1:18178"\n'
                'WHISPER_SERVER_WARMUP="1"\n'
                f'WHISPER_SERVER_WARMUP_AUDIO="{repo / "samples/jfk.wav"}"\n'
                'WHISPER_SERVER_WARMUP_TIMEOUT="3"\n'
                'WHISPER_INFERENCE_WATCHDOG_SEC="0"\n'
                'WHISPER_SERVER_RECYCLE_SEC="0"\n',
                encoding="utf-8",
            )

            write_executable(
                repo / "build-sycl/bin/whisper-server",
                "#!/usr/bin/env bash\n"
                '[[ "${FAKE_ONEAPI_READY:-0}" == "1" ]] || exit 42\n'
                'printf "%s\\n" "$$" > "${SERVER_PID_FILE}"\n'
                'printf "server-start\\n" >> "${EVENT_LOG}"\n'
                'trap \'printf "server-stop\\n" >> "${EVENT_LOG}"; exit 0\' TERM INT\n'
                "while true; do sleep 0.1; done\n",
            )
            fake_bin = temp / "bin"
            write_executable(
                fake_bin / "curl",
                "#!/usr/bin/env bash\n"
                "output=''\n"
                "warm=0\n"
                "previous=''\n"
                'for arg in "$@"; do\n'
                '  [[ "${previous}" == "-o" ]] && output="${arg}"\n'
                '  [[ "${arg}" == file=@* ]] && warm=1\n'
                '  previous="${arg}"\n'
                "done\n"
                'if [[ "${warm}" -eq 1 ]]; then\n'
                '  printf "warmup\\n" >> "${EVENT_LOG}"\n'
                '  if [[ -n "${output}" ]]; then printf \'{"text":"warm"}\' > "${output}"; else printf \'{"text":"warm"}\'; fi\n'
                "else\n"
                '  printf "probe\\n" >> "${EVENT_LOG}"\n'
                "fi\n",
            )
            write_executable(
                fake_bin / "systemd-notify",
                '#!/usr/bin/env bash\nprintf "notify-ready\\n" >> "${EVENT_LOG}"\n',
            )

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "EVENT_LOG": str(event_log),
                    "SERVER_PID_FILE": str(server_pid_file),
                    "NOTIFY_SOCKET": str(temp / "notify.sock"),
                    "PATH": f"{fake_bin}:/usr/bin:/bin",
                }
            )
            proc = subprocess.Popen(
                ["bash", str(SCRIPT_DIR / "run-server.sh")],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            try:
                deadline = time.monotonic() + 5
                events: list[str] = []
                while time.monotonic() < deadline:
                    if event_log.exists():
                        events = event_log.read_text(encoding="utf-8").splitlines()
                    if "notify-ready" in events or proc.poll() is not None:
                        break
                    time.sleep(0.05)

                stderr = ""
                if proc.poll() is not None and proc.stderr:
                    stderr = proc.stderr.read()
                self.assertIn("server-start", events, stderr)
                self.assertIn("warmup", events, stderr)
                self.assertIn("notify-ready", events, stderr)
                self.assertLess(events.index("server-start"), events.index("warmup"))
                self.assertLess(events.index("warmup"), events.index("notify-ready"))
            finally:
                if proc.poll() is None:
                    proc.terminate()
                self.assertEqual(proc.wait(timeout=5), 0)
                server_pid = int(server_pid_file.read_text(encoding="utf-8").strip())
                with self.assertRaises(ProcessLookupError):
                    os.kill(server_pid, 0)
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()

    def test_generated_launcher_loads_runtime_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home, _ = self.run_autostart_install(Path(tmp))
            launcher = (home / ".local/bin/whisper-dictation").read_text(
                encoding="utf-8"
            )
            self.assertIn("runtime-env.sh", launcher)
            self.assertIn("whisper_dictation_load_runtime", launcher)

    def test_installer_starts_warm_server_before_daemon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, events = self.run_autostart_install(Path(tmp))
            server_restart = events.index(
                "--user restart whisper-dictation-server.service"
            )
            daemon_restart = events.index("--user restart whisper-dictation.service")
            self.assertLess(server_restart, daemon_restart)

    def test_server_unit_waits_for_readiness_notification(self) -> None:
        unit = (SCRIPT_DIR / "whisper-dictation-server.service").read_text(
            encoding="utf-8"
        )
        self.assertIn("Type=notify", unit)
        self.assertIn("TimeoutStartSec=", unit)
        self.assertIn("TimeoutStopSec=", unit)
        self.assertIn("WatchdogSec=", unit)
        self.assertIn("NotifyAccess=all", unit)
        self.assertIn("KillMode=mixed", unit)

    def test_dictation_unit_allows_bounded_durable_handoff(self) -> None:
        unit = (SCRIPT_DIR / "whisper-dictation.service").read_text(encoding="utf-8")
        self.assertIn("KillMode=mixed", unit)
        self.assertIn("KillSignal=SIGTERM", unit)
        self.assertIn("TimeoutStopSec=15", unit)


class HangRecoveryTests(unittest.TestCase):
    def test_notify_replace_id_is_integer(self) -> None:
        from dictation import NOTIFY_REPLACE_ID, build_notify_cmd

        cmd = build_notify_cmd("Dictation", "Recording…", 1000)
        self.assertEqual(cmd[cmd.index("-r") + 1], NOTIFY_REPLACE_ID)
        int(cmd[cmd.index("-r") + 1])  # must not raise

    def _make_app(self, **cfg: str) -> Dictation:
        defaults = {
            "WHISPER_HOME": str(REPO_ROOT),
            "WHISPER_BUILD_DIR": "build-sycl",
            "WHISPER_BACKEND": "server",
        }
        defaults.update(cfg)
        with mock.patch("dictation.build_recorder_cmd", return_value=["parecord"]):
            return Dictation(defaults)

    def test_server_request_uses_no_timestamp_greedy_profile(self) -> None:
        for build_dir in ("build", "build-cuda", "build-sycl"):
            with self.subTest(build_dir=build_dir):
                app = self._make_app(WHISPER_BUILD_DIR=build_dir)
                which = subprocess.CompletedProcess(["which", "curl"], 0, "", "")
                response = subprocess.CompletedProcess(
                    ["curl"], 0, '{"text":"hello world"}', ""
                )
                with mock.patch(
                    "dictation.subprocess.run", side_effect=[which, response]
                ) as run:
                    text = app._transcribe_server("/tmp/clip.wav")
                self.assertEqual(text, "hello world")
                cmd = run.call_args_list[1].args[0]
                self.assertIn("no_timestamps=true", cmd)
                self.assertIn("token_timestamps=false", cmd)
                self.assertNotIn("beam_size=5", cmd)

    def test_server_request_can_select_beam_retry_profile(self) -> None:
        app = self._make_app()
        which = subprocess.CompletedProcess(["which", "curl"], 0, "", "")
        response = subprocess.CompletedProcess(
            ["curl"], 0, '{"text":"recovered words"}', ""
        )
        with mock.patch(
            "dictation.subprocess.run", side_effect=[which, response]
        ) as run:
            text = app._transcribe_server("/tmp/clip.wav", beam_size=5)
        self.assertEqual(text, "recovered words")
        cmd = run.call_args_list[1].args[0]
        self.assertIn("beam_size=5", cmd)

    def test_punctuation_only_retries_beam_on_same_server(self) -> None:
        for punctuation in (",", "()", "[]", "/", "¿?", "。", "،"):
            with self.subTest(punctuation=punctuation):
                app = self._make_app()
                app._transcribe_server = mock.Mock(
                    side_effect=[punctuation, "recovered text"]
                )
                app._restart_whisper_server = mock.Mock()
                with mock.patch.object(app, "_transcribe_cli") as cli:
                    text = app._transcribe("/tmp/clip.wav")
                self.assertEqual(text, "recovered text")
                self.assertEqual(
                    app._transcribe_server.call_args_list,
                    [
                        mock.call("/tmp/clip.wav"),
                        mock.call("/tmp/clip.wav", beam_size=5),
                    ],
                )
                app._restart_whisper_server.assert_not_called()
                cli.assert_not_called()

    def test_empty_server_text_retries_beam_on_same_server(self) -> None:
        app = self._make_app()
        app._transcribe_server = mock.Mock(side_effect=["", "recovered text"])
        app._restart_whisper_server = mock.Mock()
        with mock.patch.object(app, "_transcribe_cli") as cli:
            text = app._transcribe("/tmp/clip.wav")
        self.assertEqual(text, "recovered text")
        self.assertEqual(
            app._transcribe_server.call_args_list,
            [mock.call("/tmp/clip.wav"), mock.call("/tmp/clip.wav", beam_size=5)],
        )
        app._restart_whisper_server.assert_not_called()
        cli.assert_not_called()

    def test_semantic_failure_restarts_then_retries_beam(self) -> None:
        app = self._make_app()
        app._transcribe_server = mock.Mock(side_effect=[",", ".", "recovered text"])
        app._restart_whisper_server = mock.Mock(return_value=True)
        with mock.patch.object(app, "_notify"):
            with mock.patch.object(app, "_transcribe_cli") as cli:
                text = app._transcribe("/tmp/clip.wav")
        self.assertEqual(text, "recovered text")
        self.assertEqual(
            app._transcribe_server.call_args_list,
            [
                mock.call("/tmp/clip.wav"),
                mock.call("/tmp/clip.wav", beam_size=5),
                mock.call("/tmp/clip.wav", beam_size=5),
            ],
        )
        app._restart_whisper_server.assert_called_once()
        cli.assert_not_called()

    def test_transport_then_semantic_failure_uses_beam_before_cpu(self) -> None:
        app = self._make_app()
        app._transcribe_server = mock.Mock(side_effect=[None, ",", "recovered text"])
        app._restart_whisper_server = mock.Mock(return_value=True)
        with mock.patch.object(app, "_notify"):
            with mock.patch.object(app, "_transcribe_cli") as cli:
                text = app._transcribe("/tmp/clip.wav")
        self.assertEqual(text, "recovered text")
        self.assertEqual(
            app._transcribe_server.call_args_list,
            [
                mock.call("/tmp/clip.wav"),
                mock.call("/tmp/clip.wav"),
                mock.call("/tmp/clip.wav", beam_size=5),
            ],
        )
        app._restart_whisper_server.assert_called_once()
        cli.assert_not_called()

    def test_exhausted_semantic_retries_fall_back_to_cpu(self) -> None:
        app = self._make_app()
        app._transcribe_server = mock.Mock(side_effect=[",", ".", "-"])
        app._restart_whisper_server = mock.Mock(return_value=True)
        cpu = Path("/tmp/cpu-whisper-cli")
        with mock.patch.object(app, "_notify"):
            with mock.patch.object(app, "_cpu_fallback_cli", return_value=cpu):
                with mock.patch.object(
                    app, "_transcribe_cli", return_value="cpu transcript"
                ) as cli:
                    text = app._transcribe("/tmp/clip.wav")
        self.assertEqual(text, "cpu transcript")
        self.assertEqual(app._transcribe_server.call_count, 3)
        app._restart_whisper_server.assert_called_once()
        cli.assert_called_once_with("/tmp/clip.wav", cli=cpu)

    def test_valid_punctuation_containing_text_remains_fast_path(self) -> None:
        app = self._make_app()
        app._transcribe_server = mock.Mock(return_value="Wait, what?")
        app._restart_whisper_server = mock.Mock()
        with mock.patch.object(app, "_transcribe_cli") as cli:
            text = app._transcribe("/tmp/clip.wav")
        self.assertEqual(text, "Wait, what?")
        app._transcribe_server.assert_called_once_with("/tmp/clip.wav")
        app._restart_whisper_server.assert_not_called()
        cli.assert_not_called()

    def test_loud_punctuation_only_transcript_is_not_pasted(self) -> None:
        from session_store import ChunkJob

        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        handle.write(b"\x00" * 2000)
        handle.close()
        job = ChunkJob(Path(handle.name).parent, 0, Path(handle.name), 8.0, True, 1)
        try:
            for punctuation in (",", "()", "[]", "/", "¿?", "。", "،"):
                with self.subTest(punctuation=punctuation):
                    app = self._make_app()
                    app._store = mock.Mock()
                    with mock.patch.object(
                        app, "_transcribe", return_value=punctuation
                    ):
                        with mock.patch("dictation.wav_rms", return_value=400.0):
                            with mock.patch.object(app, "_insert_chunk") as insert:
                                with mock.patch.object(app, "_notify"):
                                    app._deliver_chunk(job)
                    insert.assert_not_called()
                    app._store.terminal.assert_called_once()
        finally:
            os.unlink(handle.name)

    def test_end_of_video_is_a_known_silence_hallucination(self) -> None:
        from dictation import is_likely_hallucination

        self.assertTrue(is_likely_hallucination("End of video."))

    def test_server_timeout_restarts_then_retries_without_gpu_cli(self) -> None:
        app = self._make_app()
        app._transcribe_server = mock.Mock(side_effect=[None, "recovered text"])
        app._restart_whisper_server = mock.Mock(return_value=True)
        with mock.patch.object(app, "_notify"):
            with mock.patch.object(app, "_transcribe_cli") as cli:
                text = app._transcribe("/tmp/clip.wav")
        self.assertEqual(text, "recovered text")
        app._restart_whisper_server.assert_called_once()
        self.assertEqual(app._transcribe_server.call_count, 2)
        cli.assert_not_called()

    def test_server_hang_falls_back_to_cpu_cli_not_sycl(self) -> None:
        app = self._make_app()
        app._transcribe_server = mock.Mock(return_value=None)
        app._restart_whisper_server = mock.Mock(return_value=False)
        cpu = Path("/tmp/cpu-whisper-cli")
        with mock.patch.object(app, "_notify"):
            with mock.patch.object(app, "_cpu_fallback_cli", return_value=cpu):
                with mock.patch.object(
                    app, "_transcribe_cli", return_value="cpu transcript"
                ) as cli:
                    text = app._transcribe("/tmp/clip.wav")
        self.assertEqual(text, "cpu transcript")
        cli.assert_called_once_with("/tmp/clip.wav", cli=cpu)

    def test_cli_timeout_is_caught(self) -> None:
        app = self._make_app(WHISPER_BACKEND="cli", WHISPER_CLI_TIMEOUT="1")
        with mock.patch.object(app, "_notify"):
            with mock.patch(
                "dictation.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="whisper-cli", timeout=1),
            ):
                text = app._transcribe_cli("/tmp/clip.wav")
        self.assertEqual(text, "")

    def test_max_record_timer_rolls_to_next_slice(self) -> None:
        app = self._make_app(MAX_RECORD_SEC="0.05")
        app._recording = True
        app._record_start = time.monotonic()
        with mock.patch.object(app, "_notify"):
            with mock.patch.object(app, "_roll_recording") as roll:
                app._arm_max_timer()
                deadline = time.monotonic() + 1.0
                while not roll.called and time.monotonic() < deadline:
                    time.sleep(0.02)
                try:
                    roll.assert_called()
                finally:
                    app._cancel_max_timer()

    def test_spawn_recorder_retries_early_exit_until_payload(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            counter = root / "starts"
            recorder = root / "recorder.py"
            write_executable(
                recorder,
                "#!/usr/bin/env python3\n"
                "import os\n"
                "import sys\n"
                "import time\n"
                "counter = os.environ['FAKE_RECORDER_COUNTER']\n"
                "try:\n"
                "    count = int(open(counter, encoding='utf-8').read()) + 1\n"
                "except FileNotFoundError:\n"
                "    count = 1\n"
                "open(counter, 'w', encoding='utf-8').write(str(count))\n"
                "if count < 3:\n"
                "    print('source acquire failed', file=sys.stderr)\n"
                "    raise SystemExit(17)\n"
                "with open(sys.argv[-1], 'wb') as output:\n"
                "    output.write(b'R' * 2048)\n"
                "while True:\n"
                "    time.sleep(1)\n",
            )
            app = self._make_app()
            app._recorder = [str(recorder)]
            real_mkstemp = tempfile.mkstemp

            def local_mkstemp(**kwargs):
                return real_mkstemp(dir=root, **kwargs)

            with mock.patch.dict(os.environ, {"FAKE_RECORDER_COUNTER": str(counter)}):
                with mock.patch("dictation.wake_audio_source"):
                    with mock.patch(
                        "dictation.tempfile.mkstemp", side_effect=local_mkstemp
                    ):
                        with mock.patch("dictation.RECORDER_START_TIMEOUT_SEC", 0.2):
                            with mock.patch("dictation.RECORDER_RETRY_DELAY_SEC", 0):
                                spawned = app._spawn_recorder()

            self.assertIsNotNone(spawned)
            proc, wav = spawned
            try:
                self.assertIsNone(proc.poll())
                self.assertGreaterEqual(Path(wav).stat().st_size, 1000)
                self.assertEqual(counter.read_text(encoding="utf-8"), "3")
                self.assertEqual(
                    list(root.glob("whisper-dictation-*.wav")), [Path(wav)]
                )
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=2)
                app._recorder_exit_detail(proc)

    def test_spawn_recorder_rejects_live_child_without_payload(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            recorder = root / "recorder.py"
            write_executable(
                recorder,
                "#!/usr/bin/env python3\nimport time\ntime.sleep(5)\n",
            )
            app = self._make_app()
            app._recorder = [str(recorder)]
            real_mkstemp = tempfile.mkstemp

            def local_mkstemp(**kwargs):
                return real_mkstemp(dir=root, **kwargs)

            with mock.patch("dictation.wake_audio_source"):
                with mock.patch(
                    "dictation.tempfile.mkstemp", side_effect=local_mkstemp
                ):
                    with mock.patch("dictation.RECORDER_START_ATTEMPTS", 2):
                        with mock.patch("dictation.RECORDER_START_TIMEOUT_SEC", 0.05):
                            with mock.patch("dictation.RECORDER_RETRY_DELAY_SEC", 0):
                                spawned = app._spawn_recorder()

            self.assertIsNone(spawned)
            self.assertEqual(list(root.glob("whisper-dictation-*.wav")), [])
            self.assertIn("no audio payload", app._last_recorder_error)

    def test_stop_tolerates_recorder_exiting_before_signal(self) -> None:
        from dictation import graceful_stop_recorder

        proc = mock.Mock(pid=123)
        proc.poll.return_value = None
        proc.send_signal.side_effect = ProcessLookupError
        proc.wait.return_value = 0
        with mock.patch("dictation.os.getpgid", side_effect=ProcessLookupError):
            graceful_stop_recorder(proc, None, 0)
        proc.wait.assert_called_once_with(timeout=5.0)

    def test_prefix_chunk_for_insert_adds_single_space(self) -> None:
        from dictation import prefix_chunk_for_insert

        text, need = prefix_chunk_for_insert(False, "  hello world  ")
        self.assertEqual(text, "hello world")
        self.assertTrue(need)
        text, need = prefix_chunk_for_insert(True, "again")
        self.assertEqual(text, " again")
        self.assertTrue(need)
        text, need = prefix_chunk_for_insert(True, ". Next")
        self.assertEqual(text, ". Next")
        empty, need = prefix_chunk_for_insert(True, "   ")
        self.assertEqual(text, ". Next")
        self.assertEqual(empty, "")
        self.assertTrue(need)

    def test_roll_releases_old_before_starting_next_recorder(self) -> None:
        app = self._make_app()
        old_proc = mock.Mock()
        new_proc = mock.Mock()
        app._recording = True
        app._session_id = 7
        app._record_proc = old_proc
        app._wav_path = "/tmp/old.wav"
        app._record_start = time.monotonic() - 45
        app._tray = mock.Mock()
        events = []

        def spawn():
            events.append("spawn-next")
            return new_proc, "/tmp/new.wav"

        def stop(*_args):
            events.append("stop-old")

        def stage(*_args):
            events.append("stage-old")

        def watch(*_args):
            events.append("watch-next")

        with mock.patch.object(app, "_spawn_recorder", side_effect=spawn):
            with mock.patch(
                "dictation.graceful_stop_recorder", side_effect=stop
            ) as stop_call:
                with mock.patch.object(
                    app, "_stage_chunk", side_effect=stage
                ) as stage_call:
                    with mock.patch.object(
                        app, "_start_recorder_watch", side_effect=watch
                    ):
                        with mock.patch.object(app, "_arm_max_timer"):
                            app._roll_recording()
        self.assertTrue(app._recording)
        self.assertIs(app._record_proc, new_proc)
        self.assertEqual(app._wav_path, "/tmp/new.wav")
        self.assertEqual(events, ["stop-old", "spawn-next", "watch-next", "stage-old"])
        stop_call.assert_called_once()
        self.assertIs(stop_call.call_args[0][0], old_proc)
        self.assertEqual(stop_call.call_args[0][1], "/tmp/old.wav")
        self.assertEqual(stop_call.call_args[0][2], 0)
        stage_call.assert_called_once()
        self.assertEqual(stage_call.call_args[0][0], "/tmp/old.wav")
        self.assertEqual(stage_call.call_args[0][3], 7)

    def test_unexpected_recorder_exit_restarts_and_stages_partial_chunk(self) -> None:
        app = self._make_app()
        old_proc = mock.Mock()
        new_proc = mock.Mock()
        app._recording = True
        app._record_proc = old_proc
        app._wav_path = "/tmp/partial.wav"
        app._record_start = time.monotonic() - 3
        app._record_generation = 4
        app._session_id = 7
        app._session_paths = {7: Path("/tmp/session")}
        app._tray = mock.Mock()
        with mock.patch.object(
            app, "_spawn_recorder", return_value=(new_proc, "/tmp/recovered.wav")
        ):
            with mock.patch.object(app, "_stage_chunk") as stage:
                with mock.patch.object(app, "_start_recorder_watch", create=True):
                    with mock.patch.object(app, "_arm_max_timer"):
                        app._handle_recorder_exit(
                            old_proc,
                            "/tmp/partial.wav",
                            4,
                            "exit 17: acquire failed",
                        )

        self.assertTrue(app._recording)
        self.assertIs(app._record_proc, new_proc)
        self.assertEqual(app._wav_path, "/tmp/recovered.wav")
        stage.assert_called_once()
        self.assertEqual(stage.call_args.args[0], "/tmp/partial.wav")
        self.assertEqual(stage.call_args.args[2:], (0, 7))

    def test_empty_failed_chunk_keeps_later_audio_in_transcript(self) -> None:
        from session_store import SessionStore

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with mock.patch.object(Dictation, "_transcribe_worker"):
                app = self._make_app()
            app._store = SessionStore(root / "sessions")
            session = app._store.start_session()
            empty_partial = root / "empty-partial.wav"
            recovered_audio = root / "recovered.wav"
            empty_partial.write_bytes(b"")
            recovered_audio.write_bytes(b"R" * 2048)
            failed_proc = mock.Mock()
            recovered_proc = mock.Mock()
            following_proc = mock.Mock()
            app._recording = True
            app._record_proc = failed_proc
            app._wav_path = str(empty_partial)
            app._record_start = time.monotonic() - 2
            app._record_generation = 3
            app._session_id = 11
            app._session_paths = {11: session}
            app._tray = mock.Mock()
            jobs = []

            def capture_job(_seq, job):
                jobs.append(job)

            with mock.patch.object(
                app,
                "_spawn_recorder",
                side_effect=[
                    (recovered_proc, str(recovered_audio)),
                    (following_proc, str(root / "following.wav")),
                ],
            ):
                with mock.patch.object(
                    app, "_submit_in_order", side_effect=capture_job
                ):
                    with mock.patch.object(app, "_start_recorder_watch"):
                        with mock.patch.object(app, "_arm_max_timer"):
                            with mock.patch("dictation.graceful_stop_recorder"):
                                app._handle_recorder_exit(
                                    failed_proc,
                                    str(empty_partial),
                                    3,
                                    "exit 17: injected",
                                )
                                app._record_start = time.monotonic() - 2
                                app._roll_recording()

            self.assertEqual([job.chunk_index for job in jobs], [0, 1])
            with mock.patch.object(app, "_notify"):
                app._deliver_chunk(jobs[0])
                with mock.patch("dictation.wav_rms", return_value=400):
                    with mock.patch.object(
                        app, "_transcribe", return_value="words after recovery"
                    ):
                        with mock.patch.object(app, "_insert_chunk"):
                            app._deliver_chunk(jobs[1])
            app._store.stop_session(session)
            transcript = (session / "transcript.txt").read_text(encoding="utf-8")
            self.assertIn("[no_text: chunk 0000", transcript)
            self.assertTrue(transcript.endswith("words after recovery\n"))

    def test_recorder_watch_routes_exit_to_expected_generation(self) -> None:
        app = self._make_app()
        proc = mock.Mock()
        proc.poll.side_effect = [None, 17]
        with mock.patch("dictation.time.sleep"):
            with mock.patch.object(
                app, "_recorder_exit_detail", return_value="exit 17"
            ):
                with mock.patch.object(app, "_handle_recorder_exit") as handle:
                    app._watch_recorder(proc, "/tmp/partial.wav", 4)
        handle.assert_called_once_with(proc, "/tmp/partial.wav", 4, "exit 17")

    def test_stale_watcher_and_timer_cannot_rotate_current_recorder(self) -> None:
        app = self._make_app()
        stale_proc = mock.Mock()
        current_proc = mock.Mock()
        app._recording = True
        app._record_proc = current_proc
        app._wav_path = "/tmp/current.wav"
        app._record_generation = 8
        with mock.patch.object(app, "_spawn_recorder") as spawn:
            with mock.patch.object(app, "_stage_chunk") as stage:
                app._handle_recorder_exit(stale_proc, "/tmp/stale.wav", 7, "exit 1")
        spawn.assert_not_called()
        stage.assert_not_called()
        with mock.patch.object(app, "_roll_recording") as roll:
            app._on_max_record(7)
        roll.assert_not_called()

    def test_rollover_wins_watcher_race_without_duplicate_stage_or_spawn(self) -> None:
        app = self._make_app()
        old_proc = mock.Mock()
        new_proc = mock.Mock()
        app._recording = True
        app._record_proc = old_proc
        app._wav_path = "/tmp/old.wav"
        app._record_start = time.monotonic() - 45
        app._record_generation = 1
        app._session_id = 5
        app._tray = mock.Mock()
        stop_entered = threading.Event()
        release_stop = threading.Event()
        watcher_attempted = threading.Event()

        def stop(*_args):
            stop_entered.set()
            release_stop.wait(timeout=2)

        def watcher():
            watcher_attempted.set()
            app._handle_recorder_exit(old_proc, "/tmp/old.wav", 1, "exit 17")

        with mock.patch.object(
            app, "_spawn_recorder", return_value=(new_proc, "/tmp/new.wav")
        ) as spawn:
            with mock.patch("dictation.graceful_stop_recorder", side_effect=stop):
                with mock.patch.object(app, "_stage_chunk") as stage:
                    with mock.patch.object(app, "_start_recorder_watch"):
                        with mock.patch.object(app, "_arm_max_timer"):
                            roll_thread = threading.Thread(target=app._roll_recording)
                            roll_thread.start()
                            self.assertTrue(stop_entered.wait(timeout=1))
                            watcher_thread = threading.Thread(target=watcher)
                            watcher_thread.start()
                            self.assertTrue(watcher_attempted.wait(timeout=1))
                            time.sleep(0.02)
                            self.assertTrue(watcher_thread.is_alive())
                            release_stop.set()
                            roll_thread.join(timeout=2)
                            watcher_thread.join(timeout=2)

        self.assertFalse(roll_thread.is_alive())
        self.assertFalse(watcher_thread.is_alive())
        spawn.assert_called_once_with()
        stage.assert_called_once()
        self.assertEqual(stage.call_args.args[2:], (0, 5))
        self.assertIs(app._record_proc, new_proc)
        self.assertEqual(app._record_generation, 2)

    def test_watcher_wins_timer_race_and_stale_timer_does_not_roll(self) -> None:
        app = self._make_app()
        old_proc = mock.Mock()
        new_proc = mock.Mock()
        app._recording = True
        app._record_proc = old_proc
        app._wav_path = "/tmp/old.wav"
        app._record_start = time.monotonic() - 2
        app._record_generation = 1
        app._session_id = 5
        app._tray = mock.Mock()
        spawn_entered = threading.Event()
        release_spawn = threading.Event()
        timer_attempted = threading.Event()

        def spawn():
            spawn_entered.set()
            release_spawn.wait(timeout=2)
            return new_proc, "/tmp/new.wav"

        def timer():
            timer_attempted.set()
            app._on_max_record(1)

        with mock.patch.object(app, "_spawn_recorder", side_effect=spawn) as spawn_call:
            with mock.patch.object(app, "_stage_chunk") as stage:
                with mock.patch.object(app, "_start_recorder_watch"):
                    with mock.patch.object(app, "_arm_max_timer"):
                        with mock.patch.object(app, "_roll_recording") as roll:
                            watcher_thread = threading.Thread(
                                target=app._handle_recorder_exit,
                                args=(old_proc, "/tmp/old.wav", 1, "exit 17"),
                            )
                            watcher_thread.start()
                            self.assertTrue(spawn_entered.wait(timeout=1))
                            timer_thread = threading.Thread(target=timer)
                            timer_thread.start()
                            self.assertTrue(timer_attempted.wait(timeout=1))
                            time.sleep(0.02)
                            self.assertTrue(timer_thread.is_alive())
                            release_spawn.set()
                            watcher_thread.join(timeout=2)
                            timer_thread.join(timeout=2)

        self.assertFalse(watcher_thread.is_alive())
        self.assertFalse(timer_thread.is_alive())
        spawn_call.assert_called_once_with()
        stage.assert_called_once()
        roll.assert_not_called()
        self.assertIs(app._record_proc, new_proc)
        self.assertEqual(app._record_generation, 2)

    def test_user_stop_wins_watcher_race_without_reconnecting(self) -> None:
        app = self._make_app()
        proc = mock.Mock()
        app._recording = True
        app._record_proc = proc
        app._wav_path = "/tmp/final.wav"
        app._record_start = time.monotonic() - 2
        app._record_generation = 1
        app._session_id = 5
        app._tray = mock.Mock()
        stop_entered = threading.Event()
        release_stop = threading.Event()

        def stop(*_args):
            stop_entered.set()
            release_stop.wait(timeout=2)

        with mock.patch.object(app, "_spawn_recorder") as spawn:
            with mock.patch("dictation.graceful_stop_recorder", side_effect=stop):
                with mock.patch.object(app, "_stage_chunk") as stage:
                    finish_thread = threading.Thread(target=app._finish_recording)
                    finish_thread.start()
                    self.assertTrue(stop_entered.wait(timeout=1))
                    watcher_thread = threading.Thread(
                        target=app._handle_recorder_exit,
                        args=(proc, "/tmp/final.wav", 1, "exit 17"),
                    )
                    watcher_thread.start()
                    watcher_thread.join(timeout=1)
                    self.assertFalse(watcher_thread.is_alive())
                    release_stop.set()
                    finish_thread.join(timeout=2)

        self.assertFalse(finish_thread.is_alive())
        spawn.assert_not_called()
        stage.assert_called_once()
        self.assertEqual(stage.call_args.args[2:], (0, 5))
        self.assertFalse(app._recording)

    def test_watcher_wins_user_stop_race_then_stop_closes_replacement(self) -> None:
        app = self._make_app()
        failed_proc = mock.Mock()
        replacement_proc = mock.Mock()
        app._recording = True
        app._record_proc = failed_proc
        app._wav_path = "/tmp/failed.wav"
        app._record_start = time.monotonic() - 2
        app._record_generation = 1
        app._session_id = 5
        app._tray = mock.Mock()
        spawn_entered = threading.Event()
        release_spawn = threading.Event()
        stop_attempted = threading.Event()

        def spawn():
            spawn_entered.set()
            release_spawn.wait(timeout=2)
            return replacement_proc, "/tmp/replacement.wav"

        def finish():
            stop_attempted.set()
            app._finish_recording()

        with mock.patch.object(app, "_spawn_recorder", side_effect=spawn) as spawn_call:
            with mock.patch("dictation.graceful_stop_recorder") as stop:
                with mock.patch.object(app, "_stage_chunk") as stage:
                    with mock.patch.object(app, "_start_recorder_watch"):
                        with mock.patch.object(app, "_arm_max_timer"):
                            watcher_thread = threading.Thread(
                                target=app._handle_recorder_exit,
                                args=(failed_proc, "/tmp/failed.wav", 1, "exit 17"),
                            )
                            watcher_thread.start()
                            self.assertTrue(spawn_entered.wait(timeout=1))
                            finish_thread = threading.Thread(target=finish)
                            finish_thread.start()
                            self.assertTrue(stop_attempted.wait(timeout=1))
                            time.sleep(0.02)
                            self.assertTrue(finish_thread.is_alive())
                            release_spawn.set()
                            watcher_thread.join(timeout=2)
                            finish_thread.join(timeout=2)

        self.assertFalse(watcher_thread.is_alive())
        self.assertFalse(finish_thread.is_alive())
        spawn_call.assert_called_once_with()
        stop.assert_called_once()
        self.assertIs(stop.call_args.args[0], replacement_proc)
        self.assertEqual(sorted(call.args[2] for call in stage.call_args_list), [0, 1])
        self.assertFalse(app._recording)

    def test_failure_at_second_third_or_later_recorder_keeps_rotating(self) -> None:
        for failed_index in (1, 2, 7, 15):
            with self.subTest(failed_index=failed_index):
                app = self._make_app()
                app._recording = True
                app._record_proc = mock.Mock(name="recorder-0")
                app._wav_path = "/tmp/chunk-0.wav"
                app._record_start = time.monotonic() - 1
                app._record_generation = 1
                app._session_id = 9
                app._session_paths = {9: Path("/tmp/session")}
                app._tray = mock.Mock()
                next_index = 1

                def spawn():
                    nonlocal next_index
                    proc = mock.Mock(name=f"recorder-{next_index}")
                    wav = f"/tmp/chunk-{next_index}.wav"
                    next_index += 1
                    return proc, wav

                staged = []

                def stage(wav, _duration, seq, session_id):
                    staged.append((seq, wav, session_id))

                with mock.patch.object(app, "_spawn_recorder", side_effect=spawn):
                    with mock.patch.object(app, "_stage_chunk", side_effect=stage):
                        with mock.patch.object(
                            app, "_start_recorder_watch", create=True
                        ):
                            with mock.patch.object(app, "_arm_max_timer"):
                                with mock.patch("dictation.graceful_stop_recorder"):
                                    for index in range(16):
                                        if index == failed_index:
                                            app._handle_recorder_exit(
                                                app._record_proc,
                                                app._wav_path,
                                                app._record_generation,
                                                "exit 17: injected",
                                            )
                                        else:
                                            app._roll_recording()

                self.assertTrue(app._recording)
                self.assertEqual([item[0] for item in staged], list(range(16)))
                self.assertEqual(len({item[0] for item in staged}), 16)
                self.assertTrue(all(item[2] == 9 for item in staged))

    def test_roll_closes_session_after_all_successor_starts_fail(self) -> None:
        from session_store import SessionStore

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with mock.patch.object(Dictation, "_transcribe_worker"):
                app = self._make_app()
            app._store = SessionStore(root / "sessions")
            session = app._store.start_session()
            wav = root / "last-good.wav"
            wav.write_bytes(b"R" * 2048)
            app._recording = True
            app._record_proc = mock.Mock()
            app._wav_path = str(wav)
            app._record_start = time.monotonic() - 45
            app._session_id = 3
            app._session_paths = {3: session}
            app._tray = mock.Mock()
            app._last_recorder_error = "source acquire failed"
            with mock.patch.object(app, "_spawn_recorder", return_value=None):
                with mock.patch("dictation.graceful_stop_recorder"):
                    with mock.patch.object(app, "_notify"):
                        app._roll_recording()

            manifest = json.loads(
                (session / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(app._recording)
            self.assertFalse(manifest["recording"])
            self.assertEqual(
                [(chunk["index"], chunk["status"]) for chunk in manifest["chunks"]],
                [(0, "queued")],
            )
            retained = session / manifest["chunks"][0]["wav"]
            self.assertEqual(retained.read_bytes(), b"R" * 2048)

    def test_short_empty_tail_is_not_reported_as_microphone_failure(self) -> None:
        from session_store import ChunkJob

        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            app = self._make_app()
            app._store = mock.Mock()
            app._recording = False
            app._session_id = 1
            app._notify = mock.Mock()
            job = ChunkJob(Path(handle.name).parent, 0, Path(handle.name), 0.1, True, 1)
            app._deliver_chunk(job)
        app._store.terminal.assert_called_once_with(
            job, "ignored", "recording was too short"
        )
        self.assertIn("Too short", app._notify.call_args.args[0])

    def test_user_stop_does_not_start_next_slice(self) -> None:
        app = self._make_app()
        proc = mock.Mock()
        app._recording = True
        app._record_proc = proc
        app._wav_path = "/tmp/last.wav"
        app._record_start = time.monotonic() - 5
        app._session_start = app._record_start
        app._tray = mock.Mock()
        with mock.patch.object(app, "_spawn_recorder") as spawn:
            with mock.patch("dictation.graceful_stop_recorder"):
                with mock.patch.object(app, "_stage_chunk") as stage:
                    app._finish_recording()
        spawn.assert_not_called()
        self.assertFalse(app._recording)
        stage.assert_called_once()
        self.assertEqual(stage.call_args[0][0], "/tmp/last.wav")

    def test_chunks_enter_worker_in_seq_order(self) -> None:
        from session_store import ChunkJob

        app = self._make_app()
        app._chunk_queue = queue.Queue()
        app._next_submit = 0
        app._staged = {}
        app._session_paths = {3: Path("/tmp/session")}
        app._store = mock.Mock()
        app._store.ingest.side_effect = lambda _session, seq, wav, duration, **kwargs: (
            ChunkJob(
                Path("/tmp/session"), seq, wav, duration, True, kwargs["paste_session"]
            )
        )
        app._stage_chunk("/tmp/b.wav", 1.0, 1, 3)
        self.assertTrue(app._chunk_queue.empty())
        app._stage_chunk("/tmp/a.wav", 1.0, 0, 3)
        self.assertEqual(app._chunk_queue.get_nowait().wav_path, Path("/tmp/a.wav"))
        self.assertEqual(app._chunk_queue.get_nowait().wav_path, Path("/tmp/b.wav"))

    def test_failed_chunk_persistence_does_not_block_later_chunk(self) -> None:
        from session_store import ChunkJob

        app = self._make_app()
        app._chunk_queue = queue.Queue()
        app._next_submit = 0
        app._staged = {}
        app._session_paths = {3: Path("/tmp/session")}
        later = ChunkJob(Path("/tmp/session"), 1, Path("/tmp/b.wav"), 1.0, True, 3)
        app._store = mock.Mock()
        app._store.ingest.side_effect = [OSError("disk error"), later]

        app._stage_chunk("/tmp/a.wav", 1.0, 0, 3)
        app._stage_chunk("/tmp/b.wav", 1.0, 1, 3)

        self.assertIs(app._chunk_queue.get_nowait(), later)
        app._store.record_gap.assert_called_once_with(
            Path("/tmp/session"), 0, 1.0, "disk error"
        )

    def test_run_server_kills_hung_inference_sockets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            home = temp / "home"
            config_dir = home / ".config/whisper-dictation"
            config_dir.mkdir(parents=True)
            repo = temp / "repo"
            (repo / "build-sycl/bin").mkdir(parents=True)
            (repo / "models").mkdir()
            (repo / "samples").mkdir()
            (repo / "models/ggml-small.en.bin").write_bytes(b"model")
            (repo / "samples/jfk.wav").write_bytes(b"wav")
            event_log = temp / "events.log"
            server_pid_file = temp / "server.pid"
            setvars = temp / "setvars.sh"
            setvars.write_text("export FAKE_ONEAPI_READY=1\n", encoding="utf-8")
            (config_dir / "install.env").write_text(
                f'WHISPER_REPO_ROOT="{repo}"\n', encoding="utf-8"
            )
            (config_dir / "config.env").write_text(
                'WHISPER_MODEL="small.en"\n'
                'WHISPER_BUILD_DIR="build-sycl"\n'
                'WHISPER_ACCELERATOR="sycl"\n'
                f'WHISPER_ONEAPI_SETVARS="{setvars}"\n'
                'WHISPER_SERVER_URL="http://127.0.0.1:18178"\n'
                'WHISPER_SERVER_WARMUP="1"\n'
                f'WHISPER_SERVER_WARMUP_AUDIO="{repo / "samples/jfk.wav"}"\n'
                'WHISPER_SERVER_WARMUP_TIMEOUT="3"\n'
                'WHISPER_INFERENCE_WATCHDOG_SEC="1"\n'
                'WHISPER_SERVER_RECYCLE_SEC="0"\n'
                'WHISPER_WATCHDOG_POLL_SEC="0.2"\n',
                encoding="utf-8",
            )
            write_executable(
                repo / "build-sycl/bin/whisper-server",
                "#!/usr/bin/env bash\n"
                'printf "%s\\n" "$$" > "${SERVER_PID_FILE}"\n'
                'trap \'printf "server-stop\\n" >> "${EVENT_LOG}"; exit 0\' TERM INT\n'
                "while true; do sleep 0.1; done\n",
            )
            fake_bin = temp / "bin"
            write_executable(
                fake_bin / "curl",
                "#!/usr/bin/env bash\n"
                "output=''\n"
                "previous=''\n"
                'for arg in "$@"; do\n'
                '  [[ "${previous}" == "-o" ]] && output="${arg}"\n'
                '  previous="${arg}"\n'
                "done\n"
                'if [[ -n "${output}" ]]; then printf \'{"text":"warm"}\' > "${output}"; else printf \'{"text":"warm"}\'; fi\n',
            )
            write_executable(
                fake_bin / "ss",
                "#!/usr/bin/env bash\nprintf 'ESTAB 0 0 127.0.0.1:18178 127.0.0.1:9\\n'\n",
            )
            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "EVENT_LOG": str(event_log),
                    "SERVER_PID_FILE": str(server_pid_file),
                    "PATH": f"{fake_bin}:/usr/bin:/bin",
                }
            )
            result = subprocess.run(
                ["bash", str(SCRIPT_DIR / "run-server.sh")],
                text=True,
                capture_output=True,
                env=env,
                timeout=8,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("inference watchdog", result.stderr)
            server_pid = int(server_pid_file.read_text(encoding="utf-8").strip())
            with self.assertRaises(ProcessLookupError):
                os.kill(server_pid, 0)


if __name__ == "__main__":
    unittest.main()
