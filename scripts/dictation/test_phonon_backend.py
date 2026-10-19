"""Production adapter contract; local HTTP fixtures, no microphone or live paste."""

import importlib
import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from dictation import Dictation
from vocab_prompt import apply_replacements, build_phonon_hotwords, load_all_vocabulary

SCRIPT_DIR = Path(__file__).resolve().parent


class PhononBackendTests(unittest.TestCase):
    def test_merged_dictionary_keeps_phrases_and_excludes_mishearings(self):
        with tempfile.TemporaryDirectory() as raw:
            repo, user = Path(raw) / "repo.txt", Path(raw) / "user.txt"
            repo.write_text("Phonon-2 => fun on 2\npull request\nVantage\n")
            user.write_text("Ada Lovelace\nvantage\n")
            with (
                mock.patch("vocab_prompt.repo_vocabulary_path", return_value=repo),
                mock.patch("vocab_prompt.user_vocabulary_path", return_value=user),
            ):
                words = build_phonon_hotwords({"test": "1"})
            self.assertEqual(
                words, ["Phonon-2", "pull request", "Vantage", "Ada Lovelace"]
            )

    def test_observed_phonon_mishearings_are_boundary_safe(self):
        terms, replacements = load_all_vocabulary()
        self.assertIn("Phonon-2", terms)
        self.assertNotIn("fun on 2", [term.lower() for term in terms])
        self.assertEqual(
            apply_replacements(
                "Fun on 2, phono-2, phonon 2, fun on two, phonon two.", replacements
            ),
            "Phonon-2, Phonon-2, Phonon-2, Phonon-2, Phonon-2.",
        )
        self.assertEqual(
            apply_replacements("Have fun on 20.", replacements), "Have fun on 20."
        )

    def app(self, **changes):
        cfg = {
            "DICTATION_ENGINE": "phonon2",
            "PHONON_DEVICE": "cpu",
            "WHISPER_BACKEND": "cli",
            "WHISPER_SESSION_DIR": self.tmp.name,
            "WHISPER_VOCABULARY_FILE": str(Path(self.tmp.name) / "no-overlay"),
        }
        cfg.update(changes)
        with mock.patch("dictation.threading.Thread.start"):
            app = Dictation(cfg)
        app._server_health_ok = mock.Mock(return_value=True)
        return app

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_phonon_selects_warm_server_and_ignores_whisper_context(self):
        app = self.app(WHISPER_AUDIO_CTX="1", WHISPER_BEAM_SIZE="-1")
        self.assertEqual(app.engine, "phonon2")
        self.assertEqual(app.backend, "server")
        self.assertIsNone(app.audio_ctx)
        self.assertIsNone(app.beam_size)

    def test_phonon_capture_does_not_require_any_whisper_assets(self):
        for continuous in ("0", "1"):
            with self.subTest(continuous=continuous):
                app = self.app(
                    CONTINUOUS_CAPTURE=continuous, WHISPER_HOME=self.tmp.name
                )
                app._notify = mock.Mock()
                proc = mock.Mock(pid=12345)
                proc.poll.return_value = None
                with (
                    mock.patch.object(
                        app, "_spawn_recorder", return_value=(proc, "/tmp/unused.wav")
                    ),
                    mock.patch.object(
                        app, "_spawn_stream_recorder", return_value=(proc, b"")
                    ),
                    mock.patch("dictation.threading.Thread.start"),
                    mock.patch.object(app, "_start_recorder_watch"),
                    mock.patch.object(app._tray, "set_recording"),
                ):
                    app._start_recording(target=None)
                self.assertTrue(app._recording, app._notify.call_args_list)

    def test_invalid_engine_and_device_refused_before_worker_start(self):
        for cfg in ({"DICTATION_ENGINE": "typo"}, {"PHONON_DEVICE": "sycl"}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                self.app(**cfg)

    def test_whisper_remains_the_default(self):
        app = self.app(DICTATION_ENGINE="whisper")
        self.assertEqual(app.engine, "whisper")
        self.assertEqual(app.backend, "cli")

    def test_no_whisper_fallback_or_beam_parameters_for_phonon(self):
        app = self.app()
        app._transcribe_server = mock.Mock(side_effect=[",", "good words"])
        with mock.patch.object(app, "_transcribe_cli") as cli:
            self.assertEqual(
                app._transcribe("/tmp/audio.wav", beam_size=5, audio_ctx=512),
                "good words",
            )
        self.assertEqual(
            app._transcribe_server.call_args_list,
            [mock.call("/tmp/audio.wav"), mock.call("/tmp/audio.wav")],
        )
        cli.assert_not_called()

    def test_exhausted_phonon_is_retryable_not_empty_success(self):
        app = self.app()
        app._transcribe_server = mock.Mock(return_value=None)
        with (
            mock.patch.object(app, "_restart_whisper_server", return_value=True),
            mock.patch.object(app, "_transcribe_cli") as cli,
        ):
            with self.assertRaisesRegex(RuntimeError, "Phonon"):
                app._transcribe("/tmp/audio.wav")
        cli.assert_not_called()

    def test_wrong_device_health_does_not_accept_a_valid_transcript(self):
        app = self.app(PHONON_DEVICE="cuda")
        app._server_health_ok.return_value = False
        app._transcribe_server = mock.Mock(return_value="valid text from wrong device")
        with mock.patch.object(app, "_restart_whisper_server", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Phonon"):
                app._transcribe("/tmp/audio.wav")
        app._transcribe_server.assert_not_called()

    def test_wrong_device_health_can_recover_only_after_matching_restart(self):
        app = self.app()
        app._server_health_ok.side_effect = [False, True]
        app._transcribe_server = mock.Mock(return_value="recovered on selected device")
        with mock.patch.object(
            app, "_restart_whisper_server", return_value=True
        ) as restart:
            self.assertEqual(
                app._transcribe("/tmp/audio.wav"), "recovered on selected device"
            )
        restart.assert_called_once()

    def test_failed_middle_chunk_keeps_audio_and_does_not_block_later_chunk(self):
        app = self.app()
        app._notify = mock.Mock()
        app.replacements = []
        session = app._store.start_session(capture_mode="continuous")
        app._transcribe_server = mock.Mock(
            side_effect=["first chunk", None, None, "last chunk"]
        )
        with mock.patch.object(app, "_restart_whisper_server", return_value=False):
            for index in range(3):
                source = Path(self.tmp.name) / f"source-{index}.wav"
                with wave.open(str(source), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(b"\x00\x10" * 16000)
                job = app._store.ingest(session, index, source, 1, paste=False)
                app._deliver_chunk(job)
        app._store.stop_session(session)
        manifest = json.loads((session / "manifest.json").read_text())
        self.assertEqual(
            [chunk["status"] for chunk in manifest["chunks"]],
            ["complete", "failed", "complete"],
        )
        self.assertTrue((session / "chunk-0001.wav").is_file())
        self.assertEqual([job.chunk_index for job in app._store.recoverable()], [1])
        transcript = (session / "transcript.txt").read_text()
        self.assertIn("first chunk", transcript)
        self.assertIn("last chunk", transcript)

    def test_canonical_dictionary_has_a_hard_limit(self):
        from vocab_prompt import build_phonon_hotwords

        with mock.patch(
            "vocab_prompt.load_all_vocabulary", return_value=(["one"] * 26, [])
        ):
            with self.assertRaisesRegex(ValueError, "25"):
                build_phonon_hotwords({"test": "1"})

    def test_health_requires_actual_model_and_device(self):
        backend = importlib.import_module("phonon_backend")
        good = {
            "status": "ok",
            "model": "FermionResearch/Phonon-2",
            "decode": {"engine": "cpu"},
        }
        self.assertTrue(backend.health_matches(good, "cpu"))
        self.assertFalse(backend.health_matches(good, "cuda"))
        self.assertFalse(backend.health_matches({"status": "ok"}, "cpu"))
        self.assertTrue(
            backend.health_matches(
                {
                    "status": "ok",
                    "model": "phonon-2",
                    "gpu": "NVIDIA GB10",
                    "path": "CUDA-graph TDT",
                },
                "cuda",
            )
        )
        self.assertFalse(backend.health_matches({**good, "model": "phonon-1"}, "cpu"))

    def test_real_http_transport_sends_only_phonon_fields(self):
        backend = importlib.import_module("phonon_backend")
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                payload = json.dumps(
                    {"status": "ok", "model": "phonon-2", "decode": {"engine": "cpu"}}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                received.append(
                    (self.path, self.rfile.read(int(self.headers["Content-Length"])))
                )
                payload = json.dumps({"text": "actual words"}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            audio = Path(self.tmp.name) / "sample.wav"
            audio.write_bytes(b"RIFF fixture audio")
            with mock.patch.dict(
                os.environ,
                {
                    "http_proxy": "http://127.0.0.1:1",
                    "ALL_PROXY": "http://127.0.0.1:1",
                    "NO_PROXY": "",
                    "no_proxy": "",
                },
            ):
                url = f"http://127.0.0.1:{server.server_port}"
                text = backend.transcribe(
                    url, str(audio), 5, ["Phonon-2", "pull request"]
                )
                self.assertTrue(backend.health(url, "cpu"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)
        self.assertEqual(text, "actual words")
        self.assertEqual(received[0][0], "/v1/audio/transcriptions")
        body = received[0][1].decode()
        self.assertIn('name="hotwords"', body)
        self.assertIn('["Phonon-2", "pull request"]', body)
        self.assertNotIn("beam_size", body)
        self.assertNotIn("audio_ctx", body)

    def test_http_errors_and_malformed_or_truncated_responses_not_text(self):
        backend = importlib.import_module("phonon_backend")
        for raw in (
            '{"text":null}',
            '{"text":23}',
            "not json",
            '{"error":{"message":"busy"}}',
            '{"text":"partial","x_fermion":{"truncated":true}}',
        ):
            with (
                self.subTest(raw=raw),
                mock.patch(
                    "subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, raw, ""),
                ),
            ):
                self.assertIsNone(
                    backend.transcribe("http://127.0.0.1:8180", "/tmp/a.wav", 5, [])
                )

    def test_non_loopback_target_refused(self):
        backend = importlib.import_module("phonon_backend")
        with self.assertRaises(ValueError):
            backend.transcribe("http://remote-host:8180", "/tmp/a.wav", 5, [])


class PhononRuntimeTests(unittest.TestCase):
    def executable(self, path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
        path.chmod(0o755)

    def fixture(self, root, device="cpu"):
        home = root / "home"
        config = home / ".config/whisper-dictation"
        config.mkdir(parents=True)
        repo = root / "repo"
        (repo / "samples").mkdir(parents=True)
        (repo / "samples/jfk.wav").write_bytes(b"fixture")
        (config / "install.env").write_text(f'WHISPER_REPO_ROOT="{repo}"\n')
        (config / "config.env").write_text(
            f'DICTATION_ENGINE="phonon2"\nPHONON_DEVICE="{device}"\n'
            f'PHONON_VENV="{root / "runtime"}"\n'
            'PHONON_CUDA_IMAGE="fixture:cuda"\n'
            'WHISPER_ACCELERATOR="sycl"\nWHISPER_ONEAPI_SETVARS="/missing-oneapi"\n'
            'WHISPER_SERVER_URL="http://127.0.0.1:18188"\n'
            'WHISPER_INFERENCE_WATCHDOG_SEC="0"\nWHISPER_SERVER_RECYCLE_SEC="0"\n'
            'WHISPER_SERVER_WARMUP_TIMEOUT="5"\n'
            'WHISPER_SERVER_WARMUP="0"\n'
        )
        bins = root / "bin"
        log = root / "events"
        env = os.environ.copy()
        # Do not inherit a user's real GPU/profile override into controlled tests.
        for key in (
            "DICTATION_ENGINE",
            "PHONON_DEVICE",
            "WHISPER_REPO_ROOT",
            "WHISPER_SERVER_URL",
            "WHISPER_BUILD_DIR",
            "WHISPER_ACCELERATOR",
        ):
            env.pop(key, None)
        env.update(
            HOME=str(home),
            PATH=f"{bins}:/usr/bin:/bin",
            EVENT_LOG=str(log),
            NOTIFY_SOCKET="fixture",
            XDG_RUNTIME_DIR=str(root),
        )
        self.executable(bins / "systemd-notify", 'echo "$*" >> "$EVENT_LOG"\n')
        payload = {"status": "ok", "model": "phonon-2", "text": "warm words"}
        if device == "cpu":
            payload["decode"] = {"engine": "cpu"}
        else:
            payload.update(gpu="NVIDIA GB10", path="CUDA-graph TDT")
        self.executable(
            bins / "curl",
            'echo "curl $*" >> "$EVENT_LOG"\n'
            + "printf '%s' '"
            + json.dumps(payload)
            + "'\n",
        )
        return env, log

    def test_cpu_and_cuda_server_readiness_and_owned_shutdown(self):
        for device in ("cpu", "cuda"):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                env, log = self.fixture(root, device)
                child = (
                    'echo "$*" >> "$EVENT_LOG"\n'
                    "trap 'exit 0' TERM INT\nwhile true; do sleep 0.1; done\n"
                )
                if device == "cpu":
                    self.executable(root / "runtime/bin/fermion", child)
                else:
                    self.executable(
                        root / "bin/docker",
                        'echo "$*" >> "$EVENT_LOG"\n'
                        'if [[ "$1" == stop ]]; then exit 0; fi\n'
                        'while [[ "$1" != --cidfile ]]; do shift; done\n'
                        'echo fixture-owned-cid > "$2"\n' + child,
                    )
                proc = subprocess.Popen(
                    ["bash", str(SCRIPT_DIR / "run-server.sh")],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline:
                        if log.exists() and "--ready" in log.read_text():
                            break
                        if proc.poll() is not None:
                            self.fail(proc.communicate())
                        time.sleep(0.05)
                    self.assertIn("--ready", log.read_text())
                finally:
                    proc.terminate()
                    stdout, stderr = proc.communicate(timeout=12)
                self.assertEqual(proc.returncode, 0, stdout + stderr)
                events = log.read_text()
                self.assertIn("/v1/audio/transcriptions", events)
                self.assertLess(
                    events.index("/v1/audio/transcriptions"), events.index("--ready")
                )
                if device == "cpu":
                    self.assertIn(
                        "serve phonon-2 --host 127.0.0.1 --port 18188", events
                    )
                else:
                    self.assertIn("--gpus all", events)
                    self.assertIn("127.0.0.1:18188:8000", events)
                    self.assertIn("stop --time 5 fixture-owned-cid", events)

    def test_disabled_warmup_cannot_publish_ready_after_failed_inference(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            env, log = self.fixture(root)
            self.executable(
                root / "runtime/bin/fermion",
                "trap 'exit 0' TERM INT\nwhile true; do sleep 0.1; done\n",
            )
            self.executable(
                root / "bin/curl",
                'echo "curl $*" >> "$EVENT_LOG"\n'
                '[[ "$*" != *"/v1/audio/transcriptions"* ]] || exit 22\n'
                'printf \'%s\' \'{"status":"ok","model":"phonon-2","decode":{"engine":"cpu"}}\'\n',
            )
            proc = subprocess.Popen(
                ["bash", str(SCRIPT_DIR / "run-server.sh")],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    if proc.poll() is not None or (
                        log.exists() and "--ready" in log.read_text()
                    ):
                        break
                    time.sleep(0.05)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                stdout, stderr = proc.communicate(timeout=12)
            self.assertNotEqual(proc.returncode, 0, stdout + stderr)
            self.assertIn("/v1/audio/transcriptions", log.read_text())
            self.assertNotIn("--ready", log.read_text())

    def test_cuda_installer_builds_native_image_and_requires_gpu(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            env, log = self.fixture(root, "cuda")
            self.executable(root / "bin/docker", 'echo "$*" >> "$EVENT_LOG"\n')
            result = subprocess.run(
                ["bash", str(SCRIPT_DIR / "install-phonon.sh"), "cuda"],
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = log.read_text()
            self.assertIn("build --pull", commands)
            self.assertNotIn("--platform linux/amd64", commands)
            self.assertIn("--gpus all --entrypoint python", commands)
            self.assertIn("torch.cuda.is_available()", commands)


if __name__ == "__main__":
    unittest.main()
