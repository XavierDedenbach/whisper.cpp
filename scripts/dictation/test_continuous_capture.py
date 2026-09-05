#!/usr/bin/env python3
"""Integration tests for the opt-in single-owner recorder path."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import sys
import tempfile
import threading
import unittest
import wave
from array import array
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from dictation import Dictation  # noqa: E402
from session_store import ChunkJob, SessionStore  # noqa: E402


def write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def wav_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as audio:
        return audio.readframes(audio.getnframes())


def write_wav(path: Path, pcm: bytes, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm)


class ContinuousCaptureTests(unittest.TestCase):
    def test_stream_recorder_command_is_raw_s16_mono_and_source_specific(self) -> None:
        from dictation import build_stream_recorder_cmd

        available = mock.Mock(returncode=0)
        with mock.patch("dictation.subprocess.run", return_value=available):
            command = build_stream_recorder_cmd(
                "test.monitor", {"RECORDER_LATENCY_MSEC": "17"}
            )
        self.assertEqual(
            command,
            [
                "parecord",
                "--latency-msec=17",
                "--rate=16000",
                "--channels=1",
                "--format=s16le",
                "--raw",
                "-d",
                "test.monitor",
            ],
        )

    def test_unreaped_stream_startup_child_stops_retrying(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            app = self._make_app(root, root / "unused-recorder")
            proc = mock.Mock(pid=4321, stdout=mock.Mock(), stderr=None)
            proc.poll.return_value = None

            with mock.patch("dictation.wake_audio_source"):
                with mock.patch(
                    "dictation.subprocess.Popen", return_value=proc
                ) as popen:
                    with mock.patch("dictation.RECORDER_START_TIMEOUT_SEC", 0):
                        with mock.patch(
                            "dictation.graceful_stop_recorder", return_value=False
                        ):
                            with mock.patch.object(
                                app, "_retain_rejected_recorder"
                            ) as retain:
                                result = app._spawn_stream_recorder()

            self.assertIsNone(result)
            popen.assert_called_once()
            retain.assert_called_once_with(proc, None)

    def _make_app(self, root: Path, recorder: Path) -> Dictation:
        cfg = {
            "WHISPER_HOME": str(REPO_ROOT),
            "WHISPER_BUILD_DIR": "build-sycl",
            "WHISPER_MODEL": "small.en",
            "WHISPER_SESSION_DIR": str(root / "sessions"),
            "CONTINUOUS_CAPTURE": "1",
            "STREAM_SEGMENT_TARGET_SEC": "0.04",
            "STREAM_SEGMENT_MIN_SEC": "0.03",
            "STREAM_SEGMENT_MAX_SEC": "0.05",
            "TRANSCRIPTION_TRAILING_SILENCE_SEC": "0.5",
            "WHISPER_BEAM_SIZE": "5",
            "MIN_AUDIO_RMS": "80",
            "TRAY_INDICATOR": "0",
        }
        with mock.patch("dictation.build_recorder_cmd", return_value=["legacy"]):
            with mock.patch(
                "dictation.build_stream_recorder_cmd",
                return_value=[str(recorder)],
                create=True,
            ):
                with mock.patch.object(Dictation, "_transcribe_worker"):
                    app = Dictation(cfg)
        app._tray = mock.Mock()
        app._notify = mock.Mock()
        return app

    def _make_staging_app(self, root: Path) -> tuple[Dictation, Path, int]:
        app = self._make_app(root, root / "unused-recorder")
        session_id = 1
        session = app._store.start_session(capture_mode="continuous")
        app._session_id = session_id
        app._session_paths[session_id] = session
        app._active_session = session
        return app, session, session_id

    def test_user_stop_drains_pipe_and_partitions_one_process_exactly_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            recorder = root / "recorder.py"
            starts = root / "starts"
            first_samples = 1600
            final_samples = 800
            expected = (b"\x10\x00" * first_samples) + (b"\x20\x00" * final_samples)
            write_executable(
                recorder,
                "#!/usr/bin/env python3\n"
                "import signal, sys, time\n"
                f"open({str(starts)!r}, 'a', encoding='utf-8').write('start\\n')\n"
                "def stop(_signum, _frame):\n"
                f"    sys.stdout.buffer.write(b'\\x20\\x00' * {final_samples})\n"
                "    sys.stdout.buffer.flush()\n"
                "    raise SystemExit(0)\n"
                "signal.signal(signal.SIGINT, stop)\n"
                f"sys.stdout.buffer.write(b'\\x10\\x00' * {first_samples})\n"
                "sys.stdout.buffer.flush()\n"
                "while True: time.sleep(0.01)\n",
            )
            app = self._make_app(root, recorder)
            with mock.patch("dictation.wake_audio_source"):
                app._start_recording()
                self.assertTrue(app._recording)
                active_pid = app._record_proc.pid
                app._finish_recording()

            self.assertTrue(app._stream_done.wait(2))
            self.assertIsNone(app._record_proc)
            self.assertFalse(app._recording)
            self.assertEqual(starts.read_text(encoding="utf-8").splitlines(), ["start"])
            self.assertEqual(app._stream_recorder_pids, [active_pid])
            session = next((root / "sessions").iterdir())
            manifest = json.loads((session / "manifest.json").read_text())
            chunks = sorted(session.glob("chunk-*.wav"))
            self.assertTrue(chunks)
            self.assertTrue(all(wav_pcm(path) for path in chunks))
            self.assertEqual(b"".join(wav_pcm(path) for path in chunks), expected)
            self.assertEqual(app._stream_input_bytes, len(expected))
            self.assertEqual(app._stream_output_bytes, len(expected))
            self.assertFalse(manifest["recording"])
            self.assertEqual(manifest["policy"]["capture_mode"], "continuous")

    def test_unexpected_exit_drains_unread_data_and_does_not_reconnect(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            recorder = root / "recorder.py"
            starts = root / "starts"
            sample_count = 128_000
            expected = b"\x33\x00" * sample_count
            write_executable(
                recorder,
                "#!/usr/bin/env python3\n"
                "import sys\n"
                f"open({str(starts)!r}, 'a', encoding='utf-8').write('start\\n')\n"
                f"sys.stdout.buffer.write(b'\\x33\\x00' * {sample_count})\n"
                "sys.stdout.buffer.flush()\n"
                "raise SystemExit(17)\n",
            )
            app = self._make_app(root, recorder)
            app.stream_segment_target = 2.0
            app.stream_segment_minimum = 1.0
            app.stream_segment_maximum = 3.0
            with mock.patch("dictation.wake_audio_source"):
                app._start_recording()

            self.assertTrue(app._stream_done.wait(5))
            session = next((root / "sessions").iterdir())
            chunks = sorted(session.glob("chunk-*.wav"))
            self.assertEqual(b"".join(wav_pcm(path) for path in chunks), expected)
            self.assertEqual(starts.read_text(encoding="utf-8").splitlines(), ["start"])
            self.assertEqual(len(app._stream_recorder_pids), 1)
            self.assertFalse(app._recording)
            self.assertIsNone(app._record_proc)
            manifest = json.loads((session / "manifest.json").read_text())
            self.assertFalse(manifest["recording"])

    def test_continuous_recorder_drains_stderr_beyond_pipe_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            recorder = root / "recorder.py"
            first_samples = 2400
            final_samples = 1600
            expected = (b"\x31\x00" * first_samples) + (b"\x32\x00" * final_samples)
            write_executable(
                recorder,
                "#!/usr/bin/env python3\n"
                "import sys, time\n"
                "sys.stderr.buffer.write(b'E' * (2 * 1024 * 1024) + b'\\n')\n"
                "sys.stderr.buffer.flush()\n"
                f"sys.stdout.buffer.write(b'\\x31\\x00' * {first_samples})\n"
                "sys.stdout.buffer.flush()\n"
                "sys.stderr.buffer.write(b'T' * (2 * 1024 * 1024) + b'\\n')\n"
                "sys.stderr.buffer.write(b'useful recorder diagnostic\\n')\n"
                "sys.stderr.buffer.flush()\n"
                f"sys.stdout.buffer.write(b'\\x32\\x00' * {final_samples})\n"
                "sys.stdout.buffer.flush()\n"
                "time.sleep(0.1)\n"
                "raise SystemExit(17)\n",
            )
            app = self._make_app(root, recorder)
            with mock.patch("dictation.wake_audio_source"):
                with mock.patch("dictation.RECORDER_START_ATTEMPTS", 1):
                    app._start_recording()
            proc = app._record_proc

            self.assertIsNotNone(proc)
            self.assertTrue(app._stream_done.wait(5))
            session = next((root / "sessions").iterdir())
            chunks = sorted(session.glob("chunk-*.wav"))
            self.assertEqual(b"".join(wav_pcm(path) for path in chunks), expected)
            self.assertIn("useful recorder diagnostic", app._recorder_exit_detail(proc))

    def test_exact_boundary_uses_final_marker_without_empty_wav(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            recorder = root / "recorder.py"
            write_executable(
                recorder,
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "sys.stdout.buffer.write(b'\\x55\\x00' * 800)\n"
                "sys.stdout.buffer.flush()\n",
            )
            app = self._make_app(root, recorder)
            with mock.patch("dictation.wake_audio_source"):
                app._start_recording()
            self.assertTrue(app._stream_done.wait(2))
            session = next((root / "sessions").iterdir())
            chunks = sorted(session.glob("chunk-*.wav"))
            self.assertTrue(chunks)
            self.assertTrue(all(path.stat().st_size > 44 for path in chunks))
            queued = []
            while not app._chunk_queue.empty():
                queued.append(app._chunk_queue.get_nowait())
            finalizers = [item for item in queued if getattr(item, "finalize", False)]
            markers = [
                item for item in queued if item.__class__.__name__ == "SessionPasteJob"
            ]
            self.assertEqual(len(finalizers) + len(markers), 1)

    def test_second_chunk_ingest_failure_preserves_it_and_all_later_chunks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            recorder = root / "recorder.py"
            sample_count = 4800
            expected = b"\x44\x00" * sample_count
            write_executable(
                recorder,
                "#!/usr/bin/env python3\n"
                "import sys\n"
                f"sys.stdout.buffer.write(b'\\x44\\x00' * {sample_count})\n"
                "sys.stdout.buffer.flush()\n",
            )
            app = self._make_app(root, recorder)
            real_ingest = app._store.ingest

            def fail_second(session, index, source, duration, **kwargs):
                if index == 1:
                    raise OSError("injected second chunk failure")
                return real_ingest(session, index, source, duration, **kwargs)

            with mock.patch.object(app._store, "ingest", side_effect=fail_second):
                with mock.patch("dictation.wake_audio_source"):
                    app._start_recording()
                self.assertTrue(app._stream_done.wait(3))

            session = next((root / "sessions").iterdir())
            chunks = sorted(session.glob("chunk-*.wav"))
            self.assertGreater(len(chunks), 2)
            self.assertEqual(b"".join(wav_pcm(path) for path in chunks), expected)
            manifest = json.loads((session / "manifest.json").read_text())
            self.assertEqual(manifest["chunks"][1]["status"], "missing")

            recovered = SessionStore(root / "sessions").recoverable()
            self.assertEqual(
                [job.chunk_index for job in recovered],
                [int(path.stem.split("-")[1]) for path in chunks],
            )

    def test_pre_persistence_failure_advances_order_and_delivers_final_marker(
        self,
    ) -> None:
        for failure_site in ("mkstemp", "write_pcm_wav"):
            with self.subTest(failure_site=failure_site):
                with tempfile.TemporaryDirectory() as raw:
                    root = Path(raw)
                    app, session, session_id = self._make_staging_app(root)
                    pcm = b"\x21\x00" * 800
                    target = (
                        "dictation.tempfile.mkstemp"
                        if failure_site == "mkstemp"
                        else "dictation.write_pcm_wav"
                    )
                    with mock.patch(
                        target, side_effect=OSError("injected wrap failure")
                    ):
                        app._stage_stream_pcm(pcm, session_id)

                    app._stage_stream_pcm(pcm, session_id)
                    app._stage_stream_marker(session_id)

                    self.assertEqual(app._next_submit, 3)
                    self.assertEqual(app._chunk_queue.qsize(), 2)
                    manifest = json.loads((session / "manifest.json").read_text())
                    self.assertEqual(
                        [
                            (int(chunk["index"]), chunk["status"])
                            for chunk in manifest["chunks"]
                        ],
                        [(0, "missing"), (1, "queued")],
                    )
                    self.assertIn(
                        "[missing: chunk 0000 — injected wrap failure]",
                        (session / "transcript.txt").read_text(encoding="utf-8"),
                    )

                    app._deliver_chunk = mock.Mock(
                        side_effect=lambda job: app._store.complete(job, "later words")
                    )
                    app._deliver_completed_session_safely = mock.Mock()
                    worker = threading.Thread(
                        target=app._transcribe_worker, daemon=True
                    )
                    worker.start()
                    app._chunk_queue.join()
                    app._chunk_queue.put(None)
                    worker.join(timeout=2)

                    app._deliver_completed_session_safely.assert_called_once_with(
                        session, session_id
                    )
                    manifest = json.loads((session / "manifest.json").read_text())
                    self.assertEqual(manifest["chunks"][1]["status"], "complete")

    def test_post_commit_rebuild_failure_requeues_live_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            app, session, session_id = self._make_staging_app(root)
            source = root / "source.wav"
            write_wav(source, b"\x22\x00" * 800)
            real_rebuild = app._store._rebuild
            rebuild_calls = 0

            def fail_once(*args, **kwargs):
                nonlocal rebuild_calls
                rebuild_calls += 1
                if rebuild_calls == 1:
                    raise OSError("injected post-commit rebuild failure")
                return real_rebuild(*args, **kwargs)

            with mock.patch.object(app._store, "_rebuild", side_effect=fail_once):
                persisted = app._stage_chunk(
                    str(source), 0.05, 0, session_id, finalize=True
                )

            self.assertTrue(persisted)
            queued = app._chunk_queue.get_nowait()
            self.assertIsInstance(queued, ChunkJob)
            self.assertEqual(queued.chunk_index, 0)
            self.assertTrue(queued.finalize)
            self.assertFalse(source.exists())
            manifest = json.loads((session / "manifest.json").read_text())
            self.assertEqual(manifest["chunks"][0]["status"], "queued")

    def test_consecutive_ingest_failures_keep_live_markers_until_restart(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            app, session, session_id = self._make_staging_app(root)
            pcm = b"\x24\x00" * 800
            real_ingest = app._store.ingest

            def fail_first_two(session_path, index, source, duration, **kwargs):
                if index < 2:
                    raise OSError(f"injected ingest failure {index}")
                return real_ingest(session_path, index, source, duration, **kwargs)

            with mock.patch.object(app._store, "ingest", side_effect=fail_first_two):
                app._stage_stream_pcm(pcm, session_id)
                app._stage_stream_pcm(pcm, session_id)
                app._stage_stream_pcm(pcm, session_id)
            app._stage_stream_marker(session_id)

            transcript = (session / "transcript.txt").read_text(encoding="utf-8")
            self.assertIn(
                "[missing: chunk 0000 — injected ingest failure 0]", transcript
            )
            self.assertIn(
                "[missing: chunk 0001 — injected ingest failure 1]", transcript
            )
            self.assertEqual(app._next_submit, 4)
            self.assertEqual(app._chunk_queue.qsize(), 2)

            app._deliver_chunk = mock.Mock(
                side_effect=lambda job: app._store.complete(job, "later words")
            )
            app._deliver_completed_session_safely = mock.Mock()
            worker = threading.Thread(target=app._transcribe_worker, daemon=True)
            worker.start()
            app._chunk_queue.join()
            app._chunk_queue.put(None)
            worker.join(timeout=2)

            app._deliver_completed_session_safely.assert_called_once_with(
                session, session_id
            )
            self.assertIn(
                "later words",
                (session / "transcript.txt").read_text(encoding="utf-8"),
            )

            recovered = SessionStore(root / "sessions").recoverable()
            self.assertEqual([job.chunk_index for job in recovered], [0, 1])
            self.assertTrue(all(job.wav_path.is_file() for job in recovered))

    def test_failed_ingest_retention_falls_back_across_filesystems(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            staging_root = root / "temporary-mount"
            staging_root.mkdir()
            app, session, session_id = self._make_staging_app(root)
            pcm = b"\x23\x00" * 800
            real_mkstemp = tempfile.mkstemp
            real_replace = os.replace
            exdev_attempts = 0

            def local_mkstemp(*args, **kwargs):
                kwargs["dir"] = staging_root
                return real_mkstemp(*args, **kwargs)

            def replace_with_cross_device_failure(source, destination):
                nonlocal exdev_attempts
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    source_path.parent == staging_root
                    and destination_path.parent == session
                ):
                    exdev_attempts += 1
                    raise OSError(errno.EXDEV, "injected cross-device move")
                return real_replace(source, destination)

            with mock.patch("dictation.tempfile.mkstemp", side_effect=local_mkstemp):
                with mock.patch.object(
                    app._store,
                    "ingest",
                    side_effect=OSError("injected ingest failure"),
                ):
                    with mock.patch(
                        "session_store.os.replace",
                        side_effect=replace_with_cross_device_failure,
                    ):
                        app._stage_stream_pcm(pcm, session_id)

            self.assertEqual(exdev_attempts, 1)
            self.assertEqual(list(staging_root.iterdir()), [])
            recovered = SessionStore(root / "sessions").recoverable()
            self.assertEqual([job.chunk_index for job in recovered], [0])
            self.assertEqual(wav_pcm(recovered[0].wav_path), pcm)

    def test_numeric_probe_defaults_match_implemented_lg_profile(self) -> None:
        from numeric_continuity_probe import parse_args

        with mock.patch.object(sys, "argv", ["numeric_continuity_probe.py"]):
            args = parse_args()

        self.assertEqual(
            (
                args.target_seconds,
                args.minimum_seconds,
                args.maximum_seconds,
                args.trailing_silence_seconds,
                args.audio_ctx,
            ),
            (8.0, 7.0, 9.0, 0.5, 512),
        )


class ContinuousDeliveryTests(unittest.TestCase):
    def _job(self, root: Path, pcm: bytes):
        store = SessionStore(root / "sessions")
        session = store.start_session(
            capture_mode="continuous",
            beam_size=5,
            audio_ctx=512,
            trailing_silence_seconds=0.5,
            minimum_audio_rms=80.0,
        )
        source = root / "source.wav"
        write_wav(source, pcm)
        return store, store.ingest(session, 0, source, len(pcm) / 32000)

    @staticmethod
    def _app(store: SessionStore) -> Dictation:
        app = object.__new__(Dictation)
        app._store = store
        app._recording = False
        app._session_id = 0
        app.audio_source = ""
        app.min_record = 0.4
        app.min_audio_rms = 9999.0
        app.replacements = {}
        app._notify = mock.Mock()
        return app

    def test_raw_is_classified_then_temporary_padding_and_beam_are_used(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            pcm = array("h", [500] * 16000).tobytes()
            store, job = self._job(root, pcm)
            app = self._app(store)
            before = hashlib.sha256(job.wav_path.read_bytes()).hexdigest()
            observed: dict[str, object] = {}

            def transcribe(
                path: str,
                *,
                beam_size: int | None = None,
                audio_ctx: int | None = None,
            ) -> str:
                observed["path"] = path
                observed["exists_during_inference"] = Path(path).is_file()
                observed["beam_size"] = beam_size
                observed["audio_ctx"] = audio_ctx
                observed["duration"] = wave.open(path, "rb").getnframes() / 16000
                return "spoken words"

            app._transcribe = mock.Mock(side_effect=transcribe)
            app._deliver_chunk(job)

            self.assertTrue(observed["exists_during_inference"])
            self.assertEqual(observed["beam_size"], 5)
            self.assertEqual(observed["audio_ctx"], 512)
            self.assertAlmostEqual(observed["duration"], 1.5)
            self.assertNotEqual(Path(str(observed["path"])), job.wav_path)
            self.assertFalse(Path(str(observed["path"])).exists())
            self.assertEqual(
                hashlib.sha256(job.wav_path.read_bytes()).hexdigest(), before
            )
            self.assertEqual(store.completed_text(job.session_path), "spoken words")

    def test_silent_raw_tail_is_retained_and_never_padded_or_inferred(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store, job = self._job(root, b"\x00\x00" * 16000)
            app = self._app(store)
            app._transcribe = mock.Mock()
            with mock.patch("dictation.prepared_padded_wav", create=True) as prepare:
                app._deliver_chunk(job)

            app._transcribe.assert_not_called()
            prepare.assert_not_called()
            self.assertTrue(job.wav_path.is_file())
            manifest = json.loads((job.session_path / "manifest.json").read_text())
            self.assertEqual(manifest["chunks"][0]["status"], "silent")

    def test_configured_beam_is_the_first_sycl_server_attempt(self) -> None:
        app = object.__new__(Dictation)
        app.backend = "server"
        app._transcribe_server = mock.Mock(return_value="recognized")
        app._valid_server_text = Dictation._valid_server_text
        app._notify = mock.Mock()

        self.assertEqual(
            app._transcribe("/tmp/chunk.wav", beam_size=5, audio_ctx=512),
            "recognized",
        )
        app._transcribe_server.assert_called_once_with(
            "/tmp/chunk.wav", beam_size=5, audio_ctx=512
        )


if __name__ == "__main__":
    unittest.main()
