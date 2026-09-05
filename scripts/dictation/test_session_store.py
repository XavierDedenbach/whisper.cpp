#!/usr/bin/env python3
"""Durability and recovery tests for long dictation sessions."""

from __future__ import annotations

import json
import queue
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def write_wav(path: Path, frames: int = 1600) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes((b"\x00\x10" * frames))


class SessionStoreTests(unittest.TestCase):
    def test_completed_fragments_join_without_spacing_before_punctuation(self) -> None:
        from session_store import join_fragments

        self.assertEqual(
            join_fragments(["  hello world  ", "again", ". Next", "   "]),
            "hello world again. Next",
        )

    def test_silent_tail_is_retained_without_polluting_transcript(self) -> None:
        from session_store import SessionStore

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = SessionStore(root / "sessions")
            session = store.start_session(capture_mode="continuous")
            speech_source = root / "speech.wav"
            write_wav(speech_source)
            speech = store.ingest(session, 0, speech_source, 1.0)
            silent_source = root / "silent.wav"
            write_wav(silent_source)
            silent = store.ingest(session, 1, silent_source, 1.0)
            ignored_source = root / "ignored.wav"
            write_wav(ignored_source)
            ignored = store.ingest(session, 2, ignored_source, 1.0)
            store.complete(speech, "complete thought")
            store.terminal(silent, "silent", "RMS 0")
            store.terminal(ignored, "ignored", "too short")

            self.assertTrue(silent.wav_path.is_file())
            self.assertTrue(ignored.wav_path.is_file())
            self.assertEqual(
                (session / "transcript.txt").read_text(encoding="utf-8"),
                "complete thought\n",
            )

    def test_legacy_and_manifest_v1_silent_chunks_keep_markers(self) -> None:
        from session_store import SessionStore

        for manifest_version in (1, 2):
            with self.subTest(manifest_version=manifest_version):
                with tempfile.TemporaryDirectory() as raw:
                    root = Path(raw)
                    store = SessionStore(root / "sessions")
                    session = store.start_session()
                    jobs = []
                    for index in range(2):
                        source = root / f"source-{index}.wav"
                        write_wav(source)
                        jobs.append(store.ingest(session, index, source, 1.0))
                    if manifest_version == 1:
                        manifest_path = session / "manifest.json"
                        manifest = json.loads(manifest_path.read_text())
                        manifest["version"] = 1
                        manifest.pop("policy", None)
                        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                    store.terminal(jobs[0], "silent", "RMS 0")
                    store.terminal(jobs[1], "ignored", "too short")

                    self.assertEqual(
                        (session / "transcript.txt").read_text(encoding="utf-8"),
                        "[silent: chunk 0000 — RMS 0]\n"
                        "[ignored: chunk 0001 — too short]\n",
                    )

    def test_preserves_audio_and_rebuilds_ordered_transcript_after_failure(
        self,
    ) -> None:
        from session_store import SessionStore

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.wav"
            write_wav(source)
            store = SessionStore(root / "sessions")
            session = store.start_session()

            first = store.ingest(session, 0, source, 45.0)
            second_source = root / "second.wav"
            write_wav(second_source)
            second = store.ingest(session, 1, second_source, 45.0)
            third_source = root / "third.wav"
            write_wav(third_source)
            third = store.ingest(session, 2, third_source, 10.0)

            store.complete(first, "first words")
            store.fail(second, "corrupt", "invalid WAV")
            store.complete(third, "last words")

            self.assertTrue(first.wav_path.is_file())
            self.assertTrue(second.wav_path.is_file())
            self.assertTrue(third.wav_path.is_file())
            self.assertEqual(
                (session / "transcript.txt").read_text(encoding="utf-8"),
                "first words\n[corrupt: chunk 0001 — invalid WAV]\nlast words\n",
            )
            self.assertEqual(store.completed_text(session), "first words last words")
            payload = json.loads((session / "manifest.json").read_text())
            self.assertEqual(
                [item["status"] for item in payload["chunks"]],
                ["complete", "corrupt", "complete"],
            )

    def test_discovers_retryable_chunks_but_not_completed_or_exhausted(self) -> None:
        from session_store import SessionStore

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = SessionStore(root / "sessions", retry_limit=3)
            session = store.start_session()
            jobs = []
            for index in range(4):
                source = root / f"source-{index}.wav"
                write_wav(source)
                jobs.append(store.ingest(session, index, source, 45.0))
            store.complete(jobs[0], "done")
            store.fail(jobs[2], "failed", "once")
            store.fail(jobs[3], "failed", "one")
            store.fail(jobs[3], "failed", "two")
            store.fail(jobs[3], "failed", "three")

            recovered = store.recoverable(limit=32)

            self.assertEqual([job.chunk_index for job in recovered], [1, 2])
            self.assertTrue(all(not job.paste for job in recovered))
            manifest = json.loads((session / "manifest.json").read_text())
            self.assertEqual(manifest["chunks"][3]["status"], "exhausted")

    def test_manifest_v2_policy_survives_restart_recovery(self) -> None:
        from session_store import SessionStore

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = SessionStore(root / "sessions")
            session = store.start_session(
                capture_mode="continuous",
                beam_size=5,
                audio_ctx=512,
                trailing_silence_seconds=0.5,
                minimum_audio_rms=93.0,
            )
            source = root / "source.wav"
            write_wav(source)
            store.ingest(session, 0, source, 8.0)

            recovered = SessionStore(
                root / "sessions", legacy_minimum_audio_rms=237.5
            ).recoverable()

            self.assertEqual(len(recovered), 1)
            job = recovered[0]
            self.assertEqual(job.capture_mode, "continuous")
            self.assertEqual(job.beam_size, 5)
            self.assertEqual(job.audio_ctx, 512)
            self.assertEqual(job.trailing_silence_seconds, 0.5)
            self.assertEqual(job.minimum_audio_rms, 93.0)
            manifest = json.loads((session / "manifest.json").read_text())
            self.assertEqual(manifest["version"], 2)

    def test_manifest_v1_recovery_uses_legacy_policy_defaults(self) -> None:
        from session_store import SessionStore

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = SessionStore(root / "sessions")
            session = store.start_session()
            source = root / "source.wav"
            write_wav(source)
            store.ingest(session, 0, source, 8.0)
            manifest_path = session / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["version"] = 1
            manifest.pop("policy", None)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            job = SessionStore(
                root / "sessions", legacy_minimum_audio_rms=237.5
            ).recoverable()[0]

            self.assertEqual(job.capture_mode, "legacy")
            self.assertIsNone(job.beam_size)
            self.assertIsNone(job.audio_ctx)
            self.assertEqual(job.trailing_silence_seconds, 0.0)
            self.assertEqual(job.minimum_audio_rms, 237.5)

    def test_default_recovery_is_exhaustive_beyond_one_hundred_chunks(self) -> None:
        from session_store import SessionStore

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = SessionStore(root / "sessions")
            session = store.start_session()
            for index in range(105):
                source = root / f"source-{index}.wav"
                write_wav(source, frames=20)
                store.ingest(session, index, source, 0.01)

            recovered = store.recoverable()

            self.assertEqual([job.chunk_index for job in recovered], list(range(105)))

    def test_orphaned_committed_wav_is_reconciled_after_manifest_write_failure(
        self,
    ) -> None:
        from session_store import SessionStore

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = SessionStore(root / "sessions")
            session = store.start_session()
            source = root / "source.wav"
            write_wav(source)
            with mock.patch.object(
                store,
                "_write_json",
                side_effect=OSError("injected manifest failure"),
            ):
                with self.assertRaisesRegex(OSError, "manifest failure"):
                    store.ingest(session, 7, source, 0.1)

            recovered = SessionStore(root / "sessions").recoverable()

            self.assertEqual([job.chunk_index for job in recovered], [7])
            self.assertTrue(recovered[0].wav_path.is_file())

    def test_orphaned_committed_wav_is_reconciled_after_directory_fsync_failure(
        self,
    ) -> None:
        from session_store import SessionStore

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = SessionStore(root / "sessions")
            session = store.start_session()
            source = root / "source.wav"
            write_wav(source)
            with mock.patch.object(
                store,
                "_fsync_directory",
                side_effect=OSError("injected directory failure"),
            ):
                with self.assertRaisesRegex(OSError, "directory failure"):
                    store.ingest(session, 9, source, 0.1)

            recovered = SessionStore(root / "sessions").recoverable()

            self.assertEqual([job.chunk_index for job in recovered], [9])
            self.assertTrue(recovered[0].wav_path.is_file())

    def test_repeated_retention_preserves_source_until_existing_wav_is_durable(
        self,
    ) -> None:
        from session_store import SessionStore

        for failure_site in ("file", "directory"):
            with self.subTest(failure_site=failure_site):
                with tempfile.TemporaryDirectory() as raw:
                    root = Path(raw)
                    store = SessionStore(root / "sessions")
                    session = store.start_session()
                    original = root / "original.wav"
                    write_wav(original)
                    destination = store.retain_failed_ingest(session, 4, original)
                    repeated = root / "repeated.wav"
                    write_wav(repeated)

                    if failure_site == "file":
                        failure = mock.patch(
                            "session_store.os.fsync",
                            side_effect=OSError("injected retained WAV fsync failure"),
                        )
                    else:
                        failure = mock.patch.object(
                            store,
                            "_fsync_directory",
                            side_effect=OSError("injected session fsync failure"),
                        )
                    with failure:
                        with self.assertRaisesRegex(OSError, "fsync failure"):
                            store.retain_failed_ingest(session, 4, repeated)

                    self.assertTrue(destination.is_file())
                    self.assertTrue(repeated.is_file())
                    self.assertEqual(
                        store.retain_failed_ingest(session, 4, repeated), destination
                    )
                    self.assertFalse(repeated.exists())

    def test_recovered_chunk_updates_transcript_without_pasting(self) -> None:
        from dictation import Dictation
        from session_store import SessionStore

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = SessionStore(root / "sessions")
            session = store.start_session()
            source = root / "source.wav"
            write_wav(source)
            store.ingest(session, 0, source, 45.0)
            recovered = store.recoverable()[0]

            app = object.__new__(Dictation)
            app._store = store
            app._recording = False
            app._session_id = 0
            app.audio_source = ""
            app.min_record = 0.4
            app.min_audio_rms = 80.0
            app.replacements = {}
            app._transcribe = mock.Mock(return_value="recovered words")
            app._insert = mock.Mock()
            app._notify = mock.Mock()

            with mock.patch("dictation.wav_rms", return_value=500.0):
                app._deliver_chunk(recovered)

            app._insert.assert_not_called()
            self.assertEqual(
                (session / "transcript.txt").read_text(encoding="utf-8"),
                "recovered words\n",
            )


class WorkerContainmentTests(unittest.TestCase):
    def test_worker_survives_failure_at_every_chunk_position(self) -> None:
        from dictation import Dictation
        from session_store import SessionStore

        for failed_index in range(16):
            with self.subTest(failed_index=failed_index):
                with tempfile.TemporaryDirectory() as raw:
                    root = Path(raw)
                    store = SessionStore(root / "sessions")
                    session = store.start_session()
                    jobs = []
                    for index in range(16):
                        source = root / f"source-{index}.wav"
                        write_wav(source)
                        jobs.append(store.ingest(session, index, source, 45.0))

                    app = object.__new__(Dictation)
                    app._chunk_queue = queue.Queue()
                    app._store = store

                    def deliver(job):
                        if job.chunk_index == failed_index:
                            raise RuntimeError("injected failure")
                        store.complete(job, f"text {job.chunk_index}")

                    app._deliver_chunk = mock.Mock(side_effect=deliver)
                    worker = threading.Thread(
                        target=app._transcribe_worker, daemon=True
                    )
                    worker.start()
                    for job in jobs:
                        app._chunk_queue.put(job)
                    app._chunk_queue.join()
                    app._chunk_queue.put(None)
                    worker.join(timeout=2)

                    manifest = json.loads(
                        (session / "manifest.json").read_text(encoding="utf-8")
                    )
                    states = [chunk["status"] for chunk in manifest["chunks"]]
                    self.assertEqual(states[failed_index], "failed")
                    later_states = states[failed_index + 1 :]
                    self.assertTrue(all(state == "complete" for state in later_states))
                    self.assertTrue(all(job.wav_path.is_file() for job in jobs))
                    transcript = (session / "transcript.txt").read_text(
                        encoding="utf-8"
                    )
                    self.assertIn(
                        f"[failed: chunk {failed_index:04d} — injected failure]",
                        transcript,
                    )
                    if failed_index < 15:
                        self.assertIn("text 15", transcript)


if __name__ == "__main__":
    unittest.main()
