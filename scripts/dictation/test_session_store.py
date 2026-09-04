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
