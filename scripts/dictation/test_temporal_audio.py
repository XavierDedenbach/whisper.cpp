#!/usr/bin/env python3
"""Tests for lossless streaming PCM ownership and inference preparation."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


class PcmSegmenterTests(unittest.TestCase):
    def test_arbitrary_odd_sized_reads_partition_every_sample_exactly_once(
        self,
    ) -> None:
        from temporal_audio import PcmSegmenter

        samples = array("h", [1200] * 350)
        pcm = samples.tobytes()
        segmenter = PcmSegmenter(
            sample_rate=100,
            target_seconds=1.0,
            minimum_seconds=0.8,
            maximum_seconds=1.2,
        )
        emitted: list[bytes] = []
        positions = [1, 4, 11, 28, 63, 126, 255, 511, len(pcm)]
        start = 0
        for stop in positions:
            emitted.extend(segmenter.feed(pcm[start:stop]))
            start = stop
        emitted.extend(segmenter.finish())

        self.assertEqual(b"".join(emitted), pcm)
        self.assertEqual(segmenter.total_input_bytes, len(pcm))
        self.assertEqual(segmenter.total_output_bytes, len(pcm))
        self.assertEqual(segmenter.input_sha256, hashlib.sha256(pcm).hexdigest())
        self.assertGreaterEqual(len(emitted), 3)
        for chunk in emitted[:-1]:
            frames = len(chunk) // 2
            self.assertGreaterEqual(frames, 80)
            self.assertLessEqual(frames, 120)
        self.assertTrue(emitted[-1])

    def test_low_energy_boundary_nearest_target_is_selected(self) -> None:
        from temporal_audio import PcmSegmenter

        samples = array("h", [2000] * 300)
        for index in range(96, 105):
            samples[index] = 0
        segmenter = PcmSegmenter(
            sample_rate=100,
            target_seconds=1.0,
            minimum_seconds=0.8,
            maximum_seconds=1.2,
        )
        emitted = segmenter.feed(samples.tobytes())
        emitted.extend(segmenter.finish())

        first_frames = len(emitted[0]) // 2
        self.assertGreaterEqual(first_frames, 96)
        self.assertLessEqual(first_frames, 104)
        self.assertEqual(b"".join(emitted), samples.tobytes())

    def test_verified_silence_beats_a_quieter_but_too_short_dip(self) -> None:
        from temporal_audio import PcmSegmenter

        sample_rate = 1000
        samples = array("h", [5000] * (sample_rate * 10))
        # A real inter-utterance pause near 7.94 seconds. Its surrounding speech
        # makes its 100 ms average louder than the short dip below.
        samples[7900:7980] = array("h", [0] * 80)
        # A quiet intra-utterance region near 8.50 seconds contains only 30 ms of
        # silence. The legacy minimum-average-energy rule incorrectly chose it.
        samples[8400:8600] = array("h", [100] * 200)
        samples[8485:8515] = array("h", [0] * 30)

        segmenter = PcmSegmenter(
            sample_rate=sample_rate,
            target_seconds=8.0,
            minimum_seconds=7.0,
            maximum_seconds=9.0,
        )
        emitted = segmenter.feed(samples.tobytes())
        emitted.extend(segmenter.finish())

        first_frames = len(emitted[0]) // 2
        self.assertGreaterEqual(first_frames, 7930)
        self.assertLessEqual(first_frames, 7950)
        self.assertEqual(b"".join(emitted), samples.tobytes())

    def test_exact_cut_stop_never_emits_an_empty_successor(self) -> None:
        from temporal_audio import PcmSegmenter

        pcm = array("h", [900] * 100).tobytes()
        segmenter = PcmSegmenter(
            sample_rate=100,
            target_seconds=1.0,
            minimum_seconds=0.8,
            maximum_seconds=1.2,
        )
        emitted = segmenter.feed(pcm)
        emitted.extend(segmenter.finish())
        self.assertEqual(emitted, [pcm])
        self.assertEqual(segmenter.finish(), [])

    def test_incomplete_final_sample_is_reported_without_discarding_it(self) -> None:
        from temporal_audio import IncompletePcmSample, PcmSegmenter

        segmenter = PcmSegmenter(
            sample_rate=100,
            target_seconds=1.0,
            minimum_seconds=0.8,
            maximum_seconds=1.2,
        )
        segmenter.feed(b"\x01")
        with self.assertRaises(IncompletePcmSample):
            segmenter.finish()
        self.assertEqual(segmenter.unaligned_tail, b"\x01")


class PreparedPaddedWavTests(unittest.TestCase):
    def test_padding_is_temporary_and_raw_wav_is_unchanged(self) -> None:
        from temporal_audio import prepared_padded_wav, write_pcm_wav

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.wav"
            pcm = array("h", range(100)).tobytes()
            write_pcm_wav(source, pcm, sample_rate=100)
            before = hashlib.sha256(source.read_bytes()).hexdigest()

            with prepared_padded_wav(source, 0.5, work_dir=root) as prepared:
                self.assertNotEqual(prepared, source)
                self.assertTrue(prepared.is_file())
                with wave.open(str(prepared), "rb") as audio:
                    padded = audio.readframes(audio.getnframes())
                self.assertEqual(padded[: len(pcm)], pcm)
                self.assertEqual(padded[len(pcm) :], b"\x00\x00" * 50)
                temporary = prepared

            self.assertFalse(temporary.exists())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_padding_cleanup_runs_when_inference_raises(self) -> None:
        from temporal_audio import prepared_padded_wav, write_pcm_wav

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.wav"
            write_pcm_wav(source, array("h", [1] * 100).tobytes(), sample_rate=100)
            temporary: Path | None = None
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with prepared_padded_wav(source, 0.5, work_dir=root) as prepared:
                    temporary = prepared
                    raise RuntimeError("injected")
            self.assertIsNotNone(temporary)
            self.assertFalse(temporary.exists())
            self.assertEqual(list(root.glob("*.partial")), [])


if __name__ == "__main__":
    unittest.main()
