"""Lossless streaming PCM segmentation and temporary inference preparation."""

from __future__ import annotations

import hashlib
import math
import os
import statistics
import sys
import tempfile
import wave
from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


SAMPLE_WIDTH = 2
CHANNELS = 1


class IncompletePcmSample(ValueError):
    """The recorder ended after only part of a signed-16 sample was read."""


class PcmSegmenter:
    """Partition a signed-16 mono stream once at bounded low-energy cuts."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        target_seconds: float = 8.0,
        minimum_seconds: float = 7.0,
        maximum_seconds: float = 9.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0 < minimum_seconds <= target_seconds <= maximum_seconds:
            raise ValueError("require 0 < minimum <= target <= maximum")
        self.sample_rate = sample_rate
        self.minimum_frames = max(1, round(sample_rate * minimum_seconds))
        self.target_frames = max(
            self.minimum_frames, round(sample_rate * target_seconds)
        )
        self.maximum_frames = max(
            self.target_frames, round(sample_rate * maximum_seconds)
        )
        self._energy_window_frames = max(1, round(sample_rate * 0.05))
        self._lookahead_frames = self._energy_window_frames
        self._silence_bin_frames = max(1, round(sample_rate * 0.01))
        self._minimum_verified_silence_frames = max(
            self._silence_bin_frames,
            round(sample_rate * 0.06),
        )
        self._buffer = bytearray()
        self._finished = False
        self.total_input_bytes = 0
        self.total_output_bytes = 0
        self.unaligned_tail = b""
        self._input_digest = hashlib.sha256()

    @property
    def input_sha256(self) -> str:
        return self._input_digest.hexdigest()

    def feed(self, data: bytes) -> list[bytes]:
        if self._finished:
            raise RuntimeError("cannot feed a finished PCM segmenter")
        if not data:
            return []
        self.total_input_bytes += len(data)
        self._input_digest.update(data)
        self._buffer.extend(data)
        emitted: list[bytes] = []
        required_frames = self.maximum_frames + self._lookahead_frames
        while len(self._buffer) // SAMPLE_WIDTH >= required_frames:
            emitted.append(self._take(self._select_cut()))
        return emitted

    def finish(self) -> list[bytes]:
        if self._finished:
            return []
        if len(self._buffer) % SAMPLE_WIDTH:
            self.unaligned_tail = bytes(self._buffer[-1:])
            raise IncompletePcmSample(
                "raw recorder stream ended with an incomplete signed-16 sample"
            )
        self._finished = True
        emitted: list[bytes] = []
        while len(self._buffer) // SAMPLE_WIDTH > self.maximum_frames:
            emitted.append(self._take(self._select_cut()))
        if self._buffer:
            emitted.append(self._take(len(self._buffer) // SAMPLE_WIDTH))
        return emitted

    def recover_aligned_tail(self) -> bytes:
        """Return complete buffered samples after a truncated final sample."""
        aligned_bytes = len(self._buffer) - (len(self._buffer) % SAMPLE_WIDTH)
        payload = bytes(self._buffer[:aligned_bytes])
        del self._buffer[:aligned_bytes]
        self.total_output_bytes += len(payload)
        self._finished = True
        return payload

    def _select_cut(self) -> int:
        available_frames = len(self._buffer) // SAMPLE_WIDTH
        high = min(self.maximum_frames, available_frames)
        low = min(self.minimum_frames, high)
        samples = array("h")
        samples.frombytes(bytes(self._buffer[: available_frames * SAMPLE_WIDTH]))
        if sys.byteorder != "little":
            samples.byteswap()

        candidate_step_frames = max(1, round(self.sample_rate * 0.02))
        candidates = range(low, high + 1, candidate_step_frames)

        def score(cut: int) -> tuple[float, int]:
            begin = max(0, cut - self._energy_window_frames)
            end = min(len(samples), cut + self._energy_window_frames)
            if begin >= end:
                energy = math.inf
            else:
                energy = sum(sample * sample for sample in samples[begin:end]) / (
                    end - begin
                )
            return energy, abs(cut - self.target_frames)

        low_energy_cut = min(candidates, key=score)
        verified_cut = self._verified_silence_cut(
            samples,
            low=low,
            high=high,
            low_energy_cut=low_energy_cut,
        )
        return verified_cut if verified_cut is not None else low_energy_cut

    def _verified_silence_cut(
        self,
        samples: array[int],
        *,
        low: int,
        high: int,
        low_energy_cut: int,
    ) -> int | None:
        """Prefer a sustained pause when the quietest point is only a short dip."""
        bins: list[tuple[int, int, float]] = []
        for begin in range(low, high, self._silence_bin_frames):
            end = min(begin + self._silence_bin_frames, high)
            energy = sum(sample * sample for sample in samples[begin:end]) / (
                end - begin
            )
            bins.append((begin, end, math.sqrt(energy)))
        if not bins:
            return None

        levels = [level for _, _, level in bins]
        quiet_count = max(1, round(len(levels) * 0.03))
        quiet_floor = statistics.median(sorted(levels)[:quiet_count])
        typical_level = statistics.median(levels)
        silence_threshold = min(
            max(80.0, quiet_floor * 2.5),
            max(80.0, typical_level * 0.35),
        )

        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for begin, end, level in bins:
            if level <= silence_threshold:
                if run_start is None:
                    run_start = begin
                continue
            if run_start is not None:
                if begin - run_start >= self._minimum_verified_silence_frames:
                    runs.append((run_start, begin))
                run_start = None
        if (
            run_start is not None
            and high - run_start >= self._minimum_verified_silence_frames
        ):
            runs.append((run_start, high))
        if not runs:
            return None

        if any(start <= low_energy_cut <= end for start, end in runs):
            return low_energy_cut

        start, end = min(
            runs,
            key=lambda run: (
                abs(((run[0] + run[1]) // 2) - self.target_frames),
                -(run[1] - run[0]),
            ),
        )
        return (start + end) // 2

    def _take(self, frame_count: int) -> bytes:
        byte_count = frame_count * SAMPLE_WIDTH
        payload = bytes(self._buffer[:byte_count])
        del self._buffer[:byte_count]
        self.total_output_bytes += len(payload)
        return payload


def write_pcm_wav(path: Path, pcm: bytes, *, sample_rate: int = 16000) -> None:
    """Atomically wrap signed-16 mono PCM in an independently valid WAV."""
    if len(pcm) % SAMPLE_WIDTH:
        raise IncompletePcmSample("cannot write a partial signed-16 sample")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    try:
        with wave.open(str(partial), "wb") as output:
            output.setnchannels(CHANNELS)
            output.setsampwidth(SAMPLE_WIDTH)
            output.setframerate(sample_rate)
            output.writeframes(pcm)
        os.chmod(partial, 0o600)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


@contextmanager
def prepared_padded_wav(
    source: Path,
    trailing_silence_seconds: float,
    *,
    work_dir: Path | None = None,
) -> Iterator[Path]:
    """Yield an inference-only padded WAV and always remove the derivative."""
    if trailing_silence_seconds <= 0:
        yield source
        return
    if work_dir is not None:
        work_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        suffix=".wav",
        prefix="whisper-inference-",
        dir=work_dir,
    )
    os.close(fd)
    temporary = Path(name)
    partial = temporary.with_name(f".{temporary.name}.partial")
    try:
        with wave.open(str(source), "rb") as input_audio:
            if (
                input_audio.getsampwidth() != SAMPLE_WIDTH
                or input_audio.getnchannels() != CHANNELS
            ):
                raise ValueError("inference padding requires signed-16 mono PCM")
            params = input_audio.getparams()
            pcm = input_audio.readframes(input_audio.getnframes())
        silence_frames = round(params.framerate * trailing_silence_seconds)
        with wave.open(str(partial), "wb") as output:
            output.setparams(params)
            output.writeframes(pcm + (b"\x00\x00" * silence_frames))
        os.chmod(partial, 0o600)
        os.replace(partial, temporary)
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
