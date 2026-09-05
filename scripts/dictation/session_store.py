"""Durable local storage for rolling dictation sessions."""

from __future__ import annotations

import errno
import json
import os
import shutil
import threading
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_NO_SPACE_PREFIX = set(".,!?;:")


def join_fragments(fragments: list[str]) -> str:
    """Normalize and join Whisper fragments without separating punctuation."""
    result = ""
    for fragment in fragments:
        fragment = " ".join(fragment.split())
        if not fragment:
            continue
        if result and fragment[0] not in _NO_SPACE_PREFIX:
            result += " "
        result += fragment
    return result


@dataclass(frozen=True)
class ChunkJob:
    session_path: Path
    chunk_index: int
    wav_path: Path
    duration: float
    paste: bool = True
    paste_session: int = -1
    finalize: bool = False
    capture_mode: str = "legacy"
    beam_size: int | None = None
    audio_ctx: int | None = None
    trailing_silence_seconds: float = 0.0
    minimum_audio_rms: float = 80.0


class SessionStore:
    """Own manifests, retained WAV chunks, and assembled transcripts."""

    def __init__(
        self,
        root: Path,
        retry_limit: int = 3,
        *,
        legacy_minimum_audio_rms: float = 80.0,
    ) -> None:
        self.root = root.expanduser()
        self.retry_limit = retry_limit
        self.legacy_minimum_audio_rms = legacy_minimum_audio_rms
        self._lock = threading.RLock()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def start_session(
        self,
        *,
        capture_mode: str = "legacy",
        beam_size: int | None = None,
        audio_ctx: int | None = None,
        trailing_silence_seconds: float = 0.0,
        minimum_audio_rms: float = 80.0,
    ) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        session = self.root / f"{stamp}-{os.getpid()}"
        session.mkdir(mode=0o700)
        self._write_json(
            session / "manifest.json",
            {
                "version": 2,
                "recording": True,
                "policy": {
                    "capture_mode": capture_mode,
                    "beam_size": beam_size,
                    "audio_ctx": audio_ctx,
                    "trailing_silence_seconds": trailing_silence_seconds,
                    "minimum_audio_rms": minimum_audio_rms,
                },
                "chunks": [],
            },
        )
        self._write_text(session / "transcript.txt", "")
        return session

    def ingest(
        self,
        session: Path,
        chunk_index: int,
        source: Path,
        duration: float,
        *,
        paste: bool = True,
        paste_session: int = -1,
        finalize: bool = False,
    ) -> ChunkJob:
        destination = session / f"chunk-{chunk_index:04d}.wav"
        temporary = destination.with_suffix(".wav.partial")
        try:
            with source.open("rb") as input_file, temporary.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
            self._fsync_directory(session)
        finally:
            temporary.unlink(missing_ok=True)
        try:
            source.unlink()
        except FileNotFoundError:
            pass

        chunk = {
            "index": chunk_index,
            "wav": destination.name,
            "duration": duration,
            "status": "queued",
            "attempts": 0,
            "error": "",
        }
        with self._lock:
            manifest = self._load(session)
            manifest["chunks"].append(chunk)
            manifest["chunks"].sort(key=lambda item: item["index"])
            self._write_json(session / "manifest.json", manifest)
            self._rebuild(session, manifest)
        return self._job_from_policy(
            manifest,
            session=session,
            chunk_index=chunk_index,
            wav_path=destination,
            duration=duration,
            paste=paste,
            paste_session=paste_session,
            finalize=finalize,
        )

    def stop_session(self, session: Path | None) -> None:
        if session is None:
            return
        with self._lock:
            manifest = self._load(session)
            manifest["recording"] = False
            self._write_json(session / "manifest.json", manifest)
            self._rebuild(session, manifest)

    def retain_capture_tail(self, session: Path, payload: bytes, detail: str) -> Path:
        """Durably retain bytes that cannot form a complete PCM sample."""
        destination = session / "capture-unaligned-tail.pcm"
        temporary = destination.with_name(f".{destination.name}.partial")
        try:
            with temporary.open("wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
            self._fsync_directory(session)
        finally:
            temporary.unlink(missing_ok=True)
        with self._lock:
            manifest = self._load(session)
            manifest["capture_tail"] = {
                "path": destination.name,
                "bytes": len(payload),
                "error": detail,
            }
            self._write_json(session / "manifest.json", manifest)
        return destination

    def record_gap(
        self,
        session: Path,
        chunk_index: int,
        duration: float,
        detail: str,
    ) -> None:
        """Best-effort manifest marker when a finalized WAV cannot be ingested."""
        chunk = {
            "index": chunk_index,
            "wav": f"chunk-{chunk_index:04d}.wav",
            "duration": duration,
            "status": "missing",
            "attempts": 0,
            "error": detail,
        }
        with self._lock:
            manifest = self._load(session)
            existing = next(
                (
                    item
                    for item in manifest["chunks"]
                    if int(item["index"]) == chunk_index
                ),
                None,
            )
            if existing is not None:
                existing_wav = session / str(existing.get("wav", ""))
                if existing_wav.is_file():
                    existing["error"] = detail
                else:
                    existing.update(chunk)
            else:
                manifest["chunks"].append(chunk)
            manifest["chunks"].sort(key=lambda item: item["index"])
            self._write_json(session / "manifest.json", manifest)
            self._rebuild(session, manifest)

    def reconcile_committed_job(
        self,
        session: Path,
        chunk_index: int,
        *,
        paste: bool = True,
        paste_session: int = -1,
        finalize: bool = False,
    ) -> ChunkJob | None:
        """Recover a live job when ingest failed after making its WAV durable."""
        with self._lock:
            manifest = self._load(session)
            changed = self._reconcile_orphaned_wav(session, manifest, chunk_index)
            if changed:
                self._write_json(session / "manifest.json", manifest)
                self._rebuild(session, manifest)
            chunk = next(
                (
                    item
                    for item in manifest["chunks"]
                    if int(item["index"]) == chunk_index
                ),
                None,
            )
            if chunk is None or chunk.get("status") != "queued":
                return None
            wav_path = session / str(chunk.get("wav", ""))
            if not wav_path.is_file():
                return None
            return self._job_from_policy(
                manifest,
                session=session,
                chunk_index=chunk_index,
                wav_path=wav_path,
                duration=float(chunk["duration"]),
                paste=paste,
                paste_session=paste_session,
                finalize=finalize,
            )

    def retain_failed_ingest(
        self, session: Path, chunk_index: int, source: Path
    ) -> Path:
        """Move a failed ingest source into its session, including across mounts."""
        destination = session / f"chunk-{chunk_index:04d}.wav"
        if destination.is_file():
            with destination.open("rb") as retained:
                os.fsync(retained.fileno())
            self._fsync_directory(session)
            source.unlink(missing_ok=True)
            return destination

        try:
            os.replace(source, destination)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            temporary = destination.with_name(f".{destination.name}.retained")
            try:
                with (
                    source.open("rb") as input_file,
                    temporary.open("wb") as output_file,
                ):
                    shutil.copyfileobj(input_file, output_file)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, destination)
                self._fsync_directory(session)
                source.unlink(missing_ok=True)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            os.chmod(destination, 0o600)
            with destination.open("rb") as retained:
                os.fsync(retained.fileno())
            self._fsync_directory(session)
        return destination

    def begin(self, job: ChunkJob) -> None:
        self._change(job, lambda chunk: chunk.update(status="transcribing"))

    def complete(self, job: ChunkJob, text: str) -> None:
        fragment = job.session_path / f"chunk-{job.chunk_index:04d}.txt"
        self._write_text(fragment, text.strip() + "\n")
        self._change(
            job,
            lambda chunk: chunk.update(status="complete", error="", text=fragment.name),
        )

    def terminal(self, job: ChunkJob, status: str, detail: str = "") -> None:
        self._change(job, lambda chunk: chunk.update(status=status, error=detail))

    def fail(self, job: ChunkJob, status: str = "failed", detail: str = "") -> None:
        def update(chunk: dict[str, Any]) -> None:
            chunk["attempts"] = int(chunk.get("attempts", 0)) + 1
            chunk["status"] = (
                "exhausted" if chunk["attempts"] >= self.retry_limit else status
            )
            chunk["error"] = detail

        self._change(job, update)

    def recoverable(self, limit: int | None = None) -> list[ChunkJob]:
        if limit is not None and limit < 0:
            raise ValueError("recovery limit cannot be negative")
        jobs: list[ChunkJob] = []
        for manifest_path in sorted(self.root.glob("*/manifest.json")):
            session = manifest_path.parent
            with self._lock:
                manifest = self._load(session)
                changed = self._reconcile_orphaned_wavs(session, manifest)
                for chunk in manifest["chunks"]:
                    if limit is not None and len(jobs) >= limit:
                        break
                    status = chunk["status"]
                    if status not in {"queued", "transcribing", "failed", "corrupt"}:
                        continue
                    wav = session / chunk["wav"]
                    attempts = int(chunk.get("attempts", 0))
                    if not wav.is_file():
                        chunk.update(status="missing", error="audio file is missing")
                        changed = True
                        continue
                    if attempts >= self.retry_limit:
                        chunk["status"] = "exhausted"
                        changed = True
                        continue
                    chunk["status"] = "queued"
                    changed = True
                    jobs.append(
                        self._job_from_policy(
                            manifest,
                            session=session,
                            chunk_index=int(chunk["index"]),
                            wav_path=wav,
                            duration=float(chunk["duration"]),
                            paste=False,
                        )
                    )
                if changed:
                    manifest["recording"] = False
                    self._write_json(manifest_path, manifest)
                    self._rebuild(session, manifest)
            if limit is not None and len(jobs) >= limit:
                break
        return jobs

    @staticmethod
    def _policy(manifest: dict[str, Any]) -> dict[str, Any]:
        policy = manifest.get("policy")
        return policy if isinstance(policy, dict) else {}

    def _job_from_policy(
        self,
        manifest: dict[str, Any],
        *,
        session: Path,
        chunk_index: int,
        wav_path: Path,
        duration: float,
        paste: bool = True,
        paste_session: int = -1,
        finalize: bool = False,
    ) -> ChunkJob:
        policy = self._policy(manifest)
        raw_beam = policy.get("beam_size")
        beam_size = int(raw_beam) if raw_beam not in (None, "") else None
        raw_audio_ctx = policy.get("audio_ctx")
        audio_ctx = int(raw_audio_ctx) if raw_audio_ctx not in (None, "") else None
        manifest_version = int(manifest.get("version", 1))
        minimum_audio_rms_default = (
            self.legacy_minimum_audio_rms if manifest_version == 1 else 80.0
        )
        return ChunkJob(
            session,
            chunk_index,
            wav_path,
            duration,
            paste,
            paste_session,
            finalize,
            str(policy.get("capture_mode", "legacy")),
            beam_size,
            audio_ctx,
            float(policy.get("trailing_silence_seconds", 0.0)),
            float(policy.get("minimum_audio_rms", minimum_audio_rms_default)),
        )

    def _reconcile_orphaned_wavs(self, session: Path, manifest: dict[str, Any]) -> bool:
        """Make a WAV durable before a manifest failure retryable on restart."""
        changed = False
        for chunk in manifest.get("chunks", []):
            wav_path = session / str(chunk.get("wav", ""))
            if chunk.get("status") == "missing" and wav_path.is_file():
                chunk["status"] = "queued"
                chunk["error"] = "recovered retained audio after ingest failure"
                changed = True
        known = {int(item["index"]) for item in manifest.get("chunks", [])}
        for wav_path in sorted(session.glob("chunk-[0-9][0-9][0-9][0-9].wav")):
            try:
                index = int(wav_path.stem.removeprefix("chunk-"))
            except ValueError:
                continue
            if index in known:
                continue
            try:
                with wave.open(str(wav_path), "rb") as audio:
                    duration = audio.getnframes() / audio.getframerate()
                status = "queued"
                error = "recovered after interrupted manifest commit"
            except (EOFError, OSError, ValueError, wave.Error) as exc:
                duration = 0.0
                status = "corrupt"
                error = str(exc)
            manifest.setdefault("chunks", []).append(
                {
                    "index": index,
                    "wav": wav_path.name,
                    "duration": duration,
                    "status": status,
                    "attempts": 0,
                    "error": error,
                }
            )
            known.add(index)
            changed = True
        if changed:
            manifest["chunks"].sort(key=lambda item: int(item["index"]))
        return changed

    def _reconcile_orphaned_wav(
        self, session: Path, manifest: dict[str, Any], chunk_index: int
    ) -> bool:
        """Reconcile only the live chunk whose ingest just raised."""
        chunk = next(
            (
                item
                for item in manifest.get("chunks", [])
                if int(item["index"]) == chunk_index
            ),
            None,
        )
        if chunk is not None:
            wav_path = session / str(chunk.get("wav", ""))
            if chunk.get("status") == "missing" and wav_path.is_file():
                chunk["status"] = "queued"
                chunk["error"] = "recovered retained audio after ingest failure"
                return True
            return False

        wav_path = session / f"chunk-{chunk_index:04d}.wav"
        if not wav_path.is_file():
            return False
        try:
            with wave.open(str(wav_path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
            status = "queued"
            error = "recovered after interrupted manifest commit"
        except (EOFError, OSError, ValueError, wave.Error) as exc:
            duration = 0.0
            status = "corrupt"
            error = str(exc)
        manifest.setdefault("chunks", []).append(
            {
                "index": chunk_index,
                "wav": wav_path.name,
                "duration": duration,
                "status": status,
                "attempts": 0,
                "error": error,
            }
        )
        manifest["chunks"].sort(key=lambda item: int(item["index"]))
        return True

    def completed_text(self, session: Path) -> str:
        """Return only successful fragments, assembled in manifest order."""
        with self._lock:
            manifest = self._load(session)
            fragments: list[str] = []
            for chunk in sorted(manifest["chunks"], key=lambda item: item["index"]):
                if chunk["status"] != "complete":
                    continue
                fragment = session / f"chunk-{int(chunk['index']):04d}.txt"
                if fragment.is_file():
                    text = fragment.read_text(encoding="utf-8").strip()
                    if text:
                        fragments.append(text)
            return join_fragments(fragments)

    def _change(self, job: ChunkJob, change: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            manifest = self._load(job.session_path)
            chunk = next(
                item
                for item in manifest["chunks"]
                if int(item["index"]) == job.chunk_index
            )
            change(chunk)
            self._write_json(job.session_path / "manifest.json", manifest)
            self._rebuild(job.session_path, manifest)

    def _rebuild(self, session: Path, manifest: dict[str, Any]) -> None:
        lines: list[str] = []
        marker_free_silence = (
            str(self._policy(manifest).get("capture_mode", "legacy")).lower()
            == "continuous"
        )
        for chunk in sorted(manifest["chunks"], key=lambda item: item["index"]):
            index = int(chunk["index"])
            fragment = session / f"chunk-{index:04d}.txt"
            if chunk["status"] == "complete" and fragment.is_file():
                lines.append(fragment.read_text(encoding="utf-8").strip())
            elif chunk["status"] in {"recording", "queued", "transcribing"}:
                continue
            elif chunk["status"] in {"silent", "ignored"} and marker_free_silence:
                continue
            else:
                detail = str(chunk.get("error", "")).strip()
                suffix = f" — {detail}" if detail else ""
                lines.append(f"[{chunk['status']}: chunk {index:04d}{suffix}]")
        text = "\n".join(line for line in lines if line)
        self._write_text(session / "transcript.txt", text + ("\n" if text else ""))

    def _load(self, session: Path) -> dict[str, Any]:
        return json.loads((session / "manifest.json").read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self._atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _write_text(self, path: Path, text: str) -> None:
        self._atomic_write(path, text)

    def _atomic_write(self, path: Path, text: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        self._fsync_directory(path.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
