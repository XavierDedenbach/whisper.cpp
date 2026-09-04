"""Durable local storage for rolling dictation sessions."""

from __future__ import annotations

import json
import os
import shutil
import threading
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


class SessionStore:
    """Own manifests, retained WAV chunks, and assembled transcripts."""

    def __init__(self, root: Path, retry_limit: int = 3) -> None:
        self.root = root.expanduser()
        self.retry_limit = retry_limit
        self._lock = threading.RLock()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def start_session(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        session = self.root / f"{stamp}-{os.getpid()}"
        session.mkdir(mode=0o700)
        self._write_json(
            session / "manifest.json",
            {"version": 1, "recording": True, "chunks": []},
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
        with source.open("rb") as input_file, temporary.open("wb") as output_file:
            shutil.copyfileobj(input_file, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        self._fsync_directory(session)
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
        return ChunkJob(
            session,
            chunk_index,
            destination,
            duration,
            paste,
            paste_session,
            finalize,
        )

    def stop_session(self, session: Path | None) -> None:
        if session is None:
            return
        with self._lock:
            manifest = self._load(session)
            manifest["recording"] = False
            self._write_json(session / "manifest.json", manifest)
            self._rebuild(session, manifest)

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
            manifest["chunks"].append(chunk)
            manifest["chunks"].sort(key=lambda item: item["index"])
            self._write_json(session / "manifest.json", manifest)
            self._rebuild(session, manifest)

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

    def recoverable(self, limit: int = 32) -> list[ChunkJob]:
        jobs: list[ChunkJob] = []
        for manifest_path in sorted(self.root.glob("*/manifest.json")):
            session = manifest_path.parent
            with self._lock:
                manifest = self._load(session)
                changed = False
                for chunk in manifest["chunks"]:
                    if len(jobs) >= limit:
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
                        ChunkJob(
                            session,
                            int(chunk["index"]),
                            wav,
                            float(chunk["duration"]),
                            paste=False,
                        )
                    )
                if changed:
                    manifest["recording"] = False
                    self._write_json(manifest_path, manifest)
                    self._rebuild(session, manifest)
            if len(jobs) >= limit:
                break
        return jobs

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
        for chunk in sorted(manifest["chunks"], key=lambda item: item["index"]):
            index = int(chunk["index"])
            fragment = session / f"chunk-{index:04d}.txt"
            if chunk["status"] == "complete" and fragment.is_file():
                lines.append(fragment.read_text(encoding="utf-8").strip())
            elif chunk["status"] in {"recording", "queued", "transcribing"}:
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
