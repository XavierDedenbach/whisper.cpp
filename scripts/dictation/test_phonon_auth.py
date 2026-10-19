"""Real authenticated loopback transport without credentials in process argv."""

import stat
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import pytest

import phonon_backend as backend
from dictation import Dictation


def test_private_key_is_created_once_and_unsafe_files_are_refused(tmp_path):
    key_file = tmp_path / "private" / "key"
    backend.ensure_key_file(key_file)
    original = key_file.read_bytes()
    assert len(original.strip()) >= 32
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    backend.ensure_key_file(key_file)
    assert key_file.read_bytes() == original
    key_file.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        backend.ensure_key_file(key_file)
    key_file.chmod(0o600)
    key_file.write_text("bad\nheader: injection")
    with pytest.raises(ValueError, match="key"):
        backend.read_key_file(key_file)
    link = tmp_path / "linked-key"
    link.symlink_to(key_file)
    with pytest.raises(OSError):
        backend.read_key_file(link)


def test_authenticated_real_http_keeps_key_out_of_argv(tmp_path):
    key_file = tmp_path / "key"
    backend.ensure_key_file(key_file)
    secret = key_file.read_text().strip()
    authenticated = []

    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            if self.headers.get("Authorization") != "Bearer " + secret:
                self.send_error(401)
                return
            authenticated.append(self.path)
            if self.command == "POST":
                self.rfile.read(int(self.headers["Content-Length"]))
            body = (
                b'{"status":"ok","model":"phonon-2","gpu":"GB10",'
                b'"path":"CUDA-graph","text":"authenticated words"}'
            )
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = handle_request

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    url = f"http://127.0.0.1:{server.server_port}"
    real_run = subprocess.run
    try:
        with mock.patch("phonon_backend.subprocess.run", wraps=real_run) as transport:
            assert backend.health(url, "cuda", key_file=key_file)
            assert (
                backend.transcribe(url, str(audio), 5, [], key_file=key_file)
                == "authenticated words"
            )
            for call in transport.call_args_list:
                assert secret not in " ".join(call.args[0])
                assert secret in call.kwargs["input"]
                assert "--noproxy" in call.args[0]
        assert backend.transcribe(url, str(audio), 5, []) is None
        assert (
            backend.transcribe(url, str(audio), 5, [], key_file=tmp_path / "missing")
            is None
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)
    assert authenticated == ["/health", "/v1/audio/transcriptions"]


def test_daemon_passes_key_file_only_for_cuda(tmp_path):
    for device in ("cpu", "cuda"):
        cfg = {
            "DICTATION_ENGINE": "phonon2",
            "PHONON_DEVICE": device,
            "PHONON_API_KEY_FILE": str(tmp_path / "key"),
            "WHISPER_SESSION_DIR": str(tmp_path / device),
        }
        with mock.patch("dictation.threading.Thread.start"):
            app = Dictation(cfg)
        expected = str(tmp_path / "key") if device == "cuda" else None
        with mock.patch("phonon_backend.health", return_value=True) as health:
            assert app._server_health_ok()
            assert health.call_args.kwargs["key_file"] == expected
        with mock.patch(
            "phonon_backend.transcribe", return_value="words"
        ) as transcribe:
            assert app._transcribe_server("audio.wav") == "words"
            assert transcribe.call_args.kwargs["key_file"] == expected
