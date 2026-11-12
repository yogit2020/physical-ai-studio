from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

import numpy as np
import pytest

from schemas.hardware import InferenceBackend
from services.system_service import SystemService
from workers.remote_predictor import (
    CONFIG_FILE,
    REMOTE_DEVICE,
    RemotePredictor,
    decode_frame,
    encode_frame,
    load_config,
    weights_fingerprint,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _export(path: Path, weights: bytes) -> Path:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"model": {"artifacts": {"openvino": "policy.xml"}}}))
    (path / "policy.xml").write_text("<net/>")
    (path / "policy.bin").write_bytes(weights)
    return path


class _FakePolicyServer:
    """Answers like ``remote_policy_server.py``, recording what it was sent."""

    def __init__(self, fingerprint: str) -> None:
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                info = {"host": "b70", "export_dir": "/models/x", "backend": "openvino", "device": "GPU"}
                self._send({**info, "fingerprint": fingerprint})

            def do_POST(self) -> None:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(request)
                state = np.asarray(request["state"])[0]
                self._send({"action": [(state + step).tolist() for step in range(1, 4)], "infer_ms": 5.0})

            def log_message(self, *_args: object) -> None:
                pass

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._http.server_address[1]}"
        threading.Thread(target=self._http.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._http.shutdown()
        self._http.server_close()


@pytest.fixture
def export(tmp_path: Path) -> Path:
    return _export(tmp_path / "exports" / "openvino", b"weights" * 1000)


@pytest.fixture
def server(export: Path) -> Iterator[_FakePolicyServer]:
    fake = _FakePolicyServer(weights_fingerprint(export))
    yield fake
    fake.close()


def test_a_frame_survives_the_round_trip_with_its_channel_order() -> None:
    frame = np.zeros((1, 48, 64, 3), np.uint8)
    frame[..., 0] = 200  # only the first channel: a swap would move it
    decoded = decode_frame(encode_frame(frame))
    assert decoded.shape == frame.shape
    assert np.abs(decoded.astype(int) - frame).max() <= 4  # JPEG is lossy, not shuffled


def test_the_fingerprint_tells_checkpoints_apart(tmp_path: Path) -> None:
    a = weights_fingerprint(_export(tmp_path / "a", b"x" * 5000))
    assert a == weights_fingerprint(_export(tmp_path / "b", b"x" * 5000))
    assert a != weights_fingerprint(_export(tmp_path / "c", b"y" * 5000))


def test_the_remote_model_infers_like_a_local_one(export: Path, server: _FakePolicyServer, tmp_path: Path) -> None:
    timing = tmp_path / "timing.csv"
    predictor = RemotePredictor(export, server.url, timing_log=str(timing))
    observation = {
        "state": np.array([[1.0, 2.0]], np.float32),
        "images.wrist": np.zeros((1, 48, 64, 3), np.uint8),
        "task": ["sort"],
    }
    chunk = predictor.predict_action_chunk(observation)

    np.testing.assert_allclose(chunk, [[2, 3], [3, 4], [4, 5]])
    (sent,) = server.requests
    assert sent["state"] == [[1.0, 2.0]]
    assert sent["task"] == "sort"
    assert decode_frame(sent["images"]["images.wrist"]).shape == (1, 48, 64, 3)
    lines = timing.read_text().splitlines()
    assert lines[0].startswith("wall_time,") and lines[1].startswith("# session") and len(lines) == 3


def test_another_model_on_the_server_is_refused(tmp_path: Path, server: _FakePolicyServer) -> None:
    other = _export(tmp_path / "other", b"different weights" * 1000)
    with pytest.raises(RuntimeError, match="another model loaded"):
        RemotePredictor(other, server.url)


def test_the_remote_device_is_listed_only_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(SystemService, "_get_torch_inference_devices", classmethod(lambda cls: []))
    monkeypatch.setattr(SystemService, "_get_openvino_inference_devices", classmethod(lambda cls: []))
    assert load_config() is None
    assert SystemService.get_inference_devices() == []

    (tmp_path / CONFIG_FILE).write_text(json.dumps({"url": "http://127.0.0.1:1", "name": "B70 (remote)"}))
    (device,) = SystemService.get_inference_devices()
    assert (device.backend, device.device, device.name) == (InferenceBackend.OPENVINO, REMOTE_DEVICE, "B70 (remote)")


def test_an_unreadable_config_is_ignored(tmp_path: Path) -> None:
    (tmp_path / CONFIG_FILE).write_text("{not json")
    assert load_config(tmp_path) is None
