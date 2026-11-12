"""Run a model's inference on another machine's accelerator.

The robot and cameras stay on this machine. A ``ModelWorker`` whose device is
:data:`REMOTE_DEVICE` sends each observation to ``scripts/remote_policy_server.py``
(usually reached through an SSH tunnel) and gets the action chunk back, so to the
rest of the session a remote model is any other model.

Enabled by ``remote_inference.json`` in the storage directory, so it needs no new
environment variables and survives restarts::

    {"url": "http://172.18.0.1:18765", "name": "Arc Pro B70 (remote)", "memory": 34242297856,
     "timing_log": "/app/storage/remote_inference_timing.csv"}
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from loguru import logger

from settings import get_settings

REMOTE_DEVICE = "REMOTE"
"""The device name a remote model is loaded with; the backend is the export it runs remotely."""

CONFIG_FILE = "remote_inference.json"
JPEG_QUALITY = 95
_FINGERPRINT_BYTES = 16 * 2**20
_TIMING_COLUMNS = (
    "wall_time,call,encode_ms,request_ms,remote_infer_ms,network_ms,total_ms,since_prev_call_ms,payload_kb\n"
)


def load_config(storage_dir: Path | None = None) -> dict[str, Any] | None:
    """The remote inference config, or ``None`` when remote inference is not set up."""
    path = (storage_dir or get_settings().storage_dir) / CONFIG_FILE
    if not path.is_file():
        return None
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Ignoring unreadable remote inference config {}: {}", path, error)
        return None
    return config if isinstance(config, dict) and config.get("url") else None


def weights_fingerprint(export_dir: Path) -> str:
    """Identify an export by its weights: their size and hashes of their first and last 16 MiB.

    Manifests are the same for every model trained on the same features, so they cannot
    tell two checkpoints apart, and hashing gigabytes of weights on every load is too slow.
    """
    manifest = json.loads((export_dir / "manifest.json").read_text())
    weights = export_dir / next(iter(manifest["model"]["artifacts"].values()))
    if weights.suffix == ".xml":  # OpenVINO keeps the weights next to the graph
        weights = weights.with_suffix(".bin")
    size = weights.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with weights.open("rb") as f:
        digest.update(f.read(_FINGERPRINT_BYTES))
        f.seek(max(size - _FINGERPRINT_BYTES, 0))
        digest.update(f.read(_FINGERPRINT_BYTES))
    return digest.hexdigest()


def encode_frame(frame: np.ndarray) -> str:
    """One ``(1, H, W, C)`` uint8 frame as base64 JPEG. Channel order is kept as is both ways."""
    ok, buffer = cv2.imencode(".jpg", np.ascontiguousarray(frame[0]), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buffer.tobytes()).decode()


def decode_frame(data: str) -> np.ndarray:
    """The inverse of :func:`encode_frame`: base64 JPEG to a ``(1, H, W, C)`` uint8 frame."""
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("not a JPEG image")
    return frame[np.newaxis]


class RemotePredictor:
    """A ``Predictor`` for ``ModelWorker`` that runs the model on a remote policy server.

    Raises on load when the server has a different model loaded than ``export_dir``,
    so the robot is never driven by another model than the one selected.
    """

    def __init__(self, export_dir: Path, url: str, *, timeout_s: float = 10.0, timing_log: str | None = None) -> None:
        started = time.perf_counter()
        # The container's HTTP(S)_PROXY would send tunnel traffic to the corporate proxy.
        self._client = httpx.Client(base_url=url, timeout=timeout_s, trust_env=False)
        info = self._client.get("/info").raise_for_status().json()
        local = weights_fingerprint(export_dir)
        if info["fingerprint"] != local:
            raise RuntimeError(
                f"The remote server at {url} has another model loaded ({info['export_dir']}, "
                f"fingerprint {info['fingerprint'][:12]}) than {export_dir} (fingerprint {local[:12]})"
            )
        logger.info("Remote inference on {} ({} {}) at {}", info["host"], info["backend"], info["device"], url)
        self._timing_log = Path(timing_log) if timing_log else None
        if self._timing_log is not None:
            new = not self._timing_log.exists()
            with self._timing_log.open("a") as f:
                if new:
                    f.write(_TIMING_COLUMNS)
                connect_ms = (time.perf_counter() - started) * 1000
                f.write(f"# session {time.strftime('%Y-%m-%dT%H:%M:%S')} connect_ms={connect_ms:.1f}\n")
        self._calls = 0
        self._previous_call: float | None = None

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        start = time.perf_counter()
        images = {name: encode_frame(value) for name, value in observation.items() if name.startswith("images")}
        task = observation.get("task")
        payload = {
            "state": np.asarray(observation["state"], dtype=np.float32).tolist(),
            "images": images,
            "task": task[0] if task else None,
        }
        encoded = time.perf_counter()
        response = self._client.post("/predict", json=payload).raise_for_status().json()
        self._record_timing(start, encoded, time.perf_counter(), response.get("infer_ms", float("nan")), images)
        return np.asarray(response["action"], dtype=np.float32)

    def reset(self) -> None:
        """The remote model keeps no state between chunks."""

    def _record_timing(self, start: float, encoded: float, done: float, remote_ms: float, images: dict) -> None:
        self._calls += 1
        since_previous = (start - self._previous_call) * 1000 if self._previous_call is not None else float("nan")
        self._previous_call = start
        if self._timing_log is None:
            return
        request_ms = (done - encoded) * 1000
        payload_kb = sum(len(image) for image in images.values()) / 1024
        with self._timing_log.open("a") as f:
            f.write(
                f"{time.time():.3f},{self._calls},{(encoded - start) * 1000:.1f},{request_ms:.1f},{remote_ms:.1f},"
                f"{request_ms - remote_ms:.1f},{(done - start) * 1000:.1f},{since_previous:.1f},{payload_kb:.0f}\n"
            )
