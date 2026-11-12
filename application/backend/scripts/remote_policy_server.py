"""Serve one exported policy over HTTP for ``workers.remote_predictor.RemotePredictor``.

Run it inside the Studio image on the machine with the accelerator, bound to
localhost, and reach it through an SSH tunnel::

    python remote_policy_server.py --export-dir /models/exports/openvino --backend openvino --device GPU

``GET /info`` names the loaded export and its weights fingerprint; ``POST /predict``
takes ``{"state": [[...]], "images": {name: base64 JPEG}, "task": str | null}`` and
returns ``{"action": [[...]], "infer_ms": float}``.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from workers.remote_predictor import decode_frame, weights_fingerprint


def main() -> None:
    """Load the export, then serve it until killed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--backend", default="openvino")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()

    from physicalai.inference import InferenceModel

    started = time.perf_counter()
    model = InferenceModel(args.export_dir, backend=args.backend, device=args.device)
    info = {
        "host": socket.gethostname(),
        "export_dir": str(args.export_dir),
        "backend": args.backend,
        "device": args.device,
        "fingerprint": weights_fingerprint(args.export_dir),
    }
    print(f"loaded in {time.perf_counter() - started:.1f}s: {info}", flush=True)
    # One inference at a time: a policy is not re-entrant, and a session has one request in flight.
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._send(200, info) if self.path == "/info" else self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/predict":
                self._send(404, {"error": "not found"})
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                observation = {"state": np.asarray(request["state"], dtype=np.float32)}
                observation.update({name: decode_frame(image) for name, image in request["images"].items()})
                if request.get("task"):
                    observation["task"] = [request["task"]]
                with lock:
                    start = time.perf_counter()
                    action = model.predict_action_chunk(observation)
                    infer_ms = (time.perf_counter() - start) * 1000
                self._send(200, {"action": np.asarray(action).tolist(), "infer_ms": infer_ms})
            except Exception as error:  # every failure goes back to the client
                self._send(500, {"error": f"{type(error).__name__}: {error}"})

        def log_message(self, *_args: object) -> None:
            pass

    print(f"serving on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
