from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from runtime_session.store import ACTION, STATE
from schemas.runtime_session import ModelControlConfig, MonitorConfig, SafetyConfig
from tests.workers.fake_model import FakeModelWorker

from .fixtures import make_environment
from .test_model import _model, _until


class _FakeVLM:
    """Answers chat completions like vLLM constrained to one word, with ``answer`` at ``confidence``."""

    def __init__(self) -> None:
        self.answer, self.confidence, self.failing = "SAFE", 0.95, False
        self.requests: list[dict] = []
        vlm = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                vlm.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if vlm.failing:
                    self.send_response(503)
                    self.end_headers()
                    return
                body = {
                    "choices": [
                        {
                            "message": {"content": vlm.answer},
                            "logprobs": {"content": [{"logprob": math.log(vlm.confidence)}]},
                        }
                    ]
                }
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args: object) -> None:
                pass

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self._http.server_address[1]}/v1"
        threading.Thread(target=self._http.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._http.shutdown()
        self._http.server_close()


@pytest.fixture
def vlm():
    fake = _FakeVLM()
    yield fake
    fake.close()


@pytest.fixture(autouse=True)
def own_storage(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))


@pytest.fixture
def fake_model_worker():
    with patch("runtime_session.model.ModelWorker", FakeModelWorker):
        yield


def _monitor(vlm: _FakeVLM, **overrides: object) -> MonitorConfig:
    fields: dict[str, object] = {"enabled": True, "endpoint": vlm.endpoint, "check_hz": 10, "fail_safe_after_s": 0.5}
    return MonitorConfig.model_validate(fields | overrides)


async def _session(new_session, tmp_path, vlm, *, drive: bool = True, home_pose=None, **monitor: object):
    session = await new_session()
    await session.load_environment(make_environment())
    await _until(lambda: session.store.read(STATE) is not None)
    await session.set_safety(SafetyConfig(monitor=_monitor(vlm, **monitor), home_pose=home_pose, home_duration_s=3))
    if drive:
        await session.load_model(_model(tmp_path), "torch", "cpu")
        await session.set_control(ModelControlConfig(hz=50, grace_s=0))
    return session


async def test_safe_verdicts_let_the_model_drive(new_session, fake_cameras, fake_model_worker, tmp_path, vlm) -> None:
    session = await _session(new_session, tmp_path, vlm)
    since = len(vlm.requests)

    await _until(lambda: len(vlm.requests) >= since + 3)

    monitor = session.state().safety.monitor
    assert (monitor.health, monitor.guarding, monitor.last_verdict.answer) == ("ok", True, "SAFE")
    assert session.state().control is not None and session.state().safety.state == "clear"
    request = vlm.requests[-1]
    assert request["structured_outputs"] == {"choice": ["SAFE", "UNSAFE_HAND", "UNSAFE_OBJECT"]}
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    content = request["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image_url", "text"]  # one camera, then the prompt
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[-1]["text"].endswith("Answer with exactly one word: SAFE, UNSAFE_HAND or UNSAFE_OBJECT.")


async def test_a_hand_stops_the_model_and_its_frames_are_kept(
    new_session, fake_cameras, fake_model_worker, tmp_path, vlm
) -> None:
    session = await _session(new_session, tmp_path, vlm)
    vlm.answer = "UNSAFE_HAND"

    await _until(lambda: session.state().safety.latched_by is not None)

    cause = session.state().safety.latched_by
    assert (cause.source, cause.hazard, cause.stopped_control) == ("monitor", "hand_or_person", "model")
    assert "hand or person (95%)" in cause.reason
    assert session.state().control is None
    (camera,) = cause.frames
    frame = session.safety_frame(cause.id, camera)
    assert frame is not None and frame.read_bytes()[:2] == b"\xff\xd8"  # a JPEG


async def test_an_unsure_unsafe_answer_does_not_stop(
    new_session, fake_cameras, fake_model_worker, tmp_path, vlm
) -> None:
    session = await _session(new_session, tmp_path, vlm)
    vlm.answer, vlm.confidence = "UNSAFE_OBJECT", 0.45
    since = len(vlm.requests)

    await _until(lambda: len(vlm.requests) >= since + 5)

    assert session.state().safety.latched_by is None
    assert session.state().safety.monitor.last_verdict.answer == "UNSAFE_OBJECT"


async def test_consecutive_unsafe_answers_can_be_required(
    new_session, fake_cameras, fake_model_worker, tmp_path, vlm
) -> None:
    session = await _session(new_session, tmp_path, vlm, unsafe_consecutive=3, check_hz=2)
    vlm.answer = "UNSAFE_OBJECT"
    await _until(lambda: len(vlm.requests) >= 1)
    first = len(vlm.requests)
    vlm.answer = "SAFE"  # one unsafe answer, then safe again: the streak resets
    await _until(lambda: len(vlm.requests) >= first + 2)
    assert session.state().safety.latched_by is None

    vlm.answer = "UNSAFE_OBJECT"
    await _until(lambda: session.state().safety.latched_by is not None)
    assert session.state().safety.latched_by.hazard == "unknown_object"


async def test_a_hazard_while_idle_is_reported_but_stops_nothing(new_session, fake_cameras, tmp_path, vlm) -> None:
    session = await _session(new_session, tmp_path, vlm, drive=False)
    vlm.answer = "UNSAFE_HAND"
    since = len(vlm.requests)

    await _until(lambda: len(vlm.requests) >= since + 5)

    safety = session.state().safety
    assert safety.latched_by is None and safety.monitor.guarding is False
    assert safety.monitor.last_verdict.answer == "UNSAFE_HAND"


async def test_an_unavailable_monitor_stops_the_model(
    new_session, fake_cameras, fake_model_worker, tmp_path, vlm
) -> None:
    session = await _session(new_session, tmp_path, vlm)
    await _until(lambda: session.state().safety.monitor.health == "ok")
    vlm.failing = True

    await _until(lambda: session.state().safety.latched_by is not None)

    assert "unavailable" in session.state().safety.latched_by.reason
    assert session.state().safety.monitor.health == "down"
    kinds = [event.kind for event in session.safety_events()]
    assert kinds[:2] == ["stop", "monitor_fault"]
    assert kinds.count("stop") == 1  # one stop per outage, not one per failed check


async def test_a_hazard_while_moving_home_freezes_the_follower(
    new_session, fake_cameras, fake_model_worker, tmp_path, vlm
) -> None:
    far = [-80.0, -80.0]
    session = await _session(new_session, tmp_path, vlm, home_pose=far)
    vlm.answer = "UNSAFE_HAND"
    await _until(lambda: session.state().safety.state == "homing")

    # Still a hand: the next stop holds instead of finishing the move.
    await _until(lambda: session.safety_status().counters.get("stop", 0) >= 2)

    assert session.state().safety.state == "latched"
    assert session.store.read(ACTION).timestamp <= time.monotonic()
    assert session.safety_events(stops_only=True)[0].motion == "hold"


async def test_setting_the_monitor_keeps_the_home_pose(new_session, fake_cameras, tmp_path, vlm) -> None:
    session = await _session(new_session, tmp_path, vlm, drive=False, home_pose=[1.0, 2.0])

    status = await session.set_safety(SafetyConfig(monitor=_monitor(vlm, enabled=False)))

    assert status.config.home_pose == [1.0, 2.0]
    assert status.config.monitor.enabled is False
    await _until(lambda: session.state().safety.monitor.health == "off")
