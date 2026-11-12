"""A vision-language model watching a session's cameras, stopping the control when it sees a hazard.

The monitor runs as a task on the session's event loop while the session is open.
At ``check_hz`` it takes the newest frame of each camera from the store, shrinks and
encodes them (on a thread), and asks the model for a one-word verdict through the
OpenAI chat API: ``SAFE``, ``UNSAFE_HAND`` or ``UNSAFE_OBJECT``, constrained by the
server so it cannot answer anything else. Its confidence is the probability of that
answer's first token, from ``logprobs``. One word instead of JSON with a reason cuts a
check from about 2.3 s to 0.5 s on a 27B model, where each output token costs ~35 ms.

It only acts while it *guards*: a control a stop ends (``stop_controls``) is driving,
or the follower is moving home. Then an unsafe answer that is confident enough
``unsafe_consecutive`` times in a row stops the session, saving the frames it judged;
an unsafe answer during a move home freezes the follower where it is; and no
successful check for ``fail_safe_after_s`` (endpoint down, timeouts, a frozen camera)
stops it too. Otherwise it only reports verdicts, so a hand setting up the bench does
not latch an idle session.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import math
import time
from typing import TYPE_CHECKING
from uuid import uuid4

import cv2
import httpx
from loguru import logger

from schemas.runtime_session import MonitorState, SafetyEvent, SafetyStopRequest
from utils.jpeg import encode_jpeg_rgb

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    import numpy as np

    from runtime_session.store import Store
    from schemas.runtime_session import MonitorConfig

ANSWERS = ("SAFE", "UNSAFE_HAND", "UNSAFE_OBJECT")
_HAZARDS = {"SAFE": "none", "UNSAFE_HAND": "hand_or_person", "UNSAFE_OBJECT": "unknown_object"}

GENERIC_PROMPT = (
    "You are the safety monitor for a robot arm work cell. The images are the cell's cameras. "
    "ALLOWED in the work area: the robot arm and its gripper with its exposed electronics and wires, "
    "the objects the robot is working with, and their containers. "
    "UNSAFE: any human hand, finger, arm, sleeve or person in the work area, or any object there that "
    "is not part of the robot's task (a cup, phone, tool, pen, bottle, box). Ignore everything beyond "
    "the work surface."
)
ANSWER_INSTRUCTION = " Answer with exactly one word: SAFE, UNSAFE_HAND or UNSAFE_OBJECT."

_IDLE_S = 0.5
"""How often a monitor that is off or has nothing to watch looks again."""


class CheckFailedError(Exception):
    """A check that gave no verdict: the endpoint, the answer, or the frames were unusable."""


class Hooks:
    """What the monitor needs of its session; the session supplies them."""

    def __init__(
        self,
        *,
        config: Callable[[], MonitorConfig],
        store: Callable[[], Store | None],
        guarding: Callable[[], bool],
        homing: Callable[[], bool],
        stop: Callable[[SafetyStopRequest, dict[str, bytes]], Awaitable[None]],
        report: Callable[[SafetyEvent], None],
    ) -> None:
        self.config = config
        self.store = store
        self.guarding = guarding
        self.homing = homing
        self.stop = stop
        """Stop the session for ``request``, keeping the JPEGs (by camera) the monitor judged."""
        self.report = report
        """Publish a verdict or a fault; the session keeps the latest in its state."""


class SafetyMonitor:
    def __init__(self, hooks: Hooks, *, frames_dir: Path) -> None:
        self._hooks = hooks
        self.frames_dir = frames_dir
        self.state = MonitorState()
        self._client: httpx.AsyncClient | None = None
        self._endpoint: str | None = None
        self._unsafe_streak = 0
        self._last_ok: float | None = None
        """``time.monotonic()`` of the last check that gave a verdict, while guarding."""
        self._faulted = False
        """A fail-safe stop was made for the current outage; the next one waits for a verdict first."""

    async def run(self) -> None:
        """Watch until cancelled."""
        try:
            while True:
                config = self._hooks.config()
                store = self._hooks.store()
                if not config.enabled or not config.endpoint or store is None:
                    self._idle()
                    await asyncio.sleep(_IDLE_S)
                    continue
                started = time.monotonic()
                await self._check_once(config, store)
                await asyncio.sleep(max(0.0, 1 / config.check_hz - (time.monotonic() - started)))
        finally:
            if self._client is not None:
                await self._client.aclose()

    def _idle(self) -> None:
        self.state = MonitorState(health="off", last_verdict=self.state.last_verdict)
        self._unsafe_streak = 0
        self._last_ok = None
        self._faulted = False

    async def _check_once(self, config: MonitorConfig, store: Store) -> None:
        guarding = self._hooks.guarding()
        if not guarding:
            self._last_ok = None  # an outage only counts while there is something to stop
            self._faulted = False
        elif self._last_ok is None:
            self._last_ok = time.monotonic()
        try:
            jpegs = await self._frames(config, store)
            answer, confidence, latency_ms = await self._ask(config, list(jpegs.values()))
        except CheckFailedError as error:
            await self._failed(config, str(error), guarding=guarding)
            return
        self._last_ok = time.monotonic()
        self._faulted = False
        verdict = SafetyEvent(
            id=uuid4().hex,
            at=datetime.datetime.now(tz=datetime.UTC),
            kind="verdict",
            source="monitor",
            hazard=_HAZARDS[answer],  # type: ignore[arg-type]
            confidence=confidence,
            answer=answer,
            latency_ms=latency_ms,
        )
        self.state = MonitorState(health="ok", guarding=guarding, last_verdict=verdict)
        self._hooks.report(verdict)
        await self._decide(config, verdict, jpegs, guarding=guarding)

    async def _decide(
        self, config: MonitorConfig, verdict: SafetyEvent, jpegs: dict[str, bytes], *, guarding: bool
    ) -> None:
        unsafe = verdict.answer != "SAFE" and (verdict.confidence or 0.0) >= config.min_confidence
        self._unsafe_streak = self._unsafe_streak + 1 if unsafe else 0
        if not guarding or self._unsafe_streak < config.unsafe_consecutive:
            return
        self._unsafe_streak = 0
        homing = self._hooks.homing()
        what = "a hand or person" if verdict.hazard == "hand_or_person" else "an unknown object"
        request = SafetyStopRequest(
            reason=f"The safety monitor saw {what} ({verdict.confidence:.0%})"
            + (" while the arm moved home" if homing else ""),
            source="monitor",
            hazard=verdict.hazard or "other",
            confidence=verdict.confidence,
            hold=homing,  # moving home into a hazard: freeze where it is
        )
        await self._hooks.stop(request, jpegs)

    async def _failed(self, config: MonitorConfig, error: str, *, guarding: bool) -> None:
        self._unsafe_streak = 0
        down = guarding and self._last_ok is not None and time.monotonic() - self._last_ok >= config.fail_safe_after_s
        self.state = MonitorState(
            health="down" if down else "degraded", guarding=guarding, last_verdict=self.state.last_verdict, error=error
        )
        if not down or self._faulted:
            return
        self._faulted = True
        logger.warning("Safety monitor has had no verdict for {:.1f} s: {}", config.fail_safe_after_s, error)
        fault = SafetyEvent(
            id=uuid4().hex,
            at=datetime.datetime.now(tz=datetime.UTC),
            kind="monitor_fault",
            source="monitor",
            reason=error,
        )
        self._hooks.report(fault)
        await self._hooks.stop(
            SafetyStopRequest(
                reason=f"The safety monitor is unavailable ({error})",
                source="monitor",
                hazard="other",
                hold=self._hooks.homing(),
            ),
            {},
        )

    async def _frames(self, config: MonitorConfig, store: Store) -> dict[str, bytes]:
        cameras = config.cameras or list(store.spec.observations)
        frames: dict[str, np.ndarray] = {}
        now = time.monotonic()
        for camera in cameras:
            if camera not in store.spec.observations:
                raise CheckFailedError(f"camera {camera!r} is not in the session")
            sample = store.read_observation(camera)
            if sample is None:
                raise CheckFailedError(f"camera {camera!r} has no frame")
            if now - sample.timestamp > config.max_frame_age_s:
                raise CheckFailedError(f"camera {camera!r} is frozen ({now - sample.timestamp:.1f} s old)")
            frames[camera] = sample.value
        if not frames:
            raise CheckFailedError("the session has no cameras")
        encoded = await asyncio.gather(
            *(asyncio.to_thread(_encode, frame, config.image_max_side) for frame in frames.values())
        )
        return dict(zip(frames, encoded, strict=True))

    async def _ask(self, config: MonitorConfig, jpegs: list[bytes]) -> tuple[str, float, float]:
        content: list[dict] = [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}
            for jpeg in jpegs
        ]
        content.append({"type": "text", "text": (config.prompt or GENERIC_PROMPT) + ANSWER_INSTRUCTION})
        body = {
            "model": config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 6,
            "logprobs": True,
            "top_logprobs": 5,
            # Qwen3.x thinks out loud by default; a verdict must be one word.
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"choice": list(ANSWERS)},
        }
        started = time.perf_counter()
        try:
            response = await self._client_for(config).post("/chat/completions", json=body, timeout=config.timeout_s)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            answer = choice["message"]["content"].strip()
            first = choice["logprobs"]["content"][0]["logprob"]
        except httpx.HTTPError as error:
            raise CheckFailedError(
                f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
            ) from error
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise CheckFailedError(f"unreadable answer: {error}") from error
        if answer not in ANSWERS:
            raise CheckFailedError(f"unexpected answer {answer[:40]!r}")
        return answer, round(math.exp(first), 3), (time.perf_counter() - started) * 1000

    def _client_for(self, config: MonitorConfig) -> httpx.AsyncClient:
        if self._client is None or self._endpoint != config.endpoint:
            if self._client is not None:
                asyncio.get_running_loop().create_task(self._client.aclose())
            # The container's HTTP(S)_PROXY would send tunnel traffic to the corporate proxy.
            self._client = httpx.AsyncClient(base_url=str(config.endpoint).rstrip("/"), trust_env=False)
            self._endpoint = config.endpoint
        return self._client

    def save_frames(self, event_id: str, jpegs: dict[str, bytes]) -> list[str]:
        """Keep the JPEGs a stop was judged on, for review and prompt tuning; returns their cameras."""
        if not jpegs:
            return []
        folder = self.frames_dir / event_id
        folder.mkdir(parents=True, exist_ok=True)
        for camera, jpeg in jpegs.items():
            (folder / f"{_safe_name(camera)}.jpg").write_bytes(jpeg)
        return list(jpegs)

    def frame_path(self, event_id: str, camera: str) -> Path | None:
        path = self.frames_dir / _safe_name(event_id) / f"{_safe_name(camera)}.jpg"
        return path if path.is_file() else None


def _encode(frame: np.ndarray, max_side: int) -> bytes:
    """An RGB frame shrunk to ``max_side`` and JPEG-encoded: fewer image tokens, a faster check."""
    scale = max_side / max(frame.shape[:2])
    if scale < 1:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return encode_jpeg_rgb(frame, quality=90)


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
